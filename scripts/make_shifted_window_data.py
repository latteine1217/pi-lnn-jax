"""make_shifted_window_data.py — 從長 DNS 切出 [t0, t0+L] 的資料組（DNS 窗 + 感測檔），時間平移到 [0, L]。

What:
    PoF 審稿追問「換一段時間窗還能不能重建」。訓練與評估流程都以 T=5、t∈[0,5] 定義
    （時間相位錨、curriculum、初期加權），所以把後段窗的時間標籤平移回 [0, L]：
    強迫與 ν 不隨時間變，系統是自治的，平移不改變物理。
      * DNS 窗：u、v、p、omega 取 [t0, t0+L] 的幀，time = t − t0；config 補記來源與平移量。
      * 感測檔：沿用既有 LES-QR 佈點（selected_coordinates 不變），從同一 DNS 窗抽值，
        npz 與既有檔同 schema（time [T]、u/v [K, T]）；json 只改值檔路徑、時間範圍與來源欄。

閘門（不過即中止、不寫任何檔）:
    同一支抽值程式套在 [0, L] 上，必須重現既有感測 npz（相對 L2 < MATCH_TOL）。
    長 DNS 的前段與原 T=5 DNS 已逐幀驗證為同一條軌跡（verify.json：最大相對差 3e-7）。

Usage:
    uv run python scripts/make_shifted_window_data.py --dns-long <T10.npy> --t0 5 --length 5 \
        --sensor-jsons <K100.json> <K200.json> --out-dns <dns_window.npy> \
        --out-sensor-dir <dir> --tag-suffix _win5to10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: 控制窗重抽值與既有 npz 的相對 L2 上限（兩者來自同一條軌跡、同一格點，差異只剩
#: CUDA/CPU 求解器的 ~3e-7 與 float32 存檔）。
MATCH_TOL = 1e-5


def window_indices(t: np.ndarray, t0: float, length: float) -> np.ndarray:
    """[t0, t0+length] 內的幀索引（含端點）；兩端都必須恰為幀時刻。"""
    i0 = int(np.argmin(np.abs(t - t0)))
    i1 = int(np.argmin(np.abs(t - (t0 + length))))
    if abs(t[i0] - t0) > 1e-6 or abs(t[i1] - (t0 + length)) > 1e-6:
        raise ValueError(f"窗 [{t0}, {t0 + length}] 的端點不是 DNS 幀時刻")
    return np.arange(i0, i1 + 1)


def grid_index(coords: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """座標 → 最近格點 (ix, iy)，與 generate_sensors_qrpivot_from_les 的抽值規則相同。"""
    ix = np.argmin(np.abs(coords[:, 0:1] - x[None, :]), axis=1)
    iy = np.argmin(np.abs(coords[:, 1:2] - y[None, :]), axis=1)
    return ix, iy


def extract(u: np.ndarray, v: np.ndarray, ix, iy) -> tuple[np.ndarray, np.ndarray]:
    """[T, x, y] 場 → (u_s, v_s)，各為 [K, T] float32（既有 npz 的 schema）。"""
    return (u[:, ix, iy].T.astype(np.float32), v[:, ix, iy].T.astype(np.float32))


def rel_l2(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns-long", required=True, type=Path)
    ap.add_argument("--t0", required=True, type=float)
    ap.add_argument("--length", required=True, type=float)
    ap.add_argument("--sensor-jsons", required=True, nargs="+", type=Path)
    ap.add_argument("--out-dns", required=True, type=Path)
    ap.add_argument("--out-sensor-dir", required=True, type=Path)
    ap.add_argument("--tag-suffix", required=True)
    a = ap.parse_args()

    d = np.load(a.dns_long, allow_pickle=True).item()
    t = np.asarray(d["time"], np.float64)
    x = np.asarray(d["x"], np.float64); y = np.asarray(d["y"], np.float64)
    ctrl = window_indices(t, 0.0, a.length)
    win = window_indices(t, a.t0, a.length)
    if len(ctrl) != len(win):
        raise ValueError(f"控制窗 {len(ctrl)} 幀、目標窗 {len(win)} 幀，長度不一致")

    # ── 閘門：控制窗重抽值必須重現每一份既有感測 npz ───────────────────────
    plans = []
    for js in a.sensor_jsons:
        meta = json.loads(js.read_text())
        coords = np.asarray(meta["selected_coordinates"], np.float64)
        ix, iy = grid_index(coords, x, y)
        npz_old = js.with_name(js.stem + "_dns_values.npz")
        old = np.load(npz_old)
        cu, cv = extract(d["u"][ctrl], d["v"][ctrl], ix, iy)
        err = max(rel_l2(cu, old["u"]), rel_l2(cv, old["v"]))
        terr = float(np.abs(np.asarray(old["time"], np.float64) - (t[ctrl] - 0.0)).max())
        print(f"[gate] {js.name}: 控制窗重抽 rel={err:.2e}  time |Δ|max={terr:.1e}")
        if err > MATCH_TOL or terr > 1e-6:
            raise SystemExit(f"[FAIL] {js.name} 控制窗重現不了既有感測值——抽值規則或軌跡不符，未寫任何檔")
        plans.append((js, meta, ix, iy))

    # ── 寫 DNS 窗 ──────────────────────────────────────────────────────────
    out = {k: d[k] for k in ("x", "y")}
    for k in ("u", "v", "p", "omega"):
        out[k] = np.asarray(d[k])[win]
    out["time"] = t[win] - a.t0
    cfg = dict(d["config"]) if isinstance(d.get("config"), dict) else {}
    cfg.update({"T_end": a.length, "time_window_source": str(a.dns_long),
                "time_window_t0": a.t0, "time_window_note":
                "frames [t0, t0+L] of the source, time relabeled to [0, L] (autonomous system)"})
    out["config"] = cfg
    out["diagnostics"] = {"note": "not carried over from the source window"}
    a.out_dns.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out_dns, out, allow_pickle=True)
    print(f"[out] DNS window {a.out_dns}  frames={len(win)} t∈[{out['time'][0]:.3f},{out['time'][-1]:.3f}]")

    # ── 寫感測檔 ───────────────────────────────────────────────────────────
    a.out_sensor_dir.mkdir(parents=True, exist_ok=True)
    for js, meta, ix, iy in plans:
        su, sv = extract(d["u"][win], d["v"][win], ix, iy)
        stem = js.stem + a.tag_suffix
        npz_new = a.out_sensor_dir / f"{stem}_dns_values.npz"
        np.savez(npz_new, time=(t[win] - a.t0).astype(np.float32), u=su, v=sv)
        m = dict(meta)
        m.update({"source_file": str(a.out_dns), "dns_values_npz": str(npz_new),
                  "time_range": [0.0, a.length],
                  "time_window_source": str(a.dns_long), "time_window_t0": a.t0})
        (a.out_sensor_dir / f"{stem}.json").write_text(json.dumps(m, indent=2))
        print(f"[out] sensors {stem}.json / _dns_values.npz  K={su.shape[0]} T={su.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
