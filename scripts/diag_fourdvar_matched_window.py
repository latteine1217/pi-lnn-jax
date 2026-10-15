"""diag_fourdvar_matched_window.py — 在 4D-Var 同一個時間窗、同一個指標下重算 learned model 的頻帶誤差。

What:
    `fourdvar_identifiability.py` 的 4D-Var 只同化 DNS（time_stride=2）的前 nf 幀，
    並以 `diag_identifiability.velocity_band_rel_l2` 在**那 nf 幀上**計分。本腳本對
    learned model 的 `final_eval/fields.npz`，以同一函式、同一 DNS 切片、同一 K，算：
      - 每個 nf 窗（`--windows`）的 low / mid 速度相對 L2（逐幀、再取窗內時間平均）；
      - 全窗 mid（對照已發表的 57.73%，確認 band 分解與時間軸無誤）；
      - 最大窗與全窗的譜同調 γ（整數 shell 頻帶 low k≤5 / mid 5<k≤16 / high k>16，
        DNS 能量加權，t>0，逐幀加權後取時間平均）；
      - 逐幀 low band 誤差（看誤差在時間上怎麼分布）；
      - `metrics.json` 的 `band_rel_err_low`（k_η 分帶的頻譜能量誤差），標明它是另一個量。
    多個 run 時另輸出 mean ± sd（ddof=1）。

Why:
    4D-Var 的數字（如 nf=20 的 low 4.75%、mid 16.63%）只涵蓋 t≤0.95 s，而先前拿來比較的
    learned 值是別的窗或別的指標：mid 57.73% 是全 101 幀，low 4.70% 是 `band_rel_err_low`
    的全窗值。`fourdvar_identifiability.py` 寫死 `E_model = 0.5773`、`diag_hybrid_refine.py`
    的 `--repro-target` 也是 0.5773，所以兩支腳本印出的 gap 都是全窗對窗內。本腳本不改
    任何 metric 定義，只把 learned 那一欄搬到 4D-Var 的窗上，讓兩欄可以直接相減。

Fail-fast:
    - fields.npz 的 `dns_path` 必須就是 `--fourdvar-dns`；
    - `load_fields_and_dns` 對齊出的 DNS 與 4D-Var 讀法（time_stride=2）在窗內必須逐位元相同，
      時間軸也必須相同；
    - 由 fields 重算的全場 uv 誤差必須對上 `metrics.json` 的 `uv_rel_err`（差 < 1e-6）。

Usage（本機可跑，只讀既有產物）:
    uv run python scripts/diag_fourdvar_matched_window.py \\
        --fields artifacts/kolmogorov/main5_s{42,1,2,3,4}/final_eval/fields.npz \\
        --fourdvar-dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --K 100 --windows 3,6,10,15,20 --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import load_dns_from_path
from pi_lnn_jax.metric_artifact import compute_energy_spectrum, spectral_coherence
from pi_lnn_jax.result_table import read_metric_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_identifiability import load_fields_and_dns, velocity_band_rel_l2

# fourdvar_identifiability.run_4dvar 以 time_stride=2 讀 DNS（該檔第 135 行）；窗的定義由它決定。
FOURDVAR_TIME_STRIDE = 2
# 整數 shell 頻帶（shell index 為 [k-0.5, k+0.5) 的 round-bin，見 metric_artifact.compute_energy_spectrum）
GAMMA_BANDS = {"low": (-1, 5), "mid": (5, 16), "high": (16, 10**9)}
UV_REL_ERR = "velocity.vector.rel_l2.v1"
BAND_REL_ERR_LOW_KETA = "spectrum.energy.rel_l2.low.v1"


def gamma_bands(u_p, v_p, u_d, v_d, frames) -> dict[str, float]:
    """逐幀以 DNS 能量加權的 band γ，再對幀取平均。"""
    per_frame: dict[str, list[float]] = {b: [] for b in GAMMA_BANDS}
    for f in frames:
        k_arr, e_dns = compute_energy_spectrum(u_d[f], v_d[f])
        _, g = spectral_coherence(u_p[f], v_p[f], u_d[f], v_d[f])
        for b, (lo, hi) in GAMMA_BANDS.items():
            sel = (k_arr > lo) & (k_arr <= hi) & np.isfinite(g)
            per_frame[b].append(float(np.sum(g[sel] * e_dns[sel]) / np.sum(e_dns[sel])))
    return {b: float(np.mean(v)) for b, v in per_frame.items()}


def per_frame_band(u_p, v_p, u_d, v_d, K: int, band: str) -> list[float]:
    """逐幀的 band 速度相對 L2（velocity_band_rel_l2 對單幀呼叫，定義不變）。"""
    return [velocity_band_rel_l2(u_p[f:f + 1], v_p[f:f + 1], u_d[f:f + 1], v_d[f:f + 1], K)
            ["bands"][band]["uv_rel"] for f in range(u_p.shape[0])]


def evaluate_run(fields_path: Path, dns_path: str, dns_ref, K: int, windows: list[int]) -> dict:
    """單一 run：4D-Var 窗與全窗的 band 誤差、γ、逐幀 low。dns_ref = 4D-Var 讀法的 (u, v, t)。"""
    u4, v4, t4 = dns_ref
    f = np.load(fields_path, allow_pickle=True)
    if str(f["dns_path"]) != dns_path:
        raise ValueError(f"{fields_path}: dns_path={f['dns_path']} ≠ --fourdvar-dns {dns_path}")
    u_p, v_p, u_d, v_d = load_fields_and_dns(fields_path)
    t_f = np.asarray(f["t"], np.float64)
    n_max = max(windows)
    if n_max > len(t_f) or n_max > len(t4):
        raise ValueError(f"{fields_path}: 窗 {n_max} 幀超過 fields {len(t_f)} / DNS {len(t4)} 幀")
    if not (np.array_equal(u_d[:n_max], u4[:n_max]) and np.array_equal(v_d[:n_max], v4[:n_max])):
        raise ValueError(f"{fields_path}: load_fields_and_dns 的 DNS 與 4D-Var 讀法在窗內不同")
    if not np.allclose(t_f[:n_max], t4[:n_max], atol=1e-5):
        raise ValueError(f"{fields_path}: fields 時間軸前 {n_max} 幀 ≠ 4D-Var 窗")

    metrics_path = fields_path.with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text())
    full = velocity_band_rel_l2(u_p, v_p, u_d, v_d, K)
    uv_metrics = read_metric_value(metrics, UV_REL_ERR)
    if abs(full["full_uv_rel"] - uv_metrics) > 1e-6:
        raise ValueError(f"{fields_path}: 重算 uv {full['full_uv_rel']} ≠ metrics.json {uv_metrics}")

    by_window = {}
    for n in windows:
        b = velocity_band_rel_l2(u_p[:n], v_p[:n], u_d[:n], v_d[:n], K)["bands"]
        by_window[str(n)] = {"t_end": float(t_f[n - 1]), "low": b["low"]["uv_rel"], "mid": b["mid"]["uv_rel"]}
    return {
        "fields": str(fields_path),
        "uv_rel_err": uv_metrics,
        "k_sensor_edge": full["k_sensor_edge"],
        "window": by_window,
        "full": {"n_frames": len(t_f), "t_end": float(t_f[-1]),
                 "low": full["bands"]["low"]["uv_rel"], "mid": full["bands"]["mid"]["uv_rel"]},
        f"gamma_win{n_max}_t>0": gamma_bands(u_p, v_p, u_d, v_d, range(1, n_max)),
        "gamma_full_t>0": gamma_bands(u_p, v_p, u_d, v_d, range(1, len(t_f))),
        "t": t_f.tolist(),
        "low_per_frame": per_frame_band(u_p, v_p, u_d, v_d, K, "low"),
        "band_rel_err_low_keta_full": read_metric_value(metrics, BAND_REL_ERR_LOW_KETA),
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fields", type=Path, nargs="+", required=True, help="final_eval/fields.npz（旁邊須有 metrics.json）")
    ap.add_argument("--fourdvar-dns", required=True, help="與 fourdvar_identifiability --dns 相同的 DNS 路徑")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--windows", required=True, help="4D-Var 的 nf 窗，逗號分隔，如 3,6,10,15,20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    windows = sorted({int(x) for x in args.windows.split(",")})
    u4, v4, t4 = load_dns_from_path(args.fourdvar_dns, time_stride=FOURDVAR_TIME_STRIDE)
    dns_ref = (np.asarray(u4, np.float64), np.asarray(v4, np.float64), np.asarray(t4, np.float64))

    runs = []
    for fp in args.fields:
        r = evaluate_run(fp, args.fourdvar_dns, dns_ref, args.K, windows)
        runs.append(r)
        t = np.asarray(r["t"]); low = np.asarray(r["low_per_frame"])
        n_max = str(max(windows))
        print(f"=== {fp.parent.parent.name} ===")
        print(f"  uv_rel_err={100*r['uv_rel_err']:.3f}%  全窗 mid={100*r['full']['mid']:.2f}% "
              f"low={100*r['full']['low']:.2f}%  band_rel_err_low(k_eta,全窗)={100*r['band_rel_err_low_keta_full']:.2f}%")
        for n, w in r["window"].items():
            print(f"  窗 nf={n:>3} (t≤{w['t_end']:.2f}): low={100*w['low']:.2f}%  mid={100*w['mid']:.2f}%")
        g = r[f"gamma_win{n_max}_t>0"]
        print(f"  γ 窗 nf={n_max} (t>0): low={g['low']:.3f} mid={g['mid']:.3f} high={g['high']:.3f}")
        print(f"  逐幀 low: t=0 {100*low[0]:.1f}%  t={t[int(n_max)-1]:.2f} {100*low[int(n_max)-1]:.1f}%  "
              f"t≥3 平均 {100*low[t >= 3.0 - 1e-9].mean():.1f}%")

    summary: dict[str, dict[str, float]] = {}
    keys = [("full", "mid"), ("full", "low")]
    keys += [("window", str(n), b) for n in windows for b in ("low", "mid")]
    keys += [(f"gamma_win{max(windows)}_t>0", b) for b in GAMMA_BANDS]
    keys += [("gamma_full_t>0", b) for b in GAMMA_BANDS]
    keys += [("band_rel_err_low_keta_full",), ("uv_rel_err",)]
    print(f"\n=== mean ± sd (ddof=1), n={len(runs)} ===")
    for path in keys:
        vals = []
        for r in runs:
            x = r
            for k in path:
                x = x[k]
            vals.append(x)
        key = "/".join(path)
        summary[key] = _mean_sd(vals)
        print(f"  {key:36s} {summary[key]['mean']:.5f} ± {summary[key]['sd']:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"K": args.K, "windows": windows, "fourdvar_dns": args.fourdvar_dns,
                                    "fourdvar_time_stride": FOURDVAR_TIME_STRIDE,
                                    "runs": runs, "summary": summary}, indent=2))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
