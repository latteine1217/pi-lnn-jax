#!/usr/bin/env python3
"""驗證「延長版 DNS」與既有 DNS 是不是**同一條軌跡**。

What:
    比對延長版（例：T=10）與參考版（例：T=5）在共同時間窗上的逐幀相對 L2 誤差，
    輸出 PASS/FAIL 判定與一份 JSON 判定檔（落在延長版旁邊，供 eval 產物追溯）。

Why:
    時間外推實驗的 sensor 來自參考版那條軌跡；延長版若不是同一條，連 t≤5 段的
    評估都沒有意義。Kolmogorov Re=10⁴ 是混沌流，換機器／換 backend 的 round-off
    會被指數放大——所以這件事**必須量**，不能靠「參數一樣所以應該一樣」。

    誤差隨時間的成長曲線本身就是證據：若在 t=5 已經 O(1)，代表兩條軌跡已完全
    去相關，延長版只能整包當成新的真實來源（sensor 值需一併重抽）。

判準（--rtol，預設 1e-6）：
    max_t rel_L2(u) 與 max_t rel_L2(v) 皆 < rtol → PASS，延長版可直接當 eval 真值。
    否則 FAIL：印出誤差跨越 1e-8 / 1e-4 / 1e-2 的時刻，讓放大速率可讀。

Usage:
    uv run python scripts/verify_dns_extension.py \\
      --extended data/dns/kolmogorov_dns_..._T10_..._ds4.npy \\
      --reference data/dns/kolmogorov_dns_..._T5_..._ds4.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: 這些 config 鍵不同 = 兩份資料在數值設定上就不是同一個問題，比誤差沒有意義。
#: 不含 T_end（本來就不同）與 backend（差異正是要量的東西）。
CRITICAL_CONFIG_KEYS = (
    "N", "L", "nu", "A", "k_f", "dt", "save_interval", "integrator",
    "dealias_mode", "seed", "ic_mode", "downsample_stride", "source_N",
)

#: 舊資料檔的鍵名。生成器現在同時寫 `ic_mode` 與 legacy alias `init_mode`，但
#: 2026-04 產的參考檔只有後者——不查別名就會把「同一個設定」判成不同，讓一次
#: 28 分鐘的生成白跑。別名是同一個量的兩個名字，查它不等於放寬判準。
CONFIG_KEY_ALIASES = {"ic_mode": ("init_mode",)}

CROSSING_LEVELS = (1e-8, 1e-4, 1e-2)


def config_value(cfg: dict, key: str):
    """取 config 值，`key` 不在時依序試 legacy 別名。全都沒有才回 None。"""
    if key in cfg:
        return cfg[key]
    for alias in CONFIG_KEY_ALIASES.get(key, ()):
        if alias in cfg:
            return cfg[alias]
    return None


def _rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐幀相對 L2：‖a−b‖ / ‖b‖，形狀 [T]。"""
    diff = np.sqrt(np.sum((a - b) ** 2, axis=(1, 2)))
    ref = np.sqrt(np.sum(b ** 2, axis=(1, 2)))
    if np.any(ref == 0.0):
        raise ValueError("參考場有整幀為零，相對誤差無定義")
    return diff / ref


def compare(extended: dict, reference: dict, rtol: float) -> dict:
    """比對兩份 DNS dict，回傳判定 payload。前提不成立一律 raise，不寬鬆對齊。"""
    cfg_e, cfg_r = extended.get("config", {}), reference.get("config", {})
    mismatched = {
        k: [config_value(cfg_e, k), config_value(cfg_r, k)] for k in CRITICAL_CONFIG_KEYS
        if config_value(cfg_e, k) != config_value(cfg_r, k)
    }
    if mismatched:
        raise ValueError(f"兩份 DNS 的數值設定不同，無法當同一條軌跡比對：{mismatched}")

    t_e = np.asarray(extended["time"], dtype=np.float64)
    t_r = np.asarray(reference["time"], dtype=np.float64)
    n = t_r.shape[0]
    if t_e.shape[0] < n:
        raise ValueError(f"延長版只有 {t_e.shape[0]} 幀，短於參考版的 {n} 幀")
    if not np.allclose(t_e[:n], t_r, atol=1e-12, rtol=0.0):
        raise ValueError("共同時間窗的時間軸不相同（save cadence 或 dt 不一致），拒絕比對")

    err = {c: _rel_l2(np.asarray(extended[c])[:n], np.asarray(reference[c]))
           for c in ("u", "v")}
    worst = {c: float(e.max()) for c, e in err.items()}
    passed = all(v < rtol for v in worst.values())

    combined = np.maximum(err["u"], err["v"])
    crossings = {}
    for level in CROSSING_LEVELS:
        idx = np.argmax(combined > level) if np.any(combined > level) else None
        crossings[f"t_first_exceeds_{level:g}"] = None if idx is None else float(t_r[idx])

    return {
        "verdict": "PASS" if passed else "FAIL",
        "rtol": rtol,
        "shared_window": [float(t_r[0]), float(t_r[-1])],
        "shared_frames": int(n),
        "max_rel_l2": worst,
        "rel_l2_at_window_end": {c: float(e[-1]) for c, e in err.items()},
        "error_growth_crossings": crossings,
        "backend": {"extended": cfg_e.get("backend"), "reference": cfg_r.get("backend")},
        "meaning": (
            "PASS：延長版與參考版是同一條軌跡，可直接當 t>window 的 eval 真值，"
            "既有 sensor 檔不需重抽。FAIL：兩條軌跡已分離，延長版只能整包當成新的"
            "真實來源——sensor 值必須從延長版重抽，且用舊 sensor 訓出的模型不可用它評估。"
        ),
        "curve": {"t": t_r.tolist(), "u": err["u"].tolist(), "v": err["v"].tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extended", required=True, help="延長版 DNS .npy（dict schema）")
    ap.add_argument("--reference", required=True, help="參考版 DNS .npy（sensor 的來源軌跡）")
    ap.add_argument("--rtol", type=float, default=1e-6, help="PASS 門檻（逐幀相對 L2 上界）")
    ap.add_argument("--out", default=None,
                    help="判定 JSON 落點；預設 <extended>.verify.json")
    args = ap.parse_args()

    for path in (args.extended, args.reference):
        if not Path(path).exists():
            raise FileNotFoundError(f"DNS .npy 不存在：{path}")

    ext = np.load(args.extended, allow_pickle=True).item()
    ref = np.load(args.reference, allow_pickle=True).item()
    report = compare(ext, ref, args.rtol)

    out = Path(args.out) if args.out else Path(str(args.extended) + ".verify.json")
    out.write_text(json.dumps(report, indent=2))

    t = np.asarray(report["curve"]["t"])
    eu, ev = np.asarray(report["curve"]["u"]), np.asarray(report["curve"]["v"])
    print("=== DNS extension trajectory check ===")
    print(f"reference  : {args.reference}")
    print(f"extended   : {args.extended}")
    print(f"backend    : extended={report['backend']['extended']} "
          f"reference={report['backend']['reference']}")
    print(f"shared     : {report['shared_frames']} frames, t ∈ {report['shared_window']}")
    print("--- rel L2 vs t (每 20 幀) ---")
    for i in range(0, len(t), 20):
        print(f"  t={t[i]:6.3f}  u={eu[i]:.3e}  v={ev[i]:.3e}")
    print(f"  t={t[-1]:6.3f}  u={eu[-1]:.3e}  v={ev[-1]:.3e}  (window end)")
    print("--- error growth ---")
    for k, v in report["error_growth_crossings"].items():
        print(f"  {k}: {'never' if v is None else f't={v:.3f}'}")
    print(f"max rel L2 : u={report['max_rel_l2']['u']:.3e}  v={report['max_rel_l2']['v']:.3e}"
          f"  (門檻 {args.rtol:g})")
    print(f"VERDICT    : {report['verdict']}")
    print(f"判定檔     : {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
