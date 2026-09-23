#!/usr/bin/env python3
"""這條 DNS 有沒有進入統計穩態？

判準（事前登錄，兩條都要成立）：
    1. **平台**：窗內 enstrophy 的線性趨勢在整個窗上的總變化量小於其標準差
       → |slope|·window < std。趨勢被自身波動蓋過，才叫平台。
    2. **仍在波動**：std/mean > `--min-cv`（預設 0.05）。否則是靜止解或週期振幅
       太小，外推會變得平庸地容易，測試在反方向失效。

Why:
    時間外推的結論目前綁在衰減暫態上。要在「統計穩態窗」重做以證明可推廣，
    前提是那個 attractor 真的是統計穩態——這支就是那道閘門，開在投入
    N=1024 DNS + 四臂重訓（半天）**之前**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np



def steady_verdict(t: np.ndarray, q: np.ndarray, min_cv: float) -> dict:
    """對一條時序下平台/波動判定。t、q 已是窗內切片。"""
    if t.size < 8:
        raise ValueError(f"窗內只有 {t.size} 個點，不足以判斷趨勢")
    slope, _ = np.polyfit(t, q, 1)
    span = float(t[-1] - t[0])
    mean, std = float(q.mean()), float(q.std())
    drift = abs(float(slope)) * span
    plateau = drift < std
    fluctuating = (std / mean) > min_cv if mean > 0 else False
    return {
        "mean": mean, "std": std, "cv": std / mean if mean > 0 else None,
        "slope_per_time": float(slope), "drift_over_window": drift,
        "plateau": bool(plateau), "fluctuating": bool(fluctuating),
        "steady": bool(plateau and fluctuating),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", required=True)
    ap.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"))
    ap.add_argument("--min-cv", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="判定 JSON；預設 <dns>.steady.json")
    args = ap.parse_args()

    obj = np.load(Path(args.dns), allow_pickle=True).item()
    t_all = np.asarray(obj["time"], dtype=float)
    dg = obj.get("diagnostics")
    if not dg or "enstrophy" not in dg:
        raise KeyError(f"{args.dns} 無 diagnostics.enstrophy —— 無法判定，拒絕用場重算充數")
    t0, t1 = args.window
    m = (t_all >= t0) & (t_all <= t1)
    if not m.any():
        raise ValueError(f"時窗 [{t0},{t1}] 不在 DNS 的 t ∈ [{t_all[0]},{t_all[-1]}] 內")

    report = {"dns": str(args.dns), "window": [t0, t1], "min_cv": args.min_cv,
              "n_frames_in_window": int(m.sum())}
    for name in ("enstrophy", "kinetic_energy"):
        q = np.asarray(dg[name], dtype=float)
        report[name] = steady_verdict(t_all[m], q[m], args.min_cv)
    # enstrophy 是主判準（KE 由 forcing 撐著，較不敏感）
    report["verdict"] = "STEADY" if report["enstrophy"]["steady"] else "NOT_STEADY"
    report["reading"] = (
        "STEADY = enstrophy 在窗內既有平台（趨勢被自身波動蓋過）又仍在波動。"
        "NOT_STEADY 時不要投入 N=1024 DNS 與四臂重訓：若仍在衰減，穩態窗根本"
        "還沒到；若波動太小，那是靜止/近層流解，外推會平庸地容易。")

    out = Path(args.out) if args.out else Path(str(args.dns) + ".steady.json")
    out.write_text(json.dumps(report, indent=2))

    print(f"=== steady-state probe: t ∈ [{t0}, {t1}]，{int(m.sum())} 幀 ===")
    for name in ("enstrophy", "kinetic_energy"):
        r = report[name]
        print(f"{name:>15}: mean={r['mean']:.4g} std={r['std']:.3g} cv={r['cv']:.3f} "
              f"drift={r['drift_over_window']:.3g} plateau={r['plateau']} "
              f"fluctuating={r['fluctuating']}")
    print(f"VERDICT: {report['verdict']}")
    print(f"[out] {out}")
    return 0 if report["verdict"] == "STEADY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
