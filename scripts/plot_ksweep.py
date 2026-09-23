#!/usr/bin/env python3
"""plot_ksweep.py — K-sweep reconstruction error（headline KE time-series MAPE）vs sensor count。

What:
    從 docs/figures/ksweep_data.json（5 點 K∈{50,100,200,400,800}，每點 5-seed mean±s.d.）畫
    headline `ke_t_mape`（E(t) MAPE，對齊 PyTorch 論文定義）vs K，log-log + error bar，標出
    deployed K=100 與 diminishing-returns knee。資料源:K=100=pv_les、其餘 ksweep_k{K}_s{seed}。

Why:
    nonlinear K-sweep 顯示誤差隨 K 單調下降並飽和（每倍 K −63/−66/−44/−6%），knee≈K200-400；
    K=100=5.73% 即 headline（同五 seed），重現 PyTorch pi-lnn 5.8%。背書 sec:placement
    「K=100 為 budget 操作點、非上限」。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo 內 scripts/journal_style.py
from journal_style import STYLE_CYCLE, figwidth, save_figure, setup_style  # noqa: E402
from pi_lnn_jax.metric_artifact import (  # noqa: E402
    KE_T_MAPE_SPATIALMEAN_V1,
    LegacyInterpretation,
    read_metric_summary,
)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "ksweep_data.json")
_OUT = "docs/figures/ksweep_ke_mape"


def main() -> None:
    rows = sorted(json.load(open(_DATA))["ksweep"], key=lambda r: r["K"])
    K = [r["K"] for r in rows]

    # 這份 published K-sweep dataset 的 campaign 定義是 spatial-mean E(t)，但檔案
    # 本身早於 semantic marker。由 caller 明示 interpretation；reader 不依日期、路徑
    # 或檔名自動猜測。
    legacy = LegacyInterpretation(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        basis="published K-sweep campaign used the spatial-mean E(t) definition",
    )
    values = [read_metric_summary(
        {"ke_t_errors": r}, KE_T_MAPE_SPATIALMEAN_V1,
        legacy_interpretation=legacy,
    ) for r in rows]
    mape = [s.value * 100.0 for s in values]  # → %
    mape_std = [r.get("ke_t_mape_std", 0.0) * 100.0 for r in rows]  # 5-seed s.d.（%）

    setup_style("tmlr")
    c, m, ls = STYLE_CYCLE[0]
    fig, ax = plt.subplots(figsize=(figwidth("tmlr", "half"), 2.7))
    ax.errorbar(K, mape, yerr=mape_std, color=c, marker=m, linestyle=ls,
                capsize=2.5, elinewidth=0.8, capthick=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # deployed K=100 標注（重現 PyTorch baseline）
    k100 = mape[K.index(100)]
    ax.scatter([100], [k100], s=70, facecolors="none", edgecolors=STYLE_CYCLE[1][0], zorder=5, linewidths=1.4)
    ax.annotate(f"deployed $K{{=}}100$\n{k100:.1f}% ($\\approx$PyTorch 5.8%)",
                xy=(100, k100), xytext=(112, k100 * 1.9), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    # diminishing-returns knee 區
    ax.axvspan(200, 400, color="0.6", alpha=0.12, lw=0)
    ax.text(283, mape[-1] * 1.15, "knee", fontsize=7, color="0.4", ha="center")

    ax.set_xlabel(r"Sensor count $K$ (–)")
    ax.set_ylabel(r"KE time-series MAPE (%)")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    save_figure(fig, _OUT)
    print("[out]", _OUT + ".pdf", "| points:", list(zip(K, [round(x, 2) for x in mape])))


if __name__ == "__main__":
    main()
