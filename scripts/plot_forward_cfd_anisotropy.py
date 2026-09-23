#!/usr/bin/env python3
"""plot_forward_cfd_anisotropy.py — forward-CFD baseline vs DNS 的 per-component velocity
std (sigma_u, sigma_v),顯示 f_x-driven anisotropy 流失(sigma_u/sigma_v 2.33→0.90)。

向量 PDF(取代舊 raster forward_cfd_anisotropy.png)。資料為 t=5 spatial std,源自
home-gpu reports/forward_cfd_baseline_T5_rank40.npz 的 4 個數值(逐欄 std)。
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_style import STYLE_CYCLE, figwidth, save_figure, setup_style  # noqa: E402

_OUT = "paper/tmlr-format/figures/results/forward_cfd_anisotropy"

# t=5 spatial std (forward_cfd_baseline_T5_rank40.npz, home-gpu): (sigma_u, sigma_v)
SIGMA = {"DNS": (0.4592, 0.1969), "Forward-CFD": (0.3277, 0.3641)}


def main() -> None:
    setup_style("tmlr")
    fig, ax = plt.subplots(figsize=(figwidth("tmlr", "half") * 0.66, 2.5))
    groups = list(SIGMA)
    x = np.arange(len(groups))
    w = 0.36
    cu, cv = STYLE_CYCLE[0][0], STYLE_CYCLE[1][0]
    ax.bar(x - w / 2, [SIGMA[g][0] for g in groups], w, label=r"$\sigma_u$", color=cu)
    ax.bar(x + w / 2, [SIGMA[g][1] for g in groups], w, label=r"$\sigma_v$", color=cv)
    for i, g in enumerate(groups):
        ax.text(i, max(SIGMA[g]) + 0.01, rf"$\sigma_u/\sigma_v={SIGMA[g][0]/SIGMA[g][1]:.2f}$",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"velocity std (m/s)")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, _OUT)
    print("[out]", _OUT + ".pdf")


if __name__ == "__main__":
    main()
