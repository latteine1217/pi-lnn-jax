#!/usr/bin/env python3
"""plot_forward_cfd_anisotropy.py — forward-CFD baseline vs DNS 的 per-component velocity
std (sigma_u, sigma_v),顯示 f_x-driven anisotropy 的流失。

向量 PDF(取代舊 raster forward_cfd_anisotropy.png)。資料為 t=5 的逐欄 spatial std,
直接由 forward-CFD baseline 的 npz（home-gpu `forward_cfd_baseline.py --save-fields` 產物，
含 u/v_pred 與 u/v_ref）計算；PoF 用真渦度佈點那份 `forward_cfd_baseline_T5_rank40_vort.npz`。
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common.units import unit
from journal_style import STYLE_CYCLE, figwidth, save_figure, setup_style  # noqa: E402

_OUT = "paper/pof-format/figures/results/forward_cfd_anisotropy"



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="forward-CFD baseline 的 t=5 場（u/v_pred、u/v_ref）")
    ap.add_argument("--out", default=_OUT, help="輸出路徑（不含副檔名）")
    ap.add_argument("--nondim", action="store_true", help="軸標籤不標 SI 單位（PoF 稿）")
    args = ap.parse_args()
    z = np.load(args.npz)
    SIGMA = {"DNS": (float(z["u_ref"].std()), float(z["v_ref"].std())),
             "Forward-CFD": (float(z["u_pred"].std()), float(z["v_pred"].std()))}
    for g, (su, sv) in SIGMA.items():
        print(f"[sigma] {g}: sigma_u={su:.4f} sigma_v={sv:.4f} ratio={su / sv:.2f}")
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
    ax.set_ylabel(r"velocity std" + unit("m/s", args.nondim))
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, args.out)
    print("[out]", args.out + ".pdf")


if __name__ == "__main__":
    main()
