#!/usr/bin/env python
"""連續-Re 物理正則：baseline vs CRP 對比圖（u_err 沒改善、div 砍半）。

讀 train5 與 train5_crp 的 full_8re.json，2-panel：(a) 速度重建相對誤差、(b) 預測散度，
各畫 baseline / CRP × train / held-out。凸顯「物理一致性改善但重建泛用性未解」。

用法: uv run python scripts/plot_multi_re_crp_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#999999"
BASE = "artifacts/kolmogorov/multi_re_train5/multi_re_eval/full_8re.json"
CRP = "artifacts/kolmogorov/multi_re_train5_crp/multi_re_eval/full_8re.json"


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def _split(path, held):
    rows = json.loads(Path(path).read_text())["rows"]
    sel = sorted((r for r in rows if r["held_out"] == held), key=lambda r: r["Re"])
    return [r["Re"] for r in sel], [r["u_rel_err"] for r in sel], [r["div_pred_l2"] for r in sel]


def main() -> int:
    _setup()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    # marker: train=填實圓、held=空心方；color: baseline=藍、CRP=橘
    for path, color, lab in [(BASE, BLUE, "baseline"), (CRP, VERM, "CRP")]:
        tr_re, tr_u, tr_d = _split(path, False)
        ho_re, ho_u, ho_d = _split(path, True)
        axL.plot(tr_re, tr_u, "-o", color=color, mfc=color, ms=4.5, lw=1.0,
                 label=f"{lab} (train)")
        axL.plot(ho_re, ho_u, "s", color=color, mfc="white", mec=color, mew=1.3,
                 ms=5.5, ls="none", label=f"{lab} (held-out)")
        axR.plot(tr_re, tr_d, "-o", color=color, mfc=color, ms=4.5, lw=1.0)
        axR.plot(ho_re, ho_d, "s", color=color, mfc="white", mec=color, mew=1.3,
                 ms=5.5, ls="none")
    axL.axhline(1.0, color=GRAY, ls="--", lw=0.7)
    axL.set_xscale("log"); axL.set_ylim(0, 1.15)
    axL.set_xlabel(r"Reynolds number $Re$")
    axL.set_ylabel(r"Velocity rel. error $\varepsilon_u$ (–)")
    axL.set_title(r"(a) Reconstruction error — unchanged")
    axL.legend(loc="center left", handlelength=1.5, ncol=1)

    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"Reynolds number $Re$")
    axR.set_ylabel(r"Pred. divergence $\Vert\nabla\!\cdot\!\mathbf{u}\Vert_2$ (–)")
    axR.set_title(r"(b) Physics consistency — improved")

    fig.tight_layout(pad=0.4, w_pad=1.6)
    out = Path("artifacts/kolmogorov/multi_re_train5_crp/multi_re_eval/crp_vs_baseline")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    print(f"[fig] {out}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
