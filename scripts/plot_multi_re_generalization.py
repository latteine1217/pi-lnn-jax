#!/usr/bin/env python
"""Multi-Re 泛用性圖（paper-quality, TMLR/ICLR serif vector）。

讀 evaluate_multi_re 的 full_8re.json，畫 train(in-dist) vs held-out 的
(a) 速度重建相對誤差、(b) 預測散度，凸顯「未見 Re 內插失敗」。

用法: uv run python scripts/plot_multi_re_generalization.py \
        artifacts/kolmogorov/multi_re_train5/multi_re_eval/full_8re.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wong (2011) colorblind-safe；train=blue 實心圓+線、held-out=vermillion 空心方（形狀亦可在 B&W 區分）
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#999999"


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "CMU Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _series(rows, held):
    sel = sorted((r for r in rows if r["held_out"] == held), key=lambda r: r["Re"])
    return ([r["Re"] for r in sel],
            [r["u_rel_err"] for r in sel],
            [r["div_pred_l2"] for r in sel])


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "artifacts/kolmogorov/multi_re_train5/multi_re_eval/full_8re.json")
    data = json.loads(src.read_text())
    rows = data["rows"]
    tr_re, tr_u, tr_div = _series(rows, held=False)
    ho_re, ho_u, ho_div = _series(rows, held=True)

    _setup()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.0, 2.5))

    train_kw = dict(color=BLUE, marker="o", mfc=BLUE, mec=BLUE, ms=5, zorder=3)
    held_kw = dict(color=VERM, marker="s", mfc="white", mec=VERM, mew=1.3, ms=5,
                   linestyle="none", zorder=4)

    # (a) 速度重建相對誤差
    axL.plot(tr_re, tr_u, label="Train (in-dist.)", **train_kw)
    axL.plot(ho_re, ho_u, label="Held-out", **held_kw)
    axL.axhline(1.0, color=GRAY, ls="--", lw=0.7, zorder=1)
    axL.text(1.1e3, 1.02, "uncorrelated", fontsize=6.5, color=GRAY, va="bottom")
    axL.set_xscale("log")
    axL.set_xlabel(r"Reynolds number $Re$")
    axL.set_ylabel(r"Velocity rel. error $\varepsilon_u$ (–)")
    axL.set_title(r"(a) Reconstruction error")
    axL.set_ylim(0, 1.2)
    axL.legend(loc="center left", handlelength=1.6)

    # (b) 預測散度（連續性一致性）
    axR.plot(tr_re, tr_div, **train_kw)
    axR.plot(ho_re, ho_div, **held_kw)
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel(r"Reynolds number $Re$")
    axR.set_ylabel(r"Pred. divergence $\Vert\nabla\!\cdot\!\mathbf{u}\Vert_2$ (–)")
    axR.set_title(r"(b) Physics consistency")

    fig.tight_layout(pad=0.4, w_pad=1.5)
    out = src.parent / "multi_re_generalization"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    print(f"[fig] {out}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
