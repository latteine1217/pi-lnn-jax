#!/usr/bin/env python3
"""plot_temporal_cadence.py — 主指標對 sensor 取樣間隔 Δt（K=200，n=5）。

What:
    畫 `docs/figures/temporal_cadence_data.json` 的五個節奏點（Δt ∈ {0.025,…,0.5} s，
    每點 5 seed mean ± s.d.），半對數 x，標出部署節奏 0.05 s 與掃掠假設的門檻
    Δt_c = 1/(2 U_rms √(K/π)) = 0.125 s。

Why:
    時間稀疏是與空間稀疏並列的一軸，但代價小一個數量級：Δt 掃 20 倍只值 3.33 pp，
    而 K 掃 16 倍值 24.3 pp。圖要能同時讀出「整段幅度小」與「膝點在 0.1 s 附近」，
    所以 y 軸不從 0 起（那會把 3.3 pp 壓成一條平線），改以文字標出整段幅度。
    背書 thesis §4.4 Temporal Sampling Cadence。

Usage:
    uv run python scripts/plot_temporal_cadence.py \
        --out paper/thesis-format/figures/results/temporal_cadence_K200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_style import figwidth, save_figure, setup_style  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "docs" / "figures" / "temporal_cadence_data.json"
# 掃掠假設的門檻：k_s = k_t 時的 Δt。U_rms 取稿內定義 t_eddy ≡ L*/U_rms = 1.99 s
# （chapter03），使圖與正文同源；DNS 殼層能譜的獨立估計 √(2ΣE) = 0.5123 與它差 1.9%。
_U_RMS = 1.0 / 1.99
_K = 200
_DEPLOYED_DT = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="輸出 stem（不含副檔名）")
    p.add_argument("--venue", default="thesis")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    payload = json.loads(_DATA.read_text())
    rows = sorted(payload["rows"], key=lambda r: r["dt"])
    dt = np.array([r["dt"] for r in rows])
    mean = np.array([r["mean"] for r in rows])
    sd = np.array([r["sd"] for r in rows])
    n = {r["n"] for r in rows}
    if len(n) != 1:
        raise ValueError(f"各點 seed 數不一致：{[r['n'] for r in rows]}——圖上只會標一個 n")

    setup_style(a.venue)
    fig, ax = plt.subplots(figsize=(figwidth(a.venue, "single"), 2.6))

    dt_c = 1.0 / (2.0 * _U_RMS * np.sqrt(_K / np.pi))
    ax.axvline(dt_c, color="0.55", lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.annotate(rf"$\Delta t_c={dt_c:.3f}$", xy=(dt_c, mean.max()), xytext=(2.5, -1),
                textcoords="offset points", fontsize=7, color="0.35",
                ha="left", va="top", rotation=90)
    ax.axvline(_DEPLOYED_DT, color="0.55", lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.annotate("deployed", xy=(_DEPLOYED_DT, mean.max()), xytext=(-3, -1),
                textcoords="offset points", fontsize=7, color="0.35",
                ha="right", va="top", rotation=90)

    ax.errorbar(dt, mean, yerr=sd, marker="o", ms=4.0, lw=1.3, capsize=2.5,
                elinewidth=0.9, color="C0", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(dt)
    ax.set_xticklabels([f"{d:g}" for d in dt])
    ax.minorticks_off()
    ax.set_xlabel(r"sensor sampling interval $\Delta t$ (s)")
    ax.set_ylabel(r"$\overline{e}_{uv}$ (%)")

    span = mean[-1] - mean[0]
    ax.annotate(rf"{span:+.2f} pp over a twentyfold change in $\Delta t$",
                xy=(0.03, 0.93), xycoords="axes fraction", fontsize=7.5, va="top")
    ax.annotate(rf"$N_t$: {rows[0]['N_t']} $\to$ {rows[-1]['N_t']} snapshots over the same $T = 5$ s",
                xy=(0.03, 0.83), xycoords="axes fraction", fontsize=7, va="top", color="0.35")

    lo, hi = mean.min() - sd.max() * 2, mean.max() + sd.max() * 2
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))

    fig.tight_layout(pad=0.25)
    written = save_figure(fig, a.out)
    for w in written:
        print(f"[out] {w}")


if __name__ == "__main__":
    main()
