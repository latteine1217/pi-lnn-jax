#!/usr/bin/env python3
"""plot_sampling_plane.py — thesis §4.4 的兩張 K–Δt 取樣平面圖。

What:
    讀 `docs/figures/sampling_plane_spg_data.json`（17 格 × n=3，全格 M=2，登錄協定），
    出兩張圖：
      * `<out-dir>/temporal_cadence`：各 K 相對自身 Δt=0.05 的主指標變化對 Δt。
      * `<out-dir>/sampling_plane_budget`：主指標對感測器數 K，每個 Δt 一條線。

Why:
    兩張圖都只畫登錄協定（各臂在自己的 sensor 幀上評分）。跨 Δt 的差值因評分相位而是一個
    範圍，那個範圍放在正文的 tab:cadence_halving，不進圖
    （knowledge/experiments/kolmogorov-sampling-plane-2026-09.md）。K 軸那張只用同一批格，
    不與 tab:ksweep_series 混畫——那批的分塊數隨 K 變、程式碼也不同代。

Usage:
    uv run python scripts/plot_sampling_plane.py --out-dir paper/thesis-format/figures/results
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
from journal_style import K_COLORS, MUTED, figwidth, save_figure, setup_style  # noqa: E402

_FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
_SEEDS = ("42", "1", "2")
_DEPLOYED_DT = 0.05
_KS = (50, 100, 200, 400)
_DTS = (0.025, 0.05, 0.1, 0.2, 0.5)
# K=50 不在共用色票裡（它只有三格、不進任何比較），給中性灰並用虛線標出
_K_STYLE = {50: (MUTED, "o", "--"), 100: (K_COLORS[100], "o", "-"),
            200: (K_COLORS[200], "^", "-"), 400: (K_COLORS[400], "D", "-")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="輸出目錄（兩張圖各寫 .pdf 與 .png）")
    p.add_argument("--venue", default="thesis")
    return p.parse_args()


def load_grid() -> dict[tuple[int, float], dict]:
    payload = json.loads((_FIG / "sampling_plane_spg_data.json").read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"spg 資料未完成：status={payload.get('status')!r}")
    grid = {(r["K"], r["dt"]): r for r in payload["rows"] if r["mean"] is not None}
    for r in grid.values():
        if set(r["per_seed"]) != set(_SEEDS):
            raise ValueError(f"K={r['K']} Δt={r['dt']} 的 seed 不是 {_SEEDS}：{sorted(r['per_seed'])}")
    return grid


def paired_pct(base: list[float], arm: list[float]) -> tuple[float, float]:
    """逐 seed 配對的相對變化：ln 比的平均轉回 %，± 為 ln 比 sd（%）。"""
    r = np.log(np.asarray(arm) / np.asarray(base))
    return 100.0 * (np.exp(r.mean()) - 1.0), 100.0 * r.std(ddof=1)


def rel_to_deployed(grid, K: int, dt: float) -> tuple[float, float]:
    a, b = grid[(K, dt)]["per_seed"], grid[(K, _DEPLOYED_DT)]["per_seed"]
    return paired_pct([b[s] for s in _SEEDS], [a[s] for s in _SEEDS])


def fig_cadence(grid, venue: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(figwidth(venue, "single") * 0.7, 2.7))
    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)
    ax.axvline(_DEPLOYED_DT, color="0.55", lw=0.9, ls=(0, (1, 2)), zorder=1)
    for K in _KS:
        dts = [d for d in _DTS if (K, d) in grid]
        m, sd = zip(*[(0.0, 0.0) if d == _DEPLOYED_DT else rel_to_deployed(grid, K, d) for d in dts])
        c, mk, ls = _K_STYLE[K]
        ax.errorbar(dts, m, yerr=sd, color=c, marker=mk, ls=ls, ms=3.6, lw=1.2,
                    capsize=2.0, elinewidth=0.8, label=rf"$K={K}$", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(_DTS)
    ax.set_xticklabels([f"{d:g}" for d in _DTS])
    ax.minorticks_off()
    ax.set_xlabel(r"sensor sampling interval $\Delta t$ (s)")
    ax.set_ylabel(r"change in $\overline{e}_{uv}$ from $\Delta t=0.05$ s (%)")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.8)

    fig.tight_layout(pad=0.3)
    for w in save_figure(fig, str(out)):
        print(f"[out] {w}")
    plt.close(fig)


def fig_budget(grid, venue: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(figwidth(venue, "single") * 0.62, 2.6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(_DTS)))   # 有序量用單調色階
    markers = ("o", "s", "^", "D", "v")
    for d, c, mk in zip(_DTS, colors, markers):
        ks = [K for K in _KS if (K, d) in grid]
        ax.errorbar(ks, [grid[(K, d)]["mean"] for K in ks], yerr=[grid[(K, d)]["sd"] for K in ks],
                    color=c, marker=mk, ms=3.4, lw=1.1, capsize=1.8, elinewidth=0.7,
                    label=rf"$\Delta t={d:g}$ s", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(_KS)
    ax.set_xticklabels([str(K) for K in _KS])
    ax.set_yticks([6, 8, 10, 15, 20, 30])
    ax.set_yticklabels(["6", "8", "10", "15", "20", "30"])
    ax.minorticks_off()
    ax.set_xlabel(r"sensor count $K$")
    ax.set_ylabel(r"$\overline{e}_{uv}$ (%)")
    ax.legend(loc="upper right", fontsize=6.8, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    for w in save_figure(fig, str(out)):
        print(f"[out] {w}")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    out_dir = Path(a.out_dir)
    grid = load_grid()
    setup_style(a.venue)
    fig_cadence(grid, a.venue, out_dir / "temporal_cadence")
    setup_style(a.venue)
    fig_budget(grid, a.venue, out_dir / "sampling_plane_budget")


if __name__ == "__main__":
    main()
