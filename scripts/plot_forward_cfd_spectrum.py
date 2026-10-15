#!/usr/bin/env python3
"""plot_forward_cfd_spectrum.py — forward-CFD baseline 與 DNS 在 t=5 的能譜（fig:forward_cfd 左格）。

What:
    讀 forward-CFD baseline 的 t=5 速度場（u_pred/v_pred）與同時刻 DNS（u_ref/v_ref），
    以 `metric_artifact.compute_energy_spectrum` 算 E(k)，畫 log-log 疊圖，附 k^{-3} 參考線
    （錨在 DNS 的 E(1)；2D 正向 enstrophy cascade，與其餘能譜圖一致）與強迫波數 k_f=2。

Why:
    原圖沒有生成腳本（docs/paper-figure-provenance.md §3），無法換成稿件的 serif 樣式。
    資料與 `plot_forward_cfd_anisotropy.py` 同源（兩支讀同一份 npz）：home-gpu
    `~/pi-lnn-cfd-baseline/reports/forward_cfd_baseline_T5_rank40_vort.npz`（PoF，真渦度佈點）。

    uv run python scripts/plot_forward_cfd_spectrum.py --npz <path/to/forward_cfd_baseline_T5_rank40_vort.npz>
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common.units import unit
from journal_style import STYLE_CYCLE, save_figure, setup_style  # noqa: E402

from pi_lnn_jax.metric_artifact import compute_energy_spectrum  # noqa: E402

_OUT = "paper/pof-format/figures/results/forward_cfd_spectrum_t5"
K_F = 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="forward-CFD baseline 的 t=5 場（--save-fields 產物）")
    ap.add_argument("--out", default=_OUT, help="輸出路徑（不含副檔名）")
    ap.add_argument("--nondim", action="store_true", help="軸標籤不標 SI 單位（PoF 稿）")
    args = ap.parse_args()

    z = np.load(args.npz)

    k, E_ref = compute_energy_spectrum(z["u_ref"], z["v_ref"])
    _, E_pred = compute_energy_spectrum(z["u_pred"], z["v_pred"])
    # 只畫到 N/2：再往上是保 Parseval 的對角 corner bin（同 plot_thesis_series_figures）
    sel = (k >= 1) & (k <= z["u_ref"].shape[0] // 2)

    setup_style("tmlr")
    # 原生寬度＝稿上 0.58\linewidth（≈3.8 in），縮放係數約 1，字級不被壓小
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    ax.loglog(k[sel], E_ref[sel], color="k", lw=1.2, label="DNS")
    ax.loglog(k[sel], E_pred[sel], color=STYLE_CYCLE[0][0], ls="--", lw=1.4,
              label="Forward-CFD baseline")
    kk = np.array([1.0, 100.0])
    ax.loglog(kk, E_ref[1] * kk ** -3.0, color="0.5", ls=":", lw=0.9, label=r"$k^{-3}$")
    ax.axvline(K_F, color="k", ls="-.", lw=0.9, label=rf"$k_f={K_F}$")
    ax.set_xlabel(r"Wavenumber $k$" + unit("1/m", args.nondim))
    ax.set_ylabel(r"Energy $E(k)$" + unit("m$^3$/s$^2$", args.nondim))
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    for p in save_figure(fig, args.out):
        print(f"[out] {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
