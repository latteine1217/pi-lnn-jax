#!/usr/bin/env python3
"""Same-IC LES 對 DNS 的渦量場與能譜（thesis fig:les_vort_compare）。

What:
    (a) 六個匹配時刻的渦量場，上排 DNS、下排 LES；每一欄各自以**該時刻 DNS 的
        RMS 渦量**正規化，色階固定 ±3。
    (b) t=5 s 的徑向能譜 E(k)，附 k∈[3,30] 的斜率擬合與 k^-3 參考線；垂直點線
        是 K=100 的取樣邊界 k_max^sensor = sqrt(K/π)。

Why 這支腳本存在:
    與 `plot_les_sameic_predictability.py` 同一批——見該檔的 docstring。

排版與標示上刻意與舊檔不同的四處:
    1. 原生寬度降到 thesis 版心 425.2 bp（舊檔 462.0 bp），字級由 6.5 pt 提到
       8.5–10 pt（`A2-M06`）。
    2. 三條灰色輔助線**全部進圖例**（`A2-N06`）。舊檔把 `k^-3` 寫成貼線的旋轉
       標註，而它正好落在圖例框下面，讀者無從把 caption 宣告的兩類線對到圖上。
    3. 圖例移到左下的空白區。舊檔在右上，壓住 k^-3 參考線。
    4. 圖內符號改與 caption 一致：`t_e` → `t_eddy`、`k_sensor` → `k_max^sensor`
       （同上 `A2-N06`）。

    能譜只畫到 k = N/2（=128）；再往上是對角 corner bin，舊檔也沒畫。

Usage:
    PILNJAX_DATA_ROOT=<大檔池> uv run python scripts/plot_dns_les_vorticity_compare.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, save_figure, OKABE_ITO  # noqa: E402
from _common.sameic_les_dns import load_sameic_pair  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pi_lnn_jax.metric_artifact import compute_energy_spectrum, compute_vorticity  # noqa: E402

TEXTWIDTH_IN = 425.2 / 72.0     # thesis 版心；見 plot_les_sameic_predictability.py
LEGEND_PT = 8.0

LES_DEFAULT = "data/les/kolmogorov_les_Re10000_N256_T5_dns_init_FIXED.npy"
DNS_DEFAULT = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"

SNAP_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
CLIM = 3.0
FIT_BAND = (3.0, 30.0)
REF_BAND = (3.0, 60.0)          # k^-3 參考線的繪製範圍（對齊舊檔量到的 3..~57）


def _index_of(t: np.ndarray, want: float) -> int:
    i = int(np.argmin(np.abs(t - want)))
    if abs(t[i] - want) > 1e-6:
        raise ValueError(
            f"時間軸上沒有 t={want} s 這一格（最近的是 {t[i]:.4f} s）——"
            "本圖標的是匹配時刻，不接受最近鄰代用。"
        )
    return i


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", default=LES_DEFAULT)
    ap.add_argument("--dns", default=DNS_DEFAULT)
    ap.add_argument("--t-eddy", type=float, default=1.99)
    ap.add_argument("--k-sensors", type=int, default=100,
                    help="取樣邊界 k_max^sensor = sqrt(K/π) 的 K")
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "results"))
    args = ap.parse_args()

    lu, lv, du, dv, t = load_sameic_pair(args.les, args.dns)
    idx = [_index_of(t, tt) for tt in SNAP_TIMES]

    wd = {i: compute_vorticity(du[i], dv[i]) for i in idx}
    wl = {i: compute_vorticity(lu[i], lv[i]) for i in idx}
    rms = {i: float(np.sqrt(np.mean(wd[i] ** 2))) for i in idx}

    i5 = idx[-1]
    n_half = du.shape[-1] // 2
    k, e_dns = compute_energy_spectrum(du[i5], dv[i5])
    _, e_les = compute_energy_spectrum(lu[i5], lv[i5])
    k, e_dns, e_les = k[1:n_half + 1], e_dns[1:n_half + 1], e_les[1:n_half + 1]
    band = (k >= FIT_BAND[0]) & (k <= FIT_BAND[1])
    fit_dns = np.polyfit(np.log(k[band]), np.log(e_dns[band]), 1)
    fit_les = np.polyfit(np.log(k[band]), np.log(e_les[band]), 1)
    k_sensor = float(np.sqrt(args.k_sensors / np.pi))
    print(f"[check] slope DNS={fit_dns[0]:.4f}  LES={fit_les[0]:.4f}  "
          f"k_max^sensor={k_sensor:.3f} 1/m")
    print("[check] omega_rms^DNS = " + ", ".join(f"{rms[i]:.2f}" for i in idx))

    setup_style("thesis")
    # bbox="tight" 會外擴，故 figsize 取略小於版心，實測輸出落在 425 bp 上下。
    fig = plt.figure(figsize=(TEXTWIDTH_IN * 0.973, 4.23))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.32, 1.0], hspace=0.42,
                             left=0.085, right=0.885, top=0.935, bottom=0.095)
    grid = outer[0].subgridspec(2, len(idx), wspace=0.07, hspace=0.07)

    im = None
    for col, i in enumerate(idx):
        for row, (field, tag) in enumerate(((wd, "DNS"), (wl, "LES"))):
            ax = fig.add_subplot(grid[row, col])
            im = ax.imshow(field[i].T / rms[i], origin="lower", extent=(0, 1, 0, 1),
                           cmap="RdBu_r", vmin=-CLIM, vmax=CLIM, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if col == 0:
                ax.set_ylabel(tag, labelpad=4)
            if row == 0:
                te = SNAP_TIMES[col] / args.t_eddy
                ax.set_title(f"$t = {SNAP_TIMES[col]:.0f}$ s\n"
                             f"$({te:.1f}\\,t_{{\\rm eddy}})$", pad=3, fontsize=8.5)
            if row == 0 and col == 0:
                ax.text(-0.42, 1.34, "(a)", transform=ax.transAxes,
                        ha="left", va="bottom")

    cax = fig.add_axes([0.898, 0.487, 0.016, 0.40])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\omega/\omega_{\rm rms}^{\rm DNS}$ (–)", labelpad=2)
    cb.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(outer[1])
    ax.loglog(k, e_dns, color=OKABE_ITO["blue"], ls="-",
              label=rf"DNS ($t = 5$ s, slope ${fit_dns[0]:.2f}$)")
    ax.loglog(k, e_les, color=OKABE_ITO["vermillion"], ls="--",
              label=rf"LES ($t = 5$ s, slope ${fit_les[0]:.2f}$)")
    kb = k[band]
    fit_line = None
    for f in (fit_dns, fit_les):
        fit_line, = ax.loglog(kb, np.exp(np.polyval(f, np.log(kb))),
                              color="#777777", ls="-", lw=0.8, zorder=1)
    fit_line.set_label(rf"slope fit (${FIT_BAND[0]:.0f} \leq k \leq {FIT_BAND[1]:.0f}$)")
    kr = np.array(REF_BAND)
    e_ref = float(np.interp(FIT_BAND[0], k, e_dns)) * (kr / FIT_BAND[0]) ** -3.0
    ax.loglog(kr, e_ref, color="#777777", ls="--", lw=0.8,
              label=r"$k^{-3}$ reference")
    ax.axvline(k_sensor, color="#808080", ls=":", lw=0.8)
    ax.text(k_sensor * 1.15, 0.90, r"$k_{\max}^{\rm sensor}$",
            transform=ax.get_xaxis_transform(), color="#666666",
            fontsize=LEGEND_PT, va="top")
    ax.set_xlabel(r"Wavenumber $k$ (1/m)")
    ax.set_ylabel(r"$E(k)$ (m$^3$/s$^2$)")
    ax.set_title("(b)", loc="left", pad=4)
    ax.legend(loc="lower left", fontsize=LEGEND_PT, handlelength=1.6,
              handletextpad=0.5, borderpad=0.35, labelspacing=0.3)

    for p in save_figure(fig, str(Path(args.out) / "dns_les_vorticity_compare")):
        print(f"[write] {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
