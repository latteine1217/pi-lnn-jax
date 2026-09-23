#!/usr/bin/env python3
"""Same-IC LES 對 DNS 的可預報時長（thesis fig:les_predictability）。

What:
    三格共用時間軸 t∈[0,5] s：
      (a) 速度誤差（%）——full-field rel-L², 低波段 rel-L²(k≤5), rel-L^∞；
          灰色點線是 √2 的完全去相關水準。
      (b) 場的 Pearson 相關——速度 u 與渦量 ω。
      (c) LES/DNS 比值——動能 KE 與 enstrophy Z=<ω²/2>。
    兩條灰色虛線是一個與兩個 eddy-turnover time。

Why 這支腳本存在:
    此圖與 `plot_dns_les_vorticity_compare.py` 原本沒有生成腳本（見
    `docs/paper-figure-provenance.md` §3），2026-09-17 的投稿前稽核查出兩條
    MAJOR（`A2-M05` 疊字、`A2-M06` 字級過小）都只能靠重產修——於是連同重建
    路徑一起補上。重產結果與舊檔的內容逐點比對一致，只有版面改變。

排版上刻意與舊檔不同的三處（都是那兩條 MAJOR 的修法）:
    1. 原生寬度降到 thesis 版心 425.2 bp（舊檔 525.6 bp，等於先放大再縮小，
       圖內字最後只剩 2.8–5.3 pt）。現在 `width=\\textwidth` 的縮放係數是 1。
    2. panel tag 走 `set_title(loc="left")`，錨在 axes 左緣。舊檔的 tag 擺在
       y 軸標籤那一欄的上方，與旋轉標籤的結尾重疊成無法辨讀的字形。
    3. y 軸標籤與圖例文字縮短（`(dimensionless)` 移進 caption——caption 本來
       就寫了；圖例的 KE_LES/KE_DNS 改為 KE，因為 y 軸已寫 "LES/DNS ratio"）。
       少掉的巢狀上下標是舊檔 2.8 pt 字形的來源。

Usage:
    PILNJAX_DATA_ROOT=<大檔池> uv run python scripts/plot_les_sameic_predictability.py
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

from pi_lnn_jax.metric_artifact import compute_vorticity  # noqa: E402

# thesis 版心：A4 減左右各 3 cm = 425.2 bp。`journal_style.figwidth("thesis")`
# 給的 5.50 in (=396 bp) 比它窄 7%，圖會被 \textwidth 放大；這裡要縮放係數
# 正好是 1，所以直接寫版心值。
TEXTWIDTH_IN = 425.2 / 72.0

# 圖例字級：`journal_style` 的 thesis profile 給 7 pt，舊檔更只有 6.5 pt（縮放後
# 5.3 pt）。7.5 pt 是這個格寬下的上限——再大圖例就會蓋掉 rel-L^∞ 的峰值，那比
# 字小更糟。位置沿用舊檔（左中），只是字變大。
LEGEND_PT = 7.5

LES_DEFAULT = "data/les/kolmogorov_les_Re10000_N256_T5_dns_init_FIXED.npy"
DNS_DEFAULT = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"


def _lowpass(f: np.ndarray, kc: float) -> np.ndarray:
    """截掉 |k| > kc 的譜分量（k 以 cycles/domain 計，L=1 m 時即 1/m）。"""
    n = f.shape[-1]
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    kmag = np.sqrt(kx ** 2 + ky ** 2)
    fh = np.fft.fft2(f)
    fh[..., kmag > kc] = 0.0
    return np.real(np.fft.ifft2(fh))


def compute_series(lu, lv, du, dv, k_lowband: float) -> dict[str, np.ndarray]:
    """逐時刻的誤差／相關／比值。回傳鍵即圖上的曲線。

    rel-L^∞ 取速度**向量長度**的最大值比（非逐分量），與 rel-L² 同樣以向量
    長度定義，兩條線才在同一個量上可比。
    """
    axes = (1, 2)
    num2 = np.sum((lu - du) ** 2 + (lv - dv) ** 2, axis=axes)
    den2 = np.sum(du ** 2 + dv ** 2, axis=axes)
    lo_l, lo_d = _lowpass(lu, k_lowband), _lowpass(du, k_lowband)
    lo_lv, lo_dv = _lowpass(lv, k_lowband), _lowpass(dv, k_lowband)
    num_lo = np.sum((lo_l - lo_d) ** 2 + (lo_lv - lo_dv) ** 2, axis=axes)
    den_lo = np.sum(lo_d ** 2 + lo_dv ** 2, axis=axes)

    err_mag = np.sqrt((lu - du) ** 2 + (lv - dv) ** 2)
    ref_mag = np.sqrt(du ** 2 + dv ** 2)
    n_t = lu.shape[0]
    linf = err_mag.reshape(n_t, -1).max(axis=1) / ref_mag.reshape(n_t, -1).max(axis=1)

    wl = np.stack([compute_vorticity(lu[i], lv[i]) for i in range(n_t)])
    wd = np.stack([compute_vorticity(du[i], dv[i]) for i in range(n_t)])

    def corr_t(a, b):
        a = a.reshape(n_t, -1) - a.reshape(n_t, -1).mean(axis=1, keepdims=True)
        b = b.reshape(n_t, -1) - b.reshape(n_t, -1).mean(axis=1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

    ke_l = 0.5 * np.mean(lu ** 2 + lv ** 2, axis=axes)
    ke_d = 0.5 * np.mean(du ** 2 + dv ** 2, axis=axes)
    z_l = 0.5 * np.mean(wl ** 2, axis=axes)
    z_d = 0.5 * np.mean(wd ** 2, axis=axes)

    return {
        "rel_l2_full": 100.0 * np.sqrt(num2 / den2),
        "rel_l2_low": 100.0 * np.sqrt(num_lo / den_lo),
        "rel_linf": 100.0 * linf,
        "corr_u": corr_t(lu, du),
        "corr_omega": corr_t(wl, wd),
        "ke_ratio": ke_l / ke_d,
        "enstrophy_ratio": z_l / z_d,
    }


def _mark_eddy_times(ax, t_eddy: float, t_max: float) -> None:
    for n in (1, 2):
        if n * t_eddy <= t_max:
            ax.axvline(n * t_eddy, color="#808080", ls="--", lw=0.8, zorder=1)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", default=LES_DEFAULT,
                    help="same-IC LES 全場 .npy（相對路徑走 PILNJAX_ROOT → PILNJAX_DATA_ROOT）")
    ap.add_argument("--dns", default=DNS_DEFAULT, help="對應的 DNS 全場 .npy")
    ap.add_argument("--k-lowband", type=float, default=5.0,
                    help="低波段 rel-L² 的截斷波數（caption 寫 k≤5）")
    ap.add_argument("--t-eddy", type=float, default=1.99,
                    help="eddy-turnover time [s]，用於兩條垂直虛線")
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "results"))
    args = ap.parse_args()

    lu, lv, du, dv, t = load_sameic_pair(args.les, args.dns)
    s = compute_series(lu, lv, du, dv, args.k_lowband)

    # stdout 印出進正文的那幾個數，讓圖與 appendix02 的句子可以互相核對
    te = args.t_eddy
    for thr in (5.0, 20.0):
        i = int(np.argmax(s["rel_l2_full"] > thr))
        print(f"[check] rel-L2(full) 首次超過 {thr:.0f}%: t={t[i]:.3f} s = {t[i]/te:.2f} t_eddy")
    i_pk = int(np.argmax(s["rel_l2_full"]))
    print(f"[check] rel-L2(full) 峰值 {s['rel_l2_full'][i_pk]:.1f}% @ t={t[i_pk]:.2f} s")
    i9 = int(np.argmax(s["corr_u"] < 0.9))
    print(f"[check] corr(u) 首次低於 0.9: t={t[i9]:.3f} s = {t[i9]/te:.2f} t_eddy")
    print(f"[check] t=5 s: corr(omega)={s['corr_omega'][-1]:.3f} "
          f"KE={s['ke_ratio'][-1]:.3f} Z={s['enstrophy_ratio'][-1]:.3f}")

    setup_style("thesis")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.45))

    ax = axes[0]
    ax.plot(t, s["rel_l2_full"], color=OKABE_ITO["blue"], ls="-",
            label=r"rel-$L^2$ (full)")
    ax.plot(t, s["rel_l2_low"], color=OKABE_ITO["green"], ls="--",
            label=rf"rel-$L^2$ ($k \leq {args.k_lowband:g}$)")
    ax.plot(t, s["rel_linf"], color=OKABE_ITO["vermillion"], ls=":",
            label=r"rel-$L^\infty$")
    ax.axhline(100.0 * np.sqrt(2.0), color="#808080", ls=":", lw=0.8)
    ax.text(0.03, 0.87, r"$\sqrt{2}$ (decorrelated)", transform=ax.transAxes,
            color="#666666", fontsize=LEGEND_PT, va="bottom", ha="left")
    ax.set_ylim(-5, 170)
    ax.set_ylabel("Velocity error (%)")
    ax.legend(loc="lower right", fontsize=LEGEND_PT, bbox_to_anchor=(1.0, 0.0),
              handlelength=1.2, handletextpad=0.4, borderpad=0.3)

    ax = axes[1]
    ax.plot(t, s["corr_u"], color=OKABE_ITO["blue"], ls="-", label=r"corr($u$)")
    ax.plot(t, s["corr_omega"], color=OKABE_ITO["vermillion"], ls="--",
            label=r"corr($\omega$)")
    ax.axhline(0.0, color="#808080", lw=0.6, zorder=1)
    ax.set_ylim(-0.1, 1.08)
    ax.set_ylabel("Pearson correlation")
    ax.legend(loc="lower left", fontsize=LEGEND_PT)

    ax = axes[2]
    ax.plot(t, s["ke_ratio"], color=OKABE_ITO["blue"], ls="-", label="KE")
    ax.plot(t, s["enstrophy_ratio"], color=OKABE_ITO["orange"], ls="--",
            label=r"enstrophy $\mathcal{Z}$")
    ax.axhline(1.0, color="#808080", lw=0.6, zorder=1)
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("LES/DNS ratio")
    ax.legend(loc="lower left", fontsize=LEGEND_PT)

    for tag, ax in zip("abc", axes):
        _mark_eddy_times(ax, te, float(t[-1]))
        ax.set_xlim(0, float(t[-1]))
        ax.set_xlabel("Time $t$ (s)")
        ax.set_title(f"({tag})", loc="left", pad=4)

    fig.tight_layout(w_pad=1.2)
    for p in save_figure(fig, str(Path(args.out) / "les_sameic_predictability")):
        print(f"[write] {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
