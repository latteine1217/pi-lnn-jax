#!/usr/bin/env python3
"""plot_nyquist_recoverability.py — 重現 fig:nyquist（K 點感測器的資訊論能量天花板）。

What:
    取主線 DNS（Re=10^4）t=5 的速度場，畫兩面板：
    (a) 逐 shell E(k)/E_tot log-log，疊 K=100/200/400 的取樣邊界 k_max^sensor=√(K/π) 與 k^{-3}；
    (b) 逐 shell 累積能量分數 F_DNS(k)，於各邊界標記圓盤 |k|≤k_max^sensor 內模態的能量占比。

Why:
    背書稿上「98.9% / 99.7% / 99.9% at K=100,200,400」。那三個數是**圓盤內模態**的能量占比，
    不是逐 shell 累積曲線在 k_max 的內插（後者得 99.3／99.8／99.9，差在 shell 寬度）；
    舊版從 docs/figures/ksweep_data.json 讀的頻譜也不是 t=5 這一幀（k=1 累積 0.40 vs 0.79），
    兩者都會讓圖上數字對不上正文。標尺單一來源為 metric_artifact.sparsity_yardsticks。
    預設輸出 *_repro 供比對；稿件圖以 --out 指定。
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from journal_style import STYLE_CYCLE, save_figure, setup_style  # noqa: E402
from _common.units import unit
from pi_lnn_jax.metric_artifact import compute_energy_spectrum, sparsity_yardsticks  # noqa: E402

_DNS = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
_T = 5.0
_OUT = "docs/figures/nyquist_recoverability_repro"
_KS = [100, 200, 400]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", default=_DNS, help="DNS .npy（dict：time/u/v）")
    ap.add_argument("--out", default=_OUT, help="輸出路徑（不含副檔名）；預設寫 *_repro 供比對")
    ap.add_argument("--nondim", action="store_true", help="軸標籤不標 SI 單位（PoF 稿）")
    args = ap.parse_args()

    obj = np.load(args.dns, allow_pickle=True).item()
    times = np.asarray(obj["time"], dtype=float)
    i = int(np.argmin(np.abs(times - _T)))
    if abs(times[i] - _T) > 1e-6:
        raise SystemExit(f"DNS 沒有 t={_T} 這一幀（最近 {times[i]}）")
    u = np.asarray(obj["u"][i], dtype=np.float64)
    v = np.asarray(obj["v"][i], dtype=np.float64)
    del obj
    N = u.shape[0]
    k, E = compute_energy_spectrum(u, v)
    E_tot = E.sum()
    cdf = np.cumsum(E) / E_tot  # 逐 shell 累積 F_DNS(k)
    cutoffs = {K: sparsity_yardsticks(K)["nyquist_kmax"] for K in _KS}
    # 圓盤 |k|<=k_max 內模態的能量占比（正文數字的定義）
    kf = np.fft.fftfreq(N, d=1.0 / N)
    kmag = np.hypot(*np.meshgrid(kf, kf, indexing="ij"))
    e2 = np.abs(np.fft.fft2(u)) ** 2 + np.abs(np.fft.fft2(v)) ** 2
    fracs = {K: float(e2[kmag <= kc].sum() / e2.sum()) for K, kc in cutoffs.items()}
    # 只畫到 N/2：再往上是保 Parseval 的對角 corner bin（同 plot_thesis_series_figures）
    sel = (k >= 1) & (k <= N // 2)

    setup_style("tmlr")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.6, 3.0))
    colors = [STYLE_CYCLE[i][0] for i in range(len(_KS))]

    axA.loglog(k[sel], (E / E_tot)[sel], color="k", lw=1.3)
    # k^{-3} 參考線錨在最小預算的取樣邊界上，延伸到 enstrophy cascade 段
    k0 = cutoffs[_KS[0]]
    kk = np.array([k0, 35.0])
    axA.loglog(kk, np.interp(k0, k, E / E_tot) * (kk / k0) ** -3.0, color="0.4", ls=":", lw=0.9)
    for (K, kc), c in zip(cutoffs.items(), colors):
        axA.axvline(kc, color=c, ls="--", lw=1.0)
    axA.set_xlim(1, k[sel][-1])
    axA.set_ylim(1e-11, 2)
    axA.set_xlabel(r"Wavenumber $k=\Vert\mathbf{k}\Vert$" + unit("1/m", args.nondim))
    axA.set_ylabel(r"Shell energy $E(k)/E_{\rm tot}$")
    axA.set_title("(a)", loc="left")

    axB.semilogx(k[sel], cdf[sel], color="k", lw=1.3)
    for (K, kc), c in zip(cutoffs.items(), colors):
        axB.axvline(kc, color=c, ls="--", lw=1.0, ymax=0.9)
        axB.plot([1, kc], [fracs[K]] * 2, color=c, ls=":", lw=0.8)
        axB.plot(kc, fracs[K], "o", color=c, ms=4)
    axB.set_xlim(1, k[sel][-1])
    axB.set_xlabel(r"Wavenumber $k$" + unit("1/m", args.nondim))
    axB.set_ylabel(r"Cumulative fraction $F_{\rm DNS}(k)$")
    axB.set_title("(b)", loc="left")

    handles = [Line2D([], [], color="k", lw=1.3, label=r"DNS $E(k)$"),
               Line2D([], [], color="0.4", ls=":", label=r"$k^{-3}$"),
               Line2D([], [], color="0.4", ls="--", label=r"$k_{\max}^{\rm sensor}(K)$")]
    handles += [Line2D([], [], color=c, ls="--", marker="o", ms=4,
                       label=rf"$K={K}$: $k_{{\max}}^{{\rm sensor}}={kc:.2f}$, $F_{{\rm DNS}}={fracs[K]*100:.1f}\%$")
                for (K, kc), c in zip(cutoffs.items(), colors)]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.26, 1, 1))

    save_figure(fig, args.out)
    # 驗證：對齊 sec:count 文字「98.9% / 99.7% / 99.9% at K=100,200,400」
    for K, kc in cutoffs.items():
        print(f"  K={K:3d}  k_max={kc:5.2f}  F_DNS={fracs[K]*100:.1f}%")
    print("[out]", args.out + ".pdf")


if __name__ == "__main__":
    main()
