#!/usr/bin/env python3
"""plot_nyquist_recoverability.py — 重現 fig:nyquist（K 點感測器的資訊論能量天花板）。

What:
    從 docs/figures/ksweep_data.json 取 DNS 逐 shell E(k)（與 K 無關），畫兩面板：
    (a) DNS E(k)/E_tot log-log，疊 K=100/200/400 的 2D Nyquist 截止 k_max=√(K/π) 垂直線；
    (b) 累積能量分數 F_DNS(k)，於各 k_max 標記可回收能量上限。

Why:
    背書 sec:count「$K$ 點感測器無法解析 Nyquist disk 外的尺度」。標尺單一真實來源為
    pi_lnn_jax.evaluate.sparsity_yardsticks（與 tab:kscaling 一致）。輸出到 *_repro 新路徑，
    視覺比對既有 baseline 後再決定是否替換（CLAUDE.md 禁止未經要求覆寫 baseline artifacts）。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from journal_style import save_figure, setup_style  # noqa: E402
from pi_lnn_jax.metric_artifact import sparsity_yardsticks  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "ksweep_data.json")
_OUT = "docs/figures/nyquist_recoverability_repro"
_KS = [100, 200, 400]


def main() -> None:
    E = np.asarray(json.load(open(_DATA))["spectrum"]["100"]["E_dns_k"], dtype=float)
    k = np.arange(len(E))
    E_tot = E.sum()
    cdf = np.cumsum(E) / E_tot  # F_DNS(k)
    cutoffs = {K: sparsity_yardsticks(K)["nyquist_kmax"] for K in _KS}

    setup_style("tmlr")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.78, len(_KS)))

    axA.loglog(k[1:], (E / E_tot)[1:], color="0.0", lw=1.3, label="DNS")
    for (K, kc), c in zip(cutoffs.items(), colors):
        axA.axvline(kc, color=c, ls="--", lw=1.0, label=rf"$k_{{\max}}(K{{=}}{K}){{=}}{kc:.2f}$")
    axA.set_xlabel(r"Wavenumber $k$ (–)")
    axA.set_ylabel(r"$E(k)/E_{\rm tot}$ (–)")
    axA.legend(fontsize=6, loc="lower left")
    axA.set_title("(a)", loc="left", fontsize=8)

    axB.semilogx(k[1:], cdf[1:], color="0.0", lw=1.3)
    for (K, kc), c in zip(cutoffs.items(), colors):
        f = float(np.interp(kc, k, cdf))
        axB.plot(kc, f, "o", color=c, ms=5)
        axB.annotate(rf"${f*100:.1f}\%$", (kc, f), textcoords="offset points",
                     xytext=(3, -9), fontsize=6, color=c)
    axB.set_xlabel(r"Wavenumber $k$ (–)")
    axB.set_ylabel(r"$F_{\rm DNS}(k)$ (–)")
    axB.set_title("(b)", loc="left", fontsize=8)

    save_figure(fig, _OUT)
    # 驗證：對齊 sec:count 文字「98.9% / 99.7% / 99.9% at K=100,200,400」
    for K, kc in cutoffs.items():
        print(f"  K={K:3d}  k_max={kc:5.2f}  F_DNS={np.interp(kc, k, cdf)*100:.1f}%")
    print("[out]", _OUT + ".pdf")


if __name__ == "__main__":
    main()
