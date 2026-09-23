#!/usr/bin/env python3
"""plot_energy_spectrum.py — DNS vs reconstruction energy spectrum E(k)（K=100 / K=800）。

What:
    從 docs/figures/ksweep_data.json 取 mid-time-slice 的 azimuthal E(k)（compute_energy_spectrum，
    shell 涵蓋到 Fourier 方盒的**對角 corner**，N=256 → 182 shells，非 N//2=128
    ——這是 Parseval 精確的必要條件），畫 DNS truth + pred@K=100 + pred@K=800，
    log-log。展示重建在能量主導大尺度（低 k）貼合 DNS，且 K 增大→高 k 小尺度逐步補上。

Why:
    背書「能量/大尺度在 K=100 已大致捕捉、小尺度隨 K 改善」的 K-sweep 結論（spectrum_rel_err
    1.68%@200→0.71%@800）。資料源:lab-server multire exp245 final_eval metrics_per_t[mid]。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_style import STYLE_CYCLE, figwidth, save_figure, setup_style  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "ksweep_data.json")
_OUT = "docs/figures/energy_spectrum"


def main() -> None:
    sp = json.load(open(_DATA))["spectrum"]
    E_dns = sp["100"]["E_dns_k"]            # DNS truth（K 無關）
    E100 = sp["100"]["E_pred_k"]
    E800 = sp["800"]["E_pred_k"]
    k = list(range(1, len(E_dns)))          # 跳過 k=0（mean/DC）

    def s(arr):
        return arr[1:len(E_dns)]

    setup_style("tmlr")
    fig, ax = plt.subplots(figsize=(figwidth("tmlr", "half"), 2.7))
    ax.loglog(k, s(E_dns), color="0.0", marker="", linestyle="-", lw=1.4, label="DNS (truth)")
    c1, m1, _ = STYLE_CYCLE[0]
    c2, m2, _ = STYLE_CYCLE[1]
    ax.loglog(k, s(E100), color=c1, marker=m1, linestyle="--", markevery=12, label="recon $K{=}100$")
    ax.loglog(k, s(E800), color=c2, marker=m2, linestyle="-.", markevery=12, label="recon $K{=}800$")

    ax.set_xlabel(r"Wavenumber $k$ (–)")
    ax.set_ylabel(r"Energy spectrum $E(k)$ (a.u.)")
    ax.legend(loc="lower left")
    save_figure(fig, _OUT)
    print("[out]", _OUT + ".pdf", "| n_shells:", len(E_dns))


if __name__ == "__main__":
    main()
