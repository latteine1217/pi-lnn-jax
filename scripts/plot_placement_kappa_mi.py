#!/usr/bin/env python3
"""plot_placement_kappa_mi.py — sec:placement 主文圖：κ(CΦ_r) 與 MI vs r（LES + DNS）。

What:
    從 diag_observability.py 的兩個 artifact（部署 K=100 sensors）讀 per-r 的 condition
    number κ 與能量加權 mutual information，畫成雙面板 TMLR 圖：
      (a) κ(CΦ_r) vs r（log-log）：LES-POD（deployed）與 DNS-POD（oracle transfer）兩條，
          標出 r_99.9 能量子空間邊界與 2K 線性觀測上限。
      (b) I(a;y) vs r：MI 在 r≈r_99.9 飽和。
    一張圖同時呈現 placement conditioning 與 LES→DNS transfer。

Usage:
    PYTHONPATH=. uv run python scripts/plot_placement_kappa_mi.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo 內 scripts/journal_style.py
from journal_style import STYLE_CYCLE, figwidth, save_figure, setup_style  # noqa: E402

_LES = "artifacts/diag/observability_re10000.json"
_DNS = "artifacts/diag/observability_dns_re10000_transfercheck.json"
_OUT = "docs/figures/placement_kappa_mi"


def _load(path: str):
    lp = json.load(open(path))["les_pod"]
    r = [e["r"] for e in lp["per_r"]]
    kappa = [e["cond"] for e in lp["per_r"]]
    mi = [e["mutual_info_bits"] for e in lp["per_r"]]
    er = {float(k): v for k, v in lp["effective_rank"].items()}
    return r, kappa, mi, er


def main() -> None:
    setup_style("tmlr")
    rL, kL, miL, erL = _load(_LES)
    rD, kD, miD, erD = _load(_DNS)
    K = json.load(open(_LES))["K"]
    twoK = 2 * K
    r999_lo, r999_hi = sorted((erL[0.999], erD[0.999]))  # r_99.9 LES–DNS 範圍帶

    cL, mL, lsL = STYLE_CYCLE[0]   # LES：blue o solid
    cD, mD, lsD = STYLE_CYCLE[1]   # DNS：vermillion ^ dashed
    fig, (axk, axm) = plt.subplots(1, 2, figsize=(figwidth("tmlr", "single"), 2.7))

    # ── (a) condition number ──
    for r, y, c, m, ls, lab in ((rL, kL, cL, mL, lsL, "LES-POD (deployed)"),
                                (rD, kD, cD, mD, lsD, "DNS-POD (oracle)")):
        axk.loglog(r, y, color=c, marker=m, linestyle=ls, label=lab)
    axk.axvspan(r999_lo, r999_hi, color="0.6", alpha=0.18, lw=0)
    axk.axvline(twoK, color="0.3", linestyle=":", lw=0.9)
    axk.text(r999_hi * 1.05, 1.5, r"$r_{99.9}$", fontsize=7, color="0.3")
    axk.text(twoK * 0.62, 4e2, r"$2K$ limit", fontsize=7, color="0.3", ha="right")
    axk.set_xlabel(r"POD modes $r$ (mode index)")
    axk.set_ylabel(r"Condition number $\kappa(C\Phi_r)$ (–)")
    axk.legend(loc="upper left")
    axk.text(-0.20, 1.04, "(a)", transform=axk.transAxes, fontweight="bold")

    # ── (b) mutual information（per-basis normalized：避免 raw bits 跨 basis 被誤讀；
    #        承重訊息是「在 r≈r_99.9 飽和」，正規化後兩條都趨近 1）──
    miL_n = [v / max(miL) for v in miL]
    miD_n = [v / max(miD) for v in miD]
    for r, y, c, m, ls, lab in ((rL, miL_n, cL, mL, lsL, "LES-POD (deployed)"),
                                (rD, miD_n, cD, mD, lsD, "DNS-POD (oracle)")):
        axm.semilogx(r, y, color=c, marker=m, linestyle=ls, label=lab)
    axm.axvspan(r999_lo, r999_hi, color="0.6", alpha=0.18, lw=0)
    axm.text(r999_hi * 1.05, 0.30, r"$r_{99.9}$", fontsize=7, color="0.3")
    axm.set_xlabel(r"POD modes $r$ (mode index)")
    axm.set_ylabel(r"Normalized MI $I(a;\,y)/I_{\max}$ (–)")
    axm.text(-0.20, 1.04, "(b)", transform=axm.transAxes, fontweight="bold")

    fig.tight_layout(pad=0.4)
    paths = save_figure(fig, _OUT)
    print("[out]", [str(p) for p in paths])


if __name__ == "__main__":
    main()
