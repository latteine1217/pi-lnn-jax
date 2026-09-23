#!/usr/bin/env python3
"""plot_exp_snap_drop.py — 2026-07 兩組實驗的 publication-quality 圖。

(a) 實驗1 data-snapshot 數量 → 重建品質（KE rel-err vs 訓練快照數 T）。
(b) 實驗2 sensor dropout robustness（KE rel-err vs dropout rate，B3 CfC vs B0 vanilla）。

數據為 lab-server worktree ~/pi-lnn-jax-expdrop 的 final_eval / dropout sweep
metrics.json 之 3-seed verified snapshot（jobs 4576-4619，2026-07-26）。
資料與繪圖分離：改樣式不需重跑實驗；改數據只動下方 DATA 區。

Venue: TMLR（single-column 6.5in，CM serif，vector PDF）。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))  # repo scripts/ 內的 journal_style
from journal_style import setup_style, figwidth, STYLE_CYCLE, save_figure  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ── DATA（3-seed ke_rel_err，seeds = 42/1/2）────────────────────────────
# 實驗1 的數據與 thesis 圖共用，故住在 _common（見該模組的 provenance 與協定說明）。
from _common.exp_snapshot_data import (  # noqa: E402
    EXP1_T, EXP1_KE, EXP1_KE_MATCHED,
)

# 實驗2：eval-time dropout rate p，B3 CfC vs B0 vanilla（clean-trained）
EXP2_P = [0.0, 0.1, 0.3, 0.5]
EXP2_B3 = {  # snap_st2_s{42,1,2}
    0.0: [0.183, 0.184, 0.183], 0.1: [0.203, 0.214, 0.205],
    0.3: [0.333, 0.363, 0.317], 0.5: [0.452, 0.512, 0.492],
}
EXP2_B0 = {  # drop_b0_s{42,1,2}（0.75M params）
    0.0: [0.262, 0.264, 0.269], 0.1: [0.354, 0.372, 0.364],
    0.3: [0.544, 0.566, 0.561], 0.5: [0.711, 0.736, 0.707],
}
# capacity-matched B0（b0cap_deep，3.08M ≈ 0.98× B3；jobs 4711-4726）
# deep 變體 clean 性能與 p0.5 皆優於 wide → 取為 B0 家族的 capacity-matched 代表
EXP2_B0CAP = {
    0.0: [0.234, 0.248, 0.250], 0.1: [0.319, 0.378, 0.408],
    0.3: [0.527, 0.657, 0.620], 0.5: [0.799, 0.894, 0.833],
}
# 實驗2b：train-time augmentation（drop_train_{arch}_p{train_rate}_s{seed}）
# [seed][eval_p index]，eval_p 序同 EXP2_P
TRAIN_B3 = {
    0.1: [[0.179, 0.179, 0.180, 0.187], [0.177, 0.177, 0.179, 0.185], [0.180, 0.180, 0.184, 0.192]],
    0.3: [[0.174, 0.174, 0.174, 0.174], [0.174, 0.174, 0.174, 0.175], [0.176, 0.176, 0.176, 0.177]],
    0.5: [[0.171, 0.171, 0.171, 0.171], [0.173, 0.173, 0.173, 0.173], [0.174, 0.174, 0.174, 0.174]],
}
TRAIN_B0 = {
    0.1: [[0.248, 0.253, 0.275, 0.315], [0.253, 0.259, 0.282, 0.329], [0.257, 0.262, 0.283, 0.332]],
    0.3: [[0.247, 0.247, 0.251, 0.264], [0.250, 0.250, 0.254, 0.270], [0.251, 0.252, 0.257, 0.275]],
    0.5: [[0.238, 0.238, 0.238, 0.243], [0.245, 0.244, 0.243, 0.248], [0.247, 0.246, 0.245, 0.249]],
}


def mean_std(dct, keys):
    m = np.array([np.mean(dct[k]) for k in keys])
    s = np.array([np.std(dct[k]) for k in keys])
    return m, s


setup_style("tmlr")

# ── Figure 1（實驗1）：單панel、half width，避免多欄擠壓 ──
fig1, axa = plt.subplots(1, 1, figsize=(figwidth("tmlr", "half"), 2.6))
# ── Figure 2（實驗2）：兩 panel、full width → 每欄 3.25in ──
fig2, (axb, axc) = plt.subplots(1, 2, figsize=(figwidth("tmlr", "single"), 2.8))

# ── (a) 實驗1：KE rel-err vs T（log-x，快照數等比 ×2）──
Ta = np.array(EXP1_T)
for i, (lab, dct) in enumerate([
        (r"matched $\Delta t$", EXP1_KE_MATCHED),
        (r"fixed eval stride", EXP1_KE)]):
    m1, s1 = mean_std(dct, EXP1_T)
    c, mk, ls = STYLE_CYCLE[i]
    axa.errorbar(Ta, m1, yerr=s1, color=c, marker=mk, linestyle=ls,
                 capsize=2.5, markersize=5, label=lab)
axa.legend(frameon=False, loc="upper right", fontsize=7)
axa.set_xscale("log")
axa.set_xticks(Ta)
axa.set_xticklabels([str(t) for t in EXP1_T])
axa.minorticks_off()
axa.axvline(101, color="0.6", linestyle=":", linewidth=0.8, zorder=0)
axa.annotate("saturation\n$T\\approx101$", xy=(101, 0.28), xytext=(150, 0.30),
             fontsize=6.5, color="0.35", ha="center")
axa.set_xlabel(r"Training snapshots $T$ (–)")
axa.set_ylabel(r"KE relative error (–)")

# ── (b) 實驗2：KE rel-err vs dropout rate，B3 vs B0 ──
Pa = np.array(EXP2_P)
for i, (lab, dct) in enumerate([
        ("B3 CfC (3.1M)", EXP2_B3),
        ("B0 vanilla (0.75M)", EXP2_B0),
        ("B0 cap-matched (3.1M)", EXP2_B0CAP)]):
    m, s = mean_std(dct, EXP2_P)
    c, mk, ls = STYLE_CYCLE[i]
    axb.errorbar(Pa, m, yerr=s, color=c, marker=mk, linestyle=ls,
                 capsize=2.5, markersize=5, label=lab)
axb.set_xlabel(r"Eval dropout rate $p$ (–)")
axb.set_ylabel(r"KE relative error (–)")
axb.set_xticks(Pa)
axb.set_title("clean-trained", fontsize=8, pad=3)
axb.legend(frameon=False, loc="upper left", fontsize=6.5)
axb.text(-0.17, 1.02, "(a)", transform=axb.transAxes, fontweight="bold")

# ── (c) train-time augmentation：eval p=0.5 誤差 vs 訓練 dropout rate ──
# x=0 為 clean-trained（未做 augmentation），顯示 augmentation 強度的單調效果
TR = [0.0, 0.1, 0.3, 0.5]
for i, (lab, clean_ke, tr) in enumerate([
        ("B3 (CfC)", EXP2_B3[0.5], TRAIN_B3),
        ("B0 (vanilla)", EXP2_B0[0.5], TRAIN_B0)]):
    m = [np.mean(clean_ke)] + [np.mean([s[3] for s in tr[t]]) for t in (0.1, 0.3, 0.5)]
    sd = [np.std(clean_ke)] + [np.std([s[3] for s in tr[t]]) for t in (0.1, 0.3, 0.5)]
    c, mk, ls = STYLE_CYCLE[i]
    axc.errorbar(TR, m, yerr=sd, color=c, marker=mk, linestyle=ls,
                 capsize=2.5, markersize=5, label=lab)
axc.set_xlabel(r"Train dropout rate (–)")
axc.set_ylabel(r"KE rel. error at $p{=}0.5$ (–)")
axc.set_xticks(TR)
axc.set_title("train-time augmentation", fontsize=8, pad=3)
axc.legend(frameon=False, loc="upper right", fontsize=7)
axc.text(-0.17, 1.02, "(b)", transform=axc.transAxes, fontweight="bold")

figdir = Path(__file__).parent.parent / "docs" / "figures"
figdir.mkdir(parents=True, exist_ok=True)
fig1.tight_layout()
fig2.tight_layout()
p1 = save_figure(fig1, str(figdir / "exp2026_07_snapshot"))
p2 = save_figure(fig2, str(figdir / "exp2026_07_dropout"))
print("saved:", [str(p) for p in p1 + p2])
