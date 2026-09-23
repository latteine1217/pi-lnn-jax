#!/usr/bin/env python3
"""thesis §7.2 的訓練快照密度 ablation 圖（NTHU thesis 樣式）。

What:
    KE 相對誤差 vs 訓練快照數 T（下軸）／取樣間隔 Δt（上軸），3 seeds 的
    mean ± std，兩條線分別是固定 eval stride 與 matched Δt 兩種評估協定。
    數據來源見 `_common.exp_snapshot_data`（與 TMLR 版同一份，不雙寫）。

Why:
    thesis 舊圖是 PyTorch 側的 single-seed 五點（201/101/51/26/11），y 軸為
    KE MAPE 與 low-band 誤差。本圖改用 JAX 側的 3-seed 資料，因此有兩處
    刻意的差異，caption 必須跟著改，不能沿用舊敘述：

      1. 少 T=11 那一點 —— JAX campaign 只掃到 stride 8（T=26）。舊圖「Δt=0.5 s
         時 low-band 衝到 16%、越過 10% 判準」的斷崖在本圖看不到。
      2. y 軸是 `ke_rel_err`（場的 KE 相對誤差），不是舊圖的 KE MAPE 與
         low-band 誤差 —— 這三者是不同的量，數值不可互相比較。舊段落的
         「5.9–6.5%」「16.0%」全部不適用於本圖。

    matched-Δt 這條線是舊圖沒有的：CfC 以實際 Δt 前進狀態，評估時餵入的 Δt
    本身就是變因，分離後可見固定-stride 曲線的退化有 82–83% 來自 Δt 分佈
    位移而非訓練資料量減少。

Usage:
    uv run python scripts/plot_temporal_density_ablation.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common.exp_snapshot_data import (  # noqa: E402
    EXP1_T, EXP1_KE, EXP1_KE_MATCHED, WINDOW_SECONDS, N_FRAMES_STORED,
)
from journal_style import (  # noqa: E402
    setup_style, figwidth, save_figure, PICON, BASELINE, PICON_LS, MUTED,
)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _mean_std(dct: dict, keys: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([np.mean(dct[k]) for k in keys]),
            np.array([np.std(dct[k]) for k in keys]))


def _dt_of(n_snapshots: int) -> float:
    """快照數 → 取樣間隔 Δt [s]（固定時窗 T=5 s，端點含首尾）。"""
    return WINDOW_SECONDS / (n_snapshots - 1)


def main() -> int:
    setup_style("thesis")
    fig, ax = plt.subplots(figsize=(figwidth("thesis", "single"), 3.0))

    T = np.array(EXP1_T)
    series = [
        (r"matched $\Delta t$", EXP1_KE_MATCHED, PICON, "o", PICON_LS),
        (r"fixed eval stride", EXP1_KE, BASELINE, "^", "-"),
    ]
    for label, dct, colour, marker, ls in series:
        m, s = _mean_std(dct, EXP1_T)
        ax.errorbar(T, 100 * m, yerr=100 * s, color=colour, marker=marker,
                    linestyle=ls, capsize=2.5, label=label)

    # 飽和點：T≈101 之後加倍快照數零收益（見 knowledge 判讀）。
    # 用 axes 座標下標註，錨點不會隨資料範圍跑到軸外被裁掉。
    ax.axvline(101, color=MUTED, linestyle=":", linewidth=0.8, zorder=0)
    ax.text(0.60, 0.55, r"saturation $N_t\approx101$", transform=ax.transAxes,
            fontsize=7, color="0.35", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in EXP1_T])
    ax.minorticks_off()
    ax.set_xlabel(r"Training snapshots $N_t$ over $T=5$ s")
    ax.set_ylabel(r"Field KE relative error $e_{\rm KE}^{\rm field}$ (%)")
    ax.legend(loc="upper right")

    # 上軸：同一組刻度改標 Δt，讀者不必自己換算
    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(T)
    secax.set_xticklabels([f"{_dt_of(t):.3g}" for t in EXP1_T])
    secax.minorticks_off()
    secax.set_xlabel(r"Sensor sampling interval $\Delta t$ (s)")

    out = (Path(__file__).resolve().parent.parent
           / "paper" / "thesis-format" / "figures" / "results"
           / "temporal_density_ablation")
    fig.tight_layout()
    paths = save_figure(fig, str(out))
    print(f"[out] {[str(p) for p in paths]}")
    print(f"[data] {N_FRAMES_STORED}-frame trajectory, 3 seeds, "
          f"Delta_t = {_dt_of(T[0]):.3g}..{_dt_of(T[-1]):.3g} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
