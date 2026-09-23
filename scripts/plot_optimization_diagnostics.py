#!/usr/bin/env python3
"""訓練收斂診斷三聯圖（thesis appendix：optimization diagnostics）。

What:
    從訓練 stdout（Slurm .out）畫三個 panel：
      (a) loss 分項軌跡（total / sensor / momentum / continuity），log-y
      (b) GradNorm 任務權重隨 step 的重新配置
      (c) Augmented-Lagrangian dual λ 與 continuity 違反量

Why:
    方法端主打 stiff multi-task + AL + GradNorm，但論文只有最終的 divergence
    ratio 這個間接證據。λ 的軌跡是「continuity constraint 確實在作用」的直接
    證據：λ 隨違反量累積而成長，收斂後才平緩。

    資料來源是 stdout 而非結構化檔案，因為訓練不寫 metrics 檔（`artifacts/
    ledger/` 記的是 RNG 重播欄位）。解析規則見 `_common.train_log`。

Usage:
    uv run python scripts/plot_optimization_diagnostics.py --log logs/jax_train_<jobid>.out
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common.train_log import parse_training_log_file  # noqa: E402
from journal_style import (  # noqa: E402
    setup_style, figwidth, save_figure, STYLE_CYCLE, PICON, MUTED,
)

import matplotlib.pyplot as plt  # noqa: E402

# GradNorm 權重欄位 w_d/u/v/c 的語意順序（三 task 時 continuity 走 AL-only）
_TASK_LABELS_4 = ["data", r"mom-$u$", r"mom-$v$", "continuity"]
_TASK_LABELS_3 = ["data", r"mom-$u$", r"mom-$v$"]

_LOSS_SERIES = [
    ("total", "total"),
    ("sensor", "sensor data"),
    ("mom_u", r"momentum $u$"),
    ("mom_v", r"momentum $v$"),
    ("cont", "continuity"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True,
                   help="訓練 stdout 檔（lab-server logs/jax_train_<jobid>.out）")
    p.add_argument("--out", default=None,
                   help="輸出 stem，預設 paper/thesis-format/figures/results/"
                        "optimization_diagnostics")
    p.add_argument("--venue", default="thesis",
                   help="繪圖樣式 venue（thesis=boxed CFD 風格，tmlr=minimal）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    d = parse_training_log_file(args.log)
    step = d["step"]

    setup_style(args.venue)
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(figwidth(args.venue, "single"), 6.4))

    # (a) loss 分項
    ax = axes[0]
    for i, (key, label) in enumerate(_LOSS_SERIES):
        colour, marker, ls = STYLE_CYCLE[i]
        ax.plot(step, d[key], color=colour, linestyle=ls, label=label)
    ax.set_yscale("log")
    ax.set_ylabel(r"loss term $\mathcal{L}_i$ (–)")
    ax.legend(ncol=3, loc="upper right")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    # (b) GradNorm 任務權重
    ax = axes[1]
    weights = d["task_weights"]
    n_task = weights.shape[1]
    labels = _TASK_LABELS_4 if n_task == 4 else _TASK_LABELS_3
    if n_task not in (3, 4):
        labels = [f"task {i}" for i in range(n_task)]
    for i in range(n_task):
        colour, marker, ls = STYLE_CYCLE[i]
        ax.plot(step, weights[:, i], color=colour, linestyle=ls, label=labels[i])
    ax.set_ylabel(r"GradNorm weight $w_i$ (–)")
    ax.legend(ncol=4, loc="upper right")
    ax.text(0.01, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    # (c) AL dual λ（左軸）與 continuity 違反量（右軸）
    ax = axes[2]
    ax.plot(step, d["lambda_al"], color=PICON, label=r"AL dual $\lambda$")
    ax.set_ylabel(r"AL dual $\lambda$ (–)")
    ax.set_xlabel(r"training step $s$")
    ax2 = ax.twinx()
    ax2.plot(step, d["cont"], color=MUTED, linestyle=":",
             label="continuity residual")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"continuity residual $\mathcal{C}$ (–)")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    ax.text(0.01, 1.02, "(c)", transform=ax.transAxes, fontweight="bold")

    stem = args.out or str(
        Path(__file__).resolve().parent.parent
        / "paper" / "thesis-format" / "figures" / "results"
        / "optimization_diagnostics"
    )
    fig.tight_layout()
    paths = save_figure(fig, stem)
    print(f"[out] {[str(p) for p in paths]}")
    print(f"[data] {len(step)} logged steps, {step[0]}..{step[-1]}, "
          f"{n_task} GradNorm tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
