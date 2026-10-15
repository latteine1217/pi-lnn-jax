#!/usr/bin/env python3
"""cross-Re × sensor-budget 圖：重建誤差 vs 感測器數 K，逐 Re 一條線。

What:
    讀 campaign 15 格（Re ∈ {100,500,1000,10⁴,10⁶} × K ∈ {10,50,100}）以現行
    指標套件重評的 metric_artifact.json（lab-server jobs 5554、5567-5570、
    5576-5585），畫兩個 panel vs K：(a) 速度向量 (u,v) 相對 L2 誤差、
    (b) 渦度相對 L2 誤差，兩者都是評估窗內的時間平均。
    對應 thesis 附錄的 fig:crossre_sensor_budget 與 pi-lnn 的 EXP-320~335。

Why:
    (a) 用速度向量誤差而非 KE：它是 thesis 登錄的 primary metric，而
    tab:crossre_kstar 三欄與 tab:crossre_baselines 都讀自同一批重評產物——圖讀
    同一批，表中的門檻才能從圖上交叉讀出。舊的 final_eval/metrics.json 只有
    u、v 分量、沒有向量誤差，因此不再讀它。

    這張圖的全部意義在於「同一條線上只有 K 在變、線與線之間只有 Re 在變」。
    因此缺一個 (Re, K) 就預設硬失敗：靜默少畫一個點會讓曲線的斜率變成另一件
    事，而圖上看不出來。要畫未跑完的中間結果必須 `--allow-partial` 明確表態，
    此時圖上會標注實際涵蓋的格子數。

Usage:
    uv run python scripts/plot_crossre_sensor_budget.py
    uv run python scripts/plot_crossre_sensor_budget.py --allow-partial
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import (  # noqa: E402
    setup_style, figwidth, save_figure, STYLE_CYCLE,
)

import matplotlib.pyplot as plt  # noqa: E402
from pi_lnn_jax.metric_artifact import (  # noqa: E402
    VORTICITY_REL_ERR_TIME_MEAN_V1,
    read_metric_summary,
)

# 與 gen_crossre_configs.py 的 RE_SPECS 對應（label 用於 artifacts 路徑）
RE_LABELS = [("100", 100.0), ("500", 500.0), ("1000", 1000.0),
             ("10000", 10000.0), ("1e6", 1e6)]
K_VALUES = [10, 50, 100]

# 重評產物落在各格的 eval_uv/（evaluate_exp245 的 OUTPUT_SUBDIR）。唯一例外是
# 這批的第一支 job 5554（Re=10³, K=100），落在 eval_uv_matched/；它宣告
# fixed_grid stride=1，而 Re=10³ 訓練的 time_strides[0] 本來就是 1，時間軸與
# 同 Re 另兩格的 follow_training 相同（T=104, t∈[0, 10.3]）。路徑照 lab-server
# 原樣鏡像，不在本機改名，rsync 回來的結構才對得上。
EVAL_SUBDIR = "eval_uv"
EVAL_SUBDIR_OVERRIDES = {("1000", 100): "eval_uv_matched"}

# canonical artifact 對同一個 metric definition（velocity.vector.rel_l2.v1）
# 同時存 time_mean 與 time_p90 兩個 summary，只給 definition id 會因歧義而
# raise；因此與 VORTICITY_REL_ERR_TIME_MEAN_V1 一樣點名 time_mean summary。
VELOCITY_REL_ERR_TIME_MEAN_V1 = "velocity.vector.rel_l2.time_mean.v1"
METRICS = [(VELOCITY_REL_ERR_TIME_MEAN_V1,
            r"Velocity $(u,v)$ relative error (%)"),
           (VORTICITY_REL_ERR_TIME_MEAN_V1,
            r"Vorticity $\omega$ relative error (%)")]


def _re_display(re_value: float) -> str:
    """legend 用次方式，與內文的 $\\{10^2, 5\\times10^2, 10^3, 10^4, 10^6\\}$ 同調。
    先前前四個印十進位、只有第五個印次方，圖例內部就不一致。"""
    exponents = {1e2: r"$Re=10^{2}$", 5e2: r"$Re=5\times10^{2}$",
                 1e3: r"$Re=10^{3}$", 1e4: r"$Re=10^{4}$", 1e6: r"$Re=10^{6}$"}
    return exponents.get(float(re_value), rf"$Re={int(re_value)}$")


def collect(root: Path) -> tuple[dict, list[str]]:
    """讀取 campaign 重評產出 → {(re_label, K): {summary id: value}}，並回報缺哪些。"""
    found, missing = {}, []
    for label, _ in RE_LABELS:
        for k in K_VALUES:
            subdir = EVAL_SUBDIR_OVERRIDES.get((label, k), EVAL_SUBDIR)
            path = root / f"crossre_re{label}_k{k}" / subdir / "metric_artifact.json"
            if not path.is_file():
                missing.append(f"Re={label} K={k}: {path}")
                continue
            with open(path) as f:
                blob = json.load(f)
            summaries = [
                read_metric_summary(blob, definition_id)
                for definition_id, _ in METRICS
            ]
            found[(label, k)] = {
                summary.definition_id: summary.value for summary in summaries
            }
    return found, missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-partial", action="store_true",
                   help="允許 campaign 未跑完就出圖；圖上會標注涵蓋的格子數")
    p.add_argument("--venue", default="thesis")
    p.add_argument("--out", default=None)
    p.add_argument("--artifacts-root", default=None,
                   help="campaign 產出的根目錄，預設 artifacts/kolmogorov/；"
                        "結果 rsync 到別處時指向該處")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = Path(args.artifacts_root) if args.artifacts_root \
        else repo / "artifacts" / "kolmogorov"
    found, missing = collect(root)

    n_total = len(RE_LABELS) * len(K_VALUES)
    if missing and not args.allow_partial:
        detail = "\n  ".join(missing)
        raise SystemExit(
            f"[ERR] campaign 未跑完：{len(found)}/{n_total} 個格子有結果，缺：\n"
            f"  {detail}\n"
            f"缺格子會改變曲線斜率而圖面看不出來。確定要畫部分結果請加 "
            f"--allow-partial。"
        )
    if not found:
        raise SystemExit("[ERR] 一個結果都沒有；campaign 尚未執行")

    setup_style(args.venue)
    fig, axes = plt.subplots(1, 2, figsize=(figwidth(args.venue, "single"), 3.0))

    for panel_tag, (ax, (key, ylabel)) in zip("ab", zip(axes, METRICS)):
        for i, (label, re_value) in enumerate(RE_LABELS):
            ks = [k for k in K_VALUES if (label, k) in found]
            if not ks:
                continue
            ys = [100 * found[(label, k)][key] for k in ks]
            colour, marker, ls = STYLE_CYCLE[i]
            ax.plot(ks, ys, color=colour, marker=marker, linestyle=ls,
                    label=_re_display(re_value))
        ax.set_xscale("log")
        ax.set_yscale("log")
        # 下限固定 1%：速度誤差的最小值 1.38% 讓自動範圍只含一個 decade，log 軸
        # 只剩 10¹ 一個刻度標籤就讀不出尺度。兩格同一下限，底部 decade 也對齊。
        ax.set_ylim(bottom=1.0)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.minorticks_off()
        ax.set_xlabel(r"Sensor count $K$")
        ax.set_ylabel(ylabel)
        # 內文以 Figure~\ref{...}(a)/(b) 與 "Panel (b)" 指稱這兩格，圖上原本沒有標記。
        ax.set_title(f"({panel_tag})", loc="left", fontweight="bold")
        # 10% 工程門檻：tab:crossre_kstar 與整章的 budget 判讀都是從這條交叉讀出來的。
        # 先前只有 decade gridline，caption 宣告的「dotted line」在圖上不存在。
        ax.axhline(10.0, color="0.25", linestyle=(0, (1, 2)), linewidth=1.2, zorder=1.5)
        ax.annotate(r"10% target", xy=(K_VALUES[0], 10.0),
                    xytext=(3, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize="small", color="0.25")
    # 兩個 panel 是同一組 Re，共用一個底部 legend：放在 panel 內會壓到曲線
    # （Re=100 那條正好穿過左下角），放圖外則兩邊都不必犧牲資料區。
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    if missing:
        fig.suptitle(f"partial campaign: {len(found)}/{n_total} cells",
                     fontsize=7, color="0.35", y=1.02)
        print(f"[warn] 部分結果出圖（{len(found)}/{n_total}）；缺：")
        for m in missing:
            print(f"    {m}")

    stem = args.out or str(
        repo / "paper" / "thesis-format" / "figures" / "results"
        / "crossre_sensor_budget"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths = save_figure(fig, stem)
    print(f"[out] {[str(x) for x in paths]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
