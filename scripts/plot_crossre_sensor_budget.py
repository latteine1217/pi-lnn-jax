#!/usr/bin/env python3
"""cross-Re × sensor-budget 圖：重建誤差 vs 感測器數 K，逐 Re 一條線。

What:
    讀 campaign 的 15 個 final_eval/metrics.json（Re ∈ {100,500,1000,10⁴,10⁶}
    × K ∈ {10,50,100}），畫兩個 panel：KE 相對誤差與渦度相對誤差 vs K。
    對應 thesis 的 cross-Re sensor-budget 圖與 pi-lnn 的 EXP-320~335。

Why:
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
    KE_T_MAPE_SPATIALMEAN_V1,
    LegacyInterpretation,
    VORTICITY_REL_ERR_TIME_MEAN_V1,
    read_metric_summary,
)

# 與 gen_crossre_configs.py 的 RE_SPECS 對應（label 用於 artifacts 路徑）
RE_LABELS = [("100", 100.0), ("500", 500.0), ("1000", 1000.0),
             ("10000", 10000.0), ("1e6", 1e6)]
K_VALUES = [10, 50, 100]

# KE 用 ke_t_mape（E(t) 軌跡的 MAPE），不是 metrics_mean 的 ke_rel_err。
# 兩者同樣叫「KE 誤差」但不是同一個量：ke_rel_err 是逐時場的 KE 相對誤差，
# 在 Re=10⁴/K=100 為 18.96%，而 thesis 該格報 4.56%——對得上的是 ke_t_mape
# 的 4.48%。用錯的那個會讓整張圖高估三到四倍。
METRICS = [(KE_T_MAPE_SPATIALMEAN_V1, "KE MAPE (%)"),
           (VORTICITY_REL_ERR_TIME_MEAN_V1,
            r"Vorticity $\omega$ relative error (%)")]

# 這張 thesis 圖定義上比較 spatial-mean KE time-series MAPE。舊 campaign
# artifacts 產於 pointwise_v2 marker 導入前，故其 unmarked ke_t_mape 明示解讀
# 為同一 spatial-mean definition；package reader 不從日期、路徑或檔名自行推測。
LEGACY_KE_INTERPRETATION = LegacyInterpretation(
    definition_id=KE_T_MAPE_SPATIALMEAN_V1,
    basis="cross-Re thesis campaign predates the pointwise_v2 definition",
)


def _re_display(re_value: float) -> str:
    """legend 用次方式，與內文的 $\\{10^2, 5\\times10^2, 10^3, 10^4, 10^6\\}$ 同調。
    先前前四個印十進位、只有第五個印次方，圖例內部就不一致。"""
    exponents = {1e2: r"$Re=10^{2}$", 5e2: r"$Re=5\times10^{2}$",
                 1e3: r"$Re=10^{3}$", 1e4: r"$Re=10^{4}$", 1e6: r"$Re=10^{6}$"}
    return exponents.get(float(re_value), rf"$Re={int(re_value)}$")


def collect(
    root: Path,
    *,
    legacy_interpretation: LegacyInterpretation | None = None,
) -> tuple[dict, list[str]]:
    """讀取 campaign 產出 → {(re_label, K): metrics_mean}，並回報缺哪些。"""
    found, missing = {}, []
    for label, _ in RE_LABELS:
        for k in K_VALUES:
            result_dir = root / f"crossre_re{label}_k{k}" / "final_eval"
            canonical_path = result_dir / "metric_artifact.json"
            compatibility_path = result_dir / "metrics.json"
            path = canonical_path if canonical_path.is_file() else compatibility_path
            if not path.is_file():
                missing.append(
                    f"Re={label} K={k}: {canonical_path} or {compatibility_path}"
                )
                continue
            with open(path) as f:
                blob = json.load(f)
            summaries = [
                read_metric_summary(
                    blob,
                    definition_id,
                    legacy_interpretation=(legacy_interpretation
                                           if definition_id == KE_T_MAPE_SPATIALMEAN_V1
                                           else None),
                )
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
    found, missing = collect(
        root,
        legacy_interpretation=LEGACY_KE_INTERPRETATION,
    )

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
