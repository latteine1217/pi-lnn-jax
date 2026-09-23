#!/usr/bin/env python
"""compare_baselines — 把 interp / gappy-POD / SHRED / PI-CON 的多-Re metrics 合成四方對照表。

讀三個 metrics JSON（皆 rows=[{Re, held_out, u_rel_err, ...}]），統一成「每 Re × 每 method」表，
印出 + 出圖 + 存合成 JSON。純讀檔合成，不重跑任何 eval。

讀取／正規化／method taxonomy 全部委派 `pi_lnn_jax.result_table.ResultTable`（deep 讀側）：
「無 method 欄 → 該 producer 的 method id」是 projection 的 `default_method`，metric 存取
一律經語意層——投影鍵被改名／漂移時**大聲失敗**，不再靜默變 None。

來源:
  --baselines : evaluate_baselines.py 輸出（含 interp_linear / gappy_pod，每 row 有 method）
  --shred     : train_baseline_shred.py 輸出（method=shred）
  --picon     : evaluate_multi_re.py 輸出（無 method 欄 → 標為 pi-con）

用法:
  uv run python scripts/compare_baselines.py \
    --baselines artifacts/baseline_eval/baselines_metrics_labserver.json \
    --shred artifacts/shred/shred_metrics.json \
    --picon artifacts/baseline_eval/picon_full_8re.json \
    --metric u_rel_err \
    --output artifacts/baseline_eval/compare_4way.json \
    --fig artifacts/baseline_eval/compare_4way.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pi_lnn_jax.result_table import ResultTable, Split, method_label, metric_definition_id

#: 合成 JSON 的 rows 一律帶這四個 metric 欄（與舊版逐位元一致）。
_METRIC_COLUMNS = ("u_rel_err", "ke_rel_err", "omega_rel_err", "low_band_rel_err")


def parse_args():
    p = argparse.ArgumentParser(description="四方 baseline 對照合成")
    p.add_argument("--baselines", required=True)
    p.add_argument("--shred", required=True)
    p.add_argument("--picon", required=True)
    p.add_argument("--metric", default="u_rel_err")
    p.add_argument("--output", default="artifacts/baseline_eval/compare_4way.json")
    p.add_argument("--fig", default="artifacts/baseline_eval/compare_4way.png")
    return p.parse_args()


def _load(path):
    return json.loads(Path(path).read_text())


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else " n/a "


def main():
    args = parse_args()
    # 「無 method 欄 → producer 的 method id」規則落在 default_method（picon → pi-con；
    # shred 的 row 本就帶 method="shred"，仍傳 default 作保險）。
    table = ResultTable.from_projections([
        _load(args.baselines),
        (_load(args.shred), "shred"),
        (_load(args.picon), "pi-con"),
    ])
    methods = list(table.methods())          # taxonomy／論文欄序
    res = list(table.reynolds_numbers())
    held = {re: table.reynolds_split(re) == Split.HELD_OUT for re in res}
    m = args.metric
    defn = metric_definition_id(m)           # 鍵漂移在此大聲失敗，不靜默回 None
    col_defns = {k: metric_definition_id(k) for k in _METRIC_COLUMNS}

    def cell(re_value, method):
        return table.value(method, re_value, defn)

    # ── 表 ──
    hdr = f"{'Re':>9} {'split':>6} " + " ".join(f"{method_label(x):>18}" for x in methods)
    print("=" * len(hdr))
    print(f"四方對照 — metric = {m}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for re_value in res:
        tag = "HELD" if held[re_value] else "train"
        line = f"{re_value:>9.0f} {tag:>6} " + " ".join(f"{_fmt(cell(re_value, x)):>18}" for x in methods)
        print(line)
    print("-" * len(hdr))

    # ── 各 method in-train / held-out 均值 ──
    def mean(method, split):
        vals = [v for u in table.units(method=method, split=split)
                if (v := u.value(defn)) is not None]
        return sum(vals) / len(vals) if vals else None
    print(f"\n{'method':>20} {'in-train':>10} {'held-out':>10}")
    summary = {}
    for x in methods:
        it, ho = mean(x, Split.IN_TRAIN), mean(x, Split.HELD_OUT)
        summary[x] = {"in_train_mean": it, "held_out_mean": ho}
        print(f"{method_label(x):>20} {_fmt(it):>10} {_fmt(ho):>10}")

    # ── rows：從 table 的 units 重建（Re / method / held_out + 四個 metric 欄，
    #    欄位順序與舊版一致，逐位元對齊）──
    rows = []
    for u in table.units():
        r = {"Re": u.reynolds, "method": u.method, "held_out": u.split == Split.HELD_OUT}
        for k in _METRIC_COLUMNS:
            r[k] = u.value(col_defns[k])
        rows.append(r)

    # ── 圖：u_err vs Re，每 method 一線；held-out Re 以垂直帶標示 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for re_value, h in held.items():
            if h:
                ax.axvspan(re_value * 0.9, re_value * 1.1, color="0.9", zorder=0)
        cmap = plt.get_cmap("tab10")
        for i, x in enumerate(methods):
            xs = [r for r in res if cell(r, x) is not None]
            ys = [cell(r, x) for r in xs]
            ax.plot(xs, ys, "-o", color=cmap(i), label=method_label(x), markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("Reynolds number")
        ax.set_ylabel(f"Relative L2 error ({m})")
        ax.set_title("Sparse reconstruction vs. Re (shaded = held-out Re)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figp = Path(args.fig)
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp, dpi=200)
        fig.savefig(figp.with_suffix(".pdf"))
        print(f"\n[fig] {figp} (+.pdf)")
    except Exception as e:  # 圖失敗不擋表/JSON
        print(f"[warn] 圖產生失敗：{e}")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"metric": m, "methods": methods, "rows": rows, "summary": summary}, indent=2))
    print(f"[out] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
