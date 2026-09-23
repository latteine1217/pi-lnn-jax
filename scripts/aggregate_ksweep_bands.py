"""EXP-510：K-sweep 重評的 band 聚合 —— 檢驗 k_s 掃頻預測。

What:
    讀 `ksweep_k{K}_s{seed}` 的重評產物，聚合每個 K 的 uv / low / mid（跨 seed
    mean±std），並從 per-t 取 high band，對照 K=100 基準（main5）。

Why:
    事前登錄的預測（knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md）：
    若中頻誤差受 sub-Nyquist 可識別性限制，把 k_s=√(K/π) 往上推應該**只在 k_s 掃過的
    頻帶**塌下去——mid 大降、high 幾乎不動。推翻條件是 high 的相對改善與 mid 相當或更大。

    本腳本只負責把數字擺成可判讀的形狀，**不下結論**。

兩個承重的實作決定（都來自事前登錄，不是這裡臨時決定的）：
  - **seed 清單寫死** `SEEDS`：缺任何一顆就 fail，不做「有幾個算幾個」的平均。
  - **high band 排除 frame 0**：初始場的參考高頻能量趨近 0，實測 K=100 為 2983。
    其餘全用，所有 K 一致。

讀側規矩（scripts/CLAUDE.md §2）：`metrics_mean` 一律經 `result_table` 的語意層
（投影鍵漂移就大聲失敗）；`metrics_per_t` 不在該 seam 範圍內，直接讀。

Usage:
    PYTHONPATH=. uv run python scripts/aggregate_ksweep_bands.py \\
        --subdir eval_v2_chunk400k --out artifacts/ksweep_band_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.result_table import metric_definition_id, read_metric_value  # noqa: E402

#: 事前宣告的 seed 清單（EXP-510）。缺一即 fail——不得用子集平均支撐宣稱。
SEEDS: tuple[int, ...] = (42, 1, 2)
#: 重評涵蓋的 K。K=100 走 baseline（main5），不在此列。
K_VALUES: tuple[int, ...] = (50, 200, 400, 800)
#: 經語意層讀的 metrics_mean 量。
_MEAN_KEYS = ("uv_rel_err", "band_rel_err_low", "band_rel_err_mid", "omega_rel_err")
#: 事前登錄的排除規則：初始場參考高頻能量趨近 0（K=100 實測 2983）。
_HIGH_BAND_SKIP_FRAMES = 1


def sensor_band_edge(K: int) -> float:
    """k_s = √(K/π)，與 diag_identifiability 同一定義。"""
    return math.sqrt(K / math.pi)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"缺 metrics.json：{path}\n  該 run 尚未重評或 job 失敗，不做部分聚合。")
    return json.loads(path.read_text())


def _mean_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """經語意層讀 metrics_mean；鍵漂移 → raise，值為 null → 明確報出。"""
    out: dict[str, float] = {}
    for key in _MEAN_KEYS:
        value = read_metric_value(payload, metric_definition_id(key))
        if value is None:
            raise SystemExit(
                f"{key} 為 null——該產物是舊指標 eval（重評未生效），不可混入聚合。")
        out[key] = float(value)
    return out


def _high_band(payload: dict[str, Any]) -> dict[str, float]:
    """per-t 的 high band，排除事前登錄的退化前導幀。"""
    per_t = payload["metrics_per_t"]
    series = [frame["band_rel_err_high"] for frame in per_t]
    kept = [v for v in series[_HIGH_BAND_SKIP_FRAMES:] if v is not None and math.isfinite(v)]
    if not kept:
        raise SystemExit("high band 逐幀序列在排除前導幀後為空——無法判讀")
    return {
        "high_mean": st.mean(kept),
        "high_median": st.median(kept),
        "high_last": kept[-1],
        "n_frames_used": len(kept),
        "frame0_dropped": series[0],
    }


def _collect(run_dir: Path, subdir: str) -> dict[str, float]:
    payload = _load(run_dir / subdir / "metrics.json")
    return {**_mean_metrics(payload), **_high_band(payload),
            "ckpt_step": payload.get("ckpt_step")}


def _agg(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-510 K-sweep band 聚合")
    p.add_argument("--root", default="artifacts/kolmogorov", help="run 目錄的父層")
    p.add_argument("--subdir", required=True, help="重評輸出子目錄（如 eval_v2_chunk400k）")
    p.add_argument("--baseline-run", default="main5", help="K=100 基準的 run 前綴")
    p.add_argument("--baseline-subdir", default="final_eval")
    p.add_argument("--out", default=None, help="JSON 輸出路徑")
    args = p.parse_args()

    root = Path(args.root)
    rows: dict[int, dict[str, Any]] = {}

    # K=100 基準（main5，與 ksweep 同架構、同佈點家族，只差 K）
    base_per_seed = [
        _collect(root / f"{args.baseline_run}_s{s}", args.baseline_subdir) for s in SEEDS
    ]
    rows[100] = {"per_seed": base_per_seed, "source": f"{args.baseline_run}_s*/{args.baseline_subdir}"}

    for K in K_VALUES:
        rows[K] = {
            "per_seed": [_collect(root / f"ksweep_k{K}_s{s}", args.subdir) for s in SEEDS],
            "source": f"ksweep_k{K}_s*/{args.subdir}",
        }

    report: dict[str, Any] = {
        "seeds": list(SEEDS),
        "subdir": args.subdir,
        "high_band_skip_frames": _HIGH_BAND_SKIP_FRAMES,
        "by_K": {},
    }
    print(f"{'K':>5} {'k_s':>6} {'uv %':>14} {'mid %':>14} {'low %':>13} {'high(逐幀均)':>15}")
    print("-" * 76)
    for K in sorted(rows):
        per_seed = rows[K]["per_seed"]
        cell: dict[str, Any] = {"k_sensor": sensor_band_edge(K), "source": rows[K]["source"],
                                "n_seeds": len(per_seed)}
        for key in (*_MEAN_KEYS, "high_mean"):
            m, s = _agg([r[key] for r in per_seed])
            cell[key] = {"mean": m, "std": s}
        report["by_K"][str(K)] = cell
        print(f"{K:>5} {cell['k_sensor']:>6.2f} "
              f"{cell['uv_rel_err']['mean'] * 100:>8.2f}±{cell['uv_rel_err']['std'] * 100:.2f} "
              f"{cell['band_rel_err_mid']['mean'] * 100:>8.2f}±{cell['band_rel_err_mid']['std'] * 100:.2f} "
              f"{cell['band_rel_err_low']['mean'] * 100:>7.2f}±{cell['band_rel_err_low']['std'] * 100:.2f} "
              f"{cell['high_mean']['mean']:>9.3f}±{cell['high_mean']['std']:.3f}")

    # 事前登錄的判別量：相對於 K=100 的改善幅度，mid vs high
    base = report["by_K"]["100"]
    print("\n相對 K=100 的改善幅度（事前登錄的判別量）")
    print(f"{'K':>5} {'mid 改善':>12} {'high 改善':>12}   判別")
    print("-" * 60)
    for K in sorted(rows):
        if K == 100:
            continue
        cell = report["by_K"][str(K)]
        d_mid = 1.0 - cell["band_rel_err_mid"]["mean"] / base["band_rel_err_mid"]["mean"]
        d_high = 1.0 - cell["high_mean"]["mean"] / base["high_mean"]["mean"]
        cell["improvement_vs_K100"] = {"mid": d_mid, "high": d_high}
        # 判定只對 k_s **高於**基準的 K 有意義：K<100 時 k_s 下降，兩個量都會是負的
        # （退化），拿「mid 改善遠大於 high」去套會誤報。
        if K < 100:
            verdict = f"k_s < 基準（退化對照，mid 惡化 {-d_mid * 100:.0f}%）"
        elif d_mid > 2 * max(d_high, 0.0):
            verdict = "mid≫high（符合 k_s 掃頻）"
        else:
            verdict = "⚠ high 改善與 mid 相當——見推翻條件"
        print(f"{K:>5} {d_mid * 100:>11.1f}% {d_high * 100:>11.1f}%   {verdict}")

    print("\n注意：K≤128 不走 PILNN_EVAL_CHUNK_BUDGET 路徑（eff_chunk=chunk_size），"
          "K≥200 才走 → K=50/100 與 K≥200 之間存在既有的 chunk 路徑差異（量級 ~1e-4，已揭露）。")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n寫出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
