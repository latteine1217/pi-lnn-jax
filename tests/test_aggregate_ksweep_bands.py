"""EXP-510 聚合腳本的承重行為 —— 三條都是「安靜給錯數字」的防線。

這支腳本的輸出會直接被拿去判讀事前登錄的預測，所以它的失敗必須是**大聲的**：
缺 seed 不得偷偷用子集平均、舊指標產物不得混入、high band 的前導幀排除規則
必須真的生效（那是事前登錄的分析選擇，不是實作細節）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_ksweep_bands import (
    SEEDS,
    _collect,
    _high_band,
    _mean_metrics,
    sensor_band_edge,
)

_MEAN = {
    "uv_rel_err": 0.06,
    "band_rel_err_low": 0.007,
    "band_rel_err_mid": 0.089,
    "omega_rel_err": 0.29,
}


def _payload(high_series: list[float], mean: dict | None = None) -> dict:
    return {
        "ckpt_step": 20000,
        "metrics_mean": dict(_MEAN if mean is None else mean),
        "metrics_per_t": [{"band_rel_err_high": v} for v in high_series],
    }


def _write(root: Path, name: str, subdir: str, payload: dict) -> Path:
    d = root / name / subdir
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(payload))
    return root / name


def test_sensor_band_edge_matches_the_identifiability_toolchain():
    """k_s 必須與 diag_identifiability 同定義——判讀整條 K 序列都建在這個數上。"""
    assert sensor_band_edge(100) == pytest.approx(5.6419, abs=1e-4)
    assert sensor_band_edge(400) == pytest.approx(11.2838, abs=1e-4)


def test_frame_zero_is_excluded_from_the_high_band():
    """事前登錄的排除規則：初始場的 high band 退化（K=100 實測 2983）。

    沒有這條，單一幀就會主宰整條序列的平均——2983 vs 其餘 ~1.0。
    """
    got = _high_band(_payload([2983.0, 0.99, 0.98, 0.97]))
    assert got["high_mean"] == pytest.approx(0.98, abs=1e-9), "frame 0 沒被排除"
    assert got["n_frames_used"] == 3
    assert got["frame0_dropped"] == 2983.0, "被丟掉的值必須留在報告裡（可稽核）"


def test_non_finite_high_band_frames_are_dropped_not_propagated():
    got = _high_band(_payload([2983.0, 0.9, float("nan"), 1.1]))
    assert got["n_frames_used"] == 2
    assert got["high_mean"] == pytest.approx(1.0, abs=1e-9)


def test_all_high_band_frames_unusable_fails_loudly():
    with pytest.raises(SystemExit, match="無法判讀"):
        _high_band(_payload([2983.0]))


def test_old_metric_artifact_is_rejected_rather_than_averaged_in():
    """舊指標 eval 的 uv_rel_err 是 null。混進聚合 = 安靜地少算一個 seed。"""
    stale = {**_MEAN, "uv_rel_err": None}
    with pytest.raises(SystemExit, match="null"):
        _mean_metrics(_payload([2983.0, 0.99], mean=stale))


def test_missing_run_fails_instead_of_partial_aggregation(tmp_path: Path):
    """缺 seed 不得用子集平均——eval-script-discipline 的 multi-seed 紅線。"""
    run = _write(tmp_path, "ksweep_k400_s42", "eval_v2_chunk400k", _payload([2983.0, 0.99, 0.98]))
    assert _collect(run, "eval_v2_chunk400k")["band_rel_err_mid"] == pytest.approx(0.089)

    with pytest.raises(SystemExit, match="缺 metrics.json"):
        _collect(tmp_path / "ksweep_k400_s1", "eval_v2_chunk400k")


def test_seed_list_is_fixed_in_source_not_discovered(tmp_path: Path):
    """seed 清單是事前宣告的常數。若有人改成「掃描目錄有幾個算幾個」，這條會紅。"""
    assert SEEDS == (42, 1, 2), (
        "SEEDS 被改動——那是 EXP-510 事前登錄的一部分，"
        "改它等於改實驗設計，須同步更新 knowledge/experiments 的登錄")
