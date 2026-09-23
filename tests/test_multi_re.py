"""load_multi_re_sensors：跨-Re T 對齊 + re_norm_scale 可配置行為。

Why: 多-Re 訓練要求所有 Re 的 sensor (T, K) 一致才能共用 jit cache。
不同 Re 的 DNS snapshot 數可能差 1（如 201 vs 200），stride 後仍差 1（51 vs 50），
loader 需顯式截到共同最小 T；同時 re_norm 上界需可配置（訓練集到 Re=1e6）。
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from pi_lnn_jax.data import load_multi_re_sensors


def _make_ds(tmp_path, name, K, T, channels=("u", "v")):
    """造一個 (json, npz) sensor fixture，回傳 json path。"""
    rng = np.random.RandomState(0)
    arrays = {"time": np.linspace(0.0, 5.0, T)}
    for c in channels:
        arrays[c] = rng.randn(K, T).astype(np.float32)
    npz_name = f"{name}_values.npz"
    np.savez(tmp_path / npz_name, **arrays)
    meta = {
        "K": K,
        "resolution": 16,
        "selected_coordinates": [[float(i), float(i)] for i in range(K)],
        "dns_values_npz": npz_name,
        "channels": list(channels),
    }
    jp = tmp_path / f"{name}.json"
    json.dump(meta, open(jp, "w"))
    return str(jp)


def test_t_alignment_off_by_one(tmp_path):
    # 模擬 201 vs 200 raw snapshots，stride=4 → 51 vs 50 → 應截到共同 T=50
    jp_a = _make_ds(tmp_path, "re1000", K=8, T=201)
    jp_b = _make_ds(tmp_path, "re50000", K=8, T=200)
    ds = load_multi_re_sensors([jp_a, jp_b], [1000.0, 50000.0], time_stride=4)
    Ts = {d["sensor_vals"].shape[0] for d in ds}
    assert len(Ts) == 1, f"T 未對齊: {Ts}"
    assert ds[0]["sensor_vals"].shape[0] == 50
    # sensor_time 也須同步截斷
    assert ds[0]["sensor_time"].shape[0] == 50


def test_re_norm_scale_configurable(tmp_path):
    jp_a = _make_ds(tmp_path, "a", K=8, T=200)
    jp_b = _make_ds(tmp_path, "b", K=8, T=200)
    ds = load_multi_re_sensors(
        [jp_a, jp_b], [1000.0, 1_000_000.0], time_stride=4, re_norm_scale=1_000_000.0
    )
    assert abs(ds[0]["re_norm"] - np.log(1000.0) / np.log(1e6)) < 1e-6
    assert abs(ds[1]["re_norm"] - 1.0) < 1e-6  # Re == Re_max → 1.0


def test_re_norm_scale_default_backward_compat(tmp_path):
    # 不傳 re_norm_scale → 預設 1e4，re_norm = log(Re)/log(1e4)（原行為）
    jp = _make_ds(tmp_path, "a", K=8, T=200)
    ds = load_multi_re_sensors([jp], [10000.0], time_stride=4)
    assert abs(ds[0]["re_norm"] - 1.0) < 1e-6


def test_large_T_mismatch_raises(tmp_path):
    # 差距 >2 → 視為 stride 設定錯誤，fail-fast（非靜默截斷）
    jp_a = _make_ds(tmp_path, "a", K=8, T=200)
    jp_b = _make_ds(tmp_path, "b", K=8, T=100)
    with pytest.raises(ValueError, match="T"):
        load_multi_re_sensors([jp_a, jp_b], [1000.0, 50000.0], time_stride=1)


def test_K_mismatch_raises(tmp_path):
    jp_a = _make_ds(tmp_path, "a", K=8, T=200)
    jp_b = _make_ds(tmp_path, "b", K=16, T=200)
    with pytest.raises(ValueError, match="K"):
        load_multi_re_sensors([jp_a, jp_b], [1000.0, 50000.0], time_stride=4)
