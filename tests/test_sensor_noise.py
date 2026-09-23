"""apply_sensor_noise 的 isolated 單元測試（純 numpy）。

量測噪音模型：sensor 讀值本身帶噪，故噪音在資料載入階段注入一次、訓練全程
看到同一份 noisy 觀測——不是每步重抽（那是 augmentation，語意不同）。

RNG 刻意用 host-side numpy 且與 training seed 分離：run.py 的主 RNG 消費順序
是 bit-identical 契約的承重面，噪音若從主 RNG 取子鍵，會讓 noise_frac=0 與
非 0 兩種跑法的後續 RNG 流錯位，連帶讓 clean baseline 不可重現。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.data import apply_sensor_noise


def _vals(T=40, K=12, C=2, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 兩個 channel 給明顯不同的尺度，才驗得出 per-channel 縮放
    u = rng.normal(0.0, 1.0, (T, K))
    v = rng.normal(0.0, 10.0, (T, K))
    return np.stack([u, v], axis=-1)


def test_zero_noise_is_identity():
    """noise_frac=0 必須 bit-identical——clean baseline 不能因為這條路徑存在而改變。"""
    x = _vals()
    out = apply_sensor_noise(x, 0.0, seed=42)
    assert out is x or np.array_equal(out, x)


def test_per_channel_sigma_scales_with_channel_std():
    """噪音幅度是各 channel 自身 std 的固定比例，不是全域單一尺度。"""
    x = _vals(T=400, K=60)
    frac = 0.10
    out = apply_sensor_noise(x, frac, seed=7)

    delta = out - x
    for c in range(x.shape[-1]):
        expected = frac * x[..., c].std()
        assert delta[..., c].std() == pytest.approx(expected, rel=0.05)


def test_same_seed_reproduces():
    x = _vals()
    a = apply_sensor_noise(x, 0.05, seed=123)
    b = apply_sensor_noise(x, 0.05, seed=123)
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs():
    x = _vals()
    a = apply_sensor_noise(x, 0.05, seed=1)
    b = apply_sensor_noise(x, 0.05, seed=2)
    assert not np.array_equal(a, b)


def test_input_not_mutated():
    """回傳新陣列；原地改會污染呼叫端持有的 clean 觀測。"""
    x = _vals()
    before = x.copy()
    apply_sensor_noise(x, 0.05, seed=3)
    np.testing.assert_array_equal(x, before)


def test_negative_fraction_raises():
    """負噪音沒有意義，是設定打錯——不要靜默當 0 處理。"""
    with pytest.raises(ValueError):
        apply_sensor_noise(_vals(), -0.01, seed=0)


def test_shape_preserved():
    x = _vals(T=17, K=5, C=2)
    out = apply_sensor_noise(x, 0.03, seed=0)
    assert out.shape == x.shape
