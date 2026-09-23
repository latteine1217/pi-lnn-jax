"""band_fourier_encode 單元測試（方向1 Increment A 地基）。

band_fourier_encode 在**任意指定波數**做 2D 週期 Fourier 編碼，是 periodic_fourier_encode
的推廣：mid-band [5.64,16] 需要 target 特定波數，而 periodic_fourier_encode 只能連續 1..n。
排列與 periodic_fourier_encode 一致：[sin_x, cos_x, sin_y, cos_y] per 波數。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 不在 module import 時全域改 x64——那會污染整個 test session（例如讓 SHRED 的
# nn.RNN scan 拿到 float64 而炸）。用 float32 + 放寬 atol 即足夠驗證數值等價。

from pi_lnn_jax.models import (
    BandFourierEmb,
    LearnableFourierEmb,
    band_fourier_encode,
    periodic_fourier_encode,
)


def test_contiguous_ints_match_periodic_fourier_encode():
    """波數 [1,2,3] 應與 periodic_fourier_encode(n=3) 逐元素相等（推廣的正確性錨）。"""
    z = jnp.asarray([[0.13, 0.47], [0.90, 0.02], [0.0, 0.5]], dtype=jnp.float32)
    L = 1.0
    got = band_fourier_encode(z, L, [1.0, 2.0, 3.0])
    ref = periodic_fourier_encode(z, L, 3)
    assert got.shape == ref.shape
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0, atol=1e-5)


def test_band_fourier_emb_single_wavenumber_matches_learnable():
    """BandFourierEmb(wavenumbers=(1.0,)) 應與 LearnableFourierEmb 逐元素相等。

    同一 RNG key、同 param name 'kernel'、同 (4,half) kernel shape → bit-identical。
    證明 BandFourierEmb 是既有單頻段 module 的忠實推廣（band=多頻段的一般化）。
    """
    key = jax.random.PRNGKey(0)
    xy = jnp.asarray([[0.13, 0.47], [0.90, 0.02]], dtype=jnp.float32)
    L = 1.0
    ref_mod = LearnableFourierEmb(embed_dim=16)
    band_mod = BandFourierEmb(embed_dim=16, wavenumbers=(1.0,))
    ref_out = ref_mod.apply(ref_mod.init(key, xy, L), xy, L)
    band_out = band_mod.apply(band_mod.init(key, xy, L), xy, L)
    assert band_out.shape == (2, 16)
    np.testing.assert_allclose(np.asarray(band_out), np.asarray(ref_out), rtol=0, atol=1e-5)


def test_band_fourier_emb_frequency_normalized_init():
    """頻率正規化 init：高波數 kernel rows 起始更小（σ_k ∝ (k_min/k)²）。

    這是 curriculum 的靜態版——高頻二階導本就大，讓它們起始更弱 → 訓練穩定、
    低頻保留強度。單一波數時 factor=1（不破壞 test_..._matches_learnable 的等價）。
    """
    key = jax.random.PRNGKey(0)
    xy = jnp.asarray([[0.1, 0.2]], dtype=jnp.float32)
    mod = BandFourierEmb(embed_dim=64, wavenumbers=(6.0, 16.0))
    kernel = np.asarray(mod.init(key, xy, 1.0)["params"]["kernel"])  # (8, 32): 0-3=k6, 4-7=k16
    std_low = np.std(kernel[0:4])   # k=6
    std_high = np.std(kernel[4:8])  # k=16
    # (6/16)^2 ≈ 0.14 → 高波數 std 應明顯小於低波數
    assert std_high < 0.4 * std_low, f"高波數未被正規化壓小: low={std_low:.3f} high={std_high:.3f}"
