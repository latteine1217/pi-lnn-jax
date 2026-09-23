"""spectrum_rel_error 的 isolated 單元測試（純 numpy，不載 model/DNS）。

驗證 energy spectrum E(k) 的 relative-L2 誤差定義（補 low_band_rel_err 成完整 ε_spectrum）。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.evaluate import spectrum_rel_error


def test_spectrum_rel_error_identical_zero():
    """相同譜 → 0。"""
    e = np.array([3.0, 2.0, 1.0, 0.5])
    assert spectrum_rel_error(e, e) == pytest.approx(0.0)


def test_spectrum_rel_error_scaled_known():
    """E_pred = 1.1·E_dns → relative L2 = ||0.1·E||/||E|| = 0.1。"""
    e = np.array([1.0, 2.0, 3.0])
    assert spectrum_rel_error(1.1 * e, e) == pytest.approx(0.1)


def test_spectrum_rel_error_length_mismatch_raises():
    """長度不符 → fail-fast（不可鬆散對齊；evalStrictness）。"""
    with pytest.raises(ValueError):
        spectrum_rel_error(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
