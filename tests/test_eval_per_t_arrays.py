"""per_t_to_arrays 的 isolated 單元測試（純 numpy）。

把 evaluate_time_series 的逐時 metrics dict list 轉成 npz-ready 欄位陣列。
繪圖端吃這份輸出，所以對齊必須是硬失敗而不是補值：一個時間點缺欄位卻靜默
補 NaN，畫出來的曲線會少一段而圖上看不出來。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.evaluate import per_t_to_arrays


def test_scalar_fields_become_1d_arrays():
    per_t = [
        {"t": 0.0, "ke_rel_err": 0.10, "u_rel_err": 0.20},
        {"t": 0.5, "ke_rel_err": 0.11, "u_rel_err": 0.21},
        {"t": 1.0, "ke_rel_err": 0.12, "u_rel_err": 0.22},
    ]
    out = per_t_to_arrays(per_t)

    assert set(out) == {"t", "ke_rel_err", "u_rel_err"}
    assert out["t"].shape == (3,)
    np.testing.assert_allclose(out["ke_rel_err"], [0.10, 0.11, 0.12])


def test_spectrum_lists_become_2d():
    """E(k) 這種 per-t list 疊成 [T, n_k]，繪圖端才能直接切片。"""
    per_t = [
        {"t": 0.0, "E_pred_k": [1.0, 2.0, 3.0]},
        {"t": 1.0, "E_pred_k": [1.5, 2.5, 3.5]},
    ]
    out = per_t_to_arrays(per_t)

    assert out["E_pred_k"].shape == (2, 3)
    np.testing.assert_allclose(out["E_pred_k"][1], [1.5, 2.5, 3.5])


def test_missing_key_in_one_timestep_raises():
    """欄位在某個 t 缺席 → fail-fast，不補 NaN。"""
    per_t = [
        {"t": 0.0, "div_ratio": 0.004},
        {"t": 1.0},  # 缺 div_ratio
    ]
    with pytest.raises(ValueError, match="div_ratio"):
        per_t_to_arrays(per_t)


def test_ragged_spectrum_length_raises():
    """E(k) 長度在時間上不一致 → fail-fast（靜默 ragged 會變 object array）。"""
    per_t = [
        {"t": 0.0, "E_pred_k": [1.0, 2.0, 3.0]},
        {"t": 1.0, "E_pred_k": [1.0, 2.0]},
    ]
    with pytest.raises(ValueError, match="E_pred_k"):
        per_t_to_arrays(per_t)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        per_t_to_arrays([])
