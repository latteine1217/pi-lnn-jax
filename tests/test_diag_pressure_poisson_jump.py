"""diag_pressure_poisson_jump 的可測部分：狀態查詢規則、左極限時刻、相對差。

跳躍量測的承重假設是「t_i − ε 查到的是前一個狀態、t_i 查到的是新狀態」。這個假設若
錯（ε 被 float32 吃掉、或查詢規則與 decoder 不同），診斷會安靜地量到 trunk 的連續變化
而不是跳躍。ckpt 相關路徑需要 GPU 寫出的 checkpoint，本機測不到；這裡釘住不需要 ckpt
的那幾塊。
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from _paths import REPO_ROOT


def _load(name: str):
    """以檔案路徑載入 `scripts/` 下的模組——不動 sys.path，也不吃 cwd。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load("diag_pressure_poisson_jump")
state_index = _mod.state_index
left_limit_times = _mod.left_limit_times
rel_diff_uv = _mod.rel_diff_uv
EPS_T = _mod.EPS_T

SENSOR_T = np.linspace(0.0, 5.0, 101).astype(np.float32)   # 主線：101 幀 @ 0.05 s


def test_state_index_matches_decoder_rule():
    """與 models.py:985 相同：Σ[st ≤ t] − 1，落在時戳上取該狀態，之間取前一個。"""
    assert state_index(SENSOR_T, 0.0) == 0
    assert state_index(SENSOR_T, float(SENSOR_T[7])) == 7
    assert state_index(SENSOR_T, float(SENSOR_T[7]) + 0.01) == 7
    assert state_index(SENSOR_T, float(SENSOR_T[7]) - 0.01) == 6
    # t 早於第一個時戳時 clip 到 0（與 decoder 的 clip 相同）
    assert state_index(SENSOR_T, -1.0) == 0
    assert state_index(SENSOR_T, 99.0) == len(SENSOR_T) - 1


def test_left_limit_times_land_on_previous_state_for_every_switch():
    """主線幾何下，每個切換點 t_i − ε 都查到狀態 i−1——含 t 接近 5 的末段（float32 間距最大處）。"""
    t_minus = left_limit_times(SENSOR_T, EPS_T)
    assert len(t_minus) == len(SENSOR_T) - 1
    for i, tm in enumerate(t_minus, start=1):
        assert state_index(SENSOR_T, float(tm)) == i - 1
        assert SENSOR_T[i] - tm == pytest.approx(EPS_T, rel=1e-2)


def test_left_limit_rejects_eps_lost_to_float32():
    """ε 小於 float32 在 t≈5 的間距時，t_i − ε == t_i，查到的是新狀態——必須 fail-fast。"""
    with pytest.raises(ValueError, match="查到狀態"):
        left_limit_times(SENSOR_T, 1e-9)


def test_left_limit_rejects_eps_not_smaller_than_spacing():
    with pytest.raises(ValueError, match="不小於最小 sensor 間距"):
        left_limit_times(SENSOR_T, 0.05)


def test_left_limit_rejects_non_monotone_time():
    with pytest.raises(ValueError, match="嚴格遞增"):
        left_limit_times(np.array([0.0, 0.1, 0.1], dtype=np.float32), 1e-4)


def test_rel_diff_uses_third_field_as_denominator():
    """分母是第三組場：跳躍以 t_i 的場正規化，不是以左極限的場。"""
    rng = np.random.default_rng(0)
    u_a, v_a = rng.normal(size=(2, 8, 8))
    u_b, v_b = u_a + 0.1, v_a.copy()
    num = np.sqrt(np.sum(0.1 ** 2 * np.ones((8, 8))))
    den_a = np.sqrt(np.sum(u_a ** 2 + v_a ** 2))
    den_b = np.sqrt(np.sum(u_b ** 2 + v_b ** 2))
    assert rel_diff_uv(u_a, v_a, u_b, v_b, u_a, v_a) == pytest.approx(num / den_a)
    assert rel_diff_uv(u_a, v_a, u_b, v_b, u_b, v_b) == pytest.approx(num / den_b)
    assert rel_diff_uv(u_a, v_a, u_a, v_a, u_a, v_a) == 0.0


# ── diag_spectral_floor 的純函式 ──────────────────────────────────────────

_floor = _load("diag_spectral_floor")


def test_band_median_uses_inclusive_integer_shells_and_rejects_overflow():
    E = np.arange(10, dtype=float)
    assert _floor.band_median(E, (2, 4)) == 3.0
    with pytest.raises(ValueError, match="超出"):
        _floor.band_median(E, (5, 10))


def test_parse_times_snaps_to_eval_times_and_rejects_unknown():
    eval_t = np.linspace(0.0, 5.0, 101)
    assert _floor.parse_times("5.0,2.5", eval_t) == [5.0, 2.5]
    with pytest.raises(ValueError, match="不在評估時刻內"):
        _floor.parse_times("2.51", eval_t)


# ── diag_unseen_window 的純函式 ───────────────────────────────────────────

_unseen = _load("diag_unseen_window")


def test_grid_indices_maps_on_grid_points_and_wraps():
    N = 256
    pos = np.array([[0.0, 0.5], [255 / 256, 1 / 256], [1.0, 0.25]])
    ix, iy = _unseen.grid_indices(pos, N)
    assert ix.tolist() == [0, 255, 0] and iy.tolist() == [128, 1, 64]


def test_grid_indices_rejects_off_grid_points():
    with pytest.raises(ValueError, match="不在"):
        _unseen.grid_indices(np.array([[0.5 + 0.3 / 256, 0.5]]), 256)


def test_window_frames_requires_exact_start_and_enough_frames():
    t = np.linspace(0.0, 10.0, 201)
    assert _unseen.window_frames(t, 5.0, 101).tolist() == list(range(100, 201))
    with pytest.raises(ValueError, match="不是 DNS 的幀時刻"):
        _unseen.window_frames(t, 5.01, 101)
    with pytest.raises(ValueError, match="超出"):
        _unseen.window_frames(t, 5.05, 101)


# ── make_shifted_window_data 的純函式 ─────────────────────────────────────

_win = _load("make_shifted_window_data")


def test_window_indices_inclusive_and_exact():
    t = np.linspace(0.0, 10.0, 401)
    idx = _win.window_indices(t, 5.0, 5.0)
    assert idx[0] == 200 and idx[-1] == 400 and len(idx) == 201
    with pytest.raises(ValueError, match="端點"):
        _win.window_indices(t, 5.01, 5.0)


def test_extract_matches_generator_schema():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 8, 8)); v = rng.normal(size=(3, 8, 8))
    x = np.arange(8) / 8
    ix, iy = _win.grid_index(np.array([[0.25, 0.5], [0.875, 0.0]]), x, x)
    su, sv = _win.extract(u, v, ix, iy)
    assert su.shape == (2, 3) and su.dtype == np.float32
    assert np.allclose(su[0], u[:, 2, 4]) and np.allclose(sv[1], v[:, 7, 0])
