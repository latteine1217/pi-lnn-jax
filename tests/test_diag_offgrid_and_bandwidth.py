"""兩支新診斷的可測部分：時間軸分組與 Fourier 頻寬換算。

這兩支腳本的結論會進稿子（A4 的 off-grid 代價、k≥16 譜地板的歸因），所以它們的
判準邏輯要有回歸保護。ckpt 相關的路徑需要 GPU 寫出的 checkpoint，本機測不到；
這裡釘住的是**不需要 ckpt 的那兩塊**——分組與波數換算——正是搞錯了會讓整份
診斷安靜給出錯數字的地方。

每個斷言都做過突變檢查（改壞實作會紅），紀錄見 commit message。
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


_offgrid = _load("diag_offgrid_query")
_bandwidth = _load("diag_spatial_emb_bandwidth")
T_MATCH_TOL = _offgrid.T_MATCH_TOL
split_on_off_grid = _offgrid.split_on_off_grid
_reach = _bandwidth._reach
_stats = _bandwidth._stats


# ── 時間軸分組 ────────────────────────────────────────────────────────────

def test_splits_dns_frames_into_on_and_off_grid():
    """主線幾何：DNS 201 幀 @0.025s、sensor 101 幀 @0.05s → 偶數幀 on、奇數幀 off。"""
    dns_t = np.linspace(0.0, 5.0, 201)
    sensor_time = np.linspace(0.0, 5.0, 101)
    on, off, gap = split_on_off_grid(dns_t, sensor_time)
    assert on.sum() == 101 and off.sum() == 100
    assert on[::2].all(), "偶數幀應全部落在 sensor 時戳上"
    assert off[1::2].all(), "奇數幀應全部落在兩個時戳之間"
    # 中點到最近時戳恰為半個 sensor 步長
    assert np.allclose(gap[off], 0.025, atol=1e-9)


def test_tolerance_does_not_swallow_midpoints():
    """容差必須遠小於半個 sensor 步長，否則中點會被誤判成 on-grid。"""
    dns_t = np.linspace(0.0, 5.0, 201)
    sensor_time = np.linspace(0.0, 5.0, 101)
    half_step = 0.025
    assert T_MATCH_TOL < half_step / 10.0
    _, off, _ = split_on_off_grid(dns_t, sensor_time, tol=half_step * 1.01)
    assert off.sum() == 0, "容差放大到超過半步時，所有幀都會被吞成 on-grid——這是要避免的失效"


def test_identical_axes_leave_no_off_grid_frames():
    """sensor 與 DNS 同格點時沒有 off-grid 對象；呼叫端據此 fail-fast。"""
    t = np.linspace(0.0, 1.0, 11)
    on, off, _ = split_on_off_grid(t, t)
    assert on.all() and off.sum() == 0


def test_rejects_non_1d_axes():
    with pytest.raises(ValueError, match="1D"):
        split_on_off_grid(np.zeros((3, 2)), np.zeros((3,)))


# ── Fourier 頻寬換算 ──────────────────────────────────────────────────────

def test_amplitude_split_follows_the_encoding_order():
    """period_enc 是 [sin(cx), cos(cx), sin(cy), cos(cy)]：前兩列是 x、後兩列是 y。

    把能量全放在 x 那兩列，R_y 必須為 0。列序搞反是這支腳本最容易犯而不會報錯的錯。
    """
    B = np.zeros((4, 8))
    B[0] = 3.0          # sin(cx)
    B[1] = 4.0          # cos(cx)  → R_x = 5
    s = _stats(B)
    assert np.isclose(s["x"]["median"], 5.0)
    assert np.isclose(s["y"]["median"], 0.0)

    B2 = np.zeros((4, 8))
    B2[2] = 6.0
    B2[3] = 8.0         # R_y = 10
    s2 = _stats(B2)
    assert np.isclose(s2["y"]["median"], 10.0)
    assert np.isclose(s2["x"]["median"], 0.0)


def test_harmonic_reach_is_monotone_and_matches_bessel():
    """|J_n(R)| 的可達階數：R 越大能表達的諧波越高，且與 scipy 的 Bessel 值一致。"""
    from scipy.special import jv

    r_small, r_large = 2.5, 16.0
    n_small, n_large = _reach(r_small), _reach(r_large)
    assert n_small < n_large, "R 變大時可達諧波階必須上升"
    # 邊界自洽：第 n 階在門檻上、第 n+1 階在門檻下
    for r, n in ((r_small, n_small), (r_large, n_large)):
        assert abs(jv(n, r)) >= 1e-3
        assert abs(jv(n + 1, r)) < 1e-3
    # init 尺度（R≈2.5）根本到不了 k=16——這是本診斷要回答的問題的前提
    assert n_small < 16


def test_reach_is_zero_when_no_harmonic_clears_the_floor():
    """R→0 時只有 J_0 存在（=常數），其餘全被門檻擋掉。"""
    assert _reach(1e-6) == 0


# ── 訓練時間軸的 fallback 解析 ────────────────────────────────────────────

resolve_training_axes = _offgrid.resolve_training_axes


def test_explicit_keys_win_and_are_labelled_as_such():
    stride, scale, prov = resolve_training_axes(
        {"time_strides": [4], "re_norm_scale": 1e6})
    assert stride == 4 and scale == 1e6
    assert prov["sensor_stride_source"] == "config.time_strides[0]"
    assert prov["re_norm_scale_source"] == "config.re_norm_scale"


def test_missing_keys_take_the_same_fallback_training_took():
    """主線 exp_main5_* 兩個鍵都沒設，訓練端自己走 fallback；診斷必須走同一個。

    這裡釘住的是**數值**：stride 2 來自 assembly.py:194、scale 1e4 來自
    config.py:306。與訓練端不同就是拿錯條件比較，而那不會有錯誤訊息。
    """
    stride, scale, prov = resolve_training_axes({})
    assert stride == 2, "訓練端 `if time_strides else 2`"
    assert scale == 1e4, "schema default"
    assert "fallback" in prov["sensor_stride_source"]
    assert "default" in prov["re_norm_scale_source"]


def test_fallback_is_recorded_not_silent():
    """用了 fallback 這件事必須進 provenance——靜默套預設正是 eval 紅線禁的。"""
    _, _, prov = resolve_training_axes({})
    assert prov["sensor_stride_source"] != prov["re_norm_scale_source"]
    assert all(isinstance(v, str) and v for v in prov.values())
