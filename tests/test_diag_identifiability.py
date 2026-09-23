"""diag_identifiability sensor-band 分解 substrate 的 isolated 單元測試（Ticket 01）。

全部用合成場（不載 DNS/fields.npz），驗證：
  - sensor band edge √(K/π) 隨 K 正確變動
  - band_filter 保相位、正確隔離 wavenumber band、low band 含 DC
  - velocity_band_rel_l2 把「只重建 low band」的 pred 正確分成 low≈0 / mid≈1
  - reconstruct_uv_rel_l2 從分量重組 == 直接 vector rel-L2
  - fail-fast：形狀 / 長度不一致 raise
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.diag_identifiability import (
    band_filter,
    band_limited_floor,
    mid_band_linear_observability,
    radial_wavenumber_grid,
    reconstruct_uv_rel_l2,
    sensor_band_edge,
    velocity_band_rel_l2,
    _per_frame_rel_l2,
    _per_frame_vec_rel_l2,
)


# ── sensor band edge ──────────────────────────────────────────────────────────

def test_sensor_band_edge_values_and_monotone():
    assert sensor_band_edge(100) == pytest.approx(np.sqrt(100 / np.pi), rel=1e-9)
    assert sensor_band_edge(400) == pytest.approx(np.sqrt(400 / np.pi), rel=1e-9)
    assert sensor_band_edge(100) < sensor_band_edge(400)  # 加 sensor → band edge 上移


def test_sensor_band_edge_rejects_nonpositive():
    with pytest.raises(ValueError):
        sensor_band_edge(0)


# ── band_filter：保相位、隔離 band、含 DC ────────────────────────────────────

def _cos_mode(N, k):
    """沿 x 的純 cos mode，|k|=(k,0)。回 [1,N,N]（沿 y 常數）。"""
    X = np.arange(N)
    row = np.cos(2 * np.pi * k * X / N)
    return np.broadcast_to(row[:, None], (N, N)).astype(np.float64)[None, :, :]


def test_band_filter_isolates_single_mode():
    N = 16
    kr = radial_wavenumber_grid(N)
    field = _cos_mode(N, 3)  # |k|=3
    k_s = sensor_band_edge(100)  # 5.64
    kept_low = band_filter(field, kr, -1.0, k_s)      # low 含 |k|=3
    kept_mid = band_filter(field, kr, k_s, 16.0)      # mid 不含 |k|=3
    np.testing.assert_allclose(kept_low, field, atol=1e-10)
    np.testing.assert_allclose(kept_mid, 0.0, atol=1e-10)


def test_band_filter_low_band_includes_dc():
    N = 16
    kr = radial_wavenumber_grid(N)
    const = np.full((1, N, N), 5.0)  # 純 DC（k=0）
    low_with_dc = band_filter(const, kr, -1.0, sensor_band_edge(100))
    low_without_dc = band_filter(const, kr, 0.0, sensor_band_edge(100))  # kr>0 排除 DC
    np.testing.assert_allclose(low_with_dc, const, atol=1e-10)
    np.testing.assert_allclose(low_without_dc, 0.0, atol=1e-10)


# ── velocity_band_rel_l2：band 隔離 ──────────────────────────────────────────

def test_velocity_band_rel_l2_isolates_bands():
    """pred 只重建 low band（cos3），mid band（cos6）全缺 → low u_rel≈0、mid u_rel≈1。"""
    N = 16
    cos3 = _cos_mode(N, 3)   # low, |k|=3 ≤ 5.64
    cos6 = _cos_mode(N, 6)   # mid, |k|=6 ∈ (5.64,16]
    u_dns = np.repeat(cos3 + cos6, 2, axis=0)   # T=2
    u_pred = np.repeat(cos3, 2, axis=0)         # 只重建 low
    v = np.repeat(cos3, 2, axis=0)              # v 完美（both cos3）

    res = velocity_band_rel_l2(u_pred, v, u_dns, v, K=100)
    assert res["k_sensor_edge"] == pytest.approx(sensor_band_edge(100))
    assert res["bands"]["low"]["u_rel"] < 0.01     # low 完美
    assert res["bands"]["mid"]["u_rel"] > 0.95      # mid 完全缺（pred=0）
    assert res["bands"]["mid"]["uv_rel"] > 0.95
    assert res["bands"]["low"]["v_rel"] < 0.01      # v 全 band 完美


def test_velocity_band_rel_l2_shape_mismatch_raises():
    a = np.zeros((2, 8, 8)); b = np.zeros((2, 8, 4))
    with pytest.raises(ValueError):
        velocity_band_rel_l2(a, a, a, b, K=100)


# ── reconstruct_uv_rel_l2：分量重組 == 直接 vector ───────────────────────────

def test_reconstruct_uv_matches_direct():
    rng = np.random.default_rng(0)
    T, N = 3, 8
    u_dns = rng.standard_normal((T, N, N)); v_dns = rng.standard_normal((T, N, N))
    u_pred = u_dns + 0.1 * rng.standard_normal((T, N, N))
    v_pred = v_dns + 0.2 * rng.standard_normal((T, N, N))

    direct = _per_frame_vec_rel_l2(u_pred, v_pred, u_dns, v_dns, eps=1e-12)

    u_rel_t = np.array([np.linalg.norm(u_pred[t] - u_dns[t]) / np.linalg.norm(u_dns[t]) for t in range(T)])
    v_rel_t = np.array([np.linalg.norm(v_pred[t] - v_dns[t]) / np.linalg.norm(v_dns[t]) for t in range(T)])
    un2 = np.sum(u_dns ** 2, axis=(1, 2)); vn2 = np.sum(v_dns ** 2, axis=(1, 2))
    recon = reconstruct_uv_rel_l2(u_rel_t, v_rel_t, un2, vn2)

    assert recon == pytest.approx(direct, rel=1e-9)


def test_reconstruct_length_mismatch_raises():
    with pytest.raises(ValueError):
        reconstruct_uv_rel_l2([0.1, 0.2], [0.1], [1.0, 1.0], [1.0, 1.0])


# ── band_limited_floor：無-PDE 資訊下界 ──────────────────────────────────────

def test_band_limited_floor_equals_unresolved_energy():
    """DNS=cos3(low)+cos6(mid)，等幅正交 → floor=‖cos6‖/‖cos3+cos6‖=1/√2。"""
    N = 16
    field = np.repeat(_cos_mode(N, 3) + _cos_mode(N, 6), 2, axis=0)  # T=2
    floor = band_limited_floor(field, field, K=100)  # k_s=5.64: cos3 resolved, cos6 unresolved
    assert floor == pytest.approx(1 / np.sqrt(2), rel=1e-6)


def test_band_limited_floor_zero_when_all_resolved():
    """DNS 只有 low band → floor≈0（無 unresolved 能量）。"""
    N = 16
    field = np.repeat(_cos_mode(N, 3), 2, axis=0)
    assert band_limited_floor(field, field, K=100) < 1e-9


def test_band_limited_floor_per_frame_shape():
    N = 16
    field = np.repeat(_cos_mode(N, 6), 3, axis=0)  # T=3
    series = band_limited_floor(field, field, K=100, per_frame=True)
    assert series.shape == (3,)


# ── mid_band_linear_observability：mid-band 欠定 → null_fraction 高 ───────────

def test_mid_band_underdetermined_high_null_fraction():
    """K=100 sensor 觀測 (2K=200) < mid-band DoF (~700) → mid-band 大量不可線性觀測。"""
    rng = np.random.default_rng(1)
    coords = rng.random((100, 2))  # K=100 隨機 sensor 位置 ∈ [0,1]²
    res = mid_band_linear_observability(coords, K=100)
    assert res["mid_dof"] > 2 * res["n_obs"] // 2  # mid dof 遠超觀測數
    assert res["null_fraction"] > 0.5              # 欠定 → 過半 mid-band 不可觀測
    assert res["rank"] <= res["n_obs"]             # rank 上界 = 觀測數


def test_mid_band_observability_more_sensors_lower_null():
    """加 sensor（K 大）→ 觀測數↑ → mid-band null_fraction 下降（更可觀測）。"""
    rng = np.random.default_rng(2)
    coords_400 = rng.random((400, 2))
    coords_100 = coords_400[:100]
    null_100 = mid_band_linear_observability(coords_100, K=100)["null_fraction"]
    null_400 = mid_band_linear_observability(coords_400, K=400)["null_fraction"]
    assert null_400 < null_100
