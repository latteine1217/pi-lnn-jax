"""J_p 局部線性化壓力可觀測性的 isolated 單元測試（合成輸入，不載 LES）。

驗證重點：
  - spectral 速度合成 == 已測 numpy loop 合成（convention 安全網）
  - spectral Poisson 解滿足 ∇²p = source（殘差 ≈ 0）
  - J_p(0) = 0（reference 不能用 zero state 的數學事實）
  - J_p autodiff ≈ finite difference
  - LES→ā 投影 round-trip 可還原 band-limited 係數
  - C_{u,p}^lin 形狀 = (3K, 2H)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as _jax_config

from scripts.diag_observability import (
    divfree_half_modes,
    divfree_observation_operator,
    velocity_field_from_psi,
)
from scripts.diag_pressure_observability import (
    _half_mode_fft_indices,
    compute_Jp,
    make_pressure_sensor_map,
    pressure_field,
    project_velocity_to_coeffs,
    streamfunction_velocity_spectral,
)


# 本檔驗證 spectral Poisson 殘差≈0 / J_p autodiff≈finite-diff，需 float64。
# 用 autouse fixture 範圍化 x64（結束還原），避免 import 副作用污染套件其他測試。
@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)


_N = 32
_KMAX = 4


def _setup():
    half = divfree_half_modes(_KMAX)
    rng = np.random.RandomState(7)
    a = rng.randn(2 * len(half))
    return half, a


def test_spectral_synth_matches_loop():
    """FFT 合成的 (u,v) 必須等於已測過的 numpy loop 合成。"""
    half, a = _setup()
    idx_pos, idx_neg, _, _ = _half_mode_fft_indices(half, _N)
    u_s, v_s = streamfunction_velocity_spectral(jnp.asarray(a), idx_pos, idx_neg, _N)
    u_l, v_l = velocity_field_from_psi(a, half, _N)
    np.testing.assert_allclose(np.asarray(u_s), u_l, atol=1e-9)
    np.testing.assert_allclose(np.asarray(v_s), v_l, atol=1e-9)


def test_pressure_solves_poisson():
    """∇²p ≈ source（零均值），spectral Laplacian 殘差到機器精度。"""
    half, a = _setup()
    idx_pos, idx_neg, _, _ = _half_mode_fft_indices(half, _N)
    u, v = streamfunction_velocity_spectral(jnp.asarray(a), idx_pos, idx_neg, _N)
    p = pressure_field(u, v, _N)
    # 重算 source
    fr = np.fft.fftfreq(_N, d=1.0 / _N)
    KX = fr[:, None] * 2 * np.pi
    KY = fr[None, :] * 2 * np.pi
    uh, vh = np.fft.fft2(np.asarray(u)), np.fft.fft2(np.asarray(v))
    ux = np.real(np.fft.ifft2(1j * KX * uh))
    uy = np.real(np.fft.ifft2(1j * KY * uh))
    vx = np.real(np.fft.ifft2(1j * KX * vh))
    vy = np.real(np.fft.ifft2(1j * KY * vh))
    source = -(ux * ux + 2 * uy * vx + vy * vy)
    ph = np.fft.fft2(np.asarray(p))
    lap_p = np.real(np.fft.ifft2(-(KX ** 2 + KY ** 2) * ph))
    resid = lap_p - (source - source.mean())
    assert np.max(np.abs(resid)) / (np.max(np.abs(source)) + 1e-30) < 1e-9


def test_Jp_zero_reference_is_zero():
    """J_p(0)=0：pressure 對速度二次 ⇒ 在 ā=0 線性化得零 Jacobian。"""
    half, _ = _setup()
    K = 5
    rng = np.random.RandomState(1)
    coords = rng.rand(K, 2)
    ix = (coords[:, 0] * _N).astype(int)
    iy = (coords[:, 1] * _N).astype(int)
    a0 = jnp.zeros(2 * len(half))
    Jp = compute_Jp(a0, half, ix, iy, _N)
    assert Jp.shape == (K, 2 * len(half))
    assert np.max(np.abs(Jp)) < 1e-10


def test_Jp_autodiff_matches_finite_difference():
    half, a = _setup()
    K = 4
    rng = np.random.RandomState(2)
    ix = rng.randint(0, _N, K)
    iy = rng.randint(0, _N, K)
    a_bar = jnp.asarray(a)
    Jp = compute_Jp(a_bar, half, ix, iy, _N)
    f = make_pressure_sensor_map(half, ix, iy, _N)
    eps = 1e-5
    for j in [0, 3, 7, 2 * len(half) - 1]:
        e = np.zeros(2 * len(half))
        e[j] = eps
        fd = (np.asarray(f(jnp.asarray(a + e))) - np.asarray(f(jnp.asarray(a - e)))) / (2 * eps)
        np.testing.assert_allclose(Jp[:, j], fd, atol=1e-6, rtol=1e-4)


def test_projection_roundtrip():
    """band-limited 速度場投影回係數應還原（synth↔project 互逆）。"""
    half, a = _setup()
    u, v = velocity_field_from_psi(a, half, _N)
    a_rec = project_velocity_to_coeffs(u, v, half, _N)
    np.testing.assert_allclose(a_rec, a, atol=1e-8)


def test_stacked_operator_shape():
    half = divfree_half_modes(_KMAX)
    K = 6
    rng = np.random.RandomState(3)
    coords = rng.rand(K, 2)
    C_u, _ = divfree_observation_operator(coords, _KMAX)
    ix = (coords[:, 0] * _N).astype(int)
    iy = (coords[:, 1] * _N).astype(int)
    Jp = compute_Jp(jnp.asarray(rng.randn(2 * len(half))), half, ix, iy, _N)
    C_up = np.concatenate([C_u, Jp], axis=0)
    assert C_u.shape == (2 * K, 2 * len(half))
    assert Jp.shape == (K, 2 * len(half))
    assert C_up.shape == (3 * K, 2 * len(half))
