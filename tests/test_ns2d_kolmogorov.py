"""ns2d_kolmogorov 前向求解器的 isolated 單元測試（Ticket 04）。

不載 DNS（大檔），用解析解 / 守恆律 / 可微性驗證求解器物理正確：
  - 單一 y-mode 純黏性衰減 = 解析 exp(-ν k² t)（該情形非線性項恆為 0）
  - vorticity_to_velocity 給 divergence-free 速度
  - dt 不整除 t_final → fail-fast
  - integrate 對初始場可微（4D-Var adjoint 前置）
（求解器對 DNS 的重現驗證見 __main__，實測渦度 rel-L2 0.04–0.40%。）
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as _jax_config

from scripts.ns2d_kolmogorov import (
    dealias_mask,
    integrate,
    vorticity_to_velocity,
    wavenumbers,
)


# x64 只在本檔測試範圍內開，結束還原。**不得改回 module-level 開啟**：
# `jax.config.update` 是全域設定，洩漏出去會讓 test_pipeline_replay（bit-identical
# 黃金測試）與 test_rwf 在完整套件下失敗、單獨跑卻通過（實測）。
@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)



def test_single_y_mode_pure_viscous_decay():
    """ω=cos(2π·3·y) 只依賴 y → 非線性項恆 0 → 純黏性衰減 exp(-ν(2π·3)²t)（解析）。"""
    N = 32
    g = np.arange(N) / N
    _, Y = np.meshgrid(g, g, indexing="ij")
    k = 3
    omega0 = np.cos(2 * np.pi * k * Y)
    nu, t = 0.01, 0.1
    out = np.asarray(integrate(omega0, t, dt=1e-3, nu=nu, k_f=2, A=0.0))
    decay = np.exp(-nu * (2 * np.pi * k) ** 2 * t)
    np.testing.assert_allclose(out, omega0 * decay, atol=2e-4)


def test_velocity_from_vorticity_is_divergence_free():
    """vorticity_to_velocity 給的 (u,v) 滿足 ∂x u + ∂y v ≈ 0（譜導數）。"""
    N = 32
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((N, N))
    omega -= omega.mean()  # 零均值
    kx, ky, k2, k2_inv = wavenumbers(N)
    u, v = vorticity_to_velocity(jnp.fft.fft2(jnp.asarray(omega)), kx, ky, k2_inv)
    div = jnp.fft.ifft2(1j * kx * jnp.fft.fft2(u) + 1j * ky * jnp.fft.fft2(v)).real
    assert float(jnp.max(jnp.abs(div))) < 1e-9


def test_integrate_dt_must_divide_t_final():
    with pytest.raises(ValueError):
        integrate(np.zeros((16, 16)), t_final=0.025, dt=0.007, nu=1e-4, k_f=2, A=0.1)


def test_integrate_differentiable_wrt_initial_field():
    """4D-Var adjoint 前置：∂(scalar of forward)/∂ω0 可由 jax.grad 求得且有限。"""
    N = 16
    omega0 = jnp.asarray(np.random.default_rng(1).standard_normal((N, N)))
    omega0 = omega0 - omega0.mean()

    def loss(w0):
        out = integrate(w0, t_final=0.01, dt=5e-3, nu=1e-4, k_f=2, A=0.1)
        return jnp.sum(out ** 2)

    grad = jax.grad(loss)(omega0)
    assert grad.shape == (N, N)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0  # 非平凡梯度


def test_dealias_mask_zeros_high_wavenumbers():
    N = 24
    m = np.asarray(dealias_mask(N))
    cut = (N // 2) * (2.0 / 3.0)
    idx = np.fft.fftfreq(N, d=1.0 / N)
    # 沿 x 軸最高波數應被遮罩
    hi = int(np.argmax(np.abs(idx)))
    assert m[hi, 0] == (np.abs(idx[hi]) <= cut)
    assert m[0, 0] == 1.0  # DC 保留
