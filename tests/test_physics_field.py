"""Unit tests for field-based vorticity-transport residual.

策略：用解析已知的場驗證 finite-diff 殘差。
"""
from __future__ import annotations

import numpy as np

from pi_lnn_jax.physics_field import vorticity_transport_residual


def _steady_kolmogorov(N: int, A: float, k_f: float, nu: float, L: float = 1.0):
    """穩態層流 Kolmogorov 解 u = (U sin(2 pi k_f y), 0)，
    U = A / (nu (2 pi k_f)^2) 使黏滯項與 forcing 平衡 → 渦量輸送殘差應 ≈ 0。
    """
    U = A / (nu * (2.0 * np.pi * k_f) ** 2)
    y = (np.arange(N) * (L / N))[None, :]          # [1, Ny]
    u2d = U * np.sin(2.0 * np.pi * k_f * y) * np.ones((N, 1))  # depends on y only
    v2d = np.zeros((N, N))
    return u2d, v2d, U


def test_steady_kolmogorov_residual_near_zero():
    N, A, k_f, nu = 128, 0.1, 2.0, 1e-4
    u2d, v2d, U = _steady_kolmogorov(N, A, k_f, nu)
    T = 5
    u = np.repeat(u2d[None], T, axis=0)            # steady → replicate in time
    v = np.repeat(v2d[None], T, axis=0)
    t = np.linspace(0.0, 0.2, T)
    R = vorticity_transport_residual(u, v, t, nu=nu, A=A, k_f=k_f)
    # 殘差量級 vs forcing-curl 特徵量級 A*(2 pi k_f)
    scale = A * (2.0 * np.pi * k_f)
    assert R.max() / scale < 0.02, f"steady residual {R.max():.3e} too large vs scale {scale:.3e}"
    print(f"✓ steady_kolmogorov_residual_near_zero: max|R|/scale = {R.max()/scale:.4e}")


def test_residual_shape_and_finite():
    rng = np.random.RandomState(0)
    T, N = 4, 32
    u = rng.randn(T, N, N)
    v = rng.randn(T, N, N)
    t = np.arange(T, dtype=float) * 0.05
    R = vorticity_transport_residual(u, v, t, nu=1e-4, A=0.1, k_f=2.0)
    assert R.shape == (T, N, N)
    assert np.all(np.isfinite(R)) and np.all(R >= 0.0)
    print("✓ residual_shape_and_finite")


def test_unsteady_term_responds_to_time_change():
    """場隨時間變化 → 殘差含非零 unsteady 貢獻（對比 steady 應更大）。"""
    N, A, k_f, nu = 64, 0.1, 2.0, 1e-4
    u2d, v2d, _ = _steady_kolmogorov(N, A, k_f, nu)
    T = 5
    t = np.linspace(0.0, 0.2, T)
    # 把振幅隨時間線性變化 → domega/dt != 0
    ramp = (1.0 + 0.5 * t)[:, None, None]
    u = ramp * u2d[None]
    v = np.repeat(v2d[None], T, axis=0)
    R_unsteady = vorticity_transport_residual(u, v, t, nu=nu, A=A, k_f=k_f)
    R_steady = vorticity_transport_residual(
        np.repeat(u2d[None], T, axis=0), v, t, nu=nu, A=A, k_f=k_f)
    assert R_unsteady.max() > R_steady.max()
    print(f"✓ unsteady_term_responds: unsteady {R_unsteady.max():.3e} > steady {R_steady.max():.3e}")
