"""Field-based (finite-difference) physics residuals for post-hoc UQ.

Why separate from physics.py: physics.py computes the NS residual via autodiff on the
*model* and includes the pressure term, so it needs a model forward pass. For conformal
prediction we only have cached (u, v) fields on a grid. We therefore use the
**vorticity-transport residual**, i.e. the curl of the momentum equation, which
eliminates pressure and is fully computable from the velocity field by finite
differences:

    R_omega = d omega/dt + u domega/dx + v domega/dy - nu lap(omega) - (curl f)_z

with omega = dv/dx - du/dy and the Kolmogorov forcing f_x = A sin(2 pi k_f y), f_y = 0,
so (curl f)_z = -d f_x/dy = -A (2 pi k_f) cos(2 pi k_f y).

This is a physics-aware difficulty signal sigma_hat: large where the reconstruction
violates the (pressure-free) momentum balance. Unlike |div u| -- which the model is
explicitly trained to suppress via the augmented-Lagrangian continuity constraint and is
therefore near-uniformly small -- the vorticity-transport residual is not directly in
the training objective, so it carries discriminative "where is the model physically
wrong" information. Conformal calibration absorbs the absolute scale, so only the
spatial structure of the residual matters.

Axis convention matches evaluate.reconstruct_field / compute_vorticity:
u[..., i, j] <-> (x_grid[i], y_grid[j]); x is axis -2, y is axis -1.
"""
from __future__ import annotations

import numpy as np

from pi_lnn_jax.metric_artifact import compute_vorticity


def _ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """Periodic central difference along x (axis -2)."""
    return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2.0 * dx)


def _ddy(f: np.ndarray, dy: float) -> np.ndarray:
    """Periodic central difference along y (axis -1)."""
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * dy)


def _laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic 5-point Laplacian (x = axis -2, y = axis -1)."""
    fxx = (np.roll(f, -1, axis=-2) - 2.0 * f + np.roll(f, 1, axis=-2)) / dx ** 2
    fyy = (np.roll(f, -1, axis=-1) - 2.0 * f + np.roll(f, 1, axis=-1)) / dy ** 2
    return fxx + fyy


def vorticity_transport_residual(
    u: np.ndarray, v: np.ndarray, t: np.ndarray,
    nu: float, A: float, k_f: float, domain_length: float = 1.0,
) -> np.ndarray:
    """|R_omega| per (time, pixel) for the pressure-free 2D Kolmogorov NS residual.

    Args:
      u, v: velocity fields [T, Nx, Ny] (physical units).
      t:    time coordinates [T] (for the unsteady term, non-uniform safe).
      nu:   kinematic viscosity (= 1/Re for the L=1 nondimensionalization).
      A, k_f: Kolmogorov forcing amplitude and wavenumber (f_x = A sin(2 pi k_f y)).
      domain_length: square domain side (default 1.0).

    Returns:
      |R_omega| with shape [T, Nx, Ny].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    T, Nx, Ny = u.shape
    dx = domain_length / Nx
    dy = domain_length / Ny

    omega = np.stack([compute_vorticity(u[k], v[k], domain_length) for k in range(T)])
    # unsteady term (np.gradient handles edges + non-uniform t)
    omega_t = np.gradient(omega, t, axis=0) if T > 1 else np.zeros_like(omega)
    omega_x = _ddx(omega, dx)
    omega_y = _ddy(omega, dy)
    lap_omega = _laplacian(omega, dx, dy)

    # forcing curl: y varies along axis -1, coordinate y_j = j * dy
    y = np.arange(Ny) * dy
    curl_f = -A * (2.0 * np.pi * k_f) * np.cos(2.0 * np.pi * k_f * y)  # [Ny]

    R = omega_t + u * omega_x + v * omega_y - nu * lap_omega - curl_f[None, None, :]
    return np.abs(R)
