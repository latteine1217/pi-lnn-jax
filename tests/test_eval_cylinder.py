"""Non-periodic + masked evaluation metrics (cylinder case)."""
import numpy as np
from pi_lnn_jax.evaluate import compute_divergence_l2, compute_vorticity, compute_metrics


def _analytic_field(H=40, W=60, Lx=2.0, Ly=1.0):
    xn = np.linspace(0, 1, W); yn = np.linspace(0, 1, H)
    gx, gy = np.meshgrid(xn, yn)            # [H,W]
    u = (gy * Ly).astype(np.float64)        # u = y_phys
    v = (-gx * Lx).astype(np.float64)       # v = -x_phys → ω = ∂v/∂x-∂u/∂y = -1-1 = -2
    return u, v, xn, yn


def test_nonperiodic_vorticity_matches_analytic():
    u, v, xn, yn = _analytic_field()
    w = compute_vorticity(u, v, periodic=False, dns_x=xn, dns_y=yn, Lx=2.0, Ly=1.0)
    assert abs(np.mean(w[5:-5, 5:-5]) - (-2.0)) < 1e-6


def test_masked_ke_ratio_detects_over_energy():
    u, v, xn, yn = _analytic_field()
    mask = np.ones_like(u, dtype=bool)
    m_same = compute_metrics(u, v, u, v, periodic=False, dns_x=xn, dns_y=yn,
                             Lx=2.0, Ly=1.0, mask=mask)
    assert abs(m_same["ke_pred_over_ref"] - 1.0) < 1e-6
    assert m_same["ke_rel_err"] < 1e-6
    m_over = compute_metrics(2 * u, 2 * v, u, v, periodic=False, dns_x=xn, dns_y=yn,
                             Lx=2.0, Ly=1.0, mask=mask)
    assert abs(m_over["ke_pred_over_ref"] - 4.0) < 1e-6


def test_periodic_default_unchanged():
    rng = np.random.RandomState(0)
    u = rng.randn(16, 16); v = rng.randn(16, 16)
    # default periodic path must still run and return a scalar div
    d = compute_divergence_l2(u, v)
    assert np.isfinite(d)
