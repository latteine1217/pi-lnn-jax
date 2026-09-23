"""Unit tests for controlled-cylinder body-trajectory recovery (synthetic data).

RealPDEBench publishes only (u, v, p) for controlled-cylinder — the solver's
body description is not released — so geometry and motion must be recovered from
the fields. These tests build synthetic fields with a KNOWN rigid moving body and
validate the recovery. No Arrow I/O.

The synthetic body is modelled the way the real one behaves: a rigid interior
moving with the wall (u ~ 0, v = v_wall(t), both spatially uniform) rather than a
plain "zero-speed disk". `test_rejects_nonuniform_wake` encodes the lesson from
the real shard: a low-|u| but NON-uniform wake must not be mistaken for the body
(that confounder is what defeated the old low-speed-centroid detector).
"""
import numpy as np
import pytest

from pi_lnn_jax.controlled import (
    body_from_field, dominant_frequency, fit_circle, fit_oscillation,
    infer_trajectory, rigid_body_mask,
)


def _grid(H=96, W=192, Lx=0.32, Ly=0.17):
    x_axis = np.linspace(0.0, Lx, W)
    y_axis = np.linspace(0.0, Ly, H)
    X, Y = np.meshgrid(x_axis, y_axis)
    return x_axis, y_axis, X, Y


def _frame(X, Y, cx, cy, r, v_wall, u_inf=0.1):
    """Rigid body (u=0, v=v_wall uniform inside) in a uniform free stream."""
    inside = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
    u = np.full(X.shape, u_inf); v = np.zeros(X.shape)
    u[inside] = 0.0
    v[inside] = v_wall
    return u, v


def test_fit_oscillation_recovers_amp_phase_and_r2():
    freq = 0.15
    A_true, phase_true = 0.02, 0.3
    t = np.linspace(0, 20, 500)
    disp = A_true * np.sin(2 * np.pi * freq * t + phase_true)
    A, phase, r2 = fit_oscillation(disp, t, freq)
    assert np.isclose(A, A_true, atol=1e-5), (A, A_true)
    assert np.isclose(np.sin(phase), np.sin(phase_true), atol=1e-5)
    assert r2 > 0.999, r2


def test_fit_circle_recovers_known_circle():
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    cx_t, cy_t, r_t = 0.0147, 0.0901, 0.0116
    cx, cy, r = fit_circle(cx_t + r_t * np.cos(th), cy_t + r_t * np.sin(th))
    assert np.isclose(cx, cx_t, atol=1e-6)
    assert np.isclose(cy, cy_t, atol=1e-6)
    assert np.isclose(r, r_t, atol=1e-6)


def test_dominant_frequency_finds_peak():
    t = np.linspace(0, 20, 800)
    sig = 0.02 * np.sin(2 * np.pi * 0.15 * t) + 0.001 * np.sin(2 * np.pi * 0.7 * t)
    f, power = dominant_frequency(sig, t)
    assert abs(f - 0.15) < 0.02, f
    assert power > 0.5, power


def test_body_from_field_locates_rigid_body():
    x_axis, y_axis, X, Y = _grid()
    cx_t, cy_t, r_t = 0.05, 0.09, 0.012
    u, v = _frame(X, Y, cx_t, cy_t, r_t, v_wall=0.01)
    cx, cy, r, n = body_from_field(u, v, x_axis, y_axis, min_cells=10)
    assert n > 10
    assert np.isclose(cx, cx_t, atol=0.004), (cx, cx_t)
    assert np.isclose(cy, cy_t, atol=0.004), (cy, cy_t)
    # mask erodes the rim by the local-std window -> radius is a slight underestimate
    assert r < r_t + 0.002 and r > r_t * 0.5, (r, r_t)


def test_rejects_nonuniform_wake():
    """Low-|u| but NON-uniform wake must not be picked as the body.

    This is the real-shard failure mode: a low-speed centroid tracks the wake
    (fit R^2 ~ 0.002). The uniformity requirement is what separates them.
    """
    x_axis, y_axis, X, Y = _grid()
    cx_t, cy_t, r_t = 0.05, 0.09, 0.012
    u, v = _frame(X, Y, cx_t, cy_t, r_t, v_wall=0.01)
    # wake: large downstream low-|u| patch, but noisy (non-uniform)
    rng = np.random.default_rng(0)
    wake = (X > 0.12) & (X < 0.28) & (np.abs(Y - 0.09) < 0.03)
    u[wake] = rng.normal(0.0, 0.01, size=int(wake.sum()))
    v[wake] = rng.normal(0.0, 0.02, size=int(wake.sum()))
    m = rigid_body_mask(u, v)
    assert m[wake].mean() < 0.15, "non-uniform wake leaked into the rigid mask"
    cx, cy, r, n = body_from_field(u, v, x_axis, y_axis, min_cells=10)
    assert np.isclose(cx, cx_t, atol=0.005), (cx, cx_t)
    assert np.isclose(cy, cy_t, atol=0.005), (cy, cy_t)


def test_infer_trajectory_measures_freq_and_amplitude():
    x_axis, y_axis, X, Y = _grid()
    base_x, r = 0.05, 0.012
    base_y = 0.09
    amp_true, freq_true, phase_true = 0.02, 0.15, 0.3
    T = 160
    t = np.linspace(0, 20, T)
    w = 2 * np.pi * freq_true
    u = np.empty((T,) + X.shape); v = np.empty((T,) + X.shape)
    for k in range(T):
        cy = base_y + amp_true * np.sin(w * t[k] + phase_true)
        v_wall = amp_true * w * np.cos(w * t[k] + phase_true)
        u[k], v[k] = _frame(X, Y, base_x, cy, r, v_wall)
    # freq is MEASURED (not passed) — the real control_freq is not the body freq
    res = infer_trajectory(u, v, t, x_axis, y_axis, axis=(0.0, 1.0), min_cells=10)
    assert abs(res["freq"] - freq_true) < 0.02, res["freq"]
    assert np.isclose(res["amp"], amp_true, atol=0.004), (res["amp"], amp_true)
    assert res["r2"] > 0.8, res["r2"]
    assert np.isclose(res["base_center"][1], base_y, atol=0.004)
    assert res["n_tracked"] == T
    assert res["centers"].shape == (T, 2)


def test_infer_trajectory_fails_loud_on_noise():
    """A body that does NOT oscillate must raise, not return a noise amplitude.

    Guards the real regression: fitting a sinusoid at the wrong frequency gave
    amp=0.0005 at R^2=0.001, which would silently become the moving-BC kinematics.
    """
    x_axis, y_axis, X, Y = _grid()
    T = 60
    t = np.linspace(0, 20, T)
    rng = np.random.default_rng(1)
    u = np.empty((T,) + X.shape); v = np.empty((T,) + X.shape)
    for k in range(T):
        # body jitters randomly -> no coherent sinusoid at any single frequency
        cy = 0.09 + rng.normal(0.0, 0.004)
        u[k], v[k] = _frame(X, Y, 0.05, cy, 0.012, v_wall=0.0)
    with pytest.raises(ValueError, match="r2"):
        infer_trajectory(u, v, t, x_axis, y_axis, axis=(0.0, 1.0),
                         min_cells=10, min_r2=0.5)
