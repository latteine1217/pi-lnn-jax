"""Controlled cylinder: recover the body trajectory from the velocity fields.

RealPDEBench's controlled-cylinder publishes only (u, v, p): the LilyPad solver's
authoritative body description (40 boundary points `bd`) is NOT in the released
Arrow shards, and `sim_params` = [Re, control_freq] carries no amplitude. So the
body geometry and motion must be recovered from the fields.

Detection principle (verified on shard 1781_0.7, 798/798 frames tracked):
    The body is RIGID. Under no-slip its interior moves with the wall, so for a
    body translating along y the interior has u ~ 0 and BOTH components are
    spatially uniform there. Free stream is uniform too but has u ~ u_inf != 0;
    the wake has small u but is NOT uniform. Requiring `|u| small AND locally
    uniform` therefore isolates the body from both confounders. A least-squares
    circle fit on the largest such blob gives (center, radius) per frame.

    A plain low-speed centroid does NOT work here: the released fields carry no
    zero-velocity body signature (min |u|+|v| ~ 8e-5, no sub-1e-4 blob), so the
    centroid tracks wake fluctuations instead (R^2 ~ 0.002 against any sinusoid).

Frequency is MEASURED, not assumed: on 1781_0.7 the body oscillates at 0.150
(88% of spectral power, fit R^2 = 0.88) while the filename's control_freq is
0.7, which carries 0.1% of the power. Passing control_freq as the oscillation
frequency injects the wrong kinematics.

Coordinates: work in the axes you pass in. The body is a true circle in PHYSICAL
coordinates, so pass physical axes for the circle fit to be meaningful; convert
the returned center/radius/amp to normalized space afterwards if needed.

numpy + scipy.ndimage only — no Arrow I/O, no training.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label, uniform_filter


def _local_std(f, size=3):
    """Local spatial std over a `size`x`size` window (rigid interior -> ~0)."""
    m = uniform_filter(f, size)
    m2 = uniform_filter(f * f, size)
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def rigid_body_mask(u, v, u_tol_frac=0.25, uniform_tol_frac=0.02, window=3):
    """Boolean mask of the rigid body for one frame.

    Args:
        u, v: [H, W] one frame.
        u_tol_frac: |u| threshold as a fraction of the frame's median speed
            (body translating along y has u ~ 0; free stream has u ~ u_inf).
        uniform_tol_frac: local-std threshold as a fraction of the frame's speed
            std (rigid interior is spatially uniform; the wake is not).
        window: local-std window size.
    Returns:
        mask [H, W] bool.
    """
    u = np.asarray(u); v = np.asarray(v)
    speed = np.sqrt(u ** 2 + v ** 2)
    u_tol = u_tol_frac * float(np.median(speed))
    s_tol = uniform_tol_frac * float(speed.std()) + 1e-9
    return (np.abs(u) < u_tol) & (_local_std(u, window) < s_tol) & \
           (_local_std(v, window) < s_tol)


def fit_circle(xs, ys):
    """Algebraic least-squares circle fit: x^2+y^2 + D x + E y + F = 0.

    Returns (cx, cy, radius). Raises ValueError if under-determined.
    """
    xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3:
        raise ValueError(f"circle fit needs >=3 points, got {xs.size}")
    A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    b = -(xs ** 2 + ys ** 2)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = -coef[0] / 2.0
    cy = -coef[1] / 2.0
    r2 = cx * cx + cy * cy - coef[2]
    return float(cx), float(cy), float(np.sqrt(max(r2, 0.0)))


def body_from_field(u, v, x_axis, y_axis, min_cells=50, **mask_kw):
    """Locate the body in one frame: (cx, cy, radius, n_cells).

    Picks the largest connected rigid-uniform blob and circle-fits it.
    Returns (nan, nan, nan, 0) when no blob of at least `min_cells` is found,
    so callers can drop untracked frames instead of getting a silent fallback.
    """
    m = rigid_body_mask(u, v, **mask_kw)
    lab, n = label(m)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    blob = (lab == k)
    n_cells = int(blob.sum())
    if n_cells < min_cells:
        return (float("nan"), float("nan"), float("nan"), n_cells)
    X, Y = np.meshgrid(np.asarray(x_axis), np.asarray(y_axis))
    cx, cy, r = fit_circle(X[blob], Y[blob])
    return (cx, cy, r, n_cells)


def dominant_frequency(sig, t):
    """Dominant (non-DC) frequency of `sig` sampled on uniform `t`."""
    sig = np.asarray(sig, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dt = float(np.mean(np.diff(t)))
    s = sig - sig.mean()
    ps = np.abs(np.fft.rfft(s)) ** 2
    ps[0] = 0.0
    freqs = np.fft.rfftfreq(len(s), d=dt)
    k = int(np.argmax(ps))
    return float(freqs[k]), float(ps[k] / (ps.sum() + 1e-30))


def fit_oscillation(disp, t, freq):
    """Least-squares fit disp(t) = A * sin(2*pi*freq*t + phase).

    Linear in (a, b): disp = a*cos(w t) + b*sin(w t), w = 2*pi*freq, with
    a = A*sin(phase), b = A*cos(phase). So A = hypot(a, b), phase = atan2(a, b).

    Returns (amp, phase, r2) where r2 is the fraction of variance explained.
    """
    w = 2.0 * np.pi * freq
    t = np.asarray(t, dtype=np.float64)
    disp = np.asarray(disp, dtype=np.float64)
    M = np.stack([np.cos(w * t), np.sin(w * t)], axis=1)
    coef, *_ = np.linalg.lstsq(M, disp, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = disp - M @ coef
    r2 = 1.0 - float(resid.var()) / (float(disp.var()) + 1e-30)
    return float(np.hypot(a, b)), float(np.arctan2(a, b)), r2


def infer_trajectory(u, v, t, x_axis, y_axis, axis=(0.0, 1.0), freq=None,
                     min_r2=0.5, min_tracked_frac=0.8, min_cells=50, **mask_kw):
    """Track the body per frame, measure its oscillation, fit a sinusoid.

    The frequency is MEASURED from the tracked displacement unless `freq` is
    given explicitly. Do not pass RealPDEBench's `control_freq` here: on
    1781_0.7 the body oscillates at 0.150 while control_freq is 0.7.

    Raises ValueError if too few frames track or the sinusoid fit is poor
    (r2 < min_r2) — a bad fit means the amplitude is noise, and silently
    returning it would inject wrong kinematics into the moving-boundary BC.

    Args:
        u, v: [T, H, W] fields.
        t: [T] physical time.
        x_axis, y_axis: [W], [H] grid axes (pass PHYSICAL axes; see module doc).
        axis: oscillation direction (unit-normalized internally).
        freq: oscillation frequency; None -> measured via FFT.
        min_r2: minimum sinusoid fit quality to accept.
        min_tracked_frac: minimum fraction of frames that must track.
    Returns:
        dict(amp, phase, freq, r2, power_frac, base_center, radius, radius_std,
             centers[T,2], disp[T], tracked[T], n_tracked, touches_border)
    """
    u = np.asarray(u); v = np.asarray(v)
    T = u.shape[0]
    out = np.array([body_from_field(u[k], v[k], x_axis, y_axis,
                                    min_cells=min_cells, **mask_kw)
                    for k in range(T)])
    centers = out[:, :2]
    radii = out[:, 2]
    tracked = np.isfinite(radii)
    n_tracked = int(tracked.sum())
    if n_tracked < min_tracked_frac * T:
        raise ValueError(
            f"body tracked in only {n_tracked}/{T} frames "
            f"(< {min_tracked_frac:.0%}); rigid-body detection failed — check "
            f"u_tol_frac/uniform_tol_frac or whether this case translates in y")

    base_center = (float(np.nanmean(centers[tracked, 0])),
                   float(np.nanmean(centers[tracked, 1])))
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    disp = (centers - np.asarray(base_center, dtype=np.float64)) @ ax

    tt = np.asarray(t, dtype=np.float64)[tracked]
    dd = disp[tracked]
    power_frac = float("nan")
    if freq is None:
        freq, power_frac = dominant_frequency(dd, tt)
    amp, phase, r2 = fit_oscillation(dd, tt, freq)
    if r2 < min_r2:
        raise ValueError(
            f"oscillation fit r2={r2:.3f} < {min_r2} at freq={freq:.4f} "
            f"(amp={amp:.5f}); the amplitude would be noise. Do not feed this "
            f"to the moving-boundary BC.")

    # Body clipped by the domain edge biases the circle fit — surface it.
    xs = np.asarray(x_axis); ys = np.asarray(y_axis)
    touches_border = bool(
        base_center[0] - np.nanmean(radii[tracked]) <= xs.min() + 1e-9
        or base_center[0] + np.nanmean(radii[tracked]) >= xs.max() - 1e-9
        or base_center[1] - np.nanmean(radii[tracked]) <= ys.min() + 1e-9
        or base_center[1] + np.nanmean(radii[tracked]) >= ys.max() - 1e-9)

    return dict(amp=amp, phase=phase, freq=float(freq), r2=r2,
                power_frac=power_frac, base_center=base_center,
                radius=float(np.nanmean(radii[tracked])),
                radius_std=float(np.nanstd(radii[tracked])),
                centers=centers, disp=disp, tracked=tracked,
                n_tracked=n_tracked, touches_border=touches_border)
