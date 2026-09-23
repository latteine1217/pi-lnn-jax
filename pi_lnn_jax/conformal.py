"""Conformal prediction core for PI-CON UQ (pure NumPy, deterministic).

Why pure NumPy: CP is post-hoc analysis on cached (pred, truth) arrays. Keeping it
free of JAX/model deps makes it locally unit-testable with synthetic data whose
coverage has an analytic expectation (split conformal guarantees 1−α marginal
coverage under exchangeability).

References: Vovk et al. 2005; Angelopoulos & Bates 2023 (gentle intro);
Romano et al. 2019 (normalized/CQR); Mondrian CP (group-conditional).
"""
from __future__ import annotations

import numpy as np


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1−α) empirical quantile of calibration scores.

    rank = ceil((n+1)(1−α)); the rank-th smallest score (1-indexed). When rank > n
    no finite calibration score is conservative enough → return +inf (interval is
    the whole real line, i.e. trivial but valid coverage).
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        raise ValueError("split_conformal_quantile: empty scores")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return np.inf
    return float(np.sort(s)[rank - 1])


def normalized_residual(pred, truth, sigma=None) -> np.ndarray:
    """Absolute residual |pred−truth|, optionally normalized by a difficulty σ̂.

    Normalization yields locally-adaptive intervals (Romano et al. 2019): the same
    conformal quantile produces wider intervals where σ̂ is large.
    """
    r = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64))
    if sigma is None:
        return r
    # + 1e-12: divide-by-zero floor for zero-difficulty pixels (σ̂=0 exactly at
    # sensor locations, where nearest-sensor distance is 0).
    return r / (np.asarray(sigma, dtype=np.float64) + 1e-12)


def local_difficulty(sensor_pos, xy, domain_length=None) -> np.ndarray:
    """σ̂ candidate: distance from each query point to its nearest sensor.

    Rationale: reconstruction error grows with distance from the nearest sensor, so
    this is a cheap, leak-free difficulty proxy (uses only sensor geometry, never the
    test labels).

    domain_length: if given, use the minimum-image (periodic) distance per axis
    `min(|Δ|, L−|Δ|)` — correct for a periodic domain such as Kolmogorov flow, where a
    point near one edge is close to a sensor across the wrap. None → plain Euclidean.
    """
    sensor_pos = np.asarray(sensor_pos, dtype=np.float64)  # [K, 2]
    xy = np.asarray(xy, dtype=np.float64)                  # [M, 2]
    diff = np.abs(xy[:, None, :] - sensor_pos[None, :, :])  # [M, K, 2]
    if domain_length is not None:
        diff = np.minimum(diff, float(domain_length) - diff)
    d = np.sqrt((diff ** 2).sum(-1))                        # [M, K]
    return d.min(axis=1)


def mondrian_quantiles(scores, groups, alpha: float) -> dict:
    """Per-group finite-sample conformal quantiles (Mondrian / group-conditional CP).

    Each group gets its own calibration set → group-conditional coverage, which is
    the honest way to handle heteroscedastic, non-i.i.d. fields. Keys are the unique
    group ids (cast to int for JSON-friendliness).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    # Why: float group ids would silently collide under int() (e.g. 1.4 and 1.6
    # both map to key 1), merging distinct calibration sets without warning.
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("mondrian_quantiles: groups must be integer-typed (float ids collide under int())")
    out: dict = {}
    for g in np.unique(groups):
        out[int(g)] = split_conformal_quantile(scores[groups == g], alpha)
    return out


def empirical_coverage(lower, upper, truth) -> float:
    """Fraction of truths inside [lower, upper]."""
    truth = np.asarray(truth, dtype=np.float64)
    # Why: silent NaN in truth would deflate the headline coverage metric
    # (NaN comparisons are False), so reject it rather than under-report coverage.
    # lo/hi are NOT checked: they may legitimately be ±inf when the conformal
    # quantile is inf (trivial-but-valid whole-line interval).
    if not np.all(np.isfinite(truth)):
        raise ValueError("empirical_coverage: non-finite values in truth")
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def interval_width(lower, upper) -> float:
    """Mean interval width (sharpness; lower is better at fixed coverage)."""
    return float(np.mean(np.asarray(upper, dtype=np.float64)
                         - np.asarray(lower, dtype=np.float64)))


def conditional_coverage(lower, upper, truth, groups) -> dict:
    """Per-group empirical coverage → dict {int(group): coverage}."""
    groups = np.asarray(groups).ravel()
    # Why: float group ids would silently collide under int() (see mondrian_quantiles).
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("conditional_coverage: groups must be integer-typed (float ids collide under int())")
    truth = np.asarray(truth, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    out: dict = {}
    for g in np.unique(groups):
        m = groups == g
        out[int(g)] = empirical_coverage(lo[m], hi[m], truth[m])
    return out


def slab_coverage(lower, upper, truth, feature, n_slabs: int = 10) -> dict:
    """Per-slab empirical coverage across equal-count bins ordered by `feature`.

    Sort by feature, cut into n_slabs contiguous equal-count bins; return the mean
    feature value and the coverage of each bin. This is the full conditional-coverage
    curve (the diagnostic behind worst-slab coverage), e.g. coverage vs sensor distance.
    """
    feature = np.asarray(feature, dtype=np.float64).ravel()
    order = np.argsort(feature)
    feat = feature[order]
    lo = np.asarray(lower, dtype=np.float64).ravel()[order]
    hi = np.asarray(upper, dtype=np.float64).ravel()[order]
    tr = np.asarray(truth, dtype=np.float64).ravel()[order]
    bins = [b for b in np.array_split(np.arange(feature.size), n_slabs) if b.size]
    return {
        "feature_mid": [float(feat[b].mean()) for b in bins],
        "coverage": [empirical_coverage(lo[b], hi[b], tr[b]) for b in bins],
    }


def worst_slab_coverage(lower, upper, truth, feature, n_slabs: int = 10) -> float:
    """Minimum coverage across equal-count slabs ordered by `feature`.

    A coarse, deterministic stand-in for worst-slab coverage (Cauchois et al. 2021):
    the min of the per-slab conditional-coverage curve (see slab_coverage). Exposes
    conditional miscoverage that the marginal number hides.
    """
    return float(min(slab_coverage(lower, upper, truth, feature, n_slabs)["coverage"]))


def average_coverage_deviation(coverages, alphas) -> float:
    """ACD: mean |empirical coverage − (1−α)| over a grid of miscoverage levels α.

    A single-scalar calibration summary (Yu et al. 2026): 0 is perfect calibration
    across all levels. `coverages[k]` is the empirical coverage measured at `alphas[k]`.
    """
    cov = np.asarray(coverages, dtype=np.float64).ravel()
    a = np.asarray(alphas, dtype=np.float64).ravel()
    if cov.shape != a.shape:
        raise ValueError("average_coverage_deviation: coverages and alphas length mismatch")
    return float(np.mean(np.abs(cov - (1.0 - a))))


def temporal_split(n_t: int, scheme: str = "interleaved", cal_frac: float = 0.5):
    """Split time indices [0, n_t) into (calibration, test).

    - "interleaved": even indices → cal, odd → test (cal_frac≈0.5). Decorrelates
      neighbouring snapshots better than a contiguous block for autocorrelated flows.
    - "block": first cal_frac fraction → cal, remainder → test (tests temporal
      extrapolation / distribution shift).
    """
    idx = np.arange(n_t)
    if scheme == "interleaved":
        cal = idx[idx % 2 == 0]
        test = idx[idx % 2 == 1]
    elif scheme == "block":
        cut = int(round(n_t * cal_frac))
        if cut <= 0 or cut >= n_t:
            raise ValueError(f"temporal_split block: cal_frac={cal_frac} yields an empty split for n_t={n_t}")
        cal, test = idx[:cut], idx[cut:]
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    return cal, test
