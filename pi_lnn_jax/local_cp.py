"""Learned local conformal prediction (spatial conformalized quantile regression).

Implements the local-CP method of Yu, Ho & Wang (JCP 2026) on top of our
physics-aware difficulty: a conditional quantile estimator g_phi(x) is fit by pinball
(quantile) regression on normalized residuals, then conformalized with a calibration
scaling ell* so the finite-sample coverage guarantee is preserved regardless of how
well g_phi fits. The final interval is

    u_hat(x) ± ell* · g_phi(x) · sigma(x).

No PINN/PINO retraining: g_phi is a lightweight auxiliary regressor fit on cached
residuals. sklearn is imported lazily so importing this module never requires it; the
conformalization itself (ell*) reuses split_conformal_quantile and is pure NumPy.

Leakage discipline (Theorem 4.1 conditions): the points fed to fit_local_quantile
(g_phi-train), to the calibration of ell*, and to evaluation MUST be disjoint.
"""
from __future__ import annotations

import numpy as np

from pi_lnn_jax.conformal import split_conformal_quantile


def fit_local_quantile(features, scores, alpha: float, backend: str = "gbr"):
    """Fit g_phi: features → the (1−α) conditional quantile of `scores` (pinball loss).

    features: [N, D] inputs (e.g. x, y, t, sensor-distance). scores: [N] normalized
    residuals |u−u_hat|/σ. Returns a fitted predictor exposing .predict(features)→[N]
    clipped to be strictly positive (a valid interval scale). backend: 'gbr' (gradient
    boosting, robust default) or 'linear' (quantile linear regression, cheapest).
    """
    features = np.asarray(features, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if backend == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor  # lazy
        model = GradientBoostingRegressor(
            loss="quantile", alpha=1.0 - alpha, n_estimators=200,
            max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    elif backend == "linear":
        from sklearn.linear_model import QuantileRegressor  # lazy
        model = QuantileRegressor(quantile=1.0 - alpha, alpha=0.0, solver="highs")
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'gbr' or 'linear'")
    model.fit(features, scores)
    return model


def predict_scale(model, features, floor: float = 1e-6) -> np.ndarray:
    """g_phi(features), floored positive so interval scale never collapses."""
    g = np.asarray(model.predict(np.asarray(features, dtype=np.float64)), dtype=np.float64)
    return np.maximum(g, floor)


def local_cp_intervals(pred_test, g_test, sigma_test, resid_cal, g_cal, sigma_cal, alpha):
    """Conformalize a learned quantile and return (lower, upper) for the test points.

    resid_cal = |u−u_hat| on the calibration set; g_cal/sigma_cal evaluated there.
    Localized calibration scores ell_j = resid_cal / (g_cal·sigma_cal); ell* is their
    finite-sample (1−α) split-conformal quantile. Interval = pred ± ell*·g_test·sigma_test.
    Returns (lower, upper, ell_star).
    """
    g_cal = np.maximum(np.asarray(g_cal, dtype=np.float64), 1e-12)
    sigma_cal = np.maximum(np.asarray(sigma_cal, dtype=np.float64), 1e-12)
    ell = np.asarray(resid_cal, dtype=np.float64) / (g_cal * sigma_cal)
    ell_star = split_conformal_quantile(ell, alpha)
    half = ell_star * np.asarray(g_test, dtype=np.float64) * np.asarray(sigma_test, dtype=np.float64)
    pred_test = np.asarray(pred_test, dtype=np.float64)
    return pred_test - half, pred_test + half, ell_star
