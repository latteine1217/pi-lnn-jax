"""Unit tests for conformal prediction core (pure NumPy, no model needed).

策略：用合成資料，覆蓋率有解析期望值 → 可斷言 coverage ≈ 1−α。
"""
from __future__ import annotations

import numpy as np

from pi_lnn_jax.conformal import split_conformal_quantile
from pi_lnn_jax.conformal import normalized_residual, local_difficulty
from pi_lnn_jax.conformal import mondrian_quantiles
from pi_lnn_jax.conformal import (
    empirical_coverage, interval_width, conditional_coverage,
)
from pi_lnn_jax.conformal import worst_slab_coverage
from pi_lnn_jax.conformal import temporal_split


def test_split_quantile_finite_sample_correction():
    # n=9, alpha=0.1 → rank = ceil(10 * 0.9) = 9 → 9th smallest = max
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    q = split_conformal_quantile(scores, alpha=0.1)
    assert q == 9.0, f"expected 9.0, got {q}"
    # n=9, alpha=0.5 → rank = ceil(10*0.5)=5 → 5th smallest = 5
    assert split_conformal_quantile(scores, alpha=0.5) == 5.0
    print("✓ split_quantile_finite_sample_correction")


def test_split_quantile_returns_inf_when_rank_exceeds_n():
    # n=5, alpha=0.1 → rank = ceil(6*0.9)=6 > 5 → no finite quantile → inf
    scores = np.arange(5, dtype=float)
    assert split_conformal_quantile(scores, alpha=0.1) == np.inf
    print("✓ split_quantile_returns_inf_when_rank_exceeds_n")


def test_split_quantile_empty_raises():
    try:
        split_conformal_quantile(np.array([]), alpha=0.1)
    except ValueError:
        print("✓ split_quantile_empty_raises")
        return
    raise AssertionError("expected ValueError on empty scores")


def test_normalized_residual_plain_and_scaled():
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.5, 1.0, 3.0])
    r = normalized_residual(pred, truth)            # absolute residual
    assert np.allclose(r, [0.5, 1.0, 0.0])
    sigma = np.array([0.5, 2.0, 1.0])
    rn = normalized_residual(pred, truth, sigma)    # divided by sigma
    assert np.allclose(rn, [1.0, 0.5, 0.0], atol=1e-6)
    print("✓ normalized_residual_plain_and_scaled")


def test_local_difficulty_is_distance_to_nearest_sensor():
    # one sensor at origin; query at (3,4) → dist 5; at (0,0) → exactly 0 (coincides with the sensor)
    sensors = np.array([[0.0, 0.0]])
    xy = np.array([[3.0, 4.0], [0.0, 0.0]])
    sig = local_difficulty(sensors, xy)
    assert abs(sig[0] - 5.0) < 1e-6
    assert sig[1] == 0.0
    # monotone: farther query → larger difficulty
    assert sig[0] > sig[1]
    print("✓ local_difficulty_is_distance_to_nearest_sensor")


def test_mondrian_quantiles_per_group():
    # group 0 small scores, group 1 large → distinct quantiles
    scores = np.concatenate([np.arange(1, 10, dtype=float),
                             np.arange(101, 110, dtype=float)])
    groups = np.concatenate([np.zeros(9, int), np.ones(9, int)])
    q = mondrian_quantiles(scores, groups, alpha=0.1)
    assert set(q.keys()) == {0, 1}
    assert q[0] == 9.0 and q[1] == 109.0
    print("✓ mondrian_quantiles_per_group")


def test_empirical_coverage_and_width():
    truth = np.array([0.0, 0.5, 2.0, -3.0])
    lower = np.array([-1.0, -1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    # covered: 0.0 yes, 0.5 yes, 2.0 no, -3.0 no → 0.5
    assert empirical_coverage(lower, upper, truth) == 0.5
    assert interval_width(lower, upper) == 2.0
    print("✓ empirical_coverage_and_width")


def test_conditional_coverage_by_group():
    truth = np.array([0.0, 0.0, 5.0, 5.0])
    lower = np.array([-1.0, -1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    groups = np.array([0, 0, 1, 1])
    cc = conditional_coverage(lower, upper, truth, groups)
    assert cc[0] == 1.0 and cc[1] == 0.0
    print("✓ conditional_coverage_by_group")


def test_worst_slab_coverage_finds_bad_region():
    # feature orders points; the high-feature slab is never covered
    n = 100
    feature = np.linspace(0, 1, n)
    truth = np.zeros(n)
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    # break coverage for the top 25% of feature: push truth outside
    upper[feature > 0.75] = -0.5  # now truth=0 lies above upper → uncovered
    wsc = worst_slab_coverage(lower, upper, truth, feature, n_slabs=4)
    assert wsc == 0.0, f"worst slab should be fully uncovered, got {wsc}"
    # marginal coverage stays high (0.75) — WSC exposes the hidden failure
    assert empirical_coverage(lower, upper, truth) == 0.75
    print("✓ worst_slab_coverage_finds_bad_region")


def test_temporal_split_interleaved_disjoint_and_covers():
    cal, test = temporal_split(10, scheme="interleaved", cal_frac=0.5)
    assert set(cal).isdisjoint(set(test))
    assert sorted(np.concatenate([cal, test])) == list(range(10))
    assert len(cal) == 5 and len(test) == 5
    print("✓ temporal_split_interleaved_disjoint_and_covers")


def test_end_to_end_split_conformal_achieves_nominal_coverage():
    """合成 i.i.d. 高斯殘差：split conformal 經驗覆蓋率應 ≈ 1−α。

    這是 CP 的核心保證；用大樣本 + 固定 seed 讓 Monte-Carlo 抖動 < 容差。
    """
    rng = np.random.RandomState(0)
    alpha = 0.1
    coverages = []
    for _ in range(200):
        cal = np.abs(rng.randn(500))          # |residual| calibration scores
        q = split_conformal_quantile(cal, alpha)
        test = np.abs(rng.randn(2000))        # fresh test residuals
        coverages.append(np.mean(test <= q))  # one-sided interval [−q, q] around 0
    mean_cov = float(np.mean(coverages))
    assert abs(mean_cov - (1 - alpha)) < 0.01, f"coverage {mean_cov} off nominal 0.9"
    print(f"✓ end_to_end coverage = {mean_cov:.4f} (nominal {1-alpha})")


def test_empirical_coverage_rejects_nonfinite_truth():
    truth = np.array([0.0, np.nan, 0.0])
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    try:
        empirical_coverage(lower, upper, truth)
    except ValueError:
        print("✓ empirical_coverage_rejects_nonfinite_truth")
        return
    raise AssertionError("expected ValueError on non-finite truth")


def test_mondrian_quantiles_rejects_float_groups():
    scores = np.arange(1, 7, dtype=float)
    groups = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # float ids → must be rejected
    try:
        mondrian_quantiles(scores, groups, alpha=0.1)
    except ValueError:
        print("✓ mondrian_quantiles_rejects_float_groups")
        return
    raise AssertionError("expected ValueError on float groups")


def test_run_cp_analysis_on_synthetic_npz(tmp_path):
    """造一個 synthetic npz（pred = dns + 已知高斯噪聲）→ 跑分析腳本 →
    cp_metrics.json 的 marginal coverage 應 ≈ 1−α。"""
    import json
    import subprocess

    rng = np.random.RandomState(0)
    T, N = 40, 16
    dns_u = rng.randn(T, N, N).astype(np.float32)
    dns_v = rng.randn(T, N, N).astype(np.float32)
    noise = 0.1
    pred_u = dns_u + noise * rng.randn(T, N, N).astype(np.float32)
    pred_v = dns_v + noise * rng.randn(T, N, N).astype(np.float32)
    x = np.linspace(0, 1, N, endpoint=False).astype(np.float32)
    npz = tmp_path / "seed42.npz"
    np.savez(npz, pred_u=pred_u, pred_v=pred_v, dns_u=dns_u, dns_v=dns_v,
             dns_t=np.arange(T, dtype=np.float32), x_grid=x, y_grid=x,
             sensor_pos=rng.rand(20, 2).astype(np.float32), seed=42, re_value=10000.0)
    out_json = tmp_path / "cp_metrics.json"
    r = subprocess.run(
        ["uv", "run", "python", "scripts/run_cp_analysis.py",
         "--npz", str(npz), "--alpha", "0.1", "--out", str(out_json), "--no-local-cp"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr
    m = json.loads(out_json.read_text())
    cov = m["vanilla"]["u"]["marginal_coverage"]
    assert abs(cov - 0.9) < 0.03, f"marginal coverage {cov} off nominal 0.9"
    # pointwise Path A/B (runs by default) → both paths near nominal coverage
    pw = json.loads((out_json.parent / "cp_pointwise.json").read_text())
    for path in ("path_A", "path_B"):
        c = pw[path]["u"]["fixed"]["coverage_mean"]
        assert abs(c - 0.9) < 0.03, f"{path} pointwise coverage {c} off nominal 0.9"
    print(f"✓ run_cp_analysis marginal={cov:.4f} pointwise A/B near nominal")


def test_run_cp_analysis_sweep_shape(tmp_path):
    """--alpha-sweep 應產生 nested-by-α JSON（plot_cp.py::reliability 消費的結構）。"""
    import json
    import subprocess

    rng = np.random.RandomState(0)
    T, N = 40, 16
    dns_u = rng.randn(T, N, N).astype(np.float32)
    dns_v = rng.randn(T, N, N).astype(np.float32)
    noise = 0.1
    pred_u = dns_u + noise * rng.randn(T, N, N).astype(np.float32)
    pred_v = dns_v + noise * rng.randn(T, N, N).astype(np.float32)
    x = np.linspace(0, 1, N, endpoint=False).astype(np.float32)
    npz = tmp_path / "seed42.npz"
    np.savez(npz, pred_u=pred_u, pred_v=pred_v, dns_u=dns_u, dns_v=dns_v,
             dns_t=np.arange(T, dtype=np.float32), x_grid=x, y_grid=x,
             sensor_pos=rng.rand(20, 2).astype(np.float32), seed=42, re_value=10000.0)
    out_json = tmp_path / "cp_metrics.json"
    sweep_json = tmp_path / "cp_metrics_sweep.json"
    r = subprocess.run(
        ["uv", "run", "python", "scripts/run_cp_analysis.py",
         "--npz", str(npz), "--alpha", "0.1", "--out", str(out_json),
         "--alpha-sweep", "0.1,0.2", "--sweep-out", str(sweep_json), "--no-local-cp"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr
    sweep = json.loads(sweep_json.read_text())
    assert {"0.1", "0.2"}.issubset(sweep.keys()), f"missing alpha keys: {sweep.keys()}"
    cov = sweep["0.1"]["vanilla"]["u"]["marginal_coverage"]
    assert isinstance(cov, float) and 0.0 <= cov <= 1.0, f"bad coverage {cov!r}"
    # ACD summary present (item: average coverage deviation)
    assert "_acd" in sweep and "acd_marginal" in sweep["_acd"]["vanilla"]["u"]
    print(f"✓ run_cp_analysis sweep shape + ACD OK; cov(0.1)={cov:.4f}")


def test_slab_coverage_curve_and_consistency_with_wsc():
    from pi_lnn_jax.conformal import slab_coverage, worst_slab_coverage
    n = 100
    feature = np.linspace(0, 1, n)
    truth = np.zeros(n)
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    upper[feature > 0.75] = -0.5            # top 25% slab uncovered
    sc = slab_coverage(lower, upper, truth, feature, n_slabs=4)
    assert len(sc["coverage"]) == 4 and len(sc["feature_mid"]) == 4
    assert sc["coverage"][-1] == 0.0 and sc["coverage"][0] == 1.0
    # worst_slab_coverage must equal the min of the curve (DRY consistency)
    assert worst_slab_coverage(lower, upper, truth, feature, n_slabs=4) == min(sc["coverage"])
    print("✓ slab_coverage_curve_and_consistency_with_wsc")


def test_local_difficulty_periodic_wraps():
    from pi_lnn_jax.conformal import local_difficulty
    sensors = np.array([[0.05, 0.5]])
    xy = np.array([[0.95, 0.5]])           # 0.90 apart non-periodic; 0.10 across the wrap
    assert abs(local_difficulty(sensors, xy)[0] - 0.90) < 1e-9          # plain Euclidean
    assert abs(local_difficulty(sensors, xy, domain_length=1.0)[0] - 0.10) < 1e-9  # periodic
    print("✓ local_difficulty_periodic_wraps")


def test_average_coverage_deviation():
    from pi_lnn_jax.conformal import average_coverage_deviation
    alphas = [0.1, 0.2]                      # targets 0.9, 0.8
    cov = [0.92, 0.75]                       # |0.92-0.9|+|0.75-0.8| = 0.02+0.05
    assert abs(average_coverage_deviation(cov, alphas) - 0.035) < 1e-12
    assert average_coverage_deviation([0.9, 0.8], alphas) == 0.0
    print("✓ average_coverage_deviation")


def test_local_cp_intervals_reduce_to_split_conformal():
    """g≡1, σ≡1 → local CP collapses to vanilla split conformal → coverage ≈ 1−α."""
    from pi_lnn_jax.local_cp import local_cp_intervals
    rng = np.random.RandomState(0)
    alpha = 0.1
    covs = []
    for _ in range(200):
        cal = np.abs(rng.randn(500))
        test = rng.randn(2000)              # truth centered at pred=0
        lo, hi, ell = local_cp_intervals(
            pred_test=np.zeros_like(test), g_test=np.ones_like(test),
            sigma_test=np.ones_like(test), resid_cal=cal,
            g_cal=np.ones_like(cal), sigma_cal=np.ones_like(cal), alpha=alpha)
        covs.append(np.mean((test >= lo) & (test <= hi)))
    assert abs(np.mean(covs) - (1 - alpha)) < 0.01, f"coverage {np.mean(covs)}"
    print(f"✓ local_cp_intervals_reduce_to_split: cov={np.mean(covs):.4f}")
