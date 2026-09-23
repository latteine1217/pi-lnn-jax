"""p 評估：load_dns_from_path(return_p)、reconstruct_field(return_p) 相容 + (後續 task) p_rel_err."""
from __future__ import annotations

import numpy as np
from _minimal_model import minimal_kwargs


def test_load_dns_return_p(tmp_path):
    from pi_lnn_jax.data import load_dns_from_path
    T, N = 3, 8
    obj = {
        "u": np.random.RandomState(0).randn(T, N, N).astype(np.float32),
        "v": np.random.RandomState(1).randn(T, N, N).astype(np.float32),
        "p": np.random.RandomState(2).randn(T, N, N).astype(np.float32),
        "time": np.linspace(0, 1, T).astype(np.float32),
    }
    fp = tmp_path / "dns.npy"
    np.save(fp, obj, allow_pickle=True)
    out3 = load_dns_from_path(fp, time_stride=1)
    assert len(out3) == 3              # 預設 3-tuple 相容
    u, v, p, t = load_dns_from_path(fp, time_stride=1, return_p=True)
    assert p.shape == u.shape


def test_reconstruct_field_return_p():
    import jax
    import jax.numpy as jnp
    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.evaluate import reconstruct_field
    model = LiquidOperator(**minimal_kwargs(3))
    K, T = 5, 3
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0, 1, T)
    sv = jax.random.normal(jax.random.PRNGKey(1), (T, K, 3))
    params = model.init(jax.random.PRNGKey(2), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(3), (4, 2)), jnp.zeros((4,)))
    ns = {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0, "p_mean": 0.0, "p_std": 1.0}
    xg = np.linspace(0, 1, 6, endpoint=False).astype(np.float32)
    yg = np.linspace(0, 1, 6, endpoint=False).astype(np.float32)
    u, v, p = reconstruct_field(model, params, sv, sp, 0.1, st, ns, xg, yg, 0.5, return_p=True)
    assert u.shape == v.shape == p.shape == (6, 6)
    # 預設 2-tuple 相容
    u2, v2 = reconstruct_field(model, params, sv, sp, 0.1, st, ns, xg, yg, 0.5)
    assert u2.shape == (6, 6)


def test_reconstruct_field_return_p_without_p_stats_raises():
    """norm_stats 無 p_mean/p_std 時 return_p=True 應明確 fail-fast（非裸 KeyError）。"""
    import jax
    import jax.numpy as jnp
    import pytest
    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.evaluate import reconstruct_field
    model = LiquidOperator(**minimal_kwargs(2))
    K, T = 5, 3
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0, 1, T)
    sv = jax.random.normal(jax.random.PRNGKey(1), (T, K, 2))
    params = model.init(jax.random.PRNGKey(2), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(3), (4, 2)), jnp.zeros((4,)))
    ns_uv = {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0}  # 無 p
    xg = np.linspace(0, 1, 6, endpoint=False).astype(np.float32)
    yg = np.linspace(0, 1, 6, endpoint=False).astype(np.float32)
    with pytest.raises(KeyError, match="p_mean/p_std"):
        reconstruct_field(model, params, sv, sp, 0.1, st, ns_uv, xg, yg, 0.5, return_p=True)


def test_p_rel_err_gauge_invariant():
    """純 gauge 常數偏移不應影響 p_rel_err（比較前各自減空間均值）。"""
    from pi_lnn_jax.evaluate import compute_metrics
    rng = np.random.RandomState(0)
    u = rng.randn(8, 8); v = rng.randn(8, 8)
    p_dns = rng.randn(8, 8)
    p_pred = p_dns + 3.7  # 純常數偏移
    m = compute_metrics(u, v, u, v, p_pred=p_pred, p_dns=p_dns)
    assert "p_rel_err" in m
    assert m["p_rel_err"] < 1e-6, m["p_rel_err"]


def test_compute_metrics_no_p_omits_key():
    """不給 p → metrics 不含 p_rel_err（baseline 不受影響）。"""
    from pi_lnn_jax.evaluate import compute_metrics
    rng = np.random.RandomState(0)
    u = rng.randn(8, 8); v = rng.randn(8, 8)
    m = compute_metrics(u, v, u, v)
    assert "p_rel_err" not in m
