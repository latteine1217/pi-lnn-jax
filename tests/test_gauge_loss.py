"""gauge 錦定正則項：gauge_loss_weight=0 退化、>0 加 w·mean(p_phys)²。"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.kolmogorov import assembly


def _build_loss_and_batch(gauge_w, p_mean=0.0, p_std=1.0):
    model = LiquidOperator(**minimal_kwargs(3))
    K, T, N = 5, 4, 8
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1,
                        sensor_time, jax.random.uniform(jax.random.PRNGKey(9), (N, 2)),
                        jnp.zeros((N,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=False, use_al=False, al_rho=0.0,
        w_poisson=0.0, gauge_loss_weight=gauge_w, T_total=1.0,
    )
    re_batch = assembly.ReBatch(
        sensor_vals=sensor_vals, sensor_pos=sensor_pos, sensor_time=sensor_time,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(p_mean), p_std=jnp.asarray(p_std),
    )
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    task_weights = jnp.array([1.0, 0.0, 0.0, 0.0])
    total, aux = loss_fn(params, cx, cy, ct, task_weights,
                         jnp.asarray(0.0), jnp.asarray(0.0), re_batch, 0.0)
    return model, params, total, (cx, cy, ct, sensor_pos, sensor_time, re_batch)


def test_gauge_zero_is_baseline():
    _, _, total0a, _ = _build_loss_and_batch(0.0)
    _, _, total0b, _ = _build_loss_and_batch(0.0)
    assert jnp.allclose(total0a, total0b)
    assert jnp.isfinite(total0a)


def test_gauge_positive_adds_penalty():
    _, _, total0, _ = _build_loss_and_batch(0.0)
    model, params, total_g, ctx = _build_loss_and_batch(0.5)
    cx, cy, ct, sp, st, rb = ctx
    h = model.apply(params, rb.sensor_vals, sp, rb.re_norm, st,
                    method=LiquidOperator.encode)
    uvp = model.apply(params, jnp.stack([cx, cy], -1), ct, h, st, sp,
                      method=LiquidOperator.decode_query)
    p_phys = uvp[:, 2] * rb.p_std + rb.p_mean
    expected = total0 + 0.5 * jnp.mean(p_phys) ** 2
    assert jnp.allclose(total_g, expected, atol=1e-5), (float(total_g), float(expected))


def test_gauge_uses_denormalized_p():
    """p_mean≠0 / p_std≠1：gauge 必須用 denormalized 物理 p（鎖住漏乘 p_std / 漏加 p_mean）。"""
    _, _, total0, _ = _build_loss_and_batch(0.0, p_mean=2.0, p_std=3.0)
    model, params, total_g, ctx = _build_loss_and_batch(0.5, p_mean=2.0, p_std=3.0)
    cx, cy, ct, sp, st, rb = ctx
    h = model.apply(params, rb.sensor_vals, sp, rb.re_norm, st,
                    method=LiquidOperator.encode)
    uvp = model.apply(params, jnp.stack([cx, cy], -1), ct, h, st, sp,
                      method=LiquidOperator.decode_query)
    p_phys = uvp[:, 2] * rb.p_std + rb.p_mean   # = out[2]*3 + 2
    expected = total0 + 0.5 * jnp.mean(p_phys) ** 2
    assert jnp.allclose(total_g, expected, atol=1e-5), (float(total_g), float(expected))
