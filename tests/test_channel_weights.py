"""data loss per-channel 權重：空 list=等權退化；[1,1,0.1] 把 p 殘差降權。"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.kolmogorov import assembly


def _sensor_loss(channel_weights):
    model = LiquidOperator(**minimal_kwargs(3))
    K, T, N = 5, 4, 8
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=False, use_al=False, al_rho=0.0,
        w_poisson=0.0, T_total=1.0, sensor_channel_weights=tuple(channel_weights),
    )
    rb = assembly.ReBatch(
        sensor_vals=sv, sensor_pos=sp, sensor_time=st, re_norm=jnp.asarray(0.1),
        nu=jnp.asarray(1e-4), u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0),
    )
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    task_weights = jnp.array([1.0, 0.0, 0.0, 0.0])  # 只留 data task → total == sensor_loss
    total, aux = loss_fn(params, cx, cy, ct, task_weights,
                         jnp.asarray(0.0), jnp.asarray(0.0), rb, 0.0)
    return float(aux[0])  # aux[0] = sensor_loss


def test_empty_weights_is_baseline():
    """空 list 與 [1,1,1] 等權 → sensor_loss 一致（bit-identical 退化）。"""
    s_none = _sensor_loss([])
    s_ones = _sensor_loss([1.0, 1.0, 1.0])
    assert jnp.allclose(jnp.asarray(s_none), jnp.asarray(s_ones), atol=1e-6), (s_none, s_ones)


def test_p_downweight_reduces_sensor_loss():
    """[1,1,0.1] 把 p 殘差降權 → sensor_loss 應 < 等權（p 殘差非零）。"""
    s_equal = _sensor_loss([1.0, 1.0, 1.0])
    s_pw = _sensor_loss([1.0, 1.0, 0.1])
    assert s_pw < s_equal, (s_pw, s_equal)
