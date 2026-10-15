"""reconstruct_field / evaluate_time_series 的 encode-once + jit 路徑與逐 chunk eager
`model.apply` 在數值上等價。

Why: eval 改為「sensor encode 整個評估只跑一次、decode 走 jit」以消除每 chunk × 每時刻
重跑 encode 的成本。jit 改變 XLA 融合 → 非逐位相同，故判準是 rtol 容差內相等。
網格點數刻意不整除 chunk，覆蓋最後一塊補齊（padding）後截掉的路徑。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.evaluate import evaluate_time_series, reconstruct_field
from pi_lnn_jax.models import (
    LiquidOperator,
    StandardPINNOperator,
    VanillaDeepONetOperator,
)

K, T, C = 5, 3, 3
NS = {"u_mean": 0.1, "u_std": 2.0, "v_mean": -0.2, "v_std": 0.5,
      "p_mean": 0.0, "p_std": 1.5}


def _inputs():
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(1), (T, K, C))
    return sv, sp, st


def _models():
    return {
        "liquid": LiquidOperator(**minimal_kwargs(C)),
        "pinn": StandardPINNOperator(num_layers=2, hidden_dim=16, fourier_embed_dim=16, d_time=4),
        "vanilla": VanillaDeepONetOperator(
            K_sensors=K, sensor_value_dim=C, hidden_dim=16, operator_rank=8,
            num_branch_layers=2, num_trunk_layers=2, fourier_embed_dim=16, d_time=4),
    }


def _reference(model, params, sv, sp, st, xg, yg, t):
    """舊實作：單次 eager model.apply（整個網格一塊），回 (u, v, p) physical。"""
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    xy = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], -1), jnp.float32)
    pred = np.asarray(model.apply(params, sv, sp, 0.3, st, xy,
                                  jnp.full((xy.shape[0],), t, jnp.float32)))
    shape = (len(xg), len(yg))
    return tuple(
        (pred[:, i] * NS[f"{c}_std"] + NS[f"{c}_mean"]).reshape(shape)
        for i, c in enumerate("uvp")
    )


@pytest.mark.parametrize("arch", ["liquid", "pinn", "vanilla"])
def test_reconstruct_field_matches_eager_apply(arch):
    model = _models()[arch]
    sv, sp, st = _inputs()
    params = model.init(jax.random.PRNGKey(2), sv, sp, 0.3, st,
                        jnp.zeros((4, 2)), jnp.zeros((4,)))
    xg = np.linspace(0, 1, 9, endpoint=False).astype(np.float32)
    yg = np.linspace(0, 1, 7, endpoint=False).astype(np.float32)
    # 63 點、chunk 16 → 4 塊，最後一塊 15 點走 padding
    got = reconstruct_field(model, params, sv, sp, 0.3, st, NS, xg, yg, 0.4,
                            chunk_size=16, return_p=True)
    want = _reference(model, params, sv, sp, st, xg, yg, 0.4)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)


def test_evaluate_time_series_fields_match_eager_apply():
    model = _models()["liquid"]
    sv, sp, st = _inputs()
    params = model.init(jax.random.PRNGKey(2), sv, sp, 0.3, st,
                        jnp.zeros((4, 2)), jnp.zeros((4,)))
    N = 8
    rng = np.random.default_rng(0)
    dns_u = rng.standard_normal((T, N, N)).astype(np.float32)
    dns_v = rng.standard_normal((T, N, N)).astype(np.float32)
    dns_t = np.asarray(st, np.float32)
    out = evaluate_time_series(model, params, sv, sp, 0.3, st, NS, dns_u, dns_v, dns_t,
                               verbose=False, collect_fields=True)
    g = np.linspace(0, 1, N, endpoint=False).astype(np.float32)
    for i, t in enumerate(dns_t):
        u, v, _ = _reference(model, params, sv, sp, st, g, g, float(t))
        np.testing.assert_allclose(out["u_fields"][i], u, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out["v_fields"][i], v, rtol=1e-5, atol=1e-6)
