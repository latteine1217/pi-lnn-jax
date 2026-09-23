"""RWF (Random Weight Factorization) on liquid trunk — unit tests.

驗 RWFDense 重參數化正確、off==nn.Dense、init 等效權重對 mean/std 不變，
以及 LiquidOperator use_rwf 只在 decoder trunk 路徑生成 rwf_g（branch 不含）。

對齊 spec: knowledge/superpowers/specs/2026-06-12-rwf-liquid-trunk-design.md
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from pi_lnn_jax.models import RWFDense, LiquidOperator


def _x(n: int = 4, d: int = 6, seed: int = 0) -> jnp.ndarray:
    return jnp.asarray(np.random.RandomState(seed).randn(n, d), jnp.float32)


# ── RWFDense unit ───────────────────────────────────────────────────────────

def test_rwf_reparam_identity():
    """forward == x @ (rwf_g * kernel) + bias。"""
    x = _x()
    m = RWFDense(features=5, use_rwf=True)
    p = m.init(jax.random.PRNGKey(0), x)
    out = m.apply(p, x)
    v = p['params']['kernel']        # (in, out)
    g = p['params']['rwf_g']         # (out,)
    b = p['params']['bias']
    manual = x @ (g * v) + b
    assert jnp.allclose(out, manual, atol=1e-6)


def test_rwf_effective_kernel_invariant_to_mean_std():
    """同一 key 下，effective kernel g*v 對 rwf_mean/rwf_stddev 不變
    （init 等效權重恆等於 base init → Never Break Userspace at init）。"""
    x = _x()
    key = jax.random.PRNGKey(3)
    m1 = RWFDense(features=5, use_rwf=True, rwf_mean=1.0, rwf_stddev=0.1)
    m2 = RWFDense(features=5, use_rwf=True, rwf_mean=5.0, rwf_stddev=0.7)
    p1 = m1.init(key, x)
    p2 = m2.init(key, x)
    eff1 = p1['params']['rwf_g'] * p1['params']['kernel']
    eff2 = p2['params']['rwf_g'] * p2['params']['kernel']
    # g（與 v）各自不同 …
    assert not jnp.allclose(p1['params']['rwf_g'], p2['params']['rwf_g'])
    # … 但乘積（effective kernel）相同
    assert jnp.allclose(eff1, eff2, atol=1e-5)


def test_rwf_disabled_equals_dense():
    """use_rwf=False 的 RWFDense 數值與 nn.Dense 完全相同（同 name path → 同 rng）。"""
    x = _x()
    key = jax.random.PRNGKey(7)
    rwf = RWFDense(features=5, use_rwf=False)
    dense = nn.Dense(5)
    p_rwf = rwf.init(key, x)
    p_dense = dense.init(key, x)
    # param 結構相同：只有 kernel/bias，無 rwf_g
    assert set(p_rwf['params'].keys()) == {'kernel', 'bias'}
    assert jnp.allclose(rwf.apply(p_rwf, x), dense.apply(p_dense, x), atol=1e-7)


def test_rwf_gradients_flow_to_g_and_v():
    """g 與 v 都要收到梯度（兩者皆為自由可訓練參數 = RWF 的精髓）。"""
    x = _x()
    m = RWFDense(features=5, use_rwf=True)
    p = m.init(jax.random.PRNGKey(0), x)

    def loss(params):
        return jnp.sum(m.apply(params, x) ** 2)

    g = jax.grad(loss)(p)
    assert float(jnp.sum(jnp.abs(g['params']['rwf_g']))) > 0
    assert float(jnp.sum(jnp.abs(g['params']['kernel']))) > 0


# ── LiquidOperator integration ──────────────────────────────────────────────

def _build_model(use_rwf: bool) -> LiquidOperator:
    return LiquidOperator(
        sensor_value_dim=2,
        d_model=32,
        d_time=8,
        num_spatial_encoder_layers=1,
        num_temporal_cfc_layers=1,
        domain_length=1.0,
        use_temporal_anchor=True,
        T_total=5.0,
        temporal_anchor_harmonics=2,
        num_token_attention_layers=1,
        token_attention_heads=4,
        num_query_mlp_layers=1,
        query_mlp_hidden_dim=32,
        operator_rank=32,
        decoder_attention_heads=1,
        use_rwf=use_rwf,
    )


def _fake_inputs():
    rng = jax.random.PRNGKey(0)
    T, K, N = 8, 12, 5
    sensor_vals = jax.random.normal(rng, (T, K, 2))
    sensor_pos = jax.random.uniform(rng, (K, 2), minval=0.0, maxval=1.0)
    sensor_time = jnp.linspace(0.0, 5.0, T)
    re_norm = 0.1
    xy = jax.random.uniform(rng, (N, 2), minval=0.0, maxval=1.0)
    t_q = jax.random.uniform(rng, (N,), minval=0.0, maxval=5.0)
    return sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q, N


def test_liquid_rwf_only_on_trunk():
    """use_rwf=True 時，trunk_out 子樹含 rwf_g，branch_proj 子樹不含。"""
    inp = _fake_inputs()
    *args, N = inp
    model = _build_model(use_rwf=True)
    params = model.init(jax.random.PRNGKey(1), *args)
    dec = params['params']['query_decoder']
    assert 'rwf_g' in dec['trunk_out'], "trunk_out 應被 RWF factorize"
    assert 'rwf_g' in dec['trunk_in'], "trunk_in 應被 RWF factorize"
    assert 'rwf_g' not in dec['branch_proj'], "branch 路徑不應被 RWF 觸及"
    # forward 跑得通
    out = model.apply(params, *args)
    assert out.shape == (N, 3)
    assert not bool(jnp.any(jnp.isnan(out)))


def test_liquid_default_has_no_rwf():
    """預設 use_rwf=False → 整個 param tree 不含任何 rwf_g（Userspace 零變化）。"""
    inp = _fake_inputs()
    *args, N = inp
    model = _build_model(use_rwf=False)
    params = model.init(jax.random.PRNGKey(1), *args)
    leaf_names = []
    for path, _ in jax.tree_util.tree_leaves_with_path(params):
        leaf_names.extend(str(p.key) for p in path if hasattr(p, 'key'))
    assert 'rwf_g' not in leaf_names


# ── config schema ───────────────────────────────────────────────────────────

def test_rwf_config_schema():
    """三個新 model key 正確解析；rwf_stddev<=0 必須 raise。"""
    import pytest
    from pi_lnn_jax.config import _validate_one, MODEL_SCHEMA

    # 2026-08-29 起預設為 True：n=5 判別顯示 rwf on 在主指標上顯著較好
    # （-0.114 pp, t=-3.92，五 seed 範圍不重疊；見 knowledge/experiments/
    # kolmogorov-rwf-multiseed-2026-08-29.md）。舊的 False 出自單 seed 與
    # 錯誤 eval stride（TD-4）疊加的「rwf 有害」判定，該判定已推翻。
    assert MODEL_SCHEMA["use_rwf"][1] is True
    assert MODEL_SCHEMA["rwf_mean"][1] == 1.0
    assert MODEL_SCHEMA["rwf_stddev"][1] == 0.1

    assert _validate_one("use_rwf", True, MODEL_SCHEMA) is True
    assert _validate_one("rwf_mean", 2.0, MODEL_SCHEMA) == 2.0
    assert _validate_one("rwf_stddev", 0.2, MODEL_SCHEMA) == 0.2

    with pytest.raises(ValueError):
        _validate_one("rwf_stddev", 0.0, MODEL_SCHEMA)
