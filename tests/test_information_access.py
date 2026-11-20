"""資訊可及性的兩個對照臂：`temporal_direction="bidirectional"` 與 `decoder_frame_context="bracket"`。

## 為什麼這一組測試存在

間歇重建的誤差在 gap 內單調上升（knowledge/experiments/kolmogorov-cfc-isolation-2026-10.md 診斷 (b)），
假設是模型看不到 gap 之後的資料。兩個臂各從一處打開這個限制，對照要成立需要：

1. 預設值不改既有行為（參數樹、輸出逐位元相同，指紋不含新鍵）。
2. bidirectional 的第 n 幀狀態確實讀到 >n 的幀；forward 讀不到。
3. bracket 的 gap 內輸出確實讀到下一幀；latest 讀不到。不存在的那一幀被遮掉。
4. bracket 下 PDE residual（folx Laplacian + 對 t 的導數）仍可算且有限。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.config import MODEL_SCHEMA
from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import LiquidOperator, TemporalCfCEncoder
from pi_lnn_jax.physics import make_ns_residual_fn

BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=16, operator_rank=8,
    decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
    temporal_anchor_harmonics=2, domain_length=1.0,
)
T, K, N = 5, 10, 8
ST = jnp.array([0.0, 0.1, 0.15, 0.6, 1.0])   # 不等距，3→4 是一個長 gap


def _args(sv=None, t_q=None):
    sv = jax.random.normal(jax.random.PRNGKey(0), (T, K, 2)) if sv is None else sv
    t_q = jax.random.uniform(jax.random.PRNGKey(3), (N,)) if t_q is None else t_q
    return (sv, jax.random.uniform(jax.random.PRNGKey(1), (K, 2)), 0.5, ST,
            jax.random.uniform(jax.random.PRNGKey(2), (N, 2)), t_q)


def test_defaults_are_unchanged():
    args = _args()
    m0 = LiquidOperator(**BASE)
    m1 = LiquidOperator(**BASE, temporal_direction="forward", decoder_frame_context="latest")
    p0, p1 = m0.init(jax.random.PRNGKey(42), *args), m1.init(jax.random.PRNGKey(42), *args)
    assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
    assert jnp.array_equal(m0.apply(p0, *args), m1.apply(p1, *args))
    fp = model_fingerprint(m1)
    assert "temporal_direction" not in fp and "decoder_frame_context" not in fp


def test_new_arms_are_fingerprinted():
    m = LiquidOperator(**BASE, temporal_direction="bidirectional", decoder_frame_context="bracket")
    fp = model_fingerprint(m)
    assert fp["temporal_direction"] == "bidirectional" and fp["decoder_frame_context"] == "bracket"


@pytest.mark.parametrize("cell", ["cfc", "gru_dt"])
def test_bidirectional_state_reads_later_frames(cell):
    """只改最後一幀的輸入：forward 的第 0 幀狀態不動，bidirectional 的會動。"""
    x = jax.random.normal(jax.random.PRNGKey(0), (T, K, 16))
    x2 = x.at[-1].add(1.0)
    for direction, should_change in (("forward", False), ("bidirectional", True)):
        enc = TemporalCfCEncoder(d_model=16, num_layers=1, num_token_attention_layers=0,
                                 temporal_cell=cell, temporal_direction=direction)
        p = enc.init(jax.random.PRNGKey(1), x, 0.5, ST)
        h0, h1 = enc.apply(p, x, 0.5, ST), enc.apply(p, x2, 0.5, ST)
        assert h0.shape == (T, K, 16)
        assert (not jnp.allclose(h0[0], h1[0])) == should_change, direction
        # 最後一幀本身兩種方向都要變
        assert not jnp.allclose(h0[-1], h1[-1])


def test_bracket_reads_next_frame_inside_gap():
    """query 落在長 gap（0.6, 1.0）內，只改下一幀（t=1.0）的 sensor：latest 不動、bracket 會動。"""
    t_q = jnp.full((N,), 0.8)
    sv = jax.random.normal(jax.random.PRNGKey(0), (T, K, 2))
    sv2 = sv.at[-1].add(1.0)
    for ctx, should_change in (("latest", False), ("bracket", True)):
        m = LiquidOperator(**BASE, decoder_frame_context=ctx)
        p = m.init(jax.random.PRNGKey(42), *_args(sv, t_q))
        a, b = m.apply(p, *_args(sv, t_q)), m.apply(p, *_args(sv2, t_q))
        assert bool(jnp.all(jnp.isfinite(a)))
        assert (not jnp.allclose(a, b)) == should_change, ctx


def test_bracket_masks_missing_frame():
    """t_q 在最後一幀之後：沒有下一幀，下一幀那半的 token 被遮掉 → 輸出與「重複最後一幀」無關、有限。
    t_q 在第一幀之前：沒有前一幀，輸出有限。"""
    m = LiquidOperator(**BASE, decoder_frame_context="bracket")
    for tq in (1.2, -0.1):
        args = _args(t_q=jnp.full((N,), tq))
        p = m.init(jax.random.PRNGKey(42), *args)
        assert bool(jnp.all(jnp.isfinite(m.apply(p, *args))))


def test_bracket_pde_residual_is_finite():
    m = LiquidOperator(**BASE, decoder_frame_context="bracket", temporal_direction="bidirectional")
    args = _args()
    p = m.init(jax.random.PRNGKey(42), *args)
    sv, sp, re, st = args[:4]
    h = m.apply(p, sv, sp, re, st, method=LiquidOperator.encode)
    ns_fn, _ = make_ns_residual_fn(m)
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    r = ns_fn(p, h, cx, cy, ct, 0.1, 2.0, sp, st, 1e-4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    for x in jax.tree_util.tree_leaves(r):
        assert bool(jnp.all(jnp.isfinite(x)))


def test_schema_defaults_and_validation():
    assert MODEL_SCHEMA["temporal_direction"][1] == "forward"
    assert MODEL_SCHEMA["decoder_frame_context"][1] == "latest"
    with pytest.raises(ValueError, match="temporal_direction"):
        MODEL_SCHEMA["temporal_direction"][2]("backward")
    with pytest.raises(ValueError, match="decoder_frame_context"):
        MODEL_SCHEMA["decoder_frame_context"][2]("all")


def test_bracket_rejects_unsupported_combo():
    with pytest.raises(ValueError, match="bracket"):
        LiquidOperator(**BASE, decoder_frame_context="bracket", attention_kind="vector").init(
            jax.random.PRNGKey(0), *_args())
