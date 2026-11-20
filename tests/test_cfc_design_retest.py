"""CfC 設計重測的三個開關：`cfc_residual`、`cfc_query_time`、`trunk_elapsed_time`。

## 為什麼這一組測試存在

knowledge/experiments/kolmogorov-cfc-design-retest-2026-10.md 用它們檢驗「是不是設計讓 CfC 發揮不了」。
每個開關都要量到它宣稱的那件事，所以各用一條恆等式釘住：

1. 預設值不改既有行為（參數樹、輸出逐位元相同，指紋不含新鍵）。
2. `cfc_residual`：開與不開的 encoder 輸出相減，恰等於輸入 token（第一層變成直通 + CfC 增量）。
3. `cfc_query_time`：不新增參數；查詢時刻落在 sensor 幀上時重現原模型輸出，幀間則隨 t_q 連續變化，
   且 PDE residual 可算。
4. `trunk_elapsed_time=False`（無 anchor）：同一區間內輸出與 t_q 無關——時間資訊確實被拿掉。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import LiquidOperator, TemporalCfCEncoder
from pi_lnn_jax.physics import make_ns_residual_fn

BASE = {
    "sensor_value_dim": 2, "d_model": 16, "d_time": 4,
    "num_spatial_encoder_layers": 1, "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 0, "num_query_mlp_layers": 1,
    "query_mlp_hidden_dim": 16, "operator_rank": 8,
    "decoder_attention_heads": 1, "use_temporal_anchor": True, "T_total": 1.0,
    "temporal_anchor_harmonics": 2, "domain_length": 1.0,
}
T, K, N = 5, 10, 8
ST = jnp.array([0.0, 0.1, 0.15, 0.6, 1.0])


def _args(t_q=None):
    t_q = jax.random.uniform(jax.random.PRNGKey(3), (N,)) if t_q is None else t_q
    return (jax.random.normal(jax.random.PRNGKey(0), (T, K, 2)),
            jax.random.uniform(jax.random.PRNGKey(1), (K, 2)), 0.5, ST,
            jax.random.uniform(jax.random.PRNGKey(2), (N, 2)), t_q)


def test_defaults_are_unchanged():
    args = _args()
    m0 = LiquidOperator(**BASE)
    m1 = LiquidOperator(**BASE, cfc_residual=False, cfc_query_time=False, trunk_elapsed_time=True)
    p0, p1 = m0.init(jax.random.PRNGKey(42), *args), m1.init(jax.random.PRNGKey(42), *args)
    assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
    assert jnp.array_equal(m0.apply(p0, *args), m1.apply(p1, *args))
    fp = model_fingerprint(m1)
    assert not {"cfc_residual", "cfc_query_time", "trunk_elapsed_time"} & set(fp)


def test_new_switches_are_fingerprinted():
    fp = model_fingerprint(LiquidOperator(**BASE, cfc_residual=True, trunk_elapsed_time=False))
    assert fp["cfc_residual"] == "True" and fp["trunk_elapsed_time"] == "False"
    assert model_fingerprint(LiquidOperator(**BASE, cfc_query_time=True))["cfc_query_time"] == "True"


def test_residual_adds_input_token_exactly():
    x = jax.random.normal(jax.random.PRNGKey(0), (T, K, 16))
    plain = TemporalCfCEncoder(d_model=16, num_layers=1, num_token_attention_layers=0)
    res = TemporalCfCEncoder(d_model=16, num_layers=1, num_token_attention_layers=0, cfc_residual=True)
    p = plain.init(jax.random.PRNGKey(1), x, 0.5, ST)
    assert jax.tree_util.tree_structure(p) == jax.tree_util.tree_structure(
        res.init(jax.random.PRNGKey(1), x, 0.5, ST))
    diff = res.apply(p, x, 0.5, ST) - plain.apply(p, x, 0.5, ST)
    assert jnp.allclose(diff, x, atol=1e-6)


def test_query_time_reproduces_frames_and_varies_between():
    m0, mq = LiquidOperator(**BASE), LiquidOperator(**BASE, cfc_query_time=True)
    p = m0.init(jax.random.PRNGKey(42), *_args())
    assert jax.tree_util.tree_structure(p) == jax.tree_util.tree_structure(
        mq.init(jax.random.PRNGKey(42), *_args()))
    on_frames = jnp.tile(ST, 2)[:N]
    assert jnp.allclose(m0.apply(p, *_args(on_frames)), mq.apply(p, *_args(on_frames)), atol=1e-5)
    h = mq.apply(p, *_args()[:4], method=LiquidOperator.encode)
    assert h.shape == (T, K, 4 * BASE["d_model"])
    # 長 gap（0.6, 1.0）內兩個時刻：原模型的 branch 狀態固定，閉式解的 branch 狀態會變
    t_a, t_b = jnp.full((N,), 0.7), jnp.full((N,), 0.9)
    ya, yb = mq.apply(p, *_args(t_a)), mq.apply(p, *_args(t_b))
    assert not jnp.allclose(ya, yb)


def test_query_time_pde_residual_is_finite():
    mq = LiquidOperator(**BASE, cfc_query_time=True)
    args = _args()
    p = mq.init(jax.random.PRNGKey(42), *args)
    sv, sp, re, st = args[:4]
    h = mq.apply(p, sv, sp, re, st, method=LiquidOperator.encode)
    ns_fn, _ = make_ns_residual_fn(mq)
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    r = ns_fn(p, h, cx, cy, ct, 0.1, 2.0, sp, st, 1e-4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    for x in jax.tree_util.tree_leaves(r):
        assert bool(jnp.all(jnp.isfinite(x)))


def test_trunk_without_elapsed_time_is_constant_within_interval():
    kw = {**BASE, "use_temporal_anchor": False}
    m = LiquidOperator(**kw, trunk_elapsed_time=False)
    p = m.init(jax.random.PRNGKey(42), *_args())
    ya, yb = m.apply(p, *_args(jnp.full((N,), 0.7))), m.apply(p, *_args(jnp.full((N,), 0.9)))
    assert jnp.allclose(ya, yb)
    m_on = LiquidOperator(**kw)
    assert not jnp.allclose(m_on.apply(p, *_args(jnp.full((N,), 0.7))),
                            m_on.apply(p, *_args(jnp.full((N,), 0.9))))


@pytest.mark.parametrize("kw", [
    {"cfc_query_time": True, "cfc_residual": True},
    {"cfc_query_time": True, "temporal_cell": "gru_dt"},
    {"cfc_query_time": True, "decoder_frame_context": "bracket"},
    {"cfc_query_time": True, "num_temporal_cfc_layers": 2},
])
def test_query_time_rejects_unsupported_combos(kw):
    with pytest.raises(ValueError, match="cfc_query_time"):
        LiquidOperator(**{**BASE, **kw}).init(jax.random.PRNGKey(0), *_args())
