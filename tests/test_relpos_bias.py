"""relpos bias 的正規化開關與 zero-init。

## 為什麼這一組測試存在

`bias_in` 特徵維 = 1 時（`relpos_bias_mode="radial"` + 無 SDF）`nn.LayerNorm` 輸出恆為 0
（mean=x, var=0）→ `rel_bias` 退化成跨 (n,k) 的常數 → softmax 前加常數是 no-op，
整條 bias 完全不生效，`relpos_bias_ln/fc1/fc2` 成為死參數。398 份 config 中 333 份吃
這個組合，**含論文 B3 主結果 `exp_245_b3_les_T50`**。

預設保留 `"layernorm"`：既有 ckpt 與已發表 run 全在此狀態下產生（Never Break Userspace）。
`"none"` 是修好的對照臂。要比較「方向資訊有沒有用」必須以 `"none"` 為基準，否則量到的是
「把壞掉的 bias 修好」而非方向本身——`gk_rel −13%` 那個站了兩個月的競爭假設就是這樣來的
（見 knowledge/experiments/cylinder-advective-attention-negative-2026-07-29.md）。

第一條測試刻意**釘住那個 no-op**：它是既有 checkpoint 的行為，不是等著被修掉的 bug。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.models import LiquidOperator

BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=16, operator_rank=8,
    decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
    temporal_anchor_harmonics=2, domain_length=1.0,
    attention_kind="scalar",          # rel_bias 只在 scalar 路徑生效
)
T, K, N = 5, 10, 8


def _args():
    return (jax.random.normal(jax.random.PRNGKey(0), (T, K, 2)),
            jax.random.uniform(jax.random.PRNGKey(1), (K, 2)), 0.5,
            jnp.linspace(0, 1, T),
            jax.random.uniform(jax.random.PRNGKey(2), (N, 2)),
            jax.random.uniform(jax.random.PRNGKey(3), (N,)))


def _perturb_fc2(model):
    """擾動 relpos_bias_fc2 的 kernel，回傳 (原輸出, 擾動後輸出)。"""
    args = _args()
    p = model.init(jax.random.PRNGKey(42), *args)
    out = model.apply(p, *args)
    p2 = jax.tree_util.tree_map(lambda x: x, p)
    k = p2["params"]["query_decoder"]["relpos_bias_fc2"]["kernel"]
    p2["params"]["query_decoder"]["relpos_bias_fc2"]["kernel"] = (
        jax.random.normal(jax.random.PRNGKey(7), k.shape) * 3.0)
    return out, model.apply(p2, *args)


def test_layernorm_default_is_still_a_noop():
    """預設 layernorm + radial + no-SDF 必須維持 no-op——那是既有 ckpt 的行為。"""
    out, out2 = _perturb_fc2(LiquidOperator(**BASE))
    assert jnp.array_equal(out, out2), "既有 no-op 行為被改變，既有 ckpt 不再可重現"


def test_norm_none_makes_bias_live():
    out, out2 = _perturb_fc2(LiquidOperator(**BASE, relpos_bias_norm="none"))
    assert not jnp.allclose(out, out2), "relpos_bias_norm='none' 下 bias 仍不生效"


def test_layernorm_stays_noop_reason_is_feature_dim_one():
    """特徵維 > 1 時 layernorm 不再塌縮——證明 no-op 的成因是維度而非 LayerNorm 本身。"""
    out, out2 = _perturb_fc2(LiquidOperator(**BASE, relpos_bias_mode="vector"))
    assert not jnp.allclose(out, out2)


def test_norm_none_drops_the_layernorm_params():
    """'none' 不該留下死參數。"""
    args = _args()
    p = LiquidOperator(**BASE, relpos_bias_norm="none").init(jax.random.PRNGKey(42), *args)
    assert "relpos_bias_ln" not in p["params"]["query_decoder"]
    p_ln = LiquidOperator(**BASE).init(jax.random.PRNGKey(42), *args)
    assert "relpos_bias_ln" in p_ln["params"]["query_decoder"]


def test_invalid_norm_value_fails_fast():
    with pytest.raises(ValueError, match="relpos_bias_norm"):
        LiquidOperator(**BASE, relpos_bias_norm="batchnorm").init(jax.random.PRNGKey(0), *_args())


def test_zero_init_starts_bias_at_zero():
    """zero-init：bias 從 0 起步再長出來（cylinder 上 1/5 seed 發散的候選穩定化）。"""
    args = _args()
    p = LiquidOperator(**BASE, relpos_bias_norm="none",
                       relpos_bias_zero_init=True).init(jax.random.PRNGKey(42), *args)
    k = p["params"]["query_decoder"]["relpos_bias_fc2"]["kernel"]
    assert jnp.all(k == 0.0), "zero_init 未生效"
    p_no = LiquidOperator(**BASE, relpos_bias_norm="none").init(jax.random.PRNGKey(42), *args)
    assert not jnp.all(p_no["params"]["query_decoder"]["relpos_bias_fc2"]["kernel"] == 0.0)


def test_zero_init_bias_is_trainable():
    """zero-init 不能讓梯度也死掉——否則 bias 永遠留在 0。"""
    args = _args()
    model = LiquidOperator(**BASE, relpos_bias_norm="none", relpos_bias_zero_init=True)
    p = model.init(jax.random.PRNGKey(42), *args)
    g = jax.grad(lambda q: jnp.sum(model.apply(q, *args) ** 2))(p)
    assert jnp.max(jnp.abs(g["params"]["query_decoder"]["relpos_bias_fc2"]["kernel"])) > 0.0
