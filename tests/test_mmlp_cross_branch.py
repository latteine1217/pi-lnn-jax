"""Wang 2021 modified DeepONet 的跨分支 U/V gating（arXiv 2110.01654 §3.4）。

原文 eq. 3.23/3.26/3.27：U 取自 branch 輸入、V 取自 trunk 輸入，branch 與 trunk
兩條網路**每層共用同一組 U/V**。既有 `use_modified_mlp` 只做到 trunk 單邊
（U/V 皆由座標產生），缺的正是跨分支那一半。本檔守住補上的兩個正交旗標：

* `mmlp_branch_u`    — U 改由 branch tokens（sensor 側）mean-pool 產生
* `mmlp_gate_branch` — branch context 也走 gated block，與 trunk 共用 U/V

**為什麼要有 no-op 守衛**：本 repo 有過 `relpos_bias` 在所有既有 config 下恆為
常數的前例——參數建了、forward 走了、對輸出卻毫無作用。旗標「有接上」不等於
「有作用」，所以下面直接擾動新增的參數並要求輸出改變。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.models import LiquidOperator

BASE_CFG = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=16, operator_rank=8,
    decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
    temporal_anchor_harmonics=2, domain_length=1.0,
)
T, K, N = 5, 10, 8


def _inputs():
    return dict(
        sensor_vals=jax.random.normal(jax.random.PRNGKey(0), (T, K, 2)),
        sensor_pos=jax.random.uniform(jax.random.PRNGKey(1), (K, 2)),
        sensor_time=jnp.linspace(0, 1, T),
        xy=jax.random.uniform(jax.random.PRNGKey(2), (N, 2)),
        t_q=jax.random.uniform(jax.random.PRNGKey(3), (N,)),
    )


def _init_apply(model):
    ins = _inputs()
    args = (ins["sensor_vals"], ins["sensor_pos"], 0.5, ins["sensor_time"], ins["xy"], ins["t_q"])
    params = model.init(jax.random.PRNGKey(42), *args)
    return params, model.apply(params, *args), args


def _decoder_params(params):
    return params["params"]["query_decoder"]


@pytest.mark.parametrize("branch_u,gate_branch", [(True, False), (False, True), (True, True)])
def test_cross_branch_variants_run(branch_u, gate_branch):
    """三種組合都要能前向且輸出有限，shape 仍是 [N, 3]。"""
    model = LiquidOperator(
        **BASE_CFG, use_modified_mlp=True, disable_cross_attention=True,
        mmlp_branch_u=branch_u, mmlp_gate_branch=gate_branch,
    )
    _, out, _ = _init_apply(model)
    assert out.shape == (N, 3)
    assert jnp.all(jnp.isfinite(out))


def test_branch_u_replaces_trunk_u_projection():
    """U 換來源是「取代」不是「並存」——留著 trunk_U_proj 會是孤兒參數。"""
    on = LiquidOperator(**BASE_CFG, use_modified_mlp=True, mmlp_branch_u=True)
    off = LiquidOperator(**BASE_CFG, use_modified_mlp=True, mmlp_branch_u=False)
    p_on = _decoder_params(_init_apply(on)[0])
    p_off = _decoder_params(_init_apply(off)[0])
    assert "branch_U_proj" in p_on and "trunk_U_proj" not in p_on
    assert "trunk_U_proj" in p_off and "branch_U_proj" not in p_off


def test_gate_branch_adds_blocks():
    on = LiquidOperator(**BASE_CFG, use_modified_mlp=True, mmlp_gate_branch=True)
    off = LiquidOperator(**BASE_CFG, use_modified_mlp=True, mmlp_gate_branch=False)
    assert any(k.startswith("branch_gate_block") for k in _decoder_params(_init_apply(on)[0]))
    assert not any(k.startswith("branch_gate_block") for k in _decoder_params(_init_apply(off)[0]))


def test_branch_u_is_not_a_noop():
    """擾動 branch_U_proj 必須改變輸出——否則 U 沒有真的參與 gating。"""
    model = LiquidOperator(
        **BASE_CFG, use_modified_mlp=True, disable_cross_attention=True, mmlp_branch_u=True)
    params, out, args = _init_apply(model)
    perturbed = jax.tree_util.tree_map(lambda x: x, params)
    perturbed["params"]["query_decoder"]["branch_U_proj"]["kernel"] += 1.0
    assert not jnp.allclose(out, model.apply(perturbed, *args))


def test_gate_branch_is_not_a_noop():
    """擾動 branch_gate_block 的 gate 權重必須改變輸出。

    擾動方式不能是「kernel 加常數」——gate 前有 LayerNorm，加常數等於乘上
    sum(LN(z))≈0，會得到一個看起來像 no-op 的假陰性。改為整組重抽。
    """
    model = LiquidOperator(
        **BASE_CFG, use_modified_mlp=True, disable_cross_attention=True, mmlp_gate_branch=True)
    params, out, args = _init_apply(model)
    perturbed = jax.tree_util.tree_map(lambda x: x, params)
    kernel = perturbed["params"]["query_decoder"]["branch_gate_block_0"]["Dense_0"]["kernel"]
    perturbed["params"]["query_decoder"]["branch_gate_block_0"]["Dense_0"]["kernel"] = (
        jax.random.normal(jax.random.PRNGKey(7), kernel.shape)
    )
    assert not jnp.allclose(out, model.apply(perturbed, *args))


def test_branch_u_makes_trunk_sensor_dependent():
    """核心行為：U 來自 branch 後，sensor 值必須能經 trunk 路徑影響輸出。

    判別觀測取在 `disable_cross_attention=True` 上——那條路的 branch readout 是
    mean-pool over K，與 attention 無關；兩個設定都會經 readout 依賴 sensor，
    所以改看**梯度是否流經 branch_U_proj**：它是 U 的唯一來源，梯度非零即證明
    sensor 資訊確實進了 trunk 的逐層 gating。
    """
    model = LiquidOperator(
        **BASE_CFG, use_modified_mlp=True, disable_cross_attention=True, mmlp_branch_u=True)
    params, _, args = _init_apply(model)
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, *args) ** 2))(params)
    g = grads["params"]["query_decoder"]["branch_U_proj"]["kernel"]
    assert jnp.max(jnp.abs(g)) > 0.0


@pytest.mark.parametrize("flag", ["mmlp_branch_u", "mmlp_gate_branch"])
def test_flags_require_modified_mlp(flag):
    """沒有 mMLP 路徑時 U/V gating 不存在——靜默忽略等於讓 config 說謊。"""
    model = LiquidOperator(**BASE_CFG, use_modified_mlp=False, **{flag: True})
    with pytest.raises(ValueError, match="use_modified_mlp"):
        _init_apply(model)
