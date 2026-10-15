"""時間 encoder 的 cell 選擇：`temporal_cell = "cfc" | "gru_dt"`。

## 為什麼這一組測試存在

`gru_dt` 是不規則時鐘實驗的對照臂：分離「CfC 的連續時間閘門」與「任何看得到 Δt 的
recurrent encoder」。對照要成立需要三件事，各由一條測試釘住：

1. 預設 `"cfc"` 不改既有行為（參數樹、輸出、指紋都與從未有此欄位時相同）。
2. `gru_dt` 真的讀 Δt（否則量到的是「不看時間的 GRU」，對照就失效）。
3. 兩臂參數量相當（差異 <1%），比較的是結構而非容量。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.config import MODEL_SCHEMA
from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import CfCStep, GRUDtStep, LiquidOperator, TemporalCfCEncoder

BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=16, operator_rank=8,
    decoder_attention_heads=1, use_temporal_anchor=True, T_total=1.0,
    temporal_anchor_harmonics=2, domain_length=1.0,
)
T, K, N = 5, 10, 8


def _args(sensor_time=None):
    st = jnp.linspace(0, 1, T) if sensor_time is None else sensor_time
    return (jax.random.normal(jax.random.PRNGKey(0), (T, K, 2)),
            jax.random.uniform(jax.random.PRNGKey(1), (K, 2)), 0.5, st,
            jax.random.uniform(jax.random.PRNGKey(2), (N, 2)),
            jax.random.uniform(jax.random.PRNGKey(3), (N,)))


def _n_params(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


def test_default_cfc_is_unchanged():
    """顯式 "cfc" 與省略欄位：參數樹、輸出逐位元相同，指紋不含此鍵。"""
    args = _args()
    m0, m1 = LiquidOperator(**BASE), LiquidOperator(**BASE, temporal_cell="cfc")
    p0, p1 = m0.init(jax.random.PRNGKey(42), *args), m1.init(jax.random.PRNGKey(42), *args)
    assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
    assert jnp.array_equal(m0.apply(p0, *args), m1.apply(p1, *args))
    assert "temporal_cell" not in model_fingerprint(m1)
    assert "log_tau_a" in p0["params"]["temporal_encoder"]["cell_0"]


def test_gru_dt_builds_and_is_fingerprinted():
    args = _args()
    m = LiquidOperator(**BASE, temporal_cell="gru_dt")
    p = m.init(jax.random.PRNGKey(42), *args)
    out = m.apply(p, *args)
    assert out.shape == (N, 3) and bool(jnp.all(jnp.isfinite(out)))
    cell = p["params"]["temporal_encoder"]["cell_0"]
    assert "gru" in cell and "log_tau_a" not in cell
    assert model_fingerprint(m)["temporal_cell"] == "gru_dt"


def test_gru_dt_encoder_reads_elapsed_time():
    """encoder 層級：同一組 token、不同（不等距）取樣時刻 → 輸出必須不同。

    在 encoder 而非整個 operator 上量：decoder 本身也吃 sensor_time（選 state、算
    query 的 elapsed time），整體輸出不同證明不了 GRU 讀到 Δt。
    """
    enc = TemporalCfCEncoder(d_model=16, num_layers=1, num_token_attention_layers=0,
                             temporal_cell="gru_dt")
    x = jax.random.normal(jax.random.PRNGKey(0), (T, K, 16))
    regular, irregular = jnp.linspace(0, 1, T), jnp.array([0.0, 0.1, 0.15, 0.6, 1.0])
    p = enc.init(jax.random.PRNGKey(1), x, 0.5, regular)
    assert not jnp.allclose(enc.apply(p, x, 0.5, regular), enc.apply(p, x, 0.5, irregular))


def test_gru_dt_step_dt_enters_update():
    """cell 層級：只改 Δt，新狀態就要變。"""
    h = jnp.zeros((K, 16))
    x = jax.random.normal(jax.random.PRNGKey(0), (K, 16))
    step = GRUDtStep(hidden_size=16)
    p = step.init(jax.random.PRNGKey(1), h, (x, jnp.float32(0.05)))
    a, _ = step.apply(p, h, (x, jnp.float32(0.05)))
    b, _ = step.apply(p, h, (x, jnp.float32(0.5)))
    assert not jnp.allclose(a, b)


def test_param_count_matched_at_production_width():
    """hidden=256（主線 d_model）時兩個 cell 的參數量差 <1%。"""
    d = 256
    h = jnp.zeros((K, d))
    x = (jnp.zeros((K, d)), jnp.float32(0.05))
    n_cfc = _n_params(CfCStep(hidden_size=d).init(jax.random.PRNGKey(0), h, x))
    n_gru = _n_params(GRUDtStep(hidden_size=d).init(jax.random.PRNGKey(0), h, x))
    assert n_cfc == 394_240 and n_gru == 395_008
    assert abs(n_gru - n_cfc) / n_cfc < 0.01


def test_schema_default_and_validation():
    assert MODEL_SCHEMA["temporal_cell"][1] == "cfc"
    validator = MODEL_SCHEMA["temporal_cell"][2]
    validator("gru_dt")
    with pytest.raises(ValueError, match="temporal_cell"):
        validator("lstm")


def test_unknown_cell_fails_fast():
    with pytest.raises(ValueError, match="temporal_cell"):
        LiquidOperator(**BASE, temporal_cell="lstm").init(jax.random.PRNGKey(0), *_args())
