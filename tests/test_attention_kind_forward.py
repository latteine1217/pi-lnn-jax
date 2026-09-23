"""`attention_kind="vector"` 的 forward —— 27 份 production config 在用，先前零測試。

問題（架構審查候選 E）：`models.py` 1403 行、沒有任何 `tests/test_models*`。
既有的旗標測試都退化成「shape 對、值有限」，因為 decoder 只能透過完整
`LiquidOperator` 的 init+apply 才碰得到。

`attention_kind` 的覆蓋原本只有 `tests/test_config.py` 檢查字串通過 `_one_of`
——那驗的是 config 解析，不是模型。實測 27 份 config 設 `"vector"`，
forward 零測試。

本檔驗**行為**而非形狀：
  1. sensor 集合的置換不變性——cross-attention 是對集合作用，換 sensor 順序
     不該改變輸出。這條會抓到索引錯配（本專案 EXP-101 的災難根因就是 sensor
     軸慣例弄反）。
  2. 旗標確實承重——`vector` 與 `scalar` 的輸出必須不同，否則這個旗標是裝飾。

另外釘住一個 production 事實：那 27 份 config **都沒設** `relpos_bias_mode`，
落到預設 `"radial"`。而 `models.py:845-846` 只在該旗標為 `"vector"` 時才把方向
向量放進 bias——所以整個 vector-attention 艦隊餵進位置路徑的只有純量半徑。
這不是 bug，但它與 `models.py:660-661` 把該路徑描述為「各向異性」不符，
值得在有人改預設值時被看見。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.models import LiquidOperator

_BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=16, operator_rank=8, decoder_attention_heads=2,
    use_temporal_anchor=True, T_total=1.0, temporal_anchor_harmonics=2,
    domain_length=1.0,
)
_T, _K, _N = 4, 6, 5


def _inputs(seed: int = 0):
    r = np.random.RandomState(seed)
    return (
        jnp.asarray(r.randn(_T, _K, 2), jnp.float32),      # sensor_vals
        jnp.asarray(r.uniform(0, 1, (_K, 2)), jnp.float32),  # sensor_pos
        0.5,                                                # re_norm
        jnp.linspace(0.0, 1.0, _T, dtype=jnp.float32),      # sensor_time
        jnp.asarray(r.uniform(0, 1, (_N, 2)), jnp.float32),  # xy
        jnp.asarray(r.uniform(0, 1, (_N,)), jnp.float32),   # t_q
    )


def _run(kind: str, inputs, *, seed: int = 42, **over):
    model = LiquidOperator(**{**_BASE, "attention_kind": kind, **over})
    params = model.init(jax.random.PRNGKey(seed), *inputs)
    return model.apply(params, *inputs), model, params


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_forward_is_finite_and_shaped(kind):
    out, _, _ = _run(kind, _inputs())
    assert out.shape == (_N, 3)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("kind", ["scalar", "vector"])
def test_sensor_permutation_does_not_change_the_output(kind):
    """cross-attention 對 sensor 集合作用——換順序不該改變輸出。

    這條抓的是索引錯配：sensor 值與位置若在某處被以不同順序取用，
    置換就會讓輸出偏移。本專案 EXP-101 的災難根因正是 sensor 軸慣例弄反。
    """
    sv, sp, re_norm, st, xy, tq = _inputs()
    perm = np.array([3, 0, 5, 1, 4, 2])
    assert len(perm) == _K and sorted(perm) == list(range(_K))

    model = LiquidOperator(**{**_BASE, "attention_kind": kind})
    params = model.init(jax.random.PRNGKey(42), sv, sp, re_norm, st, xy, tq)

    out_a = model.apply(params, sv, sp, re_norm, st, xy, tq)
    out_b = model.apply(params, sv[:, perm, :], sp[perm], re_norm, st, xy, tq)

    drift = float(jnp.abs(out_a - out_b).max())
    assert drift < 1e-4, (
        f"{kind}: 置換 sensor 順序後輸出改變（max|Δ|={drift:.3e}）"
        "——cross-attention 應對集合作用，這通常是 sensor 值與位置的索引對不上")


def test_attention_kind_is_load_bearing():
    """兩種模式的輸出必須不同，否則這個旗標是裝飾——而 27 份 config 在設它。"""
    inputs = _inputs()
    out_s, _, _ = _run("scalar", inputs)
    out_v, _, _ = _run("vector", inputs)
    assert float(jnp.abs(out_s - out_v).max()) > 1e-5, (
        "scalar 與 vector attention 的輸出相同——旗標沒有作用")


def test_relpos_bias_mode_is_load_bearing_under_vector_attention():
    """production 全部落在 radial；驗證另一個模式確實不同，那個差異才有意義。"""
    inputs = _inputs()
    out_radial, _, _ = _run("vector", inputs, relpos_bias_mode="radial")
    out_vector, _, _ = _run("vector", inputs, relpos_bias_mode="vector")
    assert float(jnp.abs(out_radial - out_vector).max()) > 1e-6, (
        "relpos_bias_mode 在 vector attention 下沒有作用")


def test_production_fleet_pairs_vector_attention_with_radial_bias():
    """釘住 production 事實：27 份 vector config 都沒設 relpos_bias_mode。

    所以那條被 `models.py:660-661` 描述為「各向異性」的位置路徑，實際只收到
    純量半徑（方向向量僅在 relpos_bias_mode="vector" 時才進 bias，見 :845-846）。
    這條紅了代表有人改了預設或某份 config 開始設它——那時該回頭確認
    §4 的措辭是否仍成立。
    """
    import pathlib
    import tomllib

    from pi_lnn_jax.config import MODEL_SCHEMA

    assert MODEL_SCHEMA["relpos_bias_mode"][1] == "radial"

    vector_cfgs, with_mode = 0, []
    for p in pathlib.Path(__file__).resolve().parent.parent.joinpath("configs").rglob("*.toml"):
        try:
            d = tomllib.loads(p.read_text())
        except Exception:
            continue
        ak = rb = None
        for body in d.values():
            if isinstance(body, dict):
                ak = body.get("attention_kind", ak)
                rb = body.get("relpos_bias_mode", rb)
        if ak == "vector":
            vector_cfgs += 1
            if rb is not None:
                with_mode.append((p.name, rb))

    assert vector_cfgs >= 20, f"只掃到 {vector_cfgs} 份 vector config——掃描壞了或情況已變"
    assert not with_mode, (
        f"有 vector config 開始明設 relpos_bias_mode：{with_mode}。"
        "更新本測試，並回頭確認 models.py:660-661 的「各向異性」措辭是否仍描述實況")
