"""非同步不規則取樣（`async_sensors`）：每個 sensor 各自缺幀，NaN＝沒有觀測。

## 為什麼這一組測試存在

knowledge/experiments/kolmogorov-cfc-design-retest-2026-10.md E5 用它檢驗 CfC 的 per-sensor 記憶在非同步缺值下
是否有價值。對照要成立需要：

1. 預設關閉時行為不變（token attention 不給 mask 時與舊版逐位元相同；載入器對無 NaN 的檔走原式）。
2. 遞迴只在該 sensor 有觀測時更新、Δt 取距上次觀測的時間；沒有觀測時狀態原樣保留，記憶確實被帶下去。
3. decoder 不讀不帶資訊的 token（遞迴臂：尚未觀測過；B2：該幀沒有觀測）。
4. data loss 只計有觀測的點，NaN 不進梯度。
5. 資料含 NaN 而模型沒開開關、或與未驗證的設定同時開啟時報錯。
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.data import load_sensors_from_path
from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import AsyncCellStep, LiquidOperator, TokenSelfAttentionBlock
from pi_lnn_jax.pipeline.kolmogorov import assembly

BASE = {
    "sensor_value_dim": 2, "d_model": 16, "d_time": 4,
    "num_spatial_encoder_layers": 1, "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 1, "token_attention_heads": 2, "num_query_mlp_layers": 1,
    "query_mlp_hidden_dim": 16, "operator_rank": 8,
    "decoder_attention_heads": 1, "use_temporal_anchor": True, "T_total": 1.0,
    "temporal_anchor_harmonics": 2, "domain_length": 1.0,
}
T, K, N = 6, 8, 5
ST = jnp.linspace(0.0, 1.0, T)
OBS = np.random.default_rng(0).random((T, K)) < 0.5
OBS[:, 0] = True                     # 每幀至少一個觀測
OBS[0, 3], OBS[1:, 3] = True, False  # sensor 3 只在第 0 幀觀測過


def _vals(seed=0, obs=OBS):
    v = np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (T, K, 2)))
    return jnp.asarray(np.where(obs[..., None], v, np.nan))


def _args(vals, t_q=None):
    t_q = jnp.full((N,), 0.95) if t_q is None else t_q
    return (vals, jax.random.uniform(jax.random.PRNGKey(1), (K, 2)), 0.5, ST,
            jax.random.uniform(jax.random.PRNGKey(2), (N, 2)), t_q)


def _init(m, vals):
    return m.init(jax.random.PRNGKey(42), *_args(vals))


def test_token_attention_without_mask_is_unchanged():
    blk = TokenSelfAttentionBlock(d_model=16, num_heads=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (T, K, 16))
    p = blk.init(jax.random.PRNGKey(1), x)
    assert jnp.array_equal(blk.apply(p, x), blk.apply(p, x, key_mask=None))
    assert "async_sensors" not in model_fingerprint(LiquidOperator(**BASE))
    assert model_fingerprint(LiquidOperator(**BASE, async_sensors=True))["async_sensors"] == "True"


@pytest.mark.parametrize("layers", [0, 1])
def test_async_forward_and_grad_are_finite(layers):
    m = LiquidOperator(**{**BASE, "num_temporal_cfc_layers": layers}, async_sensors=True)
    v = _vals()
    p = _init(m, v)
    out = m.apply(p, *_args(v))
    assert bool(jnp.all(jnp.isfinite(out)))
    g = jax.grad(lambda pp: jnp.sum(m.apply(pp, *_args(v)) ** 2))(p)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))
    h = m.apply(p, *_args(v)[:4], method=LiquidOperator.encode)
    assert h.shape == (T, K, BASE["d_model"] + 1)
    valid = np.asarray(h[..., -1]) > 0.5
    expect = (np.cumsum(OBS, axis=0) > 0) if layers else OBS
    assert np.array_equal(valid, expect)


def test_async_step_holds_state_and_uses_per_sensor_dt():
    step = AsyncCellStep(temporal_cell="cfc", hidden_size=16)
    h = jax.random.normal(jax.random.PRNGKey(0), (K, 16))
    last = jnp.zeros((K,)).at[1].set(0.4)
    x = jax.random.normal(jax.random.PRNGKey(1), (K, 16))
    obs = jnp.zeros((K,), bool).at[0].set(True).at[1].set(True)
    p = step.init(jax.random.PRNGKey(2), (h, last), (x, jnp.float32(0.5), obs))
    (h1, last1), _ = step.apply(p, (h, last), (x, jnp.float32(0.5), obs))
    assert jnp.array_equal(h1[2:], h[2:])                       # 沒有觀測：狀態不動
    assert not jnp.allclose(h1[:2], h[:2])
    assert jnp.array_equal(last1, jnp.where(obs, 0.5, last))
    # sensor 0 與 1 的輸入相同時，Δt（0.5 對 0.1）不同 → 新狀態不同
    x_same = x.at[1].set(x[0])
    h_same = h.at[1].set(h[0])
    (h2, _), _ = step.apply(p, (h_same, last), (x_same, jnp.float32(0.5), obs))
    assert not jnp.allclose(h2[0], h2[1])


def test_memory_carries_unobserved_sensor():
    """sensor 3 只在第 0 幀觀測過：改它第 0 幀的值，最後一幀的 CfC 狀態要變（記憶），B2 不受影響。"""
    v1 = _vals()
    v2 = v1.at[0, 3].add(1.0)
    for layers, should_change in ((1, True), (0, False)):
        m = LiquidOperator(**{**BASE, "num_temporal_cfc_layers": layers, "num_token_attention_layers": 0},
                           async_sensors=True)
        p = _init(m, v1)
        h1 = m.apply(p, *_args(v1)[:4], method=LiquidOperator.encode)
        h2 = m.apply(p, *_args(v2)[:4], method=LiquidOperator.encode)
        assert (not jnp.allclose(h1[-1, 3], h2[-1, 3])) == should_change, layers


def test_decoder_ignores_tokens_without_information():
    m = LiquidOperator(**BASE, async_sensors=True)
    v = _vals()
    p = _init(m, v)
    h = m.apply(p, *_args(v)[:4], method=LiquidOperator.encode)
    sp, xy, tq = _args(v)[1], _args(v)[4], _args(v)[5]
    h_bad = h.at[..., :-1].set(jnp.where(h[..., -1:] > 0.5, h[..., :-1], 1e3))
    a = m.apply(p, xy, tq, h, ST, sp, method=LiquidOperator.decode_query)
    b = m.apply(p, xy, tq, h_bad, ST, sp, method=LiquidOperator.decode_query)
    assert jnp.allclose(a, b)


def test_masked_sensor_loss_ignores_nan_targets():
    pred = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    tgt = np.array(jax.random.normal(jax.random.PRNGKey(1), (10, 2)))
    tgt[[2, 5, 7]] = np.nan
    tgt = jnp.asarray(tgt)
    ok = ~jnp.isnan(tgt).any(-1)
    ch_w = jnp.ones((2,))
    loss = assembly._masked_sensor_loss(pred, tgt, ch_w, jnp.zeros((10,)), 1.0, 0.05, 5.0)
    ref = jnp.mean(((pred - jnp.nan_to_num(tgt)) ** 2)[ok])
    assert jnp.allclose(loss, ref)
    g = jax.grad(lambda pp: assembly._masked_sensor_loss(pp, tgt, ch_w, jnp.zeros((10,)), 1.0, 0.05, 5.0))(pred)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g[~ok] == 0))


_GUARD = {"async_sensors": True, "data_has_nan": True, "supervision_sensor_json": "",
          "encoder_time_stride": 1, "target_time_stride": 1, "arch": "liquid", "n_datasets": 1,
          "sensor_cut_min_frac": 0.0, "autoreg_pseudo_dt": 0.0, "sensor_dropout_rate": 0.0,
          "use_continuous_re_physics": False, "refine_optimizer": "none"}


def test_guard():
    assembly.check_async_sensors(**_GUARD)
    assembly.check_async_sensors(**{**_GUARD, "async_sensors": False, "data_has_nan": False})
    with pytest.raises(ValueError, match="NaN"):
        assembly.check_async_sensors(**{**_GUARD, "async_sensors": False})
    for bad in ({"sensor_dropout_rate": 0.3}, {"refine_optimizer": "lbfgs"}, {"supervision_sensor_json": "x"},
                {"encoder_time_stride": 2}, {"n_datasets": 2}, {"arch": "vanilla"}):
        with pytest.raises(ValueError, match="async_sensors"):
            assembly.check_async_sensors(**{**_GUARD, **bad})


@pytest.mark.parametrize("kw", [{"cfc_query_time": True}, {"temporal_direction": "bidirectional"},
                                {"decoder_frame_context": "bracket"}])
def test_async_rejects_unsupported_combos(kw):
    with pytest.raises(ValueError, match="async_sensors"):
        m = LiquidOperator(**BASE, async_sensors=True, **kw)
        _init(m, _vals())


def test_loader_uses_observed_values_only(tmp_path):
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(4, 9)), rng.normal(size=(4, 9))
    np.savez(tmp_path / "a_dns_values.npz", time=np.linspace(0, 1, 9).astype(np.float32),
             u=u.astype(np.float32), v=v.astype(np.float32))
    meta = {"K": 4, "selected_coordinates": rng.uniform(size=(4, 2)).tolist(), "dns_values_npz": "a_dns_values.npz"}
    (tmp_path / "a.json").write_text(json.dumps(meta))
    clean = load_sensors_from_path(tmp_path / "a.json", time_stride=1)
    assert clean["norm_stats"]["u_mean"] == float(u.astype(np.float32).mean())
    u_nan = u.copy()
    u_nan[0, :3] = np.nan
    np.savez(tmp_path / "b_dns_values.npz", time=np.linspace(0, 1, 9).astype(np.float32),
             u=u_nan.astype(np.float32), v=v.astype(np.float32))
    (tmp_path / "b.json").write_text(json.dumps({**meta, "dns_values_npz": "b_dns_values.npz"}))
    d = load_sensors_from_path(tmp_path / "b.json", time_stride=1)
    assert np.isclose(d["norm_stats"]["u_mean"], float(np.nanmean(u_nan.astype(np.float32))))
    assert np.isnan(d["sensor_vals"][:3, 0, 0]).all() and np.isfinite(d["sensor_vals"][3:, 0, 0]).all()


def _ffill(vals):
    """逐 sensor 沿時間往前填最後一次觀測值；第一次觀測前維持 NaN。"""
    v = np.array(vals)
    for k in range(v.shape[1]):
        for t in range(1, v.shape[0]):
            if np.isnan(v[t, k, 0]):
                v[t, k] = v[t - 1, k]
    return jnp.asarray(v)


def test_async_hold_equals_b2_on_forward_filled_data():
    """B2-hold（E5c）＝把每個 sensor 的最後一次觀測值當作當幀觀測餵給 B2；記憶跟著保持值走，不經遞迴。"""
    obs = OBS.copy()
    obs[0, 5] = False                    # sensor 5 第 0 幀沒有觀測：保持前的空檔仍不帶資訊
    v = _vals(obs=obs)
    b2 = {**BASE, "num_temporal_cfc_layers": 0}
    hold = LiquidOperator(**b2, async_sensors=True, async_hold=True)
    ref = LiquidOperator(**b2, async_sensors=True)
    p = _init(hold, v)
    h_hold = hold.apply(p, *_args(v)[:4], method=LiquidOperator.encode)
    h_ref = ref.apply(p, *_args(_ffill(v))[:4], method=LiquidOperator.encode)
    assert jnp.array_equal(h_hold, h_ref)
    assert np.array_equal(np.asarray(h_hold[..., -1]) > 0.5, np.cumsum(obs, axis=0) > 0)
    v2 = v.at[0, 3].add(1.0)             # sensor 3 只在第 0 幀觀測過：保持值帶到最後一幀
    h2 = hold.apply(p, *_args(v2)[:4], method=LiquidOperator.encode)
    assert not jnp.allclose(h_hold[-1, 3], h2[-1, 3])
    assert jnp.array_equal(hold.apply(p, *_args(v)), ref.apply(p, *_args(_ffill(v))))


@pytest.mark.parametrize("kw", [{"async_sensors": False}, {"num_temporal_cfc_layers": 1}])
def test_async_hold_guard(kw):
    cfg = {**BASE, "num_temporal_cfc_layers": 0, "async_sensors": True, "async_hold": True, **kw}
    with pytest.raises(ValueError, match="async_hold"):
        LiquidOperator(**cfg).init(jax.random.PRNGKey(0), *_args(_vals()))
