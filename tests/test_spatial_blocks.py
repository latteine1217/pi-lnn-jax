"""空間分塊、多 case 訓練（`spatial_blocks`）：domain 切成 n×n 塊，每塊是同一個網路要擬合的一個 case。

## 為什麼這一組測試存在

knowledge/experiments/kolmogorov-cfc-design-retest-2026-10.md E6 用它檢驗：當 trunk 無法靠絕對位置把單一軌跡
記下來時，branch 的時間記憶是否變得必要。對照要成立需要：

1. 預設關閉時行為不變（attention 不給 mask、指紋不記這個鍵）。
2. 塊與塊完全隔離：改一塊的 sensor，其他塊的預測不變（同步與非同步、有無遞迴層）。
3. trunk 分不出是哪一塊：兩塊的 sensor 局部位置與數值相同時，同局部座標的預測相同。
4. 局部座標不改變座標導數：folx 的 Laplacian 與 jax.hessian 一致（PDE residual 走這條路）。
5. forcing 不滿足平移不變、或與未驗證的設定同時開啟時報錯。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.autodiff import make_fused_field_derivatives
from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import LiquidOperator, TokenSelfAttentionBlock, block_split

BASE = {
    "sensor_value_dim": 2, "d_model": 16, "d_time": 4,
    "num_spatial_encoder_layers": 1, "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 1, "token_attention_heads": 2, "num_query_mlp_layers": 1,
    "query_mlp_hidden_dim": 16, "operator_rank": 8,
    "decoder_attention_heads": 1, "use_temporal_anchor": True, "T_total": 1.0,
    "temporal_anchor_harmonics": 2, "domain_length": 1.0, "forcing_k_f_init": 2.0,
}
T = 5
ST = jnp.linspace(0.0, 1.0, T)
# 每塊 3 個 sensor；塊 0 與塊 3 的局部位置相同
LOCAL = np.array([[0.1, 0.2], [0.3, 0.05], [0.4, 0.35]])
OFFSETS = np.array([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
POS = jnp.asarray(np.concatenate([LOCAL + o for o in OFFSETS]))          # [12, 2]，塊 0,1,2,3 依序
K = POS.shape[0]
Q_LOCAL = np.array([[0.05, 0.1], [0.25, 0.4], [0.45, 0.45]])
XY = jnp.asarray(np.concatenate([Q_LOCAL + o for o in OFFSETS]))        # [12, 2]
TQ = jnp.full((XY.shape[0],), 0.7)


def _vals(seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), (T, K, 2))


def _model(**kw):
    return LiquidOperator(**{**BASE, **kw})


def _apply(m, p, vals, xy=XY, tq=TQ):
    return m.apply(p, vals, POS, 0.5, ST, xy, tq)


def test_block_split():
    local, bid = block_split(jnp.array([[0.1, 0.7], [0.6, 0.2], [1.0, 1.0], [0.5, 0.5]]), 2, 1.0)
    np.testing.assert_allclose(local, [[0.1, 0.2], [0.1, 0.2], [0.5, 0.5], [0.0, 0.0]], atol=1e-7)
    assert bid.tolist() == [1, 2, 3, 3]
    g = jax.jacobian(lambda z: block_split(z, 2, 1.0)[0])(jnp.array([0.7, 0.3]))
    np.testing.assert_allclose(g, np.eye(2))


def test_default_is_unchanged():
    blk = TokenSelfAttentionBlock(d_model=16, num_heads=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (T, K, 16))
    p = blk.init(jax.random.PRNGKey(1), x)
    assert jnp.array_equal(blk.apply(p, x), blk.apply(p, x, key_mask=None, pair_mask=None))
    assert "spatial_blocks" not in model_fingerprint(_model())
    assert model_fingerprint(_model(spatial_blocks=2))["spatial_blocks"] == "2"


@pytest.mark.parametrize("layers", [0, 1])
@pytest.mark.parametrize("async_", [False, True])
def test_blocks_are_isolated(layers, async_):
    m = _model(num_temporal_cfc_layers=layers, spatial_blocks=2, async_sensors=async_)
    v1 = _vals()
    if async_:
        obs = np.random.default_rng(0).random((T, K)) < 0.6
        obs[:, ::3] = True
        obs[2, 0:3] = False                       # 塊 0 在第 2 幀完全沒有觀測（走同塊退回路徑）
        v1 = jnp.where(jnp.asarray(obs)[..., None], v1, jnp.nan)
    p = m.init(jax.random.PRNGKey(42), v1, POS, 0.5, ST, XY, TQ)
    v2 = v1.at[:, 0:3].multiply(3.0).at[:, 0:3].add(1.0)   # 只改塊 0 的 sensor
    tq = jnp.full_like(TQ, 0.9)                            # 第 3 幀：塊 0 有觀測
    a, b = _apply(m, p, v1, tq=tq), _apply(m, p, v2, tq=tq)
    assert bool(jnp.all(jnp.isfinite(a)))
    assert not jnp.allclose(a[0:3], b[0:3])               # 塊 0 的 query 受影響
    np.testing.assert_array_equal(np.asarray(a[3:]), np.asarray(b[3:]))   # 其他塊逐位元不變
    # 第 2 幀（非同步時塊 0 全無觀測）：輸出有限、其他塊仍不受影響
    tq2 = jnp.full_like(TQ, 0.6)
    a2, b2 = _apply(m, p, v1, tq=tq2), _apply(m, p, v2, tq=tq2)
    assert bool(jnp.all(jnp.isfinite(a2)))
    np.testing.assert_array_equal(np.asarray(a2[3:]), np.asarray(b2[3:]))


def test_trunk_cannot_tell_blocks_apart():
    m = _model(spatial_blocks=2)
    v = np.array(_vals())
    v[:, 9:12] = v[:, 0:3]                         # 塊 3 的 sensor 值＝塊 0（局部位置本來就相同）
    v = jnp.asarray(v)
    p = m.init(jax.random.PRNGKey(42), v, POS, 0.5, ST, XY, TQ)
    out = _apply(m, p, v)
    np.testing.assert_allclose(out[0:3], out[9:12], rtol=1e-5, atol=1e-6)
    off = _model()                                 # 對照：不分塊時 trunk 看得到絕對位置
    po = off.init(jax.random.PRNGKey(42), v, POS, 0.5, ST, XY, TQ)
    assert not jnp.allclose(_apply(off, po, v)[0:3], _apply(off, po, v)[9:12], rtol=1e-5, atol=1e-6)


def test_laplacian_through_local_coordinates():
    m = _model(spatial_blocks=2)
    v = _vals()
    p = m.init(jax.random.PRNGKey(42), v, POS, 0.5, ST, XY, TQ)
    h = m.apply(p, v, POS, 0.5, ST, method=LiquidOperator.encode)

    def field(params, h_, x, y, t):
        return m.apply(params, jnp.stack([x, y])[None], t[None], h_, ST, POS,
                       method=LiquidOperator.decode_query)[0]

    fused = make_fused_field_derivatives(field)
    xs, ys, ts = XY[:, 0], XY[:, 1], TQ
    _, jac, d2x, d2y = fused(p, h, xs, ys, ts)
    for i in range(XY.shape[0]):
        def f(c, t_i=ts[i]):
            return field(p, h, c[0], c[1], t_i)
        c = jnp.array([xs[i], ys[i]])
        np.testing.assert_allclose(jac[i, :, :2], jax.jacobian(f)(c), rtol=1e-4, atol=1e-5)
        H = jax.hessian(f)(c)
        np.testing.assert_allclose(d2x[i] + d2y[i], H[:, 0, 0] + H[:, 1, 1], rtol=1e-4, atol=1e-5)


def test_guards():
    v = _vals()
    with pytest.raises(ValueError, match="平移不變"):
        _model(spatial_blocks=2, forcing_k_f_init=1.5).init(jax.random.PRNGKey(0), v, POS, 0.5, ST, XY, TQ)
    with pytest.raises(ValueError, match="平移不變"):
        _model(spatial_blocks=2, learn_forcing_k_f=True).init(jax.random.PRNGKey(0), v, POS, 0.5, ST, XY, TQ)
    for kw in ({"attention_kind": "vector"}, {"disable_cross_attention": True},
               {"decoder_frame_context": "bracket"}, {"mmlp_branch_u": True, "use_modified_mlp": True}):
        with pytest.raises(ValueError, match="spatial_blocks"):
            _model(spatial_blocks=2, **kw).init(jax.random.PRNGKey(0), v, POS, 0.5, ST, XY, TQ)
