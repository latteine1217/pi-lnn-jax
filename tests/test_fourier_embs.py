"""非週期 FourierEmbs（RFF）能區分 x=0 與 x=L；periodic_domain 正確路由 spatial embedding。"""
import jax
import jax.numpy as jnp
import numpy as np
from pi_lnn_jax.models import LearnableFourierEmb, FourierEmbs, SpatialSetEncoder


def test_periodic_emb_folds_boundaries():
    # LearnableFourierEmb（週期）把 x=0 與 x=1 編成相同（domain_length=1）
    m = LearnableFourierEmb(embed_dim=128)
    xy = jnp.array([[0.0, 0.5], [1.0, 0.5]], jnp.float32)
    p = m.init(jax.random.PRNGKey(0), xy, 1.0)
    out = m.apply(p, xy, 1.0)
    assert float(jnp.max(jnp.abs(out[0] - out[1]))) < 1e-5   # 折疊（週期域正確）


def test_rff_distinguishes_boundaries():
    # FourierEmbs（RFF，非週期）必須區分 x=0 與 x=1
    m = FourierEmbs(embed_dim=128)
    xy = jnp.array([[0.0, 0.5], [1.0, 0.5]], jnp.float32)
    p = m.init(jax.random.PRNGKey(0), xy, 1.0)
    out = m.apply(p, xy, 1.0)
    assert float(jnp.max(jnp.abs(out[0] - out[1]))) > 0.1    # 不折疊（非週期域必要）


def test_spatial_encoder_routes_by_periodic_domain():
    # periodic_domain=False → spatial_emb 為 FourierEmbs；True → LearnableFourierEmb
    enc_np = SpatialSetEncoder(d_model=32, num_layers=1, 
                              sensor_value_dim=2, fourier_embed_dim=128, periodic_domain=False)
    enc_p = SpatialSetEncoder(d_model=32, num_layers=1, 
                             sensor_value_dim=2, fourier_embed_dim=128, periodic_domain=True)
    sp = jnp.asarray(np.random.RandomState(0).uniform(0, 1, (10, 2)), jnp.float32)
    sv = jnp.asarray(np.random.RandomState(1).randn(5, 10, 2), jnp.float32)
    st = jnp.linspace(0, 1, 5).astype(jnp.float32)
    # init 兩者；確認非週期版的 pos encoding 對 x=0 vs x=1 不同
    enc = SpatialSetEncoder(d_model=32, num_layers=1, 
                            sensor_value_dim=2, fourier_embed_dim=128, periodic_domain=False)
    xy2 = jnp.array([[0.0, 0.5], [1.0, 0.5]], jnp.float32)
    params = enc.init(jax.random.PRNGKey(0), xy2, method=SpatialSetEncoder.encode_pos)
    pe = enc.apply(params, xy2, method=SpatialSetEncoder.encode_pos)
    assert float(jnp.max(jnp.abs(pe[0] - pe[1]))) > 0.1     # 非週期 → 區分邊界
