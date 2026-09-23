"""ns_residuals anisotropic Lx/Ly: default 1.0 reproduces today; Lx≠1 rescales derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn

CFG = dict(sensor_value_dim=2, d_model=32, d_time=8,
           num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
           num_token_attention_layers=1, token_attention_heads=2,
           num_query_mlp_layers=1, query_mlp_hidden_dim=32, operator_rank=32,
           decoder_attention_heads=2, use_temporal_anchor=True, T_total=1.0,
           temporal_anchor_harmonics=2, domain_length=1.0, fourier_embed_dim=128,
           relpos_bias_mode="radial", periodic_domain=True, use_sdf_features=False)


def _setup():
    model = LiquidOperator(**CFG)
    rng = np.random.RandomState(0)
    K, T = 8, 5
    sv = jnp.asarray(rng.randn(T, K, 2), jnp.float32)
    sp = jnp.asarray(rng.uniform(0, 1, (K, 2)), jnp.float32)
    st = jnp.asarray(np.linspace(0, 1, T), jnp.float32)
    xy = jnp.asarray(rng.uniform(0, 1, (4, 2)), jnp.float32)
    tt = jnp.asarray(rng.uniform(0, 1, (4,)), jnp.float32)
    params = model.init(jax.random.PRNGKey(0), sv, sp, 0.5, st, xy[:1], tt[:1])
    h = model.apply(params, sv, sp, 0.5, st, method=LiquidOperator.encode)
    return model, params, h, sp, st, xy, tt


def test_default_lxly_one_unchanged():
    model, params, h, sp, st, xy, tt = _setup()
    ns_fn, _ = make_ns_residual_fn(model)
    cx, cy, ct = xy[:, 0], xy[:, 1], tt
    base = ns_fn(params, h, cx, cy, ct, 0.1, 2.0, sp, st, 1e-4, 0., 1., 0., 1., 0., 1.)
    same = ns_fn(params, h, cx, cy, ct, 0.1, 2.0, sp, st, 1e-4, 0., 1., 0., 1., 0., 1., Lx=1.0, Ly=1.0)
    for a, b in zip(base, same):
        assert abs(float(a) - float(b)) < 1e-9


def test_lx_scaling_changes_residual():
    model, params, h, sp, st, xy, tt = _setup()
    ns_fn, _ = make_ns_residual_fn(model)
    cx, cy, ct = xy[:, 0], xy[:, 1], tt
    base = ns_fn(params, h, cx, cy, ct, 0.0, 2.0, sp, st, 1e-4, 0., 1., 0., 1., 0., 1.)
    scaled = ns_fn(params, h, cx, cy, ct, 0.0, 2.0, sp, st, 1e-4, 0., 1., 0., 1., 0., 1., Lx=0.5, Ly=0.5)
    # continuity (index 2) must differ once derivatives are rescaled by 1/Lx
    assert abs(float(base[2]) - float(scaled[2])) > 1e-9


def test_folx_returns_per_direction_second_derivs():
    """回歸守門（#1）：folx 必須回 per-direction d2x=∂²/∂x²、d2y=∂²/∂y²，
    而非把 collapsed laplacian 全塞 d2x、d2y=0。後者在 Lx≠Ly（cylinder）會讓
    黏性項 d2x/Lx²+d2y/Ly² 退化成 (uxx+uyy)/Lx²（錯）。"""
    from pi_lnn_jax.autodiff import make_fused_field_derivatives
    def field_fn(p, h, x, y, t):
        return jnp.array([jnp.sin(2.0 * x) + jnp.cos(5.0 * y)])
    fused = make_fused_field_derivatives(field_fn)
    xs = jnp.array([0.3, 0.7]); ys = jnp.array([0.4, 0.2]); ts = jnp.zeros(2)
    _, _, d2x, d2y = fused(None, None, xs, ys, ts)
    uxx = -4.0 * jnp.sin(2.0 * xs)
    uyy = -25.0 * jnp.cos(5.0 * ys)
    assert jnp.allclose(d2x[:, 0], uxx, atol=1e-4), f"d2x≠∂²/∂x²: {d2x[:,0]} vs {uxx}"
    assert jnp.allclose(d2y[:, 0], uyy, atol=1e-4), f"d2y≠∂²/∂y²（folx d2y=0 回歸）: {d2y[:,0]} vs {uyy}"


def test_poisson_residual_anisotropic_scaling():
    """回歸守門（round-3 F）：poisson_residual 須做 Lx/Ly 尺度化（對齊 ns_residuals），
    否則 Lx≠Ly 時 ∇²p=p_xx/Lx²+p_yy/Ly² 退化（少乘 1/Lx²、1/Ly²）。"""
    model, params, h, sp, st, xy, tt = _setup()
    _, poisson_fn = make_ns_residual_fn(model)
    cx, cy, ct = xy[:, 0], xy[:, 1], tt
    base = poisson_fn(params, h, cx, cy, ct, sp, st, 0., 1., 0., 1., 0., 1.)
    scaled = poisson_fn(params, h, cx, cy, ct, sp, st, 0., 1., 0., 1., 0., 1., Lx=0.5, Ly=2.0)
    # Lx≠1/Ly≠1 時尺度化後 residual 必須改變（不變代表沒做尺度化 = 回歸）
    assert abs(float(base) - float(scaled)) > 1e-9
