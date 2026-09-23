"""驗證重構後 physics 殘差 == 樸素參考；return_per_point 一致性。

腳本式：uv run python tests/test_physics_fused.py
"""
import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs
from jax import config as _jax_config

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn


# 本檔的 ns-vs-naive 比較需 float64（atol 1e-9）。x64 是 process-global 旗標，
# 若在 module import 時開啟會洩漏到後續測試（例如 test_refiners 的 optimistix
# scan/lstsq 在 x64 下 carry dtype 不一致而失敗）。故改用 autouse fixture：
# 只在本檔測試範圍內開 x64，結束還原，避免污染套件其他測試。
@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)


def _build():
    model = LiquidOperator(**minimal_kwargs(2))
    K, T, N = 5, 4, 12
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 2))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1,
                        sensor_time, jax.random.uniform(jax.random.PRNGKey(9), (N, 2)),
                        jnp.zeros((N,)))
    h_states = model.apply(params, sensor_vals, sensor_pos, 0.1, sensor_time,
                           method=LiquidOperator.encode)
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,), maxval=1.0)
    return model, params, h_states, sensor_pos, sensor_time, cx, cy, ct


def _naive(model, params, h_states, sp, st, cx, cy, ct, A, k_f, nu, um, us, vm, vs, pm, ps):
    def field(p, h, x, y, t):
        out = model.apply(p, jnp.array([[x, y]]), jnp.array([t]),
                          h, st, sp, method=LiquidOperator.decode_query)[0]
        return jnp.stack([out[0] * us + um, out[1] * vs + vm, out[2] * ps + pm])
    u_at = lambda p, h, x, y, t: field(p, h, x, y, t)[0]
    v_at = lambda p, h, x, y, t: field(p, h, x, y, t)[1]
    p_at = lambda p, h, x, y, t: field(p, h, x, y, t)[2]
    ia = (None, None, 0, 0, 0)
    vg = lambda fn, n: jax.vmap(jax.grad(fn, argnums=n), in_axes=ia)
    u = jax.vmap(u_at, in_axes=ia)(params, h_states, cx, cy, ct)
    v = jax.vmap(v_at, in_axes=ia)(params, h_states, cx, cy, ct)
    u_x = vg(u_at, 2)(params, h_states, cx, cy, ct); u_y = vg(u_at, 3)(params, h_states, cx, cy, ct)
    u_t = vg(u_at, 4)(params, h_states, cx, cy, ct)
    u_xx = jax.vmap(jax.grad(jax.grad(u_at, 2), 2), in_axes=ia)(params, h_states, cx, cy, ct)
    u_yy = jax.vmap(jax.grad(jax.grad(u_at, 3), 3), in_axes=ia)(params, h_states, cx, cy, ct)
    v_x = vg(v_at, 2)(params, h_states, cx, cy, ct); v_y = vg(v_at, 3)(params, h_states, cx, cy, ct)
    v_t = vg(v_at, 4)(params, h_states, cx, cy, ct)
    v_xx = jax.vmap(jax.grad(jax.grad(v_at, 2), 2), in_axes=ia)(params, h_states, cx, cy, ct)
    v_yy = jax.vmap(jax.grad(jax.grad(v_at, 3), 3), in_axes=ia)(params, h_states, cx, cy, ct)
    p_x = vg(p_at, 2)(params, h_states, cx, cy, ct); p_y = vg(p_at, 3)(params, h_states, cx, cy, ct)
    f_x = A * jnp.sin(2.0 * jnp.pi * k_f * cy)
    mom_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - f_x
    mom_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    cont = u_x + v_y
    return jnp.mean(mom_u ** 2), jnp.mean(mom_v ** 2), jnp.mean(cont ** 2)


_ARGS = (0.1, 2.0, 1e-3, 0.0, 0.4, 0.0, 0.4)
# pm, ps 對齊舊行為（p_mean=0, p_std=1 → out[2] identity）
_PARGS = _ARGS + (0.0, 1.0)


def test_ns_residual_matches_naive():
    model, params, h, sp, st, cx, cy, ct = _build()
    ns_fn, _ = make_ns_residual_fn(model)
    mu, mv, c = ns_fn(params, h, cx, cy, ct, _PARGS[0], _PARGS[1], sp, st, *_PARGS[2:])
    mu_n, mv_n, c_n = _naive(model, params, h, sp, st, cx, cy, ct, *_PARGS)
    assert jnp.allclose(mu, mu_n, atol=1e-9), (mu, mu_n)
    assert jnp.allclose(mv, mv_n, atol=1e-9), (mv, mv_n)
    assert jnp.allclose(c, c_n, atol=1e-9), (c, c_n)


def test_return_per_point_consistency():
    model, params, h, sp, st, cx, cy, ct = _build()
    ns_fn, _ = make_ns_residual_fn(model)
    mu, mv, c = ns_fn(params, h, cx, cy, ct, _PARGS[0], _PARGS[1], sp, st, *_PARGS[2:])
    mu2, mv2, c2, mu_pp, mv_pp, c_pp = ns_fn(
        params, h, cx, cy, ct, _PARGS[0], _PARGS[1], sp, st, *_PARGS[2:],
        return_per_point=True)
    assert jnp.allclose(mu, mu2) and jnp.allclose(mv, mv2) and jnp.allclose(c, c2)
    assert jnp.allclose(jnp.mean(mu_pp ** 2), mu, atol=1e-9)
    assert jnp.allclose(jnp.mean(mv_pp ** 2), mv, atol=1e-9)
    assert jnp.allclose(jnp.mean(c_pp ** 2), c, atol=1e-9)


def test_p_mean_invariance_and_p_std_scaling():
    """殘差只用 p 梯度：p_mean 不應改變殘差；p_std 應改變壓力梯度項。"""
    model, params, h, sp, st, cx, cy, ct = _build()
    ns_fn, _ = make_ns_residual_fn(model)
    A, k_f, nu, um, us, vm, vs = 0.1, 2.0, 1e-4, 0.0, 1.0, 0.0, 1.0
    r0 = ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs, 0.0, 1.0)
    r_meanshift = ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs, 5.0, 1.0)
    r_stdscale = ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs, 0.0, 2.0)
    assert jnp.allclose(jnp.array(r0[0]), jnp.array(r_meanshift[0]), atol=1e-10)
    assert jnp.allclose(jnp.array(r0[1]), jnp.array(r_meanshift[1]), atol=1e-10)
    assert not jnp.allclose(jnp.array(r0[0]), jnp.array(r_stdscale[0]), atol=1e-8)
    print("✓ p_mean invariance + p_std scaling")
