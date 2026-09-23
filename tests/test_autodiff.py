"""驗證 fused 導數工廠與樸素 jax.grad 數學等價（x64 緊容差）。

腳本式（對齊 tests/ 慣例）：uv run python tests/test_autodiff.py
"""
import jax
import jax.numpy as jnp
import pytest
from jax import config as _jax_config

from pi_lnn_jax.autodiff import make_fused_field_derivatives


# fused-vs-naive 比較需 float64（x64 緊容差）。x64 是 process-global 旗標，
# 在 module import 時開啟會洩漏到後續測試（例如 test_refiners 的 optimistix
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


def _toy_field(params, h, x, y, t, a):
    # 任意可二階微分的 [3] 場：含交互與三角，模擬 decoder 行為
    w = params["w"]
    u = jnp.sin(w[0] * x + a * y) + x * y * t
    v = jnp.cos(w[1] * y) * t + x ** 2
    p = jnp.exp(-(x ** 2 + y ** 2)) + w[2] * t
    return jnp.stack([u, v, p])


def test_fused_matches_naive():
    params = {"w": jnp.array([1.3, 0.7, 0.2])}
    h = jnp.zeros((1,))
    a = 0.5
    key = jax.random.PRNGKey(0)
    kx, ky, kt = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (16,))
    ys = jax.random.uniform(ky, (16,))
    ts = jax.random.uniform(kt, (16,))

    fused = make_fused_field_derivatives(_toy_field)
    val, jac, fxx, fyy = fused(params, h, xs, ys, ts, a)
    assert val.shape == (16, 3)
    assert jac.shape == (16, 3, 3)
    assert fxx.shape == (16, 3) and fyy.shape == (16, 3)

    def comp(c):
        def f(params, h, x, y, t, a):
            return _toy_field(params, h, x, y, t, a)[c]
        return f
    in_axes = (None, None, 0, 0, 0, None)
    for c in range(3):
        f = comp(c)
        fx = jax.vmap(jax.grad(f, argnums=2), in_axes=in_axes)(params, h, xs, ys, ts, a)
        fy = jax.vmap(jax.grad(f, argnums=3), in_axes=in_axes)(params, h, xs, ys, ts, a)
        ft = jax.vmap(jax.grad(f, argnums=4), in_axes=in_axes)(params, h, xs, ys, ts, a)
        fxx_n = jax.vmap(jax.grad(jax.grad(f, argnums=2), argnums=2), in_axes=in_axes)(params, h, xs, ys, ts, a)
        fyy_n = jax.vmap(jax.grad(jax.grad(f, argnums=3), argnums=3), in_axes=in_axes)(params, h, xs, ys, ts, a)
        assert jnp.allclose(jac[:, c, 0], fx, atol=1e-9)
        assert jnp.allclose(jac[:, c, 1], fy, atol=1e-9)
        assert jnp.allclose(jac[:, c, 2], ft, atol=1e-9)
        # Laplacian = d2x + d2y 為 mode-agnostic 物理量（physics 殘差只用此和）。
        # fof/taylor/ror 回傳逐方向 (u_xx, u_yy)；folx（aniso 修正後）回傳
        # (lap−d2y, d2y) 的等效拆分（見 test_physics_anisotropic.py）。兩者和相同。
        assert jnp.allclose(fxx[:, c] + fyy[:, c], fxx_n + fyy_n, atol=1e-9)
