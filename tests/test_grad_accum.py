"""accumulate_grads（gradient accumulation helper）核心不變式：
逐塊 scan 累積 + 平均 == 全量 value_and_grad（mean 型 loss）。M=1 為退化全量。
"""
import jax
import jax.numpy as jnp

from pi_lnn_jax.optimizers import accumulate_grads


def test_accumulate_grads_equals_full_batch_scalar_and_aux():
    N = 12
    x = jax.random.normal(jax.random.PRNGKey(0), (N,))
    y = jax.random.normal(jax.random.PRNGKey(1), (N,))
    w = jnp.array(0.37)

    def loss_fn(p, xi, yi):
        r = p * xi - yi
        return jnp.mean(r ** 2), jnp.mean(jnp.abs(r))      # (loss, aux)

    vg = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_full, aux_full), g_full = vg(w, x, y)
    for M in (1, 2, 3, 4, 6):                              # 各種等分塊數
        g, loss, aux = accumulate_grads(vg, w, (x.reshape(M, -1), y.reshape(M, -1)))
        assert jnp.allclose(g, g_full, atol=1e-5), f"M={M} grad"
        assert jnp.allclose(loss, loss_full, atol=1e-5), f"M={M} loss"
        assert jnp.allclose(aux, aux_full, atol=1e-5), f"M={M} aux"


def test_accumulate_grads_pytree_params_and_aux():
    N = 8
    x = jax.random.normal(jax.random.PRNGKey(2), (N, 3))
    y = jax.random.normal(jax.random.PRNGKey(3), (N,))
    w = {"a": jax.random.normal(jax.random.PRNGKey(4), (3,)), "b": jnp.array(0.5)}

    def loss_fn(p, xi, yi):
        r = xi @ p["a"] + p["b"] - yi
        return jnp.mean(r ** 2), (jnp.mean(r), jnp.max(jnp.abs(r)))   # aux 為 tuple

    vg = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_full, aux_full), g_full = vg(w, x, y)
    g, loss, aux = accumulate_grads(vg, w, (x.reshape(4, 2, 3), y.reshape(4, 2)))
    assert jnp.allclose(g["a"], g_full["a"], atol=1e-5)
    assert jnp.allclose(g["b"], g_full["b"], atol=1e-5)
    assert jnp.allclose(loss, loss_full, atol=1e-5)
    # aux tuple 第一項（mean(r)）對 M 塊平均 == 全量；第二項（max）為近似（per-chunk max 平均）
    assert jnp.allclose(aux[0], aux_full[0], atol=1e-5)


def test_accumulate_grads_rejects_single_array():
    # 誤傳單一 array（非 tuple/list）→ *chunk 會誤展開 → FailFast TypeError
    import pytest
    vg = jax.value_and_grad(lambda p, xi: (jnp.mean((p * xi) ** 2), 0.0), has_aux=True)
    with pytest.raises(TypeError):
        accumulate_grads(vg, jnp.array(1.0), jnp.zeros((4, 2)))   # 應為 (chunks,) 卻傳 array
