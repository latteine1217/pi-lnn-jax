"""Causal weighting 純函式性質測試。腳本式：uv run python tests/test_causal.py"""
import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.causal import causal_weights


def test_eps_zero_is_uniform():
    ct = jnp.array([0.1, 0.9, 0.5, 0.3])
    r = jnp.array([1.0, 2.0, 0.5, 3.0])
    w = causal_weights(ct, r, 0.0)
    assert jnp.allclose(w, jnp.ones_like(w))


def test_monotone_decreasing_in_time_for_sorted():
    ct = jnp.array([0.0, 1.0, 2.0, 3.0])
    r = jnp.array([1.0, 1.0, 1.0, 1.0])
    w = causal_weights(ct, r, 1.0)
    assert bool(jnp.all(w[1:] <= w[:-1] + 1e-12))
    assert jnp.allclose(w[0], 1.0)


def test_unsorted_time_correctness():
    ct = jnp.array([2.0, 0.0, 3.0, 1.0])
    r = jnp.array([5.0, 1.0, 7.0, 3.0])
    eps = 0.5
    w = causal_weights(ct, r, eps)
    order = jnp.argsort(ct)
    r_s = r[order]
    cum = jnp.cumsum(r_s) - r_s
    w_s = jnp.exp(-eps * cum)
    expected = jnp.zeros_like(w).at[order].set(w_s)
    assert jnp.allclose(w, expected, atol=1e-9)


def test_stop_gradient():
    ct = jnp.array([0.0, 1.0, 2.0])
    def loss(r):
        w = causal_weights(ct, r, 1.0)
        return jnp.sum(w * r)
    g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
    w = causal_weights(ct, jnp.array([1.0, 2.0, 3.0]), 1.0)
    assert jnp.allclose(g, w, atol=1e-9)


def test_continuous_collocation_times_do_not_collapse_the_weight():
    """連續取樣的 `ct` 經分箱後不再讓 slab 退化——TD-5 的修法。

    `run.py` 的 collocation 時間走 `jax.random.uniform`，1024 個點各不相同。
    不分箱就每點自成一個 slab，正是 `causal.py` docstring 說 slab 聚合要避免的
    情形，schema 預設的 eps=1.0 會因此關掉 99.85% 的 physics 殘差。

    `causal_weights` 現在先把 ct 等寬分箱到 `DEFAULT_N_SLABS`，所以 eps 的尺度
    回到 Wang 2022 校準的那個意義上。`n_slabs=n`（每點一箱）保留為舊行為的對照，
    兩者的落差就是這個 bug 的份量。
    """
    n = 1024
    ct = jax.random.uniform(jax.random.PRNGKey(0), (n,)) * 5.0
    residual = jnp.ones((n,))
    assert len(jnp.unique(ct)) == n, "前提：連續取樣下每點都是相異時刻"

    # eps=0 仍是關閉
    assert float(causal_weights(ct, residual, 0.0).sum()) == pytest.approx(n, rel=1e-6)

    fixed = float(causal_weights(ct, residual, 1.0).sum())
    degenerate = float(causal_weights(ct, residual, 1.0, n_slabs=n).sum())
    assert degenerate < 0.01 * n, (
        f"每點一箱時有效權重應塌掉（實測 {degenerate:.3f}/{n}）——這是修法要避開的行為")
    # 實測 54.0 vs 3.74（14.5×）；門檻取 10× 留餘裕，變動時請一併更新 causal.py 的說明
    assert fixed > 10 * degenerate, (
        f"分箱後有效權重 {fixed:.1f}/{n}，未分箱 {degenerate:.3f}——落差消失代表分箱沒生效")


def test_n_slabs_controls_granularity_monotonically():
    """slab 數越多，同一個 eps 壓得越狠（cum_excl 的項數變多）。

    這條釘住 `n_slabs` 是 eps 的尺度旋鈕：改它等同改 eps，不是無關的實作細節。
    """
    n = 512
    ct = jax.random.uniform(jax.random.PRNGKey(1), (n,)) * 5.0
    residual = jnp.ones((n,))
    sums = [float(causal_weights(ct, residual, 0.5, n_slabs=m).sum())
            for m in (4, 8, 16, 32, 64)]
    assert all(a > b for a, b in zip(sums, sums[1:])), f"非單調：{sums}"


def test_empty_slabs_do_not_decay_later_slabs():
    """沒有樣本的時段不該對它之後的 slab 施加衰減。

    ct 只落在時窗的兩端時，中間的空 slab 其 loss 記 0；若改成別的填值
    （例如全域平均），末端權重會被無中生有的殘差壓低。
    """
    ct = jnp.concatenate([jnp.zeros((8,)), jnp.ones((8,))])
    residual = jnp.ones((16,))
    w = causal_weights(ct, residual, 1.0, n_slabs=16)
    # 後半段只該被「前半段那一個非空 slab」壓一次
    assert float(w[-1]) == pytest.approx(float(jnp.exp(-1.0)), rel=1e-5)
