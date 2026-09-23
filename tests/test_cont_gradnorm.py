"""continuity 的兩條約束路徑互斥，且新路徑預設關閉。

thesis appendix G 把 AL-off 那一臂描述成「continuity 仍由 GradNorm 加權」，
但 JAX 實作裡 continuity 只走 AL：關掉 AL 就完全無約束。`cont_gradnorm`
是把那個被描述、卻從未被跑過的臂真正接上的開關。

這裡的斷言全部打在**真的** `assembly._build_loss_fn` / `_build_grad_norm_fn` /
`resolve_inputs` 上。先前版本在測試檔內複刻了一份 total 再斷言那份複刻品，
結果是實作用了 raw `cont`（而非與 momentum 同源的 `cont_eff`）整整六個測試全綠——
複刻品驗不到實作，這個檔案存在的意義就是不要再發生一次。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.losses import gradnorm_init, gradnorm_weights
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.kolmogorov import assembly

K, T, N = 5, 4, 8
W_PHYS = 0.37          # 刻意不取 1.0：1.0 會讓「漏乘 w_phys」與正確實作同值
W_CONT = 0.11


def _fixture(cont_gradnorm: bool, use_causal: bool = False):
    """真的 loss_fn + 一組固定輸入。回傳 (loss_fn, params, args)。"""
    model = LiquidOperator(**minimal_kwargs(3))
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1,
                        sensor_time, jax.random.uniform(jax.random.PRNGKey(9), (N, 2)),
                        jnp.zeros((N,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=False, use_al=False, al_rho=0.0,
        w_poisson=0.0, cont_gradnorm=cont_gradnorm, T_total=1.0,
        use_causal=use_causal,
    )
    re_batch = assembly.ReBatch(
        sensor_vals=sensor_vals, sensor_pos=sensor_pos, sensor_time=sensor_time,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0),
    )
    # ct 刻意只取兩個相異時刻：causal 的 slab prefix-sum 要有東西可分辨
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jnp.concatenate([jnp.zeros((N // 2,)), jnp.ones((N - N // 2,))])
    return loss_fn, params, (cx, cy, ct, re_batch)


def _run(loss_fn, params, args, task_weights, causal_eps=0.0):
    cx, cy, ct, re_batch = args
    total, aux = loss_fn(params, cx, cy, ct, task_weights,
                         jnp.asarray(0.0), jnp.asarray(W_PHYS), re_batch, causal_eps)
    return total, aux


TW3 = jnp.array([1.0, 0.057, 0.057])
TW4 = jnp.array([1.0, 0.057, 0.057, W_CONT])
TW4_ZERO = jnp.array([1.0, 0.057, 0.057, 0.0])


def test_flag_defaults_off_in_schema():
    """預設值必須是 False，否則主線行為被靜默改掉。"""
    from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy

    assert KolmogorovPolicy._defaults["loss.cont_gradnorm"] is False


def test_off_is_bitwise_identical_to_zero_weight():
    """關閉時多出來的項恆為 0：與「開啟但 w_cont=0」逐位元相同。"""
    off_fn, p, args = _fixture(cont_gradnorm=False)
    on_fn, p2, args2 = _fixture(cont_gradnorm=True)
    total_off, _ = _run(off_fn, p, args, TW3)
    total_on0, _ = _run(on_fn, p2, args2, TW4_ZERO)
    assert float(total_off) == float(total_on0)


def test_on_adds_continuity_inside_the_physics_schedule():
    """開啟時 total 精確多出 w_phys · w_cont · cont。

    w_phys=0.37 而非 1.0：把 continuity 放在 w_phys 括號外（或乘兩次）
    在 w_phys=1.0 下與正確實作同值，只有非 1 的 w_phys 分得出來。
    """
    off_fn, p, args = _fixture(cont_gradnorm=False)
    on_fn, p2, args2 = _fixture(cont_gradnorm=True)
    total_off, _ = _run(off_fn, p, args, TW3)
    total_on, aux = _run(on_fn, p2, args2, TW4)
    cont = aux[3]
    expected = total_off + W_PHYS * W_CONT * cont
    assert jnp.allclose(total_on, expected, atol=1e-6), (
        float(total_on), float(expected), float(cont))


def test_continuity_uses_the_causal_weighted_quantity():
    """causal 開啟時 continuity 必須用 cont_eff，與 mom_*_eff 同源。

    這是實作真正踩過的坑：原本用 raw `cont`，而同一括號內的 momentum 用
    `mom_*_eff`。eps=0 時 causal 權重全為 1（cont_eff == cont），eps>0 時兩者
    分岔——所以兩個 eps 一起斷言才鎖得住「用的是加權後的量」。
    """
    off_fn, p, args = _fixture(cont_gradnorm=False, use_causal=True)
    on_fn, p2, args2 = _fixture(cont_gradnorm=True, use_causal=True)

    # eps=0：causal 權重全 1，cont_eff 退化成 cont，差值必須精確等於 w·cont
    d0 = _run(on_fn, p2, args2, TW4, causal_eps=0.0)[0] - \
        _run(off_fn, p, args, TW3, causal_eps=0.0)[0]
    cont0 = _run(on_fn, p2, args2, TW4, causal_eps=0.0)[1][3]
    assert jnp.allclose(d0, W_PHYS * W_CONT * cont0, atol=1e-6)

    # eps>0：cont_eff 已被加權，差值不可再等於 raw cont 乘上權重
    eps = 100.0
    total_on, aux_on = _run(on_fn, p2, args2, TW4, causal_eps=eps)
    total_off, _ = _run(off_fn, p, args, TW3, causal_eps=eps)
    d = total_on - total_off
    raw = W_PHYS * W_CONT * aux_on[3]
    assert jnp.isfinite(d)
    assert not jnp.allclose(d, raw, rtol=1e-3), (
        "continuity 用了未經 causal 加權的 raw cont", float(d), float(raw))


def test_grad_norm_probe_gains_a_fourth_task():
    """開啟時 grad-norm 探針多一項，且與 ns_u/ns_v 同源。"""
    model = LiquidOperator(**minimal_kwargs(3))
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1,
                        sensor_time, jax.random.uniform(jax.random.PRNGKey(9), (N, 2)),
                        jnp.zeros((N,)))
    ns_fn, _ = make_ns_residual_fn(model)
    re_batch = assembly.ReBatch(
        sensor_vals=sensor_vals, sensor_pos=sensor_pos, sensor_time=sensor_time,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0),
    )
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))

    n3 = assembly._build_grad_norm_fn(model, ns_fn, cont_gradnorm=False)(
        params, cx, cy, ct, re_batch, 0.0)
    n4 = assembly._build_grad_norm_fn(model, ns_fn, cont_gradnorm=True)(
        params, cx, cy, ct, re_batch, 0.0)
    assert n3.shape == (3,) and n4.shape == (4,)
    assert jnp.allclose(n3, n4[:3], atol=1e-6)   # 前三項不因加第四 task 而改變
    assert jnp.isfinite(n4[3]) and float(n4[3]) > 0.0


def test_cli_flag_turns_al_off():
    """`--cont_gradnorm` 與 AL 互斥：旗標必須把 use_al 壓成 False。"""
    from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs

    base = ["--config", "configs/exp_contgn_b3.toml"]
    # 沒帶旗標就是另一個臂：兩個鍵都不在共用 TOML schema 裡，config 自己說不出來
    assert resolve_inputs(base).config.loss.cont_gradnorm is False
    r = resolve_inputs(base + ["--cont_gradnorm"])
    assert r.config.loss.cont_gradnorm is True
    assert r.config.loss.use_al is False


def test_guard_rejects_cont_gradnorm_without_gradnorm():
    """沒有 GradNorm 時 continuity 會拿到固定權重 1.0，必須擋下來。"""
    from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy

    policy = KolmogorovPolicy()

    class _Cfg:
        class loss:
            cont_gradnorm = True
            use_gradnorm = False
            data_weight = 1.0        # guard 現在也讀這個（見 test_loss_wiring_guards）

        class refinement:
            optimizer = "none"

    with pytest.raises(ValueError, match="use_gradnorm"):
        policy.guard(None, _Cfg)


def test_gradnorm_accepts_four_task_layout():
    """四 task 佈局要能用顯式 task_names 建起來，不動 losses.py 的預設表。"""
    st = gradnorm_init([1.0, 0.057, 0.057, 0.057],
                       task_names=("data", "ns_u", "ns_v", "cont"))
    assert st.task_names == ("data", "ns_u", "ns_v", "cont")
    w = gradnorm_weights(st)
    assert w.shape == (4,)
    assert jnp.allclose(w, jnp.array([1.0, 0.057, 0.057, 0.057]), atol=1e-6)


def test_four_task_layout_rejects_length_mismatch():
    with pytest.raises(ValueError):
        gradnorm_init([1.0, 0.057, 0.057],
                      task_names=("data", "ns_u", "ns_v", "cont"))
