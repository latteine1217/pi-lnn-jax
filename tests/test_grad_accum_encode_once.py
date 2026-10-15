"""grad accumulation 的 encode-once（TD-34）與「每塊各自 encode」數值等價。

accumulate_value_and_grad 把 encode 提出分塊：vjp 取 encoder 反向，各塊 ∂L/∂h 平均後
只反向一次。數學等價，只差加總順序：前向（total 與 aux）逐位相同，梯度只差捨入
（CPU 實測 ‖Δg‖/‖g‖ ≈ 8e-8–1e-7，低於 float32 機器精度 1.2e-7）。
梯度用全域量判，不逐葉用自身量級：attention 的 key bias 梯度在數學上恆為 0
（softmax 對 logit 平移不變），其值本身就是捨入雜訊，逐葉相對誤差沒有意義。
逐一覆蓋會改變 encoder 輸入的分支（CRP、sensor dropout、autoreg 延伸），以及沒有
encode 可拆、必須原樣走舊寫法的 arch。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator, StandardPINNOperator
from pi_lnn_jax.optimizers import accumulate_grads
from pi_lnn_jax.physics import make_ns_residual_fn, make_ns_residual_fn_baseline
from pi_lnn_jax.pipeline.kolmogorov import assembly

K, T, C = 5, 4, 3
N_COLLO, N_SQ, M = 8, 8, 4


def _re_batch(re_norm, seed):
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(seed), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(seed + 1), (T, K, C))
    return assembly.ReBatch(
        sensor_vals=sensor_vals, sensor_pos=sensor_pos, sensor_time=sensor_time,
        re_norm=jnp.asarray(re_norm), nu=jnp.asarray(1e-3),
        u_mean=jnp.asarray(0.1), u_std=jnp.asarray(1.5),
        v_mean=jnp.asarray(-0.1), v_std=jnp.asarray(0.8),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0))


def _setup(case):
    crp = case == "crp"
    autoreg = np.array([1.25, 1.5], np.float32) if case == "autoreg" else None
    rb = _re_batch(0.3, 0)
    if case == "pinn":
        model = StandardPINNOperator(num_layers=2, hidden_dim=16, fourier_embed_dim=16, d_time=4)
    else:
        model = LiquidOperator(**minimal_kwargs(C))
    params = model.init(jax.random.PRNGKey(3), rb.sensor_vals, rb.sensor_pos, 0.3,
                        rb.sensor_time, jnp.zeros((4, 2)), jnp.zeros((4,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=False, use_al=True, al_rho=1.0, w_poisson=0.0,
        T_total=1.0, t_early_weight=10.0, t_early_threshold=0.2,
        use_continuous_re_physics=crp, continuous_re_physics_weight=0.5,
        autoreg_pseudo_times=autoreg, autoreg_rounds=1,
        ns_fn_baseline=make_ns_residual_fn_baseline(model) if case == "pinn" else None)
    encode_fn = assembly._build_encode_fn(
        model, use_continuous_re_physics=crp, autoreg_pseudo_times=autoreg, autoreg_rounds=1)
    k = jax.random.split(jax.random.PRNGKey(7), 4)
    chunks = (jax.random.uniform(k[0], (N_COLLO,)).reshape(M, -1),
              jax.random.uniform(k[1], (N_COLLO,)).reshape(M, -1),
              jax.random.uniform(k[2], (N_COLLO,)).reshape(M, -1),
              jax.random.choice(k[3], T * K, (N_SQ,), replace=False).reshape(M, -1))
    mask = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0]) if case == "dropout" else None
    rest = {"task_w": jnp.array([1.0, 0.3, 0.2]), "al_lambda": jnp.asarray(0.4),
            "w_phys_now": jnp.asarray(0.7), "re_batch": rb, "causal_eps": 0.0,
            "re_batch_crp": _re_batch(0.6, 10) if crp else None,
            "sensor_input_mask": mask, "sensor_cut_idx": None}
    return loss_fn, encode_fn, params, chunks, rest


def _old(loss_fn, params, chunks, r):
    """TD-34 之前 step_fn 的寫法：每塊呼叫完整 loss_fn（各自 encode）。"""
    def _vg(p, cx, cy, ct, sidx):
        return jax.value_and_grad(loss_fn, has_aux=True)(
            p, cx, cy, ct, r["task_w"], r["al_lambda"], r["w_phys_now"], r["re_batch"],
            r["causal_eps"], sidx, r["re_batch_crp"], r["sensor_input_mask"],
            r["sensor_cut_idx"])
    return accumulate_grads(_vg, params, chunks)


def _new(loss_fn, encode_fn, params, chunks, r):
    return assembly.accumulate_value_and_grad(
        loss_fn, encode_fn, params, chunks, r["task_w"], r["al_lambda"], r["w_phys_now"],
        r["re_batch"], r["causal_eps"], r["re_batch_crp"], r["sensor_input_mask"],
        r["sensor_cut_idx"])


@pytest.mark.parametrize("case", ["main", "crp", "dropout", "autoreg"])
def test_encode_once_matches_per_chunk_encode(case):
    loss_fn, encode_fn, params, chunks, r = _setup(case)
    assert encode_fn is not None
    g_old, t_old, a_old = jax.jit(lambda p: _old(loss_fn, p, chunks, r))(params)
    g_new, t_new, a_new = jax.jit(lambda p: _new(loss_fn, encode_fn, p, chunks, r))(params)
    assert jnp.array_equal(t_new, t_old)
    for x, y in zip(jax.tree_util.tree_leaves(a_new), jax.tree_util.tree_leaves(a_old)):
        assert jnp.array_equal(x, y)
    L_new, L_old = jax.tree_util.tree_leaves(g_new), jax.tree_util.tree_leaves(g_old)
    norm = float(jnp.sqrt(sum(jnp.sum(y ** 2) for y in L_old)))
    d_norm = float(jnp.sqrt(sum(jnp.sum((x - y) ** 2) for x, y in zip(L_new, L_old))))
    g_max = max(float(jnp.max(jnp.abs(y))) for y in L_old)
    d_max = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(L_new, L_old))
    assert d_norm <= 1e-6 * norm, f"‖Δg‖/‖g‖={d_norm / norm:.2e}"
    assert d_max <= 1e-6 * g_max, f"max|Δg|/max|g|={d_max / g_max:.2e}"


def test_arch_without_encode_takes_old_path_unchanged():
    loss_fn, encode_fn, params, chunks, r = _setup("pinn")
    assert encode_fn is None
    g_old, t_old, _ = jax.jit(lambda p: _old(loss_fn, p, chunks, r))(params)
    g_new, t_new, _ = jax.jit(lambda p: _new(loss_fn, None, p, chunks, r))(params)
    assert jnp.array_equal(t_new, t_old)
    for x, y in zip(jax.tree_util.tree_leaves(g_new), jax.tree_util.tree_leaves(g_old)):
        assert jnp.array_equal(x, y)


def test_loss_fn_with_precomputed_h_equals_inline_encode():
    """h_pre 只是把同一次 encode 搬到外面：前向 loss 必須逐位相同。"""
    loss_fn, encode_fn, params, chunks, r = _setup("crp")
    args = (params, chunks[0][0], chunks[1][0], chunks[2][0], r["task_w"], r["al_lambda"],
            r["w_phys_now"], r["re_batch"], r["causal_eps"], chunks[3][0], r["re_batch_crp"],
            r["sensor_input_mask"], r["sensor_cut_idx"])
    # h 必須在同一個 jit 內算：在 jit 外 eager 算會因融合不同而差 ~1e-6，比的就不是本題
    t_inline, a_inline = jax.jit(lambda: loss_fn(*args))()
    t_pre, a_pre = jax.jit(lambda: loss_fn(*args, h_pre=encode_fn(
        params, r["re_batch"], r["re_batch_crp"], r["sensor_input_mask"])))()
    assert jnp.array_equal(t_inline, t_pre)
    for x, y in zip(jax.tree_util.tree_leaves(a_inline), jax.tree_util.tree_leaves(a_pre)):
        assert jnp.array_equal(x, y)
