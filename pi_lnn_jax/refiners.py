"""LBFGS + Levenberg-Marquardt + Matrix-free Gauss-Newton refinement phases。

What: PINN training literature 標準 two-stage pattern:
        Phase 1: Adam / SOAP / ScheduleFree（stochastic, mini-batch）→ ~1e-2 級 loss
        Phase 2: LBFGS / LM / GN（deterministic, full-batch）→ 收尾到 ~1e-6 級

Why:
  - LBFGS: scalar loss minimisation，limited-memory quasi-Newton。
  - LM (Levenberg-Marquardt): nonlinear least-squares via optimistix。
    LM = GN + damping；optimistix 實作記憶體較重（建完整 VJP 圖）。
  - GN matrix-free（本模組主推）：
      移植自 sci-algorithm/src/pinn_cavity/natural_gradient.py:gn_step。
      用 jax.linearize 一次前向後重用線性算子做 matvec；
      O(P) 記憶體，比 optimistix LM 省數倍 VRAM。
      (G + λI) δ = -Jᵀr 用 jax.scipy.sparse.linalg.cg 解。

API:
  refined_params, solution  = lbfgs_refine(params, loss_fn, args, max_steps, ...)
  refined_params, solution  = lm_refine(params, residual_vec_fn, args, max_steps, ...)
  refined_params, final_loss = gn_refine(params, residual_vec_fn,
                                          max_steps, lr, cg_iters, damping, ...)
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optimistix as optx


def lbfgs_refine(
    params,
    loss_fn: Callable,
    args: Any = None,
    max_steps: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    history_length: int = 10,
    verbose: bool = False,
):
    """LBFGS minimisation from `params`.

    Args:
      params: Flax params PyTree.
      loss_fn: scalar loss; signature (y, args) -> scalar (optimistix convention)
      args: 任意 PyTree of context (e.g. dict with sensor_vals, ...)
      max_steps: optimistix 預設 256；POC 用較小值（finetune phase 不該太久）
      rtol/atol: 收斂 tolerance；max_norm(grad) < atol + rtol*max_norm(grad_0) 即停
      history_length: BFGS approximation 保留的歷史步數
      verbose: 列印每步進度（debug 用）

    Returns:
      (refined_params, solution)
        refined_params: PyTree（同 params 結構）
        solution: optx.Solution，含 .value / .result / .stats
    """
    solver = optx.LBFGS(
        rtol=rtol, atol=atol,
        history_length=history_length, verbose=verbose,
    )
    sol = optx.minimise(
        loss_fn, solver, params, args=args,
        max_steps=max_steps, throw=False,
    )
    return sol.value, sol


def lm_refine(
    params,
    residual_vec_fn: Callable,
    args: Any = None,
    max_steps: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    verbose: bool = False,
):
    """Levenberg-Marquardt nonlinear least-squares from `params`.

    residual_vec_fn must return a 1D vector r ∈ R^M;
    LM minimises ||r||² (= scalar loss in PINN setup, where loss = mean(r²) up to constant).

    Args:
      residual_vec_fn: signature (y, args) -> jnp.ndarray (1D)
      其他同 lbfgs_refine

    Returns:
      (refined_params, solution)
    """
    solver = optx.LevenbergMarquardt(
        rtol=rtol, atol=atol, verbose=verbose,
    )
    sol = optx.least_squares(
        residual_vec_fn, solver, params, args=args,
        max_steps=max_steps, throw=False,
    )
    return sol.value, sol


# ─────────────────────────────────────────────────────────────────────────────
# Matrix-free Gauss-Newton（移植自 sci-algorithm/natural_gradient.py:gn_step）
# ─────────────────────────────────────────────────────────────────────────────

def _axpy(a, x, y):
    """a·x + y（pytree）。"""
    return jax.tree_util.tree_map(lambda xi, yi: a * xi + yi, x, y)


def gn_refine(
    params,
    residual_vec_fn: Callable,
    max_steps: int = 100,
    lr: float = 1.0,
    cg_iters: int = 20,
    damping: float = 1e-3,
    log_every: int = 50,
    verbose: bool = True,
):
    """Matrix-free Gauss-Newton refinement（移植自 sci-algorithm natural_gradient.py）。

    Args:
      params:           Flax params PyTree。
      residual_vec_fn:  params → 1D residual vector（同 make_pinn_residual_vector_fn）；
                        簽名可為 (params) 或 (params, args=None)。
      max_steps:        GN outer iterations。
      lr:               step size（純 GN 用 1.0；不穩定時降到 0.1-0.5）。
      cg_iters:         每步 CG inner iterations（預設 20，足夠 ill-conditioned PINN）。
      damping:          LM 正則化 λ（G + λI）；防奇異，初期可大，收斂後縮。
      log_every:        每 N 步印 loss。
      verbose:          是否印 log。

    Returns:
      (refined_params, final_loss): final_loss = mean(r²)

    Why linearize（vs VJP 重算）:
      jax.linearize 一次前向計算後，lin: pytree → R^M 是純線性算子，
      matvec 不需要重跑非線性前向，記憶體 O(P) + O(M)；
      optimistix LM 每 CG step 重建完整 VJP 圖 → VRAM 爆炸。
    """
    # 支援 (params) 或 (params, args=None) 兩種簽名
    import inspect
    sig = inspect.signature(residual_vec_fn)
    if len(sig.parameters) >= 2:
        rfn = lambda p: residual_vec_fn(p, None)
    else:
        rfn = residual_vec_fn

    def _one_gn_step(p):
        # 前向只算一次；lin: pytree_tangent → R^M（Jacobian 線性算子）
        r0, lin = jax.linearize(rfn, p)
        # 轉置算子 J^T: R^M → pytree
        jt = jax.linear_transpose(lin, p)

        def matvec(v):
            """(J^T J + λI) v — CG 用 matvec，不 materialize G。"""
            jv = lin(v)
            jtjv = jt(jv)[0]
            return _axpy(damping, v, jtjv)

        g = jt(r0)[0]                                   # J^T r
        neg_g = jax.tree_util.tree_map(lambda x: -x, g)
        delta, _ = jax.scipy.sparse.linalg.cg(matvec, neg_g, maxiter=cg_iters)
        new_p = _axpy(lr, delta, p)
        return new_p, jnp.mean(r0 ** 2)

    # JIT 單步（避免每步 retrace）
    _one_gn_step_jit = jax.jit(_one_gn_step)

    loss = jnp.inf
    for it in range(max_steps):
        params, loss = _one_gn_step_jit(params)
        if verbose and (it % log_every == 0 or it == max_steps - 1):
            print(f"  GN it={it:4d}  loss={float(loss):.4e}", flush=True)

    return params, float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# PINN-specific helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_pinn_residual_vector_fn(
    model,
    sensor_vals: jnp.ndarray,
    sensor_pos: jnp.ndarray,
    sensor_time: jnp.ndarray,
    re_norm: float,
    norm_stats: dict,
    re_value: float,
    xy_sensor_q: jnp.ndarray,
    t_sensor_q: jnp.ndarray,
    sensor_target: jnp.ndarray,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    ct: jnp.ndarray,
    w_data: float = 1.0,
    w_phys: float = 0.1,
    Lx: float = 1.0,
    Ly: float = 1.0,
):
    """Build residual_vec_fn for LM。

    Vector layout (concat):
      data_residual (sensor pred - target),  shape [T*K*sensor_dim]
      mom_u per-point,                        shape [N_collo]
      mom_v per-point,                        shape [N_collo]
      cont per-point,                         shape [N_collo]

    Weighting: data 乘 sqrt(w_data)，physics 乘 sqrt(w_phys)
    （因為 LM 最小化 ||r||² = sum(r²); 對應原 loss 形式
       L = w_data·||r_data||² + w_phys·||r_phys||²
       須拆成 r_lm = [sqrt(w_data)·r_data, sqrt(w_phys)·r_phys] 才等價）

    Note: collocation points (cx, cy, ct) 固定 — LM 是 deterministic full-batch；
          不該每步 resample（與 SGD 不同）。caller 預先 fix 一批 collocation 點。

    NS 方程不在此重寫，直接用 ``physics.make_ns_residual_fn``（2026-08-03 收斂，
    候選 C）。先前這裡有全 repo 第三份 NS 殘差，帶三個缺陷：

    1. **`p` 沒有反正規化**（`p = out[2]`）。殘差只用 ∇p，`p_mean` 微分後確實
       消失，但 `p_std` 是乘性常數、微分不會消掉它。實測 ∂p/∂x 剛好差 `p_std`
       倍；以 `p_std=0.3` 量到對 `mom_u` 的影響達其自身量級的 **134%**。
       更糟的是它讓 `norm_stats` 的 p 統計被**靜默丟棄**——傳與不傳結果完全相同。
    2. **無 anisotropic domain scaling**。`Lx≠Ly` 的域上 residual MSE 實測差 5.7–7.9 倍。
    3. **`norm_stats` 走 closure capture**——`physics.py:74-76` 記載那正是
       multi-Re 訓練 dataset 切換會 silent 失效的舊寫法。

    三者連同「sensor channel 數寫死 2」（uvp target 會被丟棄 p）一併修正。
    此前這條路徑從未執行過（``--refine_optimizer`` 預設 ``none``，且每一支已提交的
    sbatch 都明確傳 ``none``），所以那些缺陷沒有汙染過任何已發表數字。
    """
    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.physics import make_ns_residual_fn

    u_mean = float(norm_stats["u_mean"])
    u_std = float(norm_stats["u_std"])
    v_mean = float(norm_stats["v_mean"])
    v_std = float(norm_stats["v_std"])
    # p 統計與 assembly 同慣例（缺 p 的資料集 → 0.0/1.0，此時反正規化為恆等）。
    p_mean = float(norm_stats.get("p_mean", 0.0))
    p_std = float(norm_stats.get("p_std", 1.0))
    nu = 1.0 / float(re_value)
    sqrt_w_data = jnp.sqrt(w_data)
    sqrt_w_phys = jnp.sqrt(w_phys)

    # NS 方程只有一份：直接用 physics 的，不在此重寫。
    # 那份帶 p 反正規化、anisotropic domain scaling、runtime norm-stats args，
    # 且有 4 個測試檔涵蓋——本函式先前自寫的那份三者皆無。
    ns_residuals, _poisson = make_ns_residual_fn(model)

    def residual_vec_fn(params, args=None):
        # encode once（與 train_full loss_fn 同 pattern）
        h_states = model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                                method=LiquidOperator.encode)
        # ── Data residual ──
        pred = model.apply(params, xy_sensor_q, t_sensor_q, h_states, sensor_time, sensor_pos,
                            method=LiquidOperator.decode_query)
        # channel 數取自 target 而非寫死：先前寫死 2，uvp target 會被 silent 丟棄 p。
        n_ch = sensor_target.shape[-1]
        data_r = (pred[:, :n_ch] - sensor_target).reshape(-1)   # [T*K*n_ch]
        # ── Forcing ──
        A_force, k_f_force = model.apply(params, method=LiquidOperator.get_forcing)
        # ── Per-point PDE residuals (NOT mean)：LM 最小化 ||r||²，要的是逐點 signed 值 ──
        _mu, _mv, _c, mom_u, mom_v, cont = ns_residuals(
            params, h_states, cx, cy, ct, A_force, k_f_force,
            sensor_pos, sensor_time, nu,
            u_mean, u_std, v_mean, v_std, p_mean, p_std,
            Lx=Lx, Ly=Ly, return_per_point=True,
        )

        # Weighted concat (LM minimises sum of squares)
        return jnp.concatenate([
            sqrt_w_data * data_r,
            sqrt_w_phys * mom_u,
            sqrt_w_phys * mom_v,
            sqrt_w_phys * cont,
        ])

    return residual_vec_fn
