"""JAX port of pi-lnn losses.py: GradNorm dynamic 3-task weights (data/ns_u/ns_v) + Augmented Lagrangian for continuity.

對齊 pi_con/losses.py:
  - `GradNormWeights` → `GradNormState` (NamedTuple) + `gradnorm_init/step/get_weights`
  - `AugmentedLagrangianMultiplier` → `ALState` + `al_init/loss_term/update`

設計：
  1) 所有 state 是 immutable NamedTuple (functional)；訓練 loop 顯式管理 state pytree。
  2) GradNorm update 不靠 autograd，靠直接公式 + EMA assignment（與 pi-lnn 一致）。
  3) AL dual update 不靠 autograd，靠 λ ← clip(λ + ρ·EMA(C), λ_min, λ_clip)。
  4) `gradnorm_step` 接受 per-task gradient norm；caller 負責對 ref_params 算 4 個獨立 jax.grad。
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# GradNorm
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TASK_LAYOUTS = {
    3: ("data", "ns_u", "ns_v"),
    4: ("data", "ns_u", "ns_v", "bc"),
}


class GradNormState(NamedTuple):
    """GradNorm task weights state（functional, immutable）。

    log_weights: [n_tasks] jnp.ndarray，task 對應名稱見 task_names。
    task_names:  tuple of str (static metadata，不參與 grad/jit)
    """
    log_weights: jnp.ndarray
    task_names: tuple


def gradnorm_init(
    init_weights: list | tuple,
    task_names: list | tuple | None = None,
) -> GradNormState:
    """Initialize GradNorm state.

    Args:
        init_weights: e.g. [1.0, 0.01, 0.01, 0.01] (data=1, physics=0.01)
        task_names:   顯式提供 → 驗長度；否則依長度推預設 layout
    """
    w = jnp.asarray(init_weights, dtype=jnp.float32)
    if task_names is None:
        n = len(init_weights)
        if n not in _DEFAULT_TASK_LAYOUTS:
            raise ValueError(
                f"未指定 task_names 時 init_weights 長度必須 ∈ {sorted(_DEFAULT_TASK_LAYOUTS)}，收到 {n}"
            )
        task_names = _DEFAULT_TASK_LAYOUTS[n]
    else:
        if len(task_names) != len(init_weights):
            raise ValueError(
                f"task_names 長度 ({len(task_names)}) 與 init_weights ({len(init_weights)}) 不符"
            )
        task_names = tuple(task_names)
    return GradNormState(log_weights=jnp.log(w), task_names=task_names)


def gradnorm_weights(state: GradNormState) -> jnp.ndarray:
    """Return current task weights (linear scale)."""
    return jnp.exp(state.log_weights)


def gradnorm_step(
    state: GradNormState,
    task_grad_norms: jnp.ndarray,   # [n_tasks]: ||∇_W L_i|| each
    ema_momentum: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 0.0,
) -> GradNormState:
    """單步 GradNorm 權重更新（直接反比公式 + EMA）。

    公式 (對齊 pi-lnn _gradnorm_step):
      G_i      = ||∇_W L_i|| (caller 算)
      mean_G   = mean(G_i)
      w_i_raw  = mean_G / (G_i + 1e-5 * mean_G)
      w_i_norm = w_i_raw / w_i_raw[0]    # data 為基準
      w_new    = momentum * w_old + (1 - momentum) * w_i_norm
      sanity   = clamp(w_new[1:], max=max_weight); clamp_min(w_new, min_weight)

    Args:
        task_grad_norms: 各 task 對 reference params 的 L2 gradient norm
        ema_momentum:    EMA 權重保留比例（0.5 → 每次 update 走 50% 新值）
        min_weight:      所有 task weight 下限（防 GradNorm self-catalyzing pathology）
        max_weight:      physics tasks (index ≥ 1) 上限（data 不受限）；0 = 不 cap
    """
    ws_old = jnp.exp(state.log_weights)
    G = task_grad_norms
    mean_G = G.mean()
    w_raw = mean_G / (G + 1e-5 * mean_G)
    w_computed = w_raw / jnp.maximum(w_raw[0], 1e-8)
    w_new = ema_momentum * ws_old + (1.0 - ema_momentum) * w_computed
    if min_weight > 0.0:
        w_new = jnp.maximum(w_new, min_weight)
    if max_weight > 0.0:
        # 對 physics tasks (index ≥ 1) cap；data (index 0) 永保 = 1。
        # cap 不低於 min_weight，避免矛盾 config (max<min) 讓 max 壓破 min 下限。
        cap = max(float(max_weight), float(min_weight))
        w_phys = jnp.minimum(w_new[1:], cap)
        w_new = jnp.concatenate([w_new[:1], w_phys])
    return state._replace(log_weights=jnp.log(jnp.maximum(w_new, 1e-8)))


# ─────────────────────────────────────────────────────────────────────────────
# Augmented Lagrangian (for continuity)
# ─────────────────────────────────────────────────────────────────────────────

class ALState(NamedTuple):
    """AL multiplier state（functional, immutable）。

    對 continuity 純 scalar 條件 C = mean((∇·u)²) ≥ 0：
      loss_term = λ·C + (ρ/2)·C²
      dual update: λ ← clip(λ + ρ·EMA(C), λ_min, λ_clip)
    """
    lambda_: jnp.ndarray       # scalar
    ema_C: jnp.ndarray          # scalar
    initialized: jnp.ndarray    # bool scalar
    rho: float                  # static
    lambda_clip: float          # static
    ema_momentum: float         # static


def al_init(
    init_lambda: float = 0.0,
    rho: float = 1.0,
    lambda_clip: float = 10.0,
    ema_momentum: float = 0.5,
) -> ALState:
    """Initialize AL state.

    Args:
        init_lambda:    initial λ (0.0 = 純 ρ·C²/2 penalty 暖機)
        rho:            quadratic penalty coefficient (固定 hyperparameter, no schedule)
        lambda_clip:    λ upper bound (prevents runaway)
        ema_momentum:   EMA for C 平滑（0 = no smoothing; >0 = mom·old + (1-mom)·new）
    """
    return ALState(
        lambda_=jnp.asarray(float(init_lambda), dtype=jnp.float32),
        ema_C=jnp.asarray(0.0, dtype=jnp.float32),
        initialized=jnp.asarray(False),
        rho=float(rho),
        lambda_clip=float(lambda_clip),
        ema_momentum=float(ema_momentum),
    )


def al_constraint_value(
    cont_mse: jnp.ndarray,
    cont_signed: jnp.ndarray | None = None,
    mode: str = "mse",
) -> jnp.ndarray:
    """Return scalar AL constraint value for continuity。

    mode="mse" preserves the historical PI-CON behavior: C = mean(div²) >= 0.
    mode="signed_mean" uses an aggregated signed equality residual:
    C = mean(div), closer to classical equality-constraint ALM.
    """
    if mode == "mse":
        return cont_mse.reshape(())
    if mode == "signed_mean":
        if cont_signed is None:
            raise ValueError("mode='signed_mean' 需要逐點 signed continuity residual")
        return jnp.mean(cont_signed).reshape(())
    raise ValueError(f"unknown AL constraint mode: {mode!r}")


def al_update(state: ALState, C_batch: jnp.ndarray, lambda_min: float = 0.0) -> ALState:
    """Dual update: λ ← clip(λ + ρ·EMA(C), lambda_min, lambda_clip)。

    C_batch: scalar 或 1-element tensor (mean(div²) for current batch)
    lambda_min: equality-style signed constraints 可設為 -lambda_clip；預設 0 保留
                historical non-negative MSE multiplier behavior。
    """
    c_val = C_batch.reshape(()).astype(jnp.float32)
    # initialize ema_C cold-start：first call 用 c_val 取代 EMA
    use_ema = state.initialized & (state.ema_momentum > 0)
    ema_C_new = jnp.where(
        use_ema,
        state.ema_momentum * state.ema_C + (1.0 - state.ema_momentum) * c_val,
        c_val,
    )
    new_lambda = jnp.clip(
        state.lambda_ + state.rho * ema_C_new,
        float(lambda_min), state.lambda_clip,
    )
    return state._replace(
        lambda_=new_lambda,
        ema_C=ema_C_new,
        initialized=jnp.asarray(True),
    )

