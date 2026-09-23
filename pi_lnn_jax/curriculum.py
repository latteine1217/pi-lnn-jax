"""Training curriculum: physics schedule + time marching + RAR (Residual Adaptive Refinement).

對齊 pi_con/physics.py 的 schedule helpers + _rar_update_pool。

Why curriculum：
  - physics_weight 從 0 → final 線性 ramp，讓 data loss 先收斂；
  - physics_points 從 start → end 線性 ramp，避免初期 PDE 噪音；
  - time_marching 訓練初期限制 t_max，先收斂前段；
  - RAR 每 N 步從大 candidate pool 選 top-residual 點，集中學習困難區。

POC 風格：純 functional，所有 state 顯式傳入。
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# Scalar schedules (point count + weight)
# ─────────────────────────────────────────────────────────────────────────────

def physics_weight_at_step(
    step: int,
    final_weight: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """線性 physics weight warmup / ramp。
    對齊 pi_con/physics.py:physics_weight_at_step。

    Schedule:
      step ≤ warmup_steps        : weight = 0 (純 data 階段)
      ramp_steps == 0            : weight = final_weight (即時切換)
      warmup < step < warmup+ramp: weight = final * (step - warmup) / ramp
      step ≥ warmup + ramp       : weight = final_weight
    """
    if step < 1:
        raise ValueError(f"step 必須從 1 開始，收到 {step}")
    if final_weight < 0.0 or warmup_steps < 0 or ramp_steps < 0:
        raise ValueError(
            f"參數必須 ≥ 0；收到 final={final_weight}, warmup={warmup_steps}, ramp={ramp_steps}"
        )
    if final_weight == 0.0 or step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return final_weight
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    return float(final_weight * progress)


# ─────────────────────────────────────────────────────────────────────────────
# Time marching: 限制 collocation t_max 隨步數
# ─────────────────────────────────────────────────────────────────────────────

def time_marching_t_max(
    step: int,
    T_total: float,
    start_frac: float = 0.1,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
) -> float:
    """收緊 collocation 時間範圍：t ∈ [0, t_max]，t_max 從 start_frac·T_total 漸增到 T_total。

    Args:
        T_total:     total simulation time
        start_frac:  初始覆蓋比例（0 < start_frac ≤ 1.0）
        warmup_steps: 開始 ramp 前固定 start_frac
        ramp_steps:  從 start_frac → 1.0 的線性 ramp 步數；0 = 立即用 1.0

    Why: 訓練初期讓 model 先學前段 t≈0 區域，再逐步擴展，
         避免 chaotic 後段 t 早期收斂困難。
    """
    if not (0.0 < start_frac <= 1.0):
        raise ValueError(f"start_frac 必須在 (0, 1]，收到 {start_frac}")
    if step <= warmup_steps:
        return T_total * start_frac
    if ramp_steps <= 0:
        return T_total
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    frac = start_frac + (1.0 - start_frac) * progress
    return T_total * frac


# ─────────────────────────────────────────────────────────────────────────────
# RAR (Residual Adaptive Refinement)
# ─────────────────────────────────────────────────────────────────────────────

class RARState(NamedTuple):
    """RAR pool 與 rng state。"""
    rng_key: jax.Array


def rar_init(seed: int = 0) -> RARState:
    return RARState(rng_key=jax.random.PRNGKey(seed))


def rar_sample(
    state: RARState,
    residual_fn: Callable,   # (params, xs, ys, ts) -> [N] per-point residual L2
    params,
    n_select: int,
    pool_size: int = 512,
    t_min: float = 0.0,
    t_max: float = 5.0,
    exploration_ratio: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, RARState]:
    """RAR：從大 pool 選 top-residual 點 + 少數 exploration random 點。

    對齊 pi_con/physics.py:_rar_update_pool（簡化版：純 jax，無 first/second order 區分）。

    Steps:
      1. 採 pool_size 個 random collocation
      2. residual_fn 給出每點 |residual| (caller 提供，POC 用 approximation：
         ‖(u_t + u·u_x + v·u_y + p_x)² + (v_t + ...)² + cont²‖)
      3. top_k by residual magnitude（n_top = round(n_select * (1 - exploration_ratio))）
      4. 補 random (n_rand = n_select - n_top)，防退化

    Args:
        residual_fn: 已 jit 化的 per-point residual function
                     signature: residual_fn(params, xs, ys, ts) -> [N] scalar per point
        n_select:    要 select 的點數
        pool_size:   candidate pool 大小（建議 8x ~ 32x n_select）
        exploration_ratio: random fallback 比例

    Returns:
        (xs, ys, ts, new_state) where xs/ys/ts each [n_select]
    """
    if not (0.0 <= exploration_ratio <= 1.0):
        raise ValueError(f"exploration_ratio 須 ∈ [0, 1]，收到 {exploration_ratio}")
    n_top = max(1, round(n_select * (1.0 - exploration_ratio)))
    n_rand = n_select - n_top

    # 需要 6 個 subkeys (3 for pool + 3 for random fallback)；加 carry rng → 7
    rng, *subs = jax.random.split(state.rng_key, 7)
    cx_pool = jax.random.uniform(subs[0], (pool_size,), minval=0.0, maxval=1.0)
    cy_pool = jax.random.uniform(subs[1], (pool_size,), minval=0.0, maxval=1.0)
    ct_pool = jax.random.uniform(subs[2], (pool_size,), minval=t_min, maxval=t_max)

    # Per-point residuals (caller 已 vmap + jit)
    res_per_point = residual_fn(params, cx_pool, cy_pool, ct_pool)  # [pool_size]
    # top-k indices by abs value
    _, top_idx = jax.lax.top_k(jnp.abs(res_per_point), n_top)
    xs_top = cx_pool[top_idx]
    ys_top = cy_pool[top_idx]
    ts_top = ct_pool[top_idx]

    if n_rand > 0:
        xs_rand = jax.random.uniform(subs[3], (n_rand,), minval=0.0, maxval=1.0)
        ys_rand = jax.random.uniform(subs[4], (n_rand,), minval=0.0, maxval=1.0)
        ts_rand = jax.random.uniform(subs[5], (n_rand,), minval=t_min, maxval=t_max)
        xs = jnp.concatenate([xs_top, xs_rand])
        ys = jnp.concatenate([ys_top, ys_rand])
        ts = jnp.concatenate([ts_top, ts_rand])
    else:
        xs, ys, ts = xs_top, ys_top, ts_top

    return xs, ys, ts, RARState(rng_key=rng)

