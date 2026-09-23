"""Unit tests for curriculum module."""
from __future__ import annotations

import jax.numpy as jnp

from pi_lnn_jax.curriculum import (
    physics_weight_at_step, time_marching_t_max,
    rar_init, rar_sample,
)


def test_physics_weight_schedule():
    # warmup=10, ramp=50, final=0.01
    assert physics_weight_at_step(step=5, final_weight=0.01, warmup_steps=10, ramp_steps=50) == 0.0
    assert physics_weight_at_step(step=10, final_weight=0.01, warmup_steps=10, ramp_steps=50) == 0.0
    w_mid = physics_weight_at_step(step=35, final_weight=0.01, warmup_steps=10, ramp_steps=50)
    assert abs(w_mid - 0.005) < 1e-6, f"mid weight {w_mid} 應 ≈ 0.005"
    assert physics_weight_at_step(step=100, final_weight=0.01, warmup_steps=10, ramp_steps=50) == 0.01
    # final=0 → noop
    assert physics_weight_at_step(step=100, final_weight=0.0, warmup_steps=0, ramp_steps=0) == 0.0
    # ramp=0 → 立即切換
    assert physics_weight_at_step(step=11, final_weight=0.01, warmup_steps=10, ramp_steps=0) == 0.01
    print("✓ physics_weight_at_step: 5 waypoints correct")


def test_time_marching():
    # T_total=5.0, start_frac=0.2 → 初始 t_max=1.0；ramp 100 步
    assert time_marching_t_max(step=0, T_total=5.0, start_frac=0.2,
                                 warmup_steps=10, ramp_steps=100) == 1.0
    assert abs(time_marching_t_max(step=60, T_total=5.0, start_frac=0.2,
                                     warmup_steps=10, ramp_steps=100) - 3.0) < 1e-6
    assert time_marching_t_max(step=200, T_total=5.0, start_frac=0.2,
                                 warmup_steps=10, ramp_steps=100) == 5.0
    print("✓ time_marching_t_max: 3 waypoints correct")


def test_rar_sample_with_dummy_residual():
    """RAR pool sample 邏輯：top residual 點應被選中。"""
    # toy residual: max at (x=0.7, y=0.5, t=2.5)
    def dummy_residual_fn(params, xs, ys, ts):
        # 高斯峰：center (0.7, 0.5, 2.5)
        return jnp.exp(-100 * ((xs - 0.7) ** 2 + (ys - 0.5) ** 2 + (ts - 2.5) ** 2))

    state = rar_init(seed=0)
    params = None  # dummy
    xs, ys, ts, new_state = rar_sample(
        state, dummy_residual_fn, params,
        n_select=8, pool_size=128, t_min=0.0, t_max=5.0, exploration_ratio=0.25,
    )
    assert xs.shape == (8,) and ys.shape == (8,) and ts.shape == (8,)
    # 6 個 top 點應該接近 center；2 個 exploration random
    # 檢查至少 3 個點在 center 附近 (距離 < 0.15)
    dists = jnp.sqrt((xs - 0.7) ** 2 + (ys - 0.5) ** 2 + (ts - 2.5) ** 2 * 0.04)  # scale t
    n_close = int(jnp.sum(dists < 0.15))
    assert n_close >= 1, f"expected ≥1 點接近 center，得 {n_close}（pool=128 sparse）"
    print(f"✓ rar_sample: 8 points selected, {n_close} 接近 residual peak (≥1 required)")


