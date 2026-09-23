"""Unit tests for GradNorm + AL losses module.

對齊 pi-lnn losses.py 的數學語意（不需要 model；用 toy scalar function）。
"""
from __future__ import annotations

import jax.numpy as jnp

from pi_lnn_jax.losses import (
    gradnorm_init, gradnorm_weights, gradnorm_step,
    al_constraint_value, al_init, al_update,
)


def test_gradnorm_step_balances_weights():
    """如果某 task 的 G_i 比另一個大 → 反比公式 → 該 task weight 應變小。"""
    s = gradnorm_init([1.0, 1.0, 1.0])  # 起始所有 task 同權重
    # G = [1.0, 10.0, 1.0] → mean_G = 4.0
    # w_raw = mean_G / G = [4.0, 0.4, 4.0]
    # w_norm = w_raw / w_raw[0] = [1.0, 0.1, 1.0]
    # EMA 0.5: w_new = 0.5·1 + 0.5·norm = [1.0, 0.55, 1.0]
    G = jnp.array([1.0, 10.0, 1.0])
    s2 = gradnorm_step(s, G, ema_momentum=0.5, min_weight=0.0)
    w_new = gradnorm_weights(s2)
    # task 1 (ns_u) 應被 deprioritize (weight ↓)
    assert float(w_new[1]) < float(w_new[0]), f"task 1 weight {w_new[1]} should < task 0 {w_new[0]}"
    assert float(w_new[1]) < 0.7  # 從 1.0 顯著下降
    # data weight (index 0) 應接近 1（基準）
    assert abs(float(w_new[0]) - 1.0) < 0.05
    print(f"✓ gradnorm_step_balances_weights: w_new = {[f'{float(x):.3f}' for x in w_new]}")


def test_gradnorm_min_weight_floor():
    """min_weight 應對 catastrophic deprioritize 提供 floor。"""
    s = gradnorm_init([1.0, 1.0, 1.0])
    # G_1 巨大 → w_raw_1 → 0 → 若無 floor 會被 deprioritize 到 ~0
    G = jnp.array([1.0, 1e6, 1.0])
    s2 = gradnorm_step(s, G, ema_momentum=0.0, min_weight=0.1)  # 無 EMA: 直接套 computed
    w_new = gradnorm_weights(s2)
    assert float(w_new[1]) >= 0.1 - 1e-5, f"task 1 weight {w_new[1]} 應 ≥ min_weight 0.1"
    print(f"✓ gradnorm_min_weight_floor: w_new[1]={float(w_new[1]):.4f}")


def test_gradnorm_max_weight_cap_only_physics():
    """max_weight 只對 physics tasks (idx ≥ 1) cap，data (idx 0) 不受限。"""
    s = gradnorm_init([1.0, 0.01, 0.01])
    # 構造一個會讓 ns_u 想升到 5.0 的場景
    G = jnp.array([10.0, 0.1, 10.0])  # G_1 小 → w_raw_1 大 → 想 boost ns_u
    s2 = gradnorm_step(s, G, ema_momentum=0.0, min_weight=0.0, max_weight=0.5)
    w_new = gradnorm_weights(s2)
    # data (idx 0) 應 = 1（基準 normalize）
    assert abs(float(w_new[0]) - 1.0) < 0.05
    # ns_u 應被 cap 到 ≤ 0.5
    assert float(w_new[1]) <= 0.5 + 1e-5, f"ns_u {w_new[1]} 應 ≤ max_weight 0.5"
    print(f"✓ gradnorm_max_weight_cap_only_physics: w_new = {[f'{float(x):.3f}' for x in w_new]}")


def test_al_dual_update_monotonic_under_positive_C():
    """C ≥ 0 always；λ 應單調非減直到 hit clip。"""
    s = al_init(init_lambda=0.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.0)
    history = [float(s.lambda_)]
    for c in [0.1, 0.2, 0.3, 1.0, 2.0]:
        s = al_update(s, jnp.array(c))
        history.append(float(s.lambda_))
    # 確認單調非減
    for i in range(1, len(history)):
        assert history[i] >= history[i - 1] - 1e-6, f"step {i}: λ {history[i]} < {history[i-1]}"
    # 確認 clip 生效
    s = al_update(s, jnp.array(50.0))  # 大 C
    assert float(s.lambda_) <= 10.0 + 1e-5, f"λ {s.lambda_} should be clipped to ≤ 10"
    print(f"✓ al_dual_update_monotonic: history = {history}, final after clip = {float(s.lambda_):.3f}")


def test_al_ema_smoothing():
    """ema_momentum > 0 → 使 λ 對抖動較不敏感。"""
    s = al_init(init_lambda=0.0, rho=1.0, lambda_clip=10.0, ema_momentum=0.7)
    s = al_update(s, jnp.array(1.0))     # first: ema_C = 1.0, λ = 1.0
    lam_1 = float(s.lambda_)
    s = al_update(s, jnp.array(0.1))     # ema_C = 0.7·1.0 + 0.3·0.1 = 0.73, λ = 1.0 + 0.73 = 1.73
    lam_2 = float(s.lambda_)
    assert abs(lam_1 - 1.0) < 1e-5, f"λ after first update = {lam_1}, expect 1.0"
    assert abs(lam_2 - 1.73) < 1e-5, f"λ after second = {lam_2}, expect 1.73"
    print(f"✓ al_ema_smoothing: λ_1={lam_1:.3f}, λ_2={lam_2:.3f}")


def test_al_constraint_value_modes_distinguish_mse_and_signed_mean():
    """AL constraint mode：舊路徑用 cont MSE；文獻型聚合用 signed mean residual。"""
    cont_mse = jnp.array(0.25)
    cont_signed = jnp.array([-0.4, -0.2, 0.1, 0.1])

    c_mse = al_constraint_value(cont_mse, cont_signed, mode="mse")
    c_signed = al_constraint_value(cont_mse, cont_signed, mode="signed_mean")

    assert abs(float(c_mse) - 0.25) < 1e-6
    assert abs(float(c_signed) - (-0.1)) < 1e-6
    print("✓ al_constraint_value_modes_distinguish_mse_and_signed_mean")


def test_al_update_allows_negative_multiplier_for_signed_equality():
    """signed equality AL 需要 λ 可正可負；舊 MSE mode 仍可用預設 λ>=0。"""
    s = al_init(init_lambda=0.0, rho=2.0, lambda_clip=10.0, ema_momentum=0.0)
    s_signed = al_update(s, jnp.array(-0.25), lambda_min=-10.0)
    s_mse = al_update(s, jnp.array(-0.25))

    assert abs(float(s_signed.lambda_) - (-0.5)) < 1e-6
    assert abs(float(s_mse.lambda_) - 0.0) < 1e-6
    print("✓ al_update_allows_negative_multiplier_for_signed_equality")


