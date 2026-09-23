"""`scripts/diag_chaos_budget.py` 的兩個承重面。

這支腳本要背書的是「混沌下限」，判斷會決定整條外推路線收不收手，所以兩件事必須
在本機驗得動：(1) 速度↔渦度轉換是同一個場（否則 physics-oracle 的起點就錯了）、
(2) 輸出格點的前提檢查真的會擋（不寬鬆對齊）。

不跑求解器長積分（那是 lab-server 的事），只驗這兩件與資料正確性有關的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from jax import config as _jax_config

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diag_chaos_budget import roll_forward, velocity_to_vorticity  # noqa: E402


# x64 只在本檔測試範圍內開，結束還原。**不得改回 module-level 開啟**：
# `jax.config.update` 是全域設定，洩漏出去會讓 test_pipeline_replay（bit-identical
# 黃金測試）與 test_rwf 在完整套件下失敗、單獨跑卻通過（實測）。
@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)

from ns2d_kolmogorov import vorticity_to_velocity, wavenumbers  # noqa: E402


def _solenoidal_field(N=32, seed=0):
    """由流函數造的無散度場：u=ψ_y, v=−ψ_x（與求解器同 convention）。"""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    psi_hat = np.zeros((N, N), dtype=complex)
    k = np.fft.fftfreq(N, d=1.0 / N)
    for kx_i in range(-3, 4):
        for ky_i in range(-3, 4):
            if kx_i == 0 and ky_i == 0:
                continue
            psi_hat[np.argmin(abs(k - kx_i)), np.argmin(abs(k - ky_i))] = (
                rng.normal() + 1j * rng.normal())
    psi = np.fft.ifft2(psi_hat).real
    psi_hat = np.fft.fft2(psi)
    kx, ky = np.meshgrid(2 * np.pi * k, 2 * np.pi * k, indexing="ij")
    u = np.fft.ifft2(1j * ky * psi_hat).real
    v = np.fft.ifft2(-1j * kx * psi_hat).real
    return jnp.asarray(u), jnp.asarray(v)


def test_velocity_vorticity_round_trip():
    """無散度場走一圈 (u,v)→ω→(u,v) 必須回到自己。"""
    import jax.numpy as jnp

    u, v = _solenoidal_field()
    omega = velocity_to_vorticity(u, v)
    kx, ky, _, k2_inv = wavenumbers(u.shape[0])
    u2, v2 = vorticity_to_velocity(jnp.fft.fft2(omega), kx, ky, k2_inv)

    err = float(jnp.linalg.norm(u2 - u) / jnp.linalg.norm(u))
    assert err < 1e-12, f"round-trip 相對誤差 {err}"
    assert float(jnp.linalg.norm(v2 - v) / jnp.linalg.norm(v)) < 1e-12
    print("✓ velocity_vorticity_round_trip")


def test_vorticity_projects_out_divergence():
    """有散度的場轉 ω 後只留 solenoidal 部分——這是腳本 docstring 宣告的行為。"""
    import jax.numpy as jnp

    u, v = _solenoidal_field()
    N = u.shape[0]
    g = np.arange(N) / N
    X, _ = np.meshgrid(g, g, indexing="ij")
    grad = jnp.asarray(np.cos(2 * np.pi * X))          # 純梯度場（無旋）
    omega_clean = velocity_to_vorticity(u, v)
    omega_dirty = velocity_to_vorticity(u + grad, v)

    assert float(jnp.max(jnp.abs(omega_dirty - omega_clean))) < 1e-10, (
        "無旋分量不該改變渦度")
    print("✓ vorticity_projects_out_divergence")


def test_roll_forward_rejects_uneven_grid():
    """輸出格點不等距就 raise，不自行取平均（eval 紀律：不寬鬆對齊）。"""
    u, v = _solenoidal_field()
    omega = velocity_to_vorticity(u, v)
    with pytest.raises(ValueError, match="等距"):
        roll_forward(omega, [0.0, 0.05, 0.2], 2.5e-4, 1e-4, 2.0, 0.1)
    print("✓ roll_forward_rejects_uneven_grid")


def test_roll_forward_tolerates_float32_time_axis():
    """DNS 的時間軸是 float32；量化雜訊不得被當成「不等距」。

    這條是實際踩過的坑：以 1e-12 比對 float32 導出的間距，job 在第 6 秒就掛。
    """
    u, v = _solenoidal_field()
    omega = velocity_to_vorticity(u, v)
    times = np.asarray(np.arange(101) * 0.05 + 5.0, dtype=np.float32).astype(np.float64)
    assert np.ptp(np.diff(times)) > 0, "這個 fixture 必須真的帶量化雜訊，否則沒在測東西"

    u_out, _ = roll_forward(omega, times[:3], 2.5e-4, 1e-4, 2.0, 0.1)
    assert u_out.shape[0] == 3
    print("✓ roll_forward_tolerates_float32_time_axis")


def test_roll_forward_rejects_indivisible_dt():
    """段長不是 dt 的整數倍時 fail-fast，不悄悄湊到最近的步邊界。

    容差只吸收 float32 時間軸的量化雜訊（~1e-7）；真正對不上的 dt 差一整步，擋得住。
    """
    u, v = _solenoidal_field()
    omega = velocity_to_vorticity(u, v)
    with pytest.raises(ValueError, match="不是 dt"):
        roll_forward(omega, [0.0, 0.05], 3.3e-4, 1e-4, 2.0, 0.1)
    print("✓ roll_forward_rejects_indivisible_dt")
