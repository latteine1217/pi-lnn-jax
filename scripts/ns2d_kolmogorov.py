#!/usr/bin/env python3
"""ns2d_kolmogorov.py — 2D Kolmogorov 渦度形式 pseudo-spectral 前向求解器（Ticket 04）。

What:
    渦度形式 ∂ω/∂t + u·∇ω = ν∇²ω + f_ω，週期方域 [0,1)²，integrating-factor RK4
    （黏性線性項精確處理，穩健對付高波數剛性）。全 jax，可微（供 4D-Var adjoint）。
    convention 對齊 pi-lnn physics.py（f_x=A·sin(2π k_f y) → f_ω=-2π k_f A cos(2π k_f y)）
    與 metric_artifact 的整數波數（k=2π·fftfreq(N,d=1/N)）。

Why:
    Ticket 04 的 4D-Var oracle 需要一個 differentiable NS 前向 + adjoint。本 repo 無
    jax_cfd、physics.py 只有 residual 無 integrator。本檔補這個前向求解器，**唯一可用前提
    是它能重現 DNS**（tests/test_ns2d_kolmogorov.py 驗證：從 DNS 渦度前推一個 frame 對回
    DNS 下一 frame）。未驗證的求解器不得當 oracle（同 Ticket 03 gappy leakage 的教訓）。

    ⚠️ 完整 4D-Var assimilation（長優化迴圈）屬 CLAUDE.md 的「長時間診斷」，走 lab-server；
    本檔只做求解器 + adjoint gradient 的本機 syntax + unit test。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def enable_x64() -> None:
    """開 float64。**入口點才呼叫，不在 module import 時執行。**

    Why：本模組的積分器需要 fp64，但 `jax.config.update` 是**全域**設定——
    在 module 層執行會讓任何 import 它的人（包含 pytest）整個 session 被切到 x64。
    實測後果：`test_pipeline_replay`（bit-identical 黃金測試）與 `test_rwf` 在
    同一次 `pytest tests/` 中失敗，單獨跑卻通過。`tests/test_autodiff.py` 早已
    用 save/restore fixture 記錄過這個坑，但本模組仍在 import 時洩漏。

    呼叫者：本檔 `__main__`、`diag_chaos_budget.main()`；測試用 autouse fixture。
    """
    jax.config.update("jax_enable_x64", True)


def wavenumbers(N: int):
    """整數波數 grid（domain L=1，k=2π·fftfreq(N,d=1/N)）。回 (kx, ky, k2, k2_inv)。"""
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)  # [0,1,..,N/2,..,-1]·2π
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2_inv = jnp.where(k2 == 0, 0.0, 1.0 / jnp.where(k2 == 0, 1.0, k2))  # DC → 0（零均值 ψ）
    return kx, ky, k2, k2_inv


def dealias_mask(N: int):
    """2/3-rule dealiasing mask（保 |k_i| ≤ (2/3)(N/2) 沿各軸）。"""
    idx = jnp.fft.fftfreq(N, d=1.0 / N)  # 整數 index
    cut = (N // 2) * (2.0 / 3.0)
    m1 = jnp.abs(idx) <= cut
    return jnp.outer(m1, m1)


def vorticity_to_velocity(omega_hat, kx, ky, k2_inv):
    """ω_hat → (u,v) 實場。ψ_hat=ω_hat/k²，u=ψ_y=i·ky·ψ，v=-ψ_x=-i·kx·ψ。"""
    psi_hat = omega_hat * k2_inv
    u = jnp.fft.ifft2(1j * ky * psi_hat).real
    v = jnp.fft.ifft2(-1j * kx * psi_hat).real
    return u, v


def _nonlinear_and_forcing(omega_hat, kx, ky, k2_inv, mask, f_omega_hat):
    """譜空間 RHS 的非線性 + forcing 部分（黏性線性項由 integrating factor 處理）。

    N(ω) = -(u·ω_x + v·ω_y)，物理空間算積再 FFT，2/3 dealias。
    """
    u, v = vorticity_to_velocity(omega_hat, kx, ky, k2_inv)
    omega_x = jnp.fft.ifft2(1j * kx * omega_hat).real
    omega_y = jnp.fft.ifft2(1j * ky * omega_hat).real
    adv = u * omega_x + v * omega_y
    adv_hat = jnp.fft.fft2(adv) * mask
    return -adv_hat + f_omega_hat


def integrate(omega0, t_final, dt, nu, k_f, A):
    """從實渦度場 omega0 前推 t_final，integrating-factor RK4。回實渦度場。

    omega0: [N,N] 實場。回同形狀。dt 需整除 t_final（fail-fast，不寬鬆）。
    """
    omega0 = jnp.asarray(omega0, dtype=jnp.float64)
    N = omega0.shape[0]
    if omega0.shape != (N, N):
        raise ValueError(f"需方域 [N,N] 渦度場，得 {omega0.shape}")
    n_steps = round(t_final / dt)
    if abs(n_steps * dt - t_final) > 1e-12:
        raise ValueError(f"dt={dt} 無法整除 t_final={t_final}（不寬鬆對齊）")
    kx, ky, k2, k2_inv = wavenumbers(N)
    mask = dealias_mask(N)
    # forcing f_x = A sin(2π k_f y) → f_ω = -2π k_f A cos(2π k_f y)
    g = jnp.arange(N) / N
    _, Y = jnp.meshgrid(g, g, indexing="ij")
    f_omega = -2.0 * jnp.pi * k_f * A * jnp.cos(2.0 * jnp.pi * k_f * Y)
    f_omega_hat = jnp.fft.fft2(f_omega) * mask

    E = jnp.exp(-nu * k2 * dt)        # integrating factor（半步/全步）
    E2 = jnp.exp(-nu * k2 * dt / 2.0)

    def rhs(oh):
        return _nonlinear_and_forcing(oh, kx, ky, k2_inv, mask, f_omega_hat)

    def step(oh, _):
        # ETD-free integrating-factor RK4（線性項精確、非線性 RK4）
        k1 = rhs(oh)
        k2_ = rhs(E2 * (oh + 0.5 * dt * k1))
        k3 = rhs(E2 * oh + 0.5 * dt * k2_)
        k4 = rhs(E * oh + dt * E2 * k3)
        oh_next = (E * oh + (dt / 6.0) * (E * k1 + 2.0 * E2 * k2_ + 2.0 * E2 * k3 + k4))
        return oh_next, None

    omega_hat = jnp.fft.fft2(omega0)
    omega_hat, _ = jax.lax.scan(step, omega_hat, None, length=n_steps)
    return jnp.fft.ifft2(omega_hat).real


if __name__ == "__main__":
    enable_x64()
    import argparse
    import numpy as np
    from pi_lnn_jax.data import load_dns_from_path  # noqa: E402

    ap = argparse.ArgumentParser(description="驗證 NS 求解器重現 DNS")
    ap.add_argument("--dns", required=True)
    ap.add_argument("--nu", type=float, default=1e-4)
    ap.add_argument("--k_f", type=int, default=2)
    ap.add_argument("--A", type=float, default=0.1)
    ap.add_argument("--frame", type=int, default=40, help="從哪個 DNS frame 起推")
    ap.add_argument("--dt", type=float, default=2.5e-4)
    args = ap.parse_args()

    obj = np.load(args.dns, allow_pickle=True).item()
    omega = np.asarray(obj["omega"]); t = np.asarray(obj["time"])
    dt_frame = float(t[args.frame + 1] - t[args.frame])
    pred = np.asarray(integrate(omega[args.frame], dt_frame, args.dt, args.nu, args.k_f, args.A))
    ref = omega[args.frame + 1]
    rel = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    print(f"frame {args.frame}→{args.frame+1} (Δt={dt_frame:.4f}): 渦度 rel-L2 = {100*rel:.3f}%")
    print("求解器驗證：" + ("PASS（重現 DNS）" if rel < 0.02 else "FAIL（未重現，不可當 oracle）"))
