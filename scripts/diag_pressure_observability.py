#!/usr/bin/env python3
"""diag_pressure_observability.py — p-in-sensor 的「局部線性化」可觀測性診斷（JAX）。

What:
    velocity-only sensing 是線性的（見 diag_observability.py）。pressure 不是：
    不可壓縮流中 ∇²p = -[(∂_x u)² + 2(∂_y u)(∂_x v) + (∂_y v)²]，p 是速度的二次泛函。
    因此「velocity+pressure 直接 stack rows」不成立；正確物件是 reference state ā 附近的
    locally-linearized sensing operator
        C_{u,p}^lin(ā) = [ C_u ; J_p(ā) ],   J_p(ā) = ∂p_s/∂a |_ā。
    J_p 用 JAX autodiff 取，不手推 Fourier triad。

    forward map:  a → ψ(a) → (u,v) → source = -[(u_x)²+2 u_y v_x+(v_y)²] → p = ∇⁻²source → p_s

Why:
    給 p-in-sensor ablation 一個可信、可複現、可寫進論文的局部可觀測性診斷。
    重點規則（見模組末 INTERPRETATION）：
      - reference 不可用 ā=0（J_p(0)=0，會錯誤得出「pressure 無用」）。
      - 對多個 LES snapshot 的 ā 報分佈，避免 cherry-pick。
      - 報 singular spectrum 而非只報 rank（pressure 可能只增 weakly-observable 方向）。
      - p 不改變 sensor Nyquist ceiling（位置沒變）；只可能注入非局部 Poisson 約束。

    basis / C_u / observability_metrics 全部重用 scripts/diag_observability.py，
    保證 C_u 與 J_p 的 columns 對應同一組 stream-function modal coefficients。

Convention:
    domain [0,1)²，axis_1=x, axis_2=y。stream-function 實基底 (α,β) per half-mode，
    排列與 divfree_observation_operator 完全一致（round-trip test 守住）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.diag_observability import (
    coords_to_xy_indices,
    divfree_half_modes,
    divfree_observation_operator,
    observability_metrics,
)

# NOTE: x64 只在 standalone 執行時開（見 __main__）。不可在 module import 時全域開，
# 否則被測試 import 會洩漏 x64 到整個 pytest session，污染其他測試（如 test_refiners
# 的 optimistix scan/lstsq 在 x64 下 carry dtype 不一致而失敗）。
_TWO_PI = 2.0 * jnp.pi


# ── spectral 基礎設施 ─────────────────────────────────────────────────────────

def _int_freqs(N: int) -> jnp.ndarray:
    """整數波數 [0,1,...,N/2-1,-N/2,...,-1]（fft 排序）。"""
    return jnp.fft.fftfreq(N, d=1.0 / N)


def _half_mode_fft_indices(half_modes: list, N: int):
    """預先算好 half-mode 在 fft grid 上的 (q, -q) 索引（static，不進 autodiff）。"""
    kx = np.array([m[0] for m in half_modes])
    ky = np.array([m[1] for m in half_modes])
    return (kx % N, ky % N), ((-kx) % N, (-ky) % N), kx, ky


def streamfunction_velocity_spectral(a: jnp.ndarray, idx_pos, idx_neg, N: int):
    """a (2H 實係數 α,β) → (u, v) 全場 (N,N)，spectral 合成（JAX-diffable）。

    ψ_hat(q)=(α-iβ)/2、ψ_hat(-q)=(α+iβ)/2 ⇒ 實場與 velocity_field_from_psi 一致。
    u=∂ψ/∂y, v=-∂ψ/∂x ⇒ ∇·u≡0。
    """
    alpha = a[0::2]
    beta = a[1::2]
    psi_q = (alpha - 1j * beta) / 2.0
    psi_negq = (alpha + 1j * beta) / 2.0
    psi_hat = jnp.zeros((N, N), dtype=jnp.complex128)
    psi_hat = psi_hat.at[idx_pos[0], idx_pos[1]].add(psi_q)
    psi_hat = psi_hat.at[idx_neg[0], idx_neg[1]].add(psi_negq)
    fr = _int_freqs(N)
    KX = fr[:, None]
    KY = fr[None, :]
    u_hat = 1j * _TWO_PI * KY * psi_hat
    v_hat = -1j * _TWO_PI * KX * psi_hat
    u = jnp.real(jnp.fft.ifft2(u_hat)) * (N * N)
    v = jnp.real(jnp.fft.ifft2(v_hat)) * (N * N)
    return u, v


def _spectral_grad(f_hat, K):
    return jnp.real(jnp.fft.ifft2(1j * _TWO_PI * K * f_hat))


def pressure_field(u: jnp.ndarray, v: jnp.ndarray, N: int) -> jnp.ndarray:
    """∇²p = -[(u_x)²+2 u_y v_x+(v_y)²] 的 spectral Poisson 解（k=0 gauge=0）。"""
    fr = _int_freqs(N)
    KX = fr[:, None]
    KY = fr[None, :]
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    ux = _spectral_grad(u_hat, KX)
    uy = _spectral_grad(u_hat, KY)
    vx = _spectral_grad(v_hat, KX)
    vy = _spectral_grad(v_hat, KY)
    source = -(ux * ux + 2.0 * uy * vx + vy * vy)
    s_hat = jnp.fft.fft2(source)
    k2 = (_TWO_PI ** 2) * (KX ** 2 + KY ** 2)
    k2 = k2.at[0, 0].set(1.0)  # 避免除零；k=0 mode 另設 0
    p_hat = -s_hat / k2
    p_hat = p_hat.at[0, 0].set(0.0)
    return jnp.real(jnp.fft.ifft2(p_hat))


def make_pressure_sensor_map(half_modes: list, ix: np.ndarray, iy: np.ndarray, N: int):
    """closure 固定 basis/sensor，回傳 a → p_sensor[K] 的純函式（供 jax.jacobian）。"""
    idx_pos, idx_neg, _, _ = _half_mode_fft_indices(half_modes, N)
    ix_j = jnp.asarray(ix)
    iy_j = jnp.asarray(iy)

    def f(a):
        u, v = streamfunction_velocity_spectral(a, idx_pos, idx_neg, N)
        p = pressure_field(u, v, N)
        return p[ix_j, iy_j]

    return f


def compute_Jp(a_bar: jnp.ndarray, half_modes: list, ix, iy, N: int) -> np.ndarray:
    """J_p(ā)=∂p_s/∂a ∈ R^{K×2H}。"""
    f = make_pressure_sensor_map(half_modes, ix, iy, N)
    return np.asarray(jax.jacobian(f)(a_bar))


# ── LES snapshot → reference coefficients ā ──────────────────────────────────

def project_velocity_to_coeffs(u: np.ndarray, v: np.ndarray, half_modes: list, N: int) -> np.ndarray:
    """速度場 (N,N) spectral 投影到截斷 stream-function 基底 → ā (2H,)。

    ω=∂_x v-∂_y u=-∇²ψ ⇒ ψ_hat=ω_hat/((2π)²k²)；對每 half-mode 讀 α=2Re ψ_hat, β=-2Im ψ_hat。
    convention 與 streamfunction_velocity_spectral 互逆（round-trip test 守住）。
    """
    fr = np.fft.fftfreq(N, d=1.0 / N)
    KX = fr[:, None]
    KY = fr[None, :]
    # 係數慣例：合成 field = N²·ifft2(C) ⇒ 反推 C = fft2(field)/N²
    u_hat = np.fft.fft2(u) / (N * N)
    v_hat = np.fft.fft2(v) / (N * N)
    omega_hat = 1j * 2 * np.pi * KX * v_hat - 1j * 2 * np.pi * KY * u_hat
    k2 = (2 * np.pi) ** 2 * (KX ** 2 + KY ** 2)
    k2[0, 0] = 1.0
    psi_hat = omega_hat / k2
    a = np.zeros(2 * len(half_modes))
    for j, (kx, ky) in enumerate(half_modes):
        c = psi_hat[kx % N, ky % N]
        a[2 * j] = 2.0 * np.real(c)
        a[2 * j + 1] = -2.0 * np.imag(c)
    return a


# ── driver ────────────────────────────────────────────────────────────────────

def pressure_observability(coords: np.ndarray, kmax: int, les_uv_snaps, N: int,
                           tols=(1e-6, 1e-8, 1e-10)) -> dict:
    """對多個 LES snapshot ā 報 C_u vs C_{u,p}^lin 的可觀測性分佈。

    les_uv_snaps: list of (u,v) 各 (N,N) physical 場。
    """
    half = divfree_half_modes(kmax)
    ix, iy = coords_to_xy_indices(coords, N)
    C_u, _ = divfree_observation_operator(coords, kmax)  # (2K, 2H)
    base = observability_metrics(C_u)

    per_snap = []
    for m, (u, v) in enumerate(les_uv_snaps):
        a_bar = jnp.asarray(project_velocity_to_coeffs(u, v, half, N))
        Jp = compute_Jp(a_bar, half, ix, iy, N)            # (K, 2H)
        C_up = np.concatenate([C_u, Jp], axis=0)           # (3K, 2H)
        m_up = observability_metrics(C_up)
        ranks_tol = {f"rank@{t:g}": int((np.asarray(m_up["singular_values"]) >
                                         t * m_up["sigma_max"]).sum()) for t in tols}
        per_snap.append({
            "snapshot": m,
            "rank_u": base["rank"], "rank_up": m_up["rank"],
            "delta_rank": m_up["rank"] - base["rank"],
            "null_u": base["null_fraction"], "null_up": m_up["null_fraction"],
            "cond_up": m_up["cond"], "eff_rank_up": m_up["effective_rank"],
            **ranks_tol,
        })

    dr = np.array([s["delta_rank"] for s in per_snap], dtype=float)
    return {
        "kmax": kmax, "real_dof": C_u.shape[1], "n_obs_u": C_u.shape[0],
        "baseline_velocity_only": {k: base[k] for k in
                                   ("rank", "cond", "null_fraction", "effective_rank")},
        "n_snapshots": len(per_snap),
        "delta_rank_stats": {"mean": float(dr.mean()), "std": float(dr.std()),
                             "min": float(dr.min()), "max": float(dr.max()),
                             "median": float(np.median(dr))} if dr.size else {},
        "per_snapshot": per_snap,
    }


def _load_les_snapshots(les_path: Path, n_snap: int, t_spinup: float):
    raw = np.load(les_path, allow_pickle=True).item()
    t = np.asarray(raw["time"], dtype=np.float64)
    mask = t >= t_spinup
    u = np.asarray(raw["u"], dtype=np.float64)[mask]
    v = np.asarray(raw["v"], dtype=np.float64)[mask]
    N = u.shape[-1]
    idx = np.linspace(0, u.shape[0] - 1, n_snap).astype(int)
    return [(u[i], v[i]) for i in idx], N


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor-json", default="data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json")
    ap.add_argument("--les", required=True, help="LES 全場 .npy（提供 reference snapshots；DNS-free）")
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--n-snapshots", type=int, default=8)
    ap.add_argument("--t-spinup", type=float, default=0.0)
    ap.add_argument("--out", default="artifacts/diag/pressure_observability.json")
    args = ap.parse_args()

    coords = np.asarray(json.load(open(args.sensor_json))["selected_coordinates"], dtype=np.float64)
    snaps, N = _load_les_snapshots(Path(args.les), args.n_snapshots, args.t_spinup)
    print(f"[sensors] K={coords.shape[0]}  [LES] {N}x{N}  snapshots={len(snaps)}  kmax={args.kmax}")

    rep = pressure_observability(coords, args.kmax, snaps, N)
    b = rep["baseline_velocity_only"]
    print(f"\nvelocity-only: rank={b['rank']}  null={b['null_fraction']*100:.2f}%  κ={b['cond']:.3e}")
    print(f"Δrank over {rep['n_snapshots']} ā: {rep['delta_rank_stats']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(f"[out] {out}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # standalone 診斷需 float64 精度
    main()
