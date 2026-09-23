#!/usr/bin/env python3
"""fourdvar_identifiability.py — Ticket 04：nonlinear NS 4D-Var mid-band 可識別性 oracle。

What:
    strong-constraint 4D-Var：控制變數 = 初始渦度 ω0，前向由 validated NS solver
    （ns2d_kolmogorov）決定整條軌跡；cost = K=100 sensor 速度軌跡對 DNS 觀測的 mismatch。
    從「只含 low-band、mid-band 清零」的初始猜測出發，用 L-BFGS 優化 ω0，看 sensor+NS
    能否**唯一恢復 mid-band**。收斂後重建的 mid-band rel-L2 = **nonlinear 可識別性上限**。

Why（Ticket 03 → 04）:
    Ticket 03 已證 linear projection 無法作為上限（mid-band linear null_fraction 0.714，
    model 57% 已超越 linear 上限 ≥85%）。floor/gap 只能由 nonlinear NS oracle 裁定——
    NS 把 mid-band 耦合到 low-band 的資訊是非線性的，linear observability 測不到。

判定（事前登錄，凍結；σ=0.325pp 由 Ticket 02 鎖定）:
    gap = E_model(mid 57.73%) − E_oracle。
    gap ≥ max(15pp, 4σ)=15pp 且 headline 改善 ≥2pp → gap-confirmed（解鎖方向 1-3）。
    gap ≤ max(3pp, 2σ)=3pp → floor（加 K / negative result）。之間 → 部分可識別。

⚠️ 完整收斂優化屬 CLAUDE.md「長診斷」，走 lab-server r740 sbatch，不在本機跑。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from pi_lnn_jax.data import load_dns_from_path  # noqa: E402
from scripts.ns2d_kolmogorov import (  # noqa: E402
    _nonlinear_and_forcing,
    dealias_mask,
    vorticity_to_velocity,
    wavenumbers,
)
from scripts.diag_identifiability import (  # noqa: E402
    radial_wavenumber_grid,
    sensor_band_edge,
    band_filter,
    velocity_band_rel_l2,
)


def _sensor_indices(coords: np.ndarray, N: int) -> np.ndarray:
    """sensor 物理座標 → flat grid index（u[ix,iy] → ix*N+iy），對齊 baselines convention。"""
    g = np.arange(N) / N
    ix = np.abs(coords[:, 0][:, None] - g[None, :]).argmin(axis=1)
    iy = np.abs(coords[:, 1][:, None] - g[None, :]).argmin(axis=1)
    return ix * N + iy


def make_forward(N, dt, steps_per_frame, nu, k_f, A, sensor_flat):
    """回一個 jit forward_observe(omega0)→ sensor 速度軌跡 [n_frames,2K]（frame-wise IF-RK4）。"""
    kx, ky, k2, k2_inv = wavenumbers(N)
    mask = dealias_mask(N)
    g = jnp.arange(N) / N
    _, Y = jnp.meshgrid(g, g, indexing="ij")
    f_omega = -2.0 * jnp.pi * k_f * A * jnp.cos(2.0 * jnp.pi * k_f * Y)
    f_omega_hat = jnp.fft.fft2(f_omega) * mask
    E = jnp.exp(-nu * k2 * dt); E2 = jnp.exp(-nu * k2 * dt / 2.0)
    sensor_flat = jnp.asarray(sensor_flat)

    def rhs(oh):
        return _nonlinear_and_forcing(oh, kx, ky, k2_inv, mask, f_omega_hat)

    def solver_step(oh, _):
        k1 = rhs(oh)
        k2_ = rhs(E2 * (oh + 0.5 * dt * k1))
        k3 = rhs(E2 * oh + 0.5 * dt * k2_)
        k4 = rhs(E * oh + dt * E2 * k3)
        return E * oh + (dt / 6.0) * (E * k1 + 2.0 * E2 * k2_ + 2.0 * E2 * k3 + k4), None

    def frame_step(oh, _):
        oh, _ = jax.lax.scan(solver_step, oh, None, length=steps_per_frame)
        u, v = vorticity_to_velocity(oh, kx, ky, k2_inv)
        sens = jnp.concatenate([u.reshape(-1)[sensor_flat], v.reshape(-1)[sensor_flat]])
        return oh, sens

    # gradient checkpointing：長 scan（n_frames×steps_per_frame≈4000 steps）的 reverse-mode
    # grad 若存全部中間 activation 會 OOM（實測 31GB > 24GB）。frame-level remat → backward
    # 只存 frame 邊界（n_frames 個），frame 內 steps 重算。峰值記憶體降到 O(steps_per_frame·N²)。
    frame_step = jax.checkpoint(frame_step)

    def forward_observe(omega0, n_frames):
        oh0 = jnp.fft.fft2(omega0)
        # frame 0 的觀測（t=0，未演化）
        u0, v0 = vorticity_to_velocity(oh0, kx, ky, k2_inv)
        s0 = jnp.concatenate([u0.reshape(-1)[sensor_flat], v0.reshape(-1)[sensor_flat]])
        _, rest = jax.lax.scan(frame_step, oh0, None, length=n_frames - 1)
        return jnp.concatenate([s0[None, :], rest], axis=0)  # [n_frames, 2K]

    return forward_observe


#: cold-start 的初始猜測來源。**這不是實作細節，是結論的承重面**——
#: `dns_lowband` 讓 oracle 從**真實 DNS 初始渦度的低頻**出發，因此 8.8% 的正確讀法是
#: 「**在給定真實低頻的前提下**，中頻可由 sensor+NS 恢復」，比「從 sensor 單獨可識別」弱。
#: 在崎嶇的 chaotic landscape 上（見下方 tolerance 註解）起點不是無關緊要的細節，
#: 而 `--n-frames-list` 的 incremental warm-start 只會沿著鏈條放大這個依賴。
#: `zero` 是把該依賴量出來的控制組：完全不給低頻。
INIT_MODES = ("dns_lowband", "zero")


def make_cold_init(mode: str, omega_dns0: np.ndarray, kr: np.ndarray, k_s: float) -> np.ndarray:
    if mode == "dns_lowband":
        return band_filter(omega_dns0[None], kr, -1.0, k_s)[0]
    if mode == "zero":
        return np.zeros_like(omega_dns0)
    raise ValueError(f"unknown init mode {mode!r}; 可用：{INIT_MODES}")


#: EXP-512 天花板控制組的評估點。`dns_true` = DNS 真值 ω₀ 本身：
#: 它在真實觀測上的 cost 就是 **model-error 地板**（前向 IF-RK4 ≠ 產生 DNS 的 ETDRK4），
#: 由它前向積分得到的 band 誤差就是 **oracle 天花板**——任何反演都不可能更好。
#: 沒有這兩個數，`E_oracle_mid` 無法區分「資訊下限 / 優化未收斂 / 軌跡漂移」。
EVAL_AT_MODES = ("dns_true", "dns_lowband", "zero")


def make_eval_field(mode: str, omega_dns0: np.ndarray, kr: np.ndarray, k_s: float) -> np.ndarray:
    if mode == "dns_true":
        return np.asarray(omega_dns0, np.float64)
    if mode in INIT_MODES:
        return make_cold_init(mode, omega_dns0, kr, k_s)
    raise ValueError(f"unknown eval-at mode {mode!r}; 可用：{EVAL_AT_MODES}")


def run_4dvar(dns_path, sensor_json, n_frames, nu, k_f, A, dt, maxiter, out_path=None,
              w0_init_field=None, init_mode="dns_lowband", eval_at=None):
    u_dns, v_dns, t_dns = load_dns_from_path(dns_path, time_stride=2)
    u_dns = np.asarray(u_dns, np.float64); v_dns = np.asarray(v_dns, np.float64)
    obj = np.load(_resolve(dns_path), allow_pickle=True).item()
    omega_dns = np.asarray(obj["omega"], np.float64)[::2]  # 對齊 velocity 的 time_stride=2
    N = u_dns.shape[-1]
    steps_per_frame = round(float(t_dns[1] - t_dns[0]) / dt)

    meta = json.load(open(sensor_json))
    coords = np.asarray(meta["selected_coordinates"], np.float64)
    sensor_flat = _sensor_indices(coords, N)
    K = coords.shape[0]

    # 觀測軌跡 = DNS 在 sensor 的速度（window 內）
    obs = np.stack([
        np.concatenate([u_dns[f].reshape(-1)[sensor_flat], v_dns[f].reshape(-1)[sensor_flat]])
        for f in range(n_frames)
    ])  # [n_frames, 2K]
    obs_j = jnp.asarray(obs)

    forward = make_forward(N, dt, steps_per_frame, nu, k_f, A, sensor_flat)
    scale = float(np.linalg.norm(obs) ** 2)

    def cost(w0_flat):
        w0 = w0_flat.reshape(N, N)
        pred = forward(w0, n_frames)
        return jnp.sum((pred - obs_j) ** 2) / scale

    vg = jax.jit(jax.value_and_grad(cost))

    def scipy_obj(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, np.float64)

    # 初始猜測：DNS[0] 的 low-band（含可觀測部分），mid/high 清零 → 測 sensor+NS 能否恢復 mid
    kr = radial_wavenumber_grid(N); k_s = sensor_band_edge(K)
    if w0_init_field is not None:
        w0_init = np.asarray(w0_init_field)  # warm-start：前一個 window 的收斂解（incremental）
    else:
        w0_init = make_cold_init(init_mode, omega_dns[0], kr, k_s)

    print(f"[4D-Var] N={N} K={K} n_frames={n_frames} steps/frame={steps_per_frame} "
          f"window t=[0,{t_dns[n_frames-1]:.3f}]  maxiter={maxiter}", flush=True)
    print(f"[init] cost={scipy_obj(w0_init.reshape(-1))[0]:.4e}", flush=True)

    # 收緊 tolerance：chaotic 4D-Var 的 cost landscape 崎嶇，預設 gtol/ftol 會讓 L-BFGS
    # 在局部平坦區早停（實測 nit=14、cost 卡 0.17、連 low band 都沒收斂）。強制跑到底。
    if eval_at is not None:
        # 天花板控制組：不呼叫 minimize，直接在指定 ω₀ 上求值，其餘管線與優化路徑共用
        # （共用是重點——分開寫兩條重建路徑會讓天花板與被比較的數字不可比）。
        w0_star = make_eval_field(eval_at, omega_dns[0], kr, k_s)
        cost_final, nit, success = float(scipy_obj(w0_star.reshape(-1))[0]), 0, True
        print(f"[eval-at {eval_at}] cost={cost_final:.4e}（未優化）", flush=True)
    else:
        res = minimize(scipy_obj, np.asarray(w0_init).reshape(-1), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 20 * maxiter, "ftol": 1e-15, "gtol": 1e-12})
        w0_star = np.asarray(res.x).reshape(N, N)
        cost_final, nit, success = float(res.fun), int(res.nit), bool(res.success)
        print(f"[done] cost={cost_final:.4e} nit={nit} success={success}", flush=True)

    # 重建整條軌跡（oracle）→ 各 band rel-L2 vs DNS（window 內）
    # 逐 frame integrate 存全場（非只 sensor），供 band 分解
    from scripts.ns2d_kolmogorov import integrate
    u_rec = np.empty((n_frames, N, N)); v_rec = np.empty((n_frames, N, N))
    kx, ky, _, k2_inv = wavenumbers(N)
    for f in range(n_frames):
        # 用整數步 × dt（float64 一致），避免 t_dns 的 float32 捨入踩 integrate 整除檢查
        w = w0_star if f == 0 else np.asarray(integrate(w0_star, f * steps_per_frame * dt, dt, nu, k_f, A))
        uu, vv = vorticity_to_velocity(jnp.fft.fft2(jnp.asarray(w)), kx, ky, k2_inv)
        u_rec[f], v_rec[f] = np.asarray(uu), np.asarray(vv)

    res_band = velocity_band_rel_l2(u_rec, v_rec, u_dns[:n_frames], v_dns[:n_frames], K)
    E_oracle = res_band["bands"]["mid"]["uv_rel"]
    E_model = 0.5773
    gap = E_model - E_oracle

    # headline 換算（band 正交）
    f_mid = 0.0443
    u_cur = 0.1503
    du = u_cur - float(np.sqrt(max(u_cur**2 - (E_model**2 - E_oracle**2) * f_mid, 0.0)))

    print("\n=== Ticket 04 判定（nonlinear 4D-Var oracle）===")
    print(f"  E_model(mid) = {100*E_model:.2f}%   E_oracle(mid) = {100*E_oracle:.2f}%   gap = {100*gap:.2f}pp")
    print(f"  headline 預期改善 ≈ {100*du:.2f}pp")
    if eval_at is not None:
        # 天花板控制組沒有優化，gap 判定不適用；照原判準印會得到誤導性的 GAP-CONFIRMED。
        verdict = f"EVAL-ONLY（eval_at={eval_at}；未優化，gap 判定不適用）"
    elif gap >= 0.15 and du >= 0.02:
        verdict = "GAP-CONFIRMED（解鎖方向 1-3）"
    elif gap <= 0.03:
        verdict = "FLOOR-CONFIRMED（加 K / negative result）"
    else:
        verdict = "部分可識別 / 邊界（不強行二分）"
    print(f"  → {verdict}")

    # 收斂診斷（nit/success/maxiter）必須落進產物：cost 的最小值不隨初始猜測改變，
    # 故兩臂 cost_final 不同即代表至少一臂沒到全域最小。沒有這些欄位，
    # 「資訊下限」與「優化未收斂」在事後無法分辨——這正是 2026-08 稽核踩到的坑。
    out = {"E_model_mid": E_model, "E_oracle_mid": float(E_oracle), "gap_pp": 100*float(gap),
           "headline_improve_pp": 100*float(du), "verdict": verdict, "n_frames": n_frames,
           "cost_init": scipy_obj(w0_init.reshape(-1))[0], "cost_final": cost_final,
           "nit": nit, "success": success, "maxiter": int(maxiter),
           "eval_at": eval_at, "band_full": res_band,
           "init_mode": (init_mode if w0_init_field is None else "warm_start"),
           "cold_init_mode": init_mode}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2, default=float))
        print(f"[out] {out_path}", flush=True)
    return out, w0_star


def _resolve(p):
    from pi_lnn_jax.data import _resolve_data_path
    return _resolve_data_path(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--n-frames-list", default=None,
                    help="逗號分隔 window sweep（incremental warm-start，如 3,6,10,15,20）")
    ap.add_argument("--nu", type=float, default=1e-4)
    ap.add_argument("--k_f", type=int, default=2)
    ap.add_argument("--A", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=2.5e-4)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--init-mode", choices=INIT_MODES, default="dns_lowband",
                    help="cold-start 初始猜測；預設 dns_lowband = 既有行為")
    ap.add_argument("--eval-at", choices=EVAL_AT_MODES, default=None,
                    help="EXP-512 天花板控制組：不優化，只在指定 ω₀ 上求值。"
                         "dns_true = DNS 真值 ω₀（給 model-error 地板與 oracle 天花板）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.n_frames_list:
        windows = [int(x) for x in args.n_frames_list.split(",")]
        w0 = None; sweep = []
        for nf in windows:
            print(f"\n########## window n_frames={nf} (warm-start={'yes' if w0 is not None else 'cold'}) ##########", flush=True)
            out, w0 = run_4dvar(args.dns, args.sensor_json, nf, args.nu, args.k_f, args.A,
                                args.dt, args.maxiter, None,
                                w0_init_field=(None if args.eval_at else w0),
                                init_mode=args.init_mode, eval_at=args.eval_at)
            sweep.append({"n_frames": nf, "E_oracle_mid": out["E_oracle_mid"],
                          "gap_pp": out["gap_pp"],
                          "cost_init": out["cost_init"], "cost_final": out["cost_final"],
                          "nit": out["nit"], "success": out["success"],
                          "maxiter": out["maxiter"], "init_mode": out["init_mode"],
                          "eval_at": out["eval_at"],
                          "low": out["band_full"]["bands"]["low"]["uv_rel"]})
        print("\n=== window sweep 趨勢（gap 隨 window 增長）===", flush=True)
        for s in sweep:
            if s.get("eval_at"):
                budget, conv = "", f"  [EVAL-AT {s['eval_at']}]"
            else:
                budget = "  [HIT MAXITER]" if s["nit"] >= s["maxiter"] else ""
                conv = "" if s["success"] else "  [NOT CONVERGED]"
            print(f"  nf={s['n_frames']:3d}: cost={s['cost_init']:.1e}->{s['cost_final']:.1e} "
                  f"nit={s['nit']}/{s['maxiter']} low={100*s['low']:5.1f}% "
                  f"oracle_mid={100*s['E_oracle_mid']:.2f}% gap={s['gap_pp']:+.2f}pp"
                  f"{conv}{budget}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(sweep, indent=2, default=float))
            print(f"[out] {args.out}", flush=True)
    else:
        run_4dvar(args.dns, args.sensor_json, args.n_frames, args.nu, args.k_f, args.A,
                  args.dt, args.maxiter, args.out, init_mode=args.init_mode,
                  eval_at=args.eval_at)
