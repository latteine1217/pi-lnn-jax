#!/usr/bin/env python3
"""混沌預算：把 t₀ 的場交給**真實求解器**往前積，誤差最少會長到多少。

What:
    從 t₀ 出發積分到 t_end，量兩條曲線對 DNS 真值的相對誤差：
      A) 從 DNS 自己的 t₀ 場出發   → 求解器保真度（解析度誤差）。這是控制組，
         它若自己就長到 O(1)，本實驗無效，必須先修解析度再談別的。
      B) 從**模型在 t₀ 的重建場**出發 → physics-oracle continuation：
         模型把手上的狀態交給真實物理之後，最好能到哪裡。

Why:
    EXP-530/531 的模型外推在 τ≈1 之後就輸給「凍住不動」。有兩個互斥解釋：
      (i)  模型離開了軌跡（方法問題，可修）；
      (ii) 它在 t₀ 的 14% 誤差經混沌放大，任何方法都到不了更好（物理上限）。
    曲線 B 正是 (ii) 的量化：**任何**正確的外推方式，起點誤差一樣時都跑不贏它。
    模型曲線遠高於 B → (i) 成立；貼著 B → (ii) 成立，該收手寫 limitation。

    這比用 λ_max 推論可靠：λ_max 量在 attractor 穩態幀、N=256，而本段是 enstrophy
    衰減的非穩態前段，且模型誤差不是白噪聲（集中在 mid-band），白噪聲擾動會低估成長。
    直接用模型自己的誤差場當擾動，沒有這兩個外插。

Notes:
    求解器是 `scripts/ns2d_kolmogorov.py`（integrating-factor RK4，已驗證能重現 DNS）。
    它吃渦度；模型的場不完全無散度，轉成 ω 等於取其 solenoidal 投影——這對本實驗
    有利於模型（去掉了它的散度誤差），故結論方向不受影響。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import figwidth, save_figure, setup_style  # noqa: E402
from ns2d_kolmogorov import (  # noqa: E402
    enable_x64,
    integrate,
    vorticity_to_velocity,
    wavenumbers,
)

from pi_lnn_jax.data import load_dns_from_path  # noqa: E402


def velocity_to_vorticity(u: np.ndarray, v: np.ndarray) -> jnp.ndarray:
    """(u,v) → ω = ∂v/∂x − ∂u/∂y（譜空間）。回實場 [N,N]。

    convention 對齊 `ns2d_kolmogorov`：axis 0 = x、axis 1 = y。
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    if u.shape != v.shape or u.shape[0] != u.shape[1]:
        raise ValueError(f"需要同形狀方域場，得 u{u.shape} v{v.shape}")
    kx, ky, _, _ = wavenumbers(u.shape[0])
    omega_hat = 1j * kx * jnp.fft.fft2(v) - 1j * ky * jnp.fft.fft2(u)
    return jnp.fft.ifft2(omega_hat).real


def _uv_rel(u_a, v_a, u_b, v_b) -> float:
    """與 metric_artifact.uv_rel_err 同定義（u,v 合併的相對 L2）。"""
    num = np.sqrt(np.sum((np.asarray(u_a) - u_b) ** 2) + np.sum((np.asarray(v_a) - v_b) ** 2))
    den = np.sqrt(np.sum(np.asarray(u_b) ** 2) + np.sum(np.asarray(v_b) ** 2))
    return float(num / den)


def roll_forward(omega0, times, dt, nu, k_f, A):
    """從 times[0] 逐段積到 times[-1]，回每個 times 上的 (u, v)。

    每段長度必須相同（等距輸出格點）且被 dt 整除——不整除時 `integrate` 會 raise，
    不做寬鬆對齊。
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        raise ValueError("times 至少要兩個時刻")
    # DNS 的時間軸是 float32（見 data.load_dns_from_path），間距帶 ~1e-7 的量化雜訊。
    # 拿它當積分段長會讓 `integrate` 的整除檢查必然失敗，所以：**用標稱格點去積、
    # 拿實際時間軸去驗**。容差 1e-5 對應 float32 在 t~10 的解析度（eps≈1e-6）；
    # 真正不等距（例如漏幀）會差一整個 span，仍然擋得住。
    # 標稱段長 = 最接近的「整數個求解器步」——輸出節奏本來就只能落在步的邊界上，
    # 直接 round 到小數位無法消掉 float32 的雜訊（實測 0.049999952 仍不被 dt 整除）。
    span_raw = (times[-1] - times[0]) / (times.size - 1)
    n_steps = int(round(span_raw / dt))
    if n_steps < 1:
        raise ValueError(f"輸出間距 {span_raw:g} 小於一個求解器步 dt={dt:g}")
    span = n_steps * dt
    if abs(span - span_raw) > 1e-5 * max(span, 1.0):
        raise ValueError(
            f"輸出間距 {span_raw:g} 不是 dt={dt:g} 的整數倍（最近的是 {span:g}）")
    expected = times[0] + span * np.arange(times.size)
    drift = float(np.max(np.abs(times - expected)))
    if drift > 1e-5:
        raise ValueError(
            f"輸出時間格點必須等距：與標稱格點（span={span}）最大偏離 {drift:.3e}")
    kx, ky, _, k2_inv = wavenumbers(omega0.shape[0])
    step = jax.jit(lambda w: integrate(w, span, dt, nu, k_f, A))

    omega = jnp.asarray(omega0, dtype=jnp.float64)
    out_u, out_v = [], []
    for i in range(len(times)):
        if i > 0:
            omega = step(omega)
        u, v = vorticity_to_velocity(jnp.fft.fft2(omega), kx, ky, k2_inv)
        out_u.append(np.asarray(u))
        out_v.append(np.asarray(v))
    return np.stack(out_u), np.stack(out_v)


def main() -> int:
    # 先前靠 `ns2d_kolmogorov` 在 module import 時開 x64（已改為入口點才開），
    # 這裡明確啟用以保住本腳本的 fp64 數值——少了它會靜默掉進 float32。
    enable_x64()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", required=True, help="真值 DNS .npy（涵蓋 [t0, t_end]）")
    ap.add_argument("--fields", required=True,
                    help="evaluate_exp245.py --export-fields 的 fields.npz（模型重建場）")
    ap.add_argument("--t0", type=float, default=5.0, help="出發時刻（= 資料末端）")
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--out-stride", type=int, default=2,
                    help="輸出格點取 DNS 每第幾幀（預設 2 → 對齊 eval 的 Δt=0.05）")
    ap.add_argument("--dt", type=float, default=2.5e-4, help="求解器時間步")
    ap.add_argument("--re", type=float, default=10000.0)
    ap.add_argument("--k-f", type=float, default=2.0)
    ap.add_argument("--amp", type=float, default=0.1, help="forcing 振幅 A")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dns_u, dns_v, dns_t = load_dns_from_path(args.dns, time_stride=1)
    dns_t = np.asarray(dns_t, dtype=np.float64)
    i0 = int(np.argmin(np.abs(dns_t - args.t0)))
    i1 = int(np.argmin(np.abs(dns_t - args.t_end)))
    for name, want, got in (("t0", args.t0, dns_t[i0]), ("t_end", args.t_end, dns_t[i1])):
        if abs(float(got) - want) > 1e-6:
            raise ValueError(f"DNS 時間軸上沒有 {name}={want}（最近 {float(got)}）；不寬鬆對齊")
    idx = np.arange(i0, i1 + 1, args.out_stride)
    times = dns_t[idx]

    f = np.load(args.fields)
    f_t = np.asarray(f["t"], dtype=np.float64)
    j0 = int(np.argmin(np.abs(f_t - args.t0)))
    if abs(float(f_t[j0]) - args.t0) > 1e-6:
        raise ValueError(f"fields.npz 時間軸上沒有 t0={args.t0}；不寬鬆對齊")

    nu = 1.0 / float(args.re)
    print(f"[setup] N={dns_u.shape[1]}  ν={nu:g}  k_f={args.k_f}  A={args.amp}  dt={args.dt:g}")
    print(f"[setup] 輸出 {len(times)} 個時刻，t ∈ [{times[0]:.3f}, {times[-1]:.3f}]")

    runs = {}
    for tag, (u0, v0) in {
        "solver_from_dns": (dns_u[i0], dns_v[i0]),
        "solver_from_model": (f["u_pred"][j0], f["v_pred"][j0]),
    }.items():
        print(f"[run] {tag} …", flush=True)
        omega0 = velocity_to_vorticity(u0, v0)
        pu, pv = roll_forward(omega0, times, args.dt, nu, args.k_f, args.amp)
        runs[tag] = np.array([_uv_rel(pu[k], pv[k], dns_u[i], dns_v[i])
                              for k, i in enumerate(idx)])

    tau = times - args.t0
    report = {
        "t0": args.t0, "t_end": args.t_end, "re": args.re, "dt": args.dt,
        "grid": int(dns_u.shape[1]), "dns": str(args.dns), "fields": str(args.fields),
        "tau": tau.tolist(),
        "solver_from_dns": runs["solver_from_dns"].tolist(),
        "solver_from_model": runs["solver_from_model"].tolist(),
        "at_lead_times": {
            f"{lt:g}": {
                "solver_from_dns": float(np.interp(lt, tau, runs["solver_from_dns"])),
                "solver_from_model": float(np.interp(lt, tau, runs["solver_from_model"])),
            } for lt in (0.0, 0.5, 1.0, 2.0, 5.0) if lt <= tau.max()
        },
        "reading": (
            "solver_from_dns 是控制組（求解器在此網格的保真度）；它若不遠小於 "
            "solver_from_model，本實驗無效。solver_from_model 是 physics-oracle "
            "continuation：起點誤差相同時，任何正確的外推方式都跑不贏它，故它是"
            "混沌下限。模型自身曲線遠高於它 = 模型離開了軌跡（方法問題）；"
            "貼著它 = 已達物理上限。"
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chaos_budget.json").write_text(json.dumps(report, indent=2))

    setup_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(figwidth("iclr", "single"), 2.6))
    ax.plot(tau, runs["solver_from_model"], lw=1.4, label="Physics from model state")
    ax.plot(tau, runs["solver_from_dns"], lw=1.1, ls="--", label="Physics from DNS state (control)")
    ax.set_xlabel(r"Lead time $\tau$")
    ax.set_ylabel("Relative $L_2$ error of $(u,v)$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_figure(fig, str(out_dir / "chaos_budget"))

    print("\n=== chaos budget ===")
    print(f"  {'tau':>5}  {'from model':>11}  {'from DNS (ctrl)':>16}")
    for lt, row in report["at_lead_times"].items():
        print(f"  {lt:>5}  {row['solver_from_model']:11.4f}  {row['solver_from_dns']:16.4f}")
    print(f"[out] {out_dir / 'chaos_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
