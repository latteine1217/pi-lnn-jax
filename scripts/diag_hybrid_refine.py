#!/usr/bin/env python3
"""diag_hybrid_refine.py — learned 初始 + 迭代精修，量「單次前向 vs 迭代」的分界（EXP-514）。

What:
    取 learned model 已發表的重建場（`final_eval/fields.npz`），把 t=0 的 (u,v) 轉成
    渦度 ω0，以它為 4D-Var 的初始猜測跑固定預算的迭代，並沿途記錄 ω0 快照，
    後處理成「mid-band 誤差 vs 迭代次數」的曲線。

Why:
    EXP-512 顯示 oracle 的數字是（問題＋優化器＋預算）的性質；EXP-513 顯示中頻佔渦度
    範數 64%，故「從零恢復中頻」是全域非凸問題、線性化框架不適用。兩者共同指向一個
    比原本「感測 vs 模型」更站得住的主張：**中頻可能只能由迭代／全域方法恢復，
    任何 amortized 單次前向算子都做不到**——那是方法類別的陳述，不是誰的錯。
    本診斷直接測那條分界：learned 的輸出若落在正確 basin，少量迭代就該讓中頻大幅下降。

    對照組已存在（EXP-512 Stage 3a，job 5345）：**同樣 nf=20、非鏈式、maxiter=200，
    僅初始猜測不同**——零起點得 E_mid=34.87%。故本實驗是乾淨的單變因對照。

驗證階梯（不過關即中止）:
    1. curl round-trip：ω0 = i(k_x v̂ − k_y û) 必須逐位元反轉 `vorticity_to_velocity`。
    2. 重現已發表數字：由 fields.npz 算出的 learned mid-band 誤差須落在 57.73% 附近
       （容差由 --repro-tol 給），否則 band 分解或時間軸對齊有誤。
    3. div-free 投影損失：模型的 (u,v) 經 curl→velocity 往返後的相對變化。
       這量的是「以 ω0 為控制變數」丟掉了多少模型輸出，必須報告而非忽略。

⚠️ 長診斷，走 lab-server r740 sbatch，不在本機跑。
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

from scripts.diag_identifiability import (  # noqa: E402
    load_fields_and_dns,
    sensor_band_edge,
    velocity_band_rel_l2,
)
from scripts.fourdvar_identifiability import (  # noqa: E402
    _resolve,
    _sensor_indices,
    make_forward,
)
from scripts.ns2d_kolmogorov import (  # noqa: E402
    integrate,
    vorticity_to_velocity,
    wavenumbers,
)


def velocity_to_vorticity(u: np.ndarray, v: np.ndarray, kx, ky) -> np.ndarray:
    """(u,v) → ω，作為 `vorticity_to_velocity` 的逆（由 round-trip gate 驗證，非靠推導）。"""
    uh = jnp.fft.fft2(jnp.asarray(u))
    vh = jnp.fft.fft2(jnp.asarray(v))
    return np.asarray(jnp.fft.ifft2(1j * (kx * vh - ky * uh)).real)


def gate_curl_roundtrip(N: int, rng, k_cut_frac: float = 0.25) -> dict:
    """關卡 1：從隨機 ω 出發，velocity→vorticity 必須還原它。

    測試場**帶限**在 k ≤ k_cut_frac·N，理由不是為了讓它過關：
    實場在 Nyquist（k=N/2）的 Fourier 係數為實數，乘 ik 後不再是實場的變換，
    `ifft.real` 會丟掉該模態——這是 ω↔(u,v) 在 Nyquist 上的**結構性**限制，
    不是實作錯誤。白噪聲在該處有滿能量（實測 N=64 的往返誤差 0.148，恰為
    Nyquist 模態占比的開根號 0.176 的量級），但實際流場在 k=N/2 的能量可忽略
    （附錄 D：k≤16 已含 99.91% 動能）。本關驗的是**代數正確性**；
    真實場上的實際損失由 gate3 在原始資料上量測並報告，不藏。

    ⚠️ 2026-08-30：白噪聲測試訊號在本系列已誤用兩次（EXP-513 gate2 亦然）。
    通則——測試方向必須落在被測物實際作用的子空間內。
    """
    kx, ky, k2, k2_inv = wavenumbers(N)
    kr = np.sqrt(np.asarray(kx) ** 2 + np.asarray(ky) ** 2) / (2.0 * np.pi)
    w = rng.standard_normal((N, N))
    wh = np.fft.fft2(w)
    wh[kr > k_cut_frac * N] = 0.0
    wh[0, 0] = 0.0                                  # DC 不在 ω 的像空間內
    w = np.fft.ifft2(wh).real
    u, v = vorticity_to_velocity(jnp.fft.fft2(jnp.asarray(w)), kx, ky, k2_inv)
    w_back = velocity_to_vorticity(np.asarray(u), np.asarray(v), kx, ky)
    rel = float(np.linalg.norm(w_back - w) / np.linalg.norm(w))
    return {"rel_err": rel, "k_cut": k_cut_frac * N, "pass": rel < 1e-10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True, help="learned model 的 final_eval/fields.npz")
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--nu", type=float, default=1e-4)
    ap.add_argument("--k_f", type=int, default=2)
    ap.add_argument("--A", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=2.5e-4)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--trace-every", type=int, default=5)
    ap.add_argument("--repro-target", type=float, default=0.5773,
                    help="已發表的 learned mid-band 誤差；gate2 以此比對")
    ap.add_argument("--repro-tol", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    u_pred, v_pred, u_dns_f, v_dns_f = load_fields_and_dns(args.fields)
    N = u_pred.shape[-1]
    kx, ky, k2, k2_inv = wavenumbers(N)
    meta = json.load(open(_resolve(args.sensor_json)))
    coords = np.asarray(meta["selected_coordinates"], np.float64)
    K = int(meta["K"])
    k_s = sensor_band_edge(K)
    out: dict = {"N": N, "K": K, "k_s": k_s, "n_frames": args.n_frames,
                 "maxiter": args.maxiter, "fields": str(args.fields)}
    print(f"[cfg] N={N} K={K} k_s={k_s:.4f} frames_in_fields={u_pred.shape[0]}", flush=True)

    # ---- gate 1：curl round-trip ----
    g1 = gate_curl_roundtrip(N, rng)
    out["gate_curl_roundtrip"] = g1
    print(f"[gate1 curl] rel_err={g1['rel_err']:.3e} pass={g1['pass']}", flush=True)
    if not g1["pass"]:
        raise SystemExit("gate1 FAILED：velocity→vorticity 非 vorticity_to_velocity 的逆。")

    # ---- gate 2：重現已發表的 learned mid-band 誤差 ----
    band_learned = velocity_band_rel_l2(u_pred, v_pred, u_dns_f, v_dns_f, K)
    e_learned = float(band_learned["bands"]["mid"]["uv_rel"])
    dev = abs(e_learned - args.repro_target)
    out["gate_repro"] = {"e_learned_mid": e_learned, "target": args.repro_target,
                         "dev": dev, "pass": dev <= args.repro_tol}
    out["band_learned"] = band_learned
    print(f"[gate2 repro] learned mid={100*e_learned:.2f}% "
          f"(目標 {100*args.repro_target:.2f}%, 差 {100*dev:.2f}pp) "
          f"pass={out['gate_repro']['pass']}", flush=True)
    if not out["gate_repro"]["pass"]:
        raise SystemExit("gate2 FAILED：無法重現已發表數字，band 分解或時間軸對齊有誤。")

    # ---- gate 3：div-free 投影損失（報告，不 gate）----
    w0_learned = velocity_to_vorticity(u_pred[0], v_pred[0], kx, ky)
    u_rt, v_rt = vorticity_to_velocity(jnp.fft.fft2(jnp.asarray(w0_learned)), kx, ky, k2_inv)
    proj = float(np.linalg.norm(np.stack([np.asarray(u_rt) - u_pred[0],
                                          np.asarray(v_rt) - v_pred[0]]))
                 / np.linalg.norm(np.stack([u_pred[0], v_pred[0]])))
    out["divfree_projection_loss"] = proj
    print(f"[gate3 proj] 以 ω0 為控制變數丟掉的模型輸出 = {100*proj:.2f}%", flush=True)

    # ---- 觀測與前向 ----
    from pi_lnn_jax.data import load_dns_from_path
    obj_u, obj_v, t_dns = load_dns_from_path(str(np.load(_resolve(args.fields),
                                                        allow_pickle=True)["dns_path"]),
                                             time_stride=2)
    obj_u = np.asarray(obj_u, np.float64); obj_v = np.asarray(obj_v, np.float64)
    spf = round(float(t_dns[1] - t_dns[0]) / args.dt)
    sensor_flat = _sensor_indices(coords, N)
    forward = make_forward(N, args.dt, spf, args.nu, args.k_f, args.A, sensor_flat)
    obs = np.stack([np.concatenate([obj_u[f].reshape(-1)[sensor_flat],
                                    obj_v[f].reshape(-1)[sensor_flat]])
                    for f in range(args.n_frames)])
    obs_j = jnp.asarray(obs)
    scale = float(np.linalg.norm(obs) ** 2)

    def cost(w0_flat):
        pred = forward(w0_flat.reshape(N, N), args.n_frames)
        return jnp.sum((pred - obs_j) ** 2) / scale

    vg = jax.jit(jax.value_and_grad(cost))

    def scipy_obj(x):
        val, g = vg(jnp.asarray(x))
        return float(val), np.asarray(g, np.float64)

    # ---- traced 4D-Var，從 learned ω0 出發 ----
    traces: list[dict] = []
    state = {"it": 0}

    def record(w0, it):
        traces.append({"iter": it, "omega0": np.asarray(w0).copy()})

    record(w0_learned, 0)

    def cb(xk):
        state["it"] += 1
        if state["it"] % args.trace_every == 0:
            record(xk.reshape(N, N), state["it"])

    print(f"[4dvar] 從 learned ω0 出發，maxiter={args.maxiter} "
          f"trace_every={args.trace_every}，init cost={scipy_obj(w0_learned.reshape(-1))[0]:.4e}",
          flush=True)
    res = minimize(scipy_obj, w0_learned.reshape(-1), jac=True, method="L-BFGS-B",
                   callback=cb,
                   options={"maxiter": args.maxiter, "maxfun": 20 * args.maxiter,
                            "ftol": 1e-15, "gtol": 1e-12})
    if not traces or traces[-1]["iter"] != res.nit:
        record(np.asarray(res.x).reshape(N, N), int(res.nit))
    out["cost_final"] = float(res.fun)
    out["nit"] = int(res.nit)
    out["success"] = bool(res.success)
    print(f"[4dvar] cost={res.fun:.4e} nit={res.nit}/{args.maxiter} success={res.success}",
          flush=True)

    # ---- 後處理：每個快照的 band 誤差 ----
    curve = []
    for tr in traces:
        w = tr["omega0"]
        u_rec = np.empty((args.n_frames, N, N)); v_rec = np.empty((args.n_frames, N, N))
        for f in range(args.n_frames):
            wf = w if f == 0 else np.asarray(integrate(w, f * spf * args.dt, args.dt,
                                                       args.nu, args.k_f, args.A))
            uu, vv = vorticity_to_velocity(jnp.fft.fft2(jnp.asarray(wf)), kx, ky, k2_inv)
            u_rec[f], v_rec[f] = np.asarray(uu), np.asarray(vv)
        b = velocity_band_rel_l2(u_rec, v_rec, obj_u[:args.n_frames], obj_v[:args.n_frames], K)
        row = {"iter": tr["iter"],
               "mid": float(b["bands"]["mid"]["uv_rel"]),
               "low": float(b["bands"]["low"]["uv_rel"])}
        curve.append(row)
        print(f"  [trace] iter={row['iter']:4d}  mid={100*row['mid']:6.2f}%  "
              f"low={100*row['low']:6.2f}%", flush=True)
    out["curve"] = curve
    out["e_zero_init_ref"] = 0.3487   # EXP-512 Stage 3a（job 5345），同 nf/maxiter，僅初始不同

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
