#!/usr/bin/env python
"""EnKF 資料同化基線 —— 從稀疏 (u,v) 感測重建 Kolmogorov 全場。

定位（比較時必須連同這點一起報告）：EnKF **知道精確的 PDE 離散化**——forward
model 就是 solver 本身——而 PI-CON 只有 residual 形式的物理約束。所以它不是
「同條件下的更強方法」，是「拿到更多資訊的方法」。

與 PI-CON 用同一份 sensor_json/DNS、同一個評估協定（`load_for_evaluation`）與
同一道 metric seam（`EvaluationRunRecorder`）→ fair by construction，與
`classical_baselines_fair.py` 同一條產線。

⚠️ **所有濾波器與物理旋鈕都必填、無預設。**
thesis `chapter04.tex:512` 原本報的 EnKF 數字（96 members / 2.53% KE / 26.29%
velocity）出自一個已無跡可循的程式：driver、config、artifact、log 在
pi-lnn-jax 與 pi-lnn 的全 git 歷史、lab-server 與 home-gpu 上都不存在
（見 `paper/thesis-format/AUDIT.md` §4.1）。在這裡給任何預設值，只會再製造
一個同樣無法追溯的數字。ensemble 大小、膨脹、局域化半徑、觀測誤差、solver
步長、forcing 參數一律由呼叫端明示並寫進 artifact 的 provenance。

觀測噪音的約定：標準 sensor 檔存的是乾淨的 DNS 取樣值。本腳本**不對觀測加噪**，
`--obs-err-frac` 只決定 R = (frac * RMS)^2——即「濾波器假設觀測有多不準」。
要評估含噪觀測請改傳 `gen_noisy_sensors.py` 產出的 sensor 檔，噪音便來自資料
本身而非這裡偷偷加的。

初始 ensemble 不碰真值（由獨立隨機場演化而來），否則同化結果會因洩漏而虛高。
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

from pi_lnn_jax.evaluation_protocol import ProtocolMode


def band_limited_ensemble(rng, N: int, n_members: int, k_cut: int,
                          amplitude: float) -> np.ndarray:
    """帶限隨機渦度 ensemble [Ne, N, N]，零均值、實數。

    不從真值擾動：真值資訊一旦進入初始 ensemble，同化誤差就不再衡量
    「從觀測恢復狀態」的能力。
    """
    k1 = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    mask = (np.sqrt(KX**2 + KY**2) <= k_cut) & ((KX != 0) | (KY != 0))
    out = np.empty((n_members, N, N), dtype=np.float64)
    for m in range(n_members):
        noise = rng.normal(size=(N, N))
        w_hat = np.fft.fft2(noise) * mask
        w = np.real(np.fft.ifft2(w_hat))
        s = w.std()
        if s <= 0:
            raise RuntimeError("初始 ensemble member 為常數場——k_cut 可能過小")
        out[m] = w / s * amplitude
    return out


def positions_to_grid_indices(pos: np.ndarray, N: int, *, tol: float) -> np.ndarray:
    """物理座標 [K,2] ∈ [0,1) → 網格索引 [K,2]。落不到格點就失敗，不靜默 snap。

    DNS 的格點慣例是 `linspace(0, L, N, endpoint=False)`（見 dns_generator 的
    輸出 `x`/`y`），故 i = x*N 必須是整數。感測點若不在格點上，觀測算子取的
    就不是它宣稱的那一點——那是靜默錯配，不是可容忍的近似。
    """
    scaled = np.asarray(pos, dtype=np.float64) * N
    idx = np.rint(scaled).astype(int)
    drift = np.abs(scaled - idx)
    bad = drift > tol
    if bad.any():
        k = int(np.argmax(drift.max(axis=1)))
        raise ValueError(
            f"感測點未落在 {N}x{N} 格點上：第 {k} 點 pos={pos[k]} → "
            f"scaled={scaled[k]}（最大偏移 {drift.max():.3e} > tol {tol:g}）。"
            "sensor 檔的解析度與 DNS 評估網格不一致時會出現這種偏移。")
    if (idx < 0).any() or (idx >= N).any():
        raise ValueError(f"感測點索引越界 [0,{N})：{idx.min()}..{idx.max()}")
    return idx


def _selftest() -> None:
    """孿生實驗：真值與 ensemble 由同一 solver 生成，故無 model error。

    判準是「同化後誤差顯著低於未同化的自由跑」，不是絕對誤差——後者取決於
    觀測密度與濾波器參數，訂一個門檻等於把調參結果寫死進測試。
    """
    from pi_lnn_jax.enkf import EnKF
    from pi_lnn_jax.kolmogorov_solver import KolmogorovSolver

    N, Ne, dt, n_sub = 32, 24, 2.0e-3, 25
    rng = np.random.default_rng(0)
    solver = KolmogorovSolver(N=N, nu=1e-3, L=1.0, k_f=2, forcing_amplitude=0.1)

    truth = band_limited_ensemble(rng, N, 1, k_cut=6, amplitude=1.0)[0]
    truth = solver.integrate(truth, dt=dt, n_steps=400)          # 甩掉初始暫態

    obs_ij = np.stack(np.unravel_index(
        rng.choice(N * N, 64, replace=False), (N, N)), axis=1)
    f = EnKF(solver, obs_ij, obs_std=0.05, inflation=1.05,
             loc_radius=0.3, seed=1)

    ens = band_limited_ensemble(rng, N, Ne, k_cut=6, amplitude=1.0)
    ens = np.stack([solver.integrate(w, dt=dt, n_steps=400) for w in ens])
    free = ens.mean(axis=0).copy()                               # 未同化對照

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    for _ in range(30):
        truth = solver.integrate(truth, dt=dt, n_steps=n_sub)
        ens = f.forecast(ens, dt=dt, n_steps=n_sub)
        ens = f.analyze(ens, f.observe(truth))
        free = solver.integrate(free, dt=dt, n_steps=n_sub)

    e_da, e_free = rel(ens.mean(axis=0), truth), rel(free, truth)
    spread = ens.std(axis=0).mean()
    print(f"[selftest] 同化後 rel-L2={e_da:.3e}  自由跑 rel-L2={e_free:.3e}  "
          f"ensemble spread={spread:.3e}")
    assert e_da < 0.5 * e_free, f"同化未改善狀態估計：{e_da:.3e} vs 自由跑 {e_free:.3e}"
    assert spread > 0.0, "ensemble 已塌陷"
    print("[selftest] PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor-json")
    ap.add_argument("--dns-path")
    ap.add_argument("--config", default=None,
                    help="訓練 config TOML：follow_training 由它的 time_strides 決定取樣")
    ap.add_argument("--protocol", choices=[m.value for m in ProtocolMode],
                    help="評估協定（必填，無預設）")
    ap.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    ap.add_argument("--sensor-time-stride", type=int, default=None,
                    help="明示 stride；follow_training 下與 config 不一致即失敗")
    ap.add_argument("--sensor-T", type=int, default=200)

    # ── 物理：必須與產生 DNS 的設定一致，不從檔名猜 ──────────────────
    ap.add_argument("--re", type=float, help="Reynolds number；ν=1/Re（solver 與 band/γ 共用）")
    ap.add_argument("--k-f", type=int, help="forcing 波數，須同 DNS")
    ap.add_argument("--forcing-amplitude", type=float, help="forcing 振幅 A，須同 DNS")

    # ── 濾波器：全部必填（見模組 docstring 的理由）──────────────────
    ap.add_argument("--n-members", type=int, help="ensemble 大小 N_e")
    ap.add_argument("--obs-err-frac", type=float,
                    help="觀測誤差佔感測 RMS 的比例，決定 R；不對觀測加噪")
    ap.add_argument("--inflation", type=float, help="乘性膨脹係數")
    ap.add_argument("--loc-radius", type=float, help="Gaspari–Cohn 局域化半徑（域長單位）")
    ap.add_argument("--solver-dt", type=float, help="forward model 時間步")
    ap.add_argument("--spinup-frac", type=float,
                    help="評估時捨棄的前段比例（thesis 用 0.5）")
    ap.add_argument("--init-k-cut", type=int, help="初始 ensemble 的帶限波數")
    ap.add_argument("--init-amplitude", type=float, help="初始 ensemble 的渦度標準差")
    ap.add_argument("--init-burnin-steps", type=int,
                    help="初始 ensemble 起跑前的自由演化步數")
    ap.add_argument("--seed", type=int, help="ensemble 與擾動觀測的亂數種子")

    ap.add_argument("--grid-index-tol", type=float, default=1e-6,
                    help="感測點落點容差（格點單位）；超過即失敗")
    ap.add_argument("--forecast-drift-only", action="store_true",
                    help="只量 forward model 的單區間漂移，不跑同化。用於回答"
                         "「從參考快照出發，一個觀測間隔後偏離多少」")
    ap.add_argument("--output", default="artifacts/baseline_eval/enkf_metrics.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    required = ["sensor_json", "dns_path", "protocol", "re", "k_f", "forcing_amplitude",
                "solver_dt"]
    if not a.forecast_drift_only:
        # 濾波器旋鈕只在真的要同化時必填；drift 量測不牽涉 ensemble
        required += ["n_members", "obs_err_frac", "inflation", "loc_radius",
                     "spinup_frac", "init_k_cut", "init_amplitude",
                     "init_burnin_steps", "seed"]
    missing = [f"--{n.replace('_', '-')}" for n in required if getattr(a, n) is None]
    if missing:
        ap.error("以下參數必填且無預設（見模組 docstring）：" + " ".join(missing))
    if not a.forecast_drift_only and not 0.0 <= a.spinup_frac < 1.0:
        ap.error(f"--spinup-frac 需在 [0,1)，收到 {a.spinup_frac}")

    from pi_lnn_jax.enkf import EnKF
    from pi_lnn_jax.evaluation_protocol import (
        load_for_evaluation,
        resolve_protocol,
        training_time_strides_from_config,
    )
    from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity
    from pi_lnn_jax.kolmogorov_solver import KolmogorovSolver
    from pi_lnn_jax.metric_artifact import write_compatibility_projection

    nu = 1.0 / float(a.re)
    protocol = resolve_protocol(
        mode=a.protocol,
        training_time_strides=training_time_strides_from_config(a.config) if a.config else [],
        cli_time_stride=a.sensor_time_stride, sensor_T=a.sensor_T,
        reason=a.protocol_reason)
    # grid_stride/max_grid 刻意不開放：forward model 必須跑在真實網格上，
    # 子取樣後的評估網格與 solver 網格不一致會讓觀測算子取錯點。
    d = load_for_evaluation(a.sensor_json, a.dns_path, protocol=protocol,
                            viscosity=nu, grid_stride=1, max_grid=0)

    sensor_phys, pos = d.sensor_phys, d.sensor_pos
    dns_u, dns_v = d.dns_u_eval, d.dns_v_eval
    T, Np, K = d.T, d.Nprime, pos.shape[0]
    times = np.asarray(d.context.times, dtype=np.float64)
    if times.shape[0] != T:
        raise RuntimeError(f"context.times 長度 {times.shape[0]} != T {T}")

    obs_ij = positions_to_grid_indices(pos, Np, tol=a.grid_index_tol)
    solver = KolmogorovSolver(N=Np, nu=nu, L=1.0, k_f=a.k_f,
                              forcing_amplitude=a.forcing_amplitude)

    # 觀測間隔 → 每個同化區間的 solver 步數。非整數即失敗：湊整會讓同化時刻
    # 與觀測時刻錯開，而錯開的後果是誤差看起來像模型誤差。
    dts = np.diff(times)
    if dts.size == 0:
        raise RuntimeError("評估時間軸只有一個時刻，無法同化")
    # 容差依 float32 的 ULP 訂，不用固定常數。sensor 時間戳以 float32 儲存
    # （sensor_dt 實際是 0.10000000149011612 = float32(0.1)），故等距軸的相鄰
    # 間隔本來就會有量化抖動，且**抖動大小隨 |t| 增長**——Re=100 的視窗到
    # t≈55.8，那裡的 float32 間距是 t≈10 處的五倍以上。訂固定常數會在長視窗上
    # 假紅、或在短視窗上失去鑑別力。以 ULP 為尺度則兩端都成立，且仍能抓出真正
    # 的不等距：漏幀或 stride 改變會讓間隔差至少一個完整區間。
    dt_obs = float(dts.mean())
    spread = float(dts.max() - dts.min())
    quantum = float(np.spacing(np.float32(np.abs(times).max())))
    tol_abs = max(8.0 * quantum, 1e-12)
    if dt_obs <= 0:
        raise RuntimeError(f"觀測間隔非正：{dt_obs:g}")
    if spread > tol_abs:
        raise RuntimeError(
            f"觀測時間非等距（{dts.min():.9g}..{dts.max():.9g}，散布 {spread:.3e} > "
            f"容差 {tol_abs:.3e} ＝ 8 float32-ULP@t_max）——本 driver 假設等距同化區間")
    n_sub_f = dt_obs / a.solver_dt
    n_sub = int(round(n_sub_f))
    # 同上：比值帶著同一份量化誤差，故判準也用絕對時間尺度而非固定相對值
    if n_sub < 1 or abs(n_sub_f - n_sub) * a.solver_dt > tol_abs:
        raise RuntimeError(
            f"觀測間隔 {dt_obs:.9g} 不是 solver-dt {a.solver_dt:g} 的整數倍"
            f"（比值 {n_sub_f:.6f}，偏差 {abs(n_sub_f - n_sub) * a.solver_dt:.3e} > "
            f"容差 {tol_abs:.3e}）")

    if a.forecast_drift_only:
        # 「從參考快照出發，一個觀測間隔後偏離多少」——量的是 forward model 相對
        # 參考 DNS 的模型誤差，與 ensemble、局域化、觀測噪音都無關。
        # 每一幀各自從真值起跑（不累積），故量到的是單區間漂移而非長期發散。
        drifts = []
        for t_i in range(T - 1):
            w0 = solver.curl(np.asarray(dns_u[t_i], dtype=np.float64),
                             np.asarray(dns_v[t_i], dtype=np.float64))
            w1 = solver.integrate(w0, dt=a.solver_dt, n_steps=n_sub)
            u1, v1 = solver.velocity(w1)
            ur, vr = np.asarray(dns_u[t_i + 1], np.float64), np.asarray(dns_v[t_i + 1], np.float64)
            num = np.sqrt(np.sum((u1 - ur) ** 2 + (v1 - vr) ** 2))
            den = np.sqrt(np.sum(ur ** 2 + vr ** 2))
            drifts.append(float(num / den))
        d = np.asarray(drifts)
        # curl→velocity 的往返誤差是這個量測的地板；不報它就無法判斷 drift 是否可信
        u0, v0 = solver.velocity(solver.curl(np.asarray(dns_u[0], np.float64),
                                             np.asarray(dns_v[0], np.float64)))
        rt = float(np.sqrt(np.sum((u0 - dns_u[0]) ** 2 + (v0 - dns_v[0]) ** 2))
                   / np.sqrt(np.sum(np.asarray(dns_u[0], np.float64) ** 2
                                    + np.asarray(dns_v[0], np.float64) ** 2)))
        print(f"[drift] 單觀測間隔（{dt_obs:.4g} s，{n_sub} 步）相對 L2 漂移，"
              f"逐幀各自從真值起跑，n={d.size}")
        print(f"[drift]   mean={d.mean()*100:.4f}%  median={np.median(d)*100:.4f}%  "
              f"min={d.min()*100:.4f}%  max={d.max()*100:.4f}%")
        print(f"[drift]   curl→velocity 往返誤差（量測地板）={rt*100:.4g}%")
        out_path = Path(a.output); out_path.parent.mkdir(parents=True, exist_ok=True)
        write_compatibility_projection(out_path, {"forecast_drift": {
            "definition": "relative L2 of (u,v) after integrating the forward model "
                          "one observation interval from a reference DNS snapshot",
            "assimilation_interval_s": dt_obs, "solver_steps": n_sub,
            "solver_dt": a.solver_dt, "n_frames": int(d.size),
            "mean": float(d.mean()), "median": float(np.median(d)),
            "min": float(d.min()), "max": float(d.max()),
            "curl_velocity_roundtrip_floor": rt,
            "reynolds": a.re, "k_f": a.k_f, "forcing_amplitude": a.forcing_amplitude,
            "sensor_json": str(Path(a.sensor_json).resolve()),
            "dns_path": str(Path(a.dns_path).resolve()),
        }})
        print(f"[out] {a.output}")
        return

    obs_rms = float(np.sqrt(np.mean(np.asarray(sensor_phys, dtype=np.float64) ** 2)))
    obs_std = a.obs_err_frac * obs_rms
    if not np.isfinite(obs_std) or obs_std <= 0:
        raise RuntimeError(f"觀測誤差不合法：obs_rms={obs_rms:g} frac={a.obs_err_frac:g}")

    print(f"[data] T={T} K={K} grid={Np}x{Np} protocol={protocol.mode.value} "
          f"stride={protocol.sensor_time_stride}")
    print(f"[filter] Ne={a.n_members} dt_obs={dt_obs:g} n_sub={n_sub} "
          f"obs_std={obs_std:.4g} (={a.obs_err_frac:g}×RMS {obs_rms:.4g}) "
          f"infl={a.inflation:g} loc={a.loc_radius:g}")

    rng = np.random.default_rng(a.seed)
    f = EnKF(solver, obs_ij, obs_std=obs_std, inflation=a.inflation,
             loc_radius=a.loc_radius, seed=a.seed)

    ens = band_limited_ensemble(rng, Np, a.n_members,
                                k_cut=a.init_k_cut, amplitude=a.init_amplitude)
    if a.init_burnin_steps > 0:
        ens = np.stack([solver.integrate(w, dt=a.solver_dt, n_steps=a.init_burnin_steps)
                        for w in ens])

    u_rec = np.empty((T, Np, Np), dtype=np.float64)
    v_rec = np.empty((T, Np, Np), dtype=np.float64)
    t0 = time.perf_counter()
    for t in range(T):
        if t > 0:
            ens = f.forecast(ens, dt=a.solver_dt, n_steps=n_sub)
        y_obs = np.concatenate([sensor_phys[t, :, 0], sensor_phys[t, :, 1]])
        ens = f.analyze(ens, y_obs)
        u_rec[t], v_rec[t] = solver.velocity(ens.mean(axis=0))
        if not np.isfinite(u_rec[t]).all() or not np.isfinite(v_rec[t]).all():
            raise RuntimeError(f"濾波器在 t={t}（time={times[t]:g}）發散——"
                               "檢查 --solver-dt 與 --inflation")
    wall_s = time.perf_counter() - t0

    # spin-up：只切評估窗，不改同化過程（濾波器仍看過整段觀測）
    t_start = int(round(a.spinup_frac * T))
    if t_start >= T:
        raise RuntimeError(f"--spinup-frac {a.spinup_frac} 切光了評估窗（T={T}）")
    sl = slice(t_start, T)
    context = dataclasses.replace(d.context, times=tuple(times[sl].tolist()))
    n_eval = T - t_start
    print(f"[eval] 捨棄前 {t_start}/{T} 幀為 spin-up，評估 {n_eval} 幀 "
          f"（t={times[t_start]:g}..{times[-1]:g}）")

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recorder = EvaluationRunRecorder("scripts/enkf_baseline.py", out_path)
    projection = recorder.record(
        u_rec[sl], v_rec[sl], dns_u[sl], dns_v[sl],
        context=context, protocol=protocol,
        identity=RunArtifactIdentity(method="enkf"),
        inputs=(
            ("sensor", str(Path(a.sensor_json).resolve())),
            ("dns", str(Path(a.dns_path).resolve())),
        ),
        details_extras={
            "method": "enkf",
            "reynolds": a.re,
            "viscosity": nu,
            "k_f": a.k_f,
            "forcing_amplitude": a.forcing_amplitude,
            "K": int(K),
            "n_members": a.n_members,
            "obs_err_frac": a.obs_err_frac,
            "obs_rms": obs_rms,
            "obs_std": obs_std,
            "observations_perturbed": False,
            "inflation": a.inflation,
            "loc_radius": a.loc_radius,
            "solver_dt": a.solver_dt,
            "assimilation_interval": dt_obs,
            "assimilation_interval_spread": spread,
            "assimilation_interval_tol": tol_abs,
            "solver_steps_per_interval": n_sub,
            "forward_model": "pi_lnn_jax.kolmogorov_solver.KolmogorovSolver (IF-RK4, 2/3 dealias)",
            "init_k_cut": a.init_k_cut,
            "init_amplitude": a.init_amplitude,
            "init_burnin_steps": a.init_burnin_steps,
            "init_from_truth": False,
            "seed": a.seed,
            "spinup_frac": a.spinup_frac,
            "spinup_frames_discarded": t_start,
            "frames_evaluated": n_eval,
            "sensor_time_stride": a.sensor_time_stride,
            "sensor_T": a.sensor_T,
        },
        run_measurements={"wall_s": wall_s,
                          "recon_s_per_field": wall_s / float(T)},
    )
    print("[out] canonical artifact written (enkf)")

    mean, ket = projection["metrics_mean"], projection["ke_t_errors"]
    keys = ["uv_rel_err", "u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err",
            "ke_pw_mape", "ke_pw_nmae", "band_rel_err_low", "band_rel_err_mid",
            "gamma_low", "gamma_mid"]
    agg = {k: (float(mean[k]) if k in mean and np.isfinite(mean[k]) else None)
           for k in keys}
    agg["ke_t_mape"] = float(ket["ke_t_mape"])
    agg["ke_t_mape_spatialmean"] = float(ket["ke_t_mape_spatialmean"])
    agg["ke_t_rel_l2"] = float(ket["ke_t_rel_l2"])
    agg["ke_mape_def"] = ket["ke_mape_def"]
    agg["wall_s"] = wall_s

    # 主指標是 uv_rel_err（2026-08 指標重設計：KE 降級為 QoI）；KE 的 MAPE 一律
    # 用 pointwise。`ke_t_mape_spatialmean` 仍寫進 artifact，但只為與改定義之前
    # 已發表的數字對照，不作為 headline——它先把整場塌成純量再比，空間分布錯誤
    # 會被抵消（見 metric_artifact.energy_timeseries_errors 的 docstring）。
    print(f"[enkf] uv={agg['uv_rel_err']*100:.2f}%  u={agg['u_rel_err']*100:.2f}%  "
          f"v={agg['v_rel_err']*100:.2f}%  w={agg['omega_rel_err']*100:.2f}%  "
          f"KE-pw-MAPE={agg['ke_t_mape']*100:.2f}%  wall={wall_s:.1f}s")
    write_compatibility_projection(out_path, {"enkf": agg})
    print(f"[out] {a.output}")


if __name__ == "__main__":
    main()
