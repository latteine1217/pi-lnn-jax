#!/usr/bin/env python3
"""diag_spacetime_observability.py — 時空觀測算子的奇異譜（EXP-513）。

What:
    在真實軌跡上線性化「初始渦度 → sensor 速度軌跡」這個映射，
        Φ : δω0 ↦ {H(∂S_{t_j} δω0)}_{j=1..nf} ∈ R^{2·K·nf}
    分別把它限制在 low / mid 兩個 sensor-band 的輸入子空間上顯式建構，取奇異譜。
    Φ 的 frame 下界即 dynamical sampling 的 β_{K,T}（見
    knowledge/deep-research/spacetime-observability-nse/），**與優化器無關**。

Why:
    EXP-512 證明 4D-Var oracle 的數字是（問題＋優化器＋預算）的性質——加預算會讓
    E_oracle_mid 從 16.63% 惡化到 22.27%。要回答「中頻可不可識別」，必須拿掉優化器。
    根因是控制變數選錯：整個 ω0 對 4000 個觀測欠定 16.4×，但 k≤16（整個目標頻帶）
    只有 796 個自由度，其實過定 5.03×；欠定全來自 sensor 看不到的 k>16。

Convention:
    - band 邊界與 `diag_identifiability` 一致：k_s = √(K/π)，mid = (k_s, mid_hi]。
    - 基底是**實場**的正交基：每個共軛對出 cos/sin 兩個實自由度，故
      「實自由度數 = 該 band 的格點數」，與事前登錄的計數一致。
    - 前向算子直接重用 `fourdvar_identifiability.make_forward`，確保與 oracle 同一條路徑。

驗證階梯（`--gate` 逐關跑，不過不得讀後續數字）:
    1. adjoint：⟨Φv,u⟩ = ⟨v,Φᵀu⟩ 至 float64 精度。
    2. linearization：ε→0 時 Φ(εδ) 與有限差分的相對誤差 →0。窗長 t=5 僅 2.5 個渦翻轉、
       λ_max=0.135 ⟹ 擾動放大 e^{0.675}≈2，先驗上線性化應成立——但必須實測。
    3. spectrum：low / mid 的奇異譜與有效秩 r(η)。

⚠️ 長診斷，走 lab-server r740 sbatch，不在本機跑。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pi_lnn_jax.data import _resolve_data_path, load_dns_from_path  # noqa: E402
from scripts.diag_identifiability import (  # noqa: E402
    band_filter,
    radial_wavenumber_grid,
    sensor_band_edge,
)
from scripts.fourdvar_identifiability import (  # noqa: E402
    _resolve,
    _sensor_indices,
    make_forward,
)

_HIGH_BAND_LO_DEFAULT = 16.0


def band_real_basis(N: int, kr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """該 band 的實場正交基 [n_dof, N, N]。

    每個共軛對 (k, -k) 貢獻 cos/sin 兩個實自由度，故 n_dof == band 內格點數
    ——與事前登錄的自由度計數同一個定義，不得改用其他計數。
    """
    kx = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    # DC（k=0）排除：渦度場 ∮ω=0 恆成立，均值不是自由度。`band_filter` 以 lo=-1.0
    # 表示低頻帶時會含入 DC，此處刻意不含——事前登錄的 low=96 / mid=700 即此定義。
    sel = (kr > max(lo, 0.0)) & (kr <= hi)
    # 每個共軛對只取一個代表：kx>0，或 kx==0 且 ky>0
    canon = sel & ((KX > 0) | ((KX == 0) & (KY > 0)))
    idx = np.argwhere(canon)
    g = np.arange(N) / N
    X, Y = np.meshgrid(g, g, indexing="ij")
    cols = []
    for i, j in idx:
        phase = 2.0 * np.pi * (KX[i, j] * X + KY[i, j] * Y)
        for f in (np.cos(phase), np.sin(phase)):
            n = np.linalg.norm(f)
            if n > 0:
                cols.append(f / n)
    B = np.asarray(cols, dtype=np.float64)
    n_lattice = int(sel.sum())
    if B.shape[0] != n_lattice:
        raise RuntimeError(
            f"basis 維度 {B.shape[0]} != band 格點數 {n_lattice}（lo={lo}, hi={hi}）"
            "——共軛對代表選取有誤，計數與事前登錄不一致，中止。"
        )
    return B


def _load(dns_path: str, sensor_json: str, n_frames: int, dt: float, nu, k_f, A):
    u_dns, v_dns, t_dns = load_dns_from_path(dns_path, time_stride=2)
    obj = np.load(_resolve(dns_path), allow_pickle=True).item()
    omega0 = np.asarray(obj["omega"], np.float64)[::2][0]
    N = np.asarray(u_dns).shape[-1]
    steps_per_frame = round(float(t_dns[1] - t_dns[0]) / dt)
    # 偏離 `scripts/CLAUDE.md` §2「讀 sensor 用 data.load_sensors_from_path」的理由：
    # 那支會強制定位並載入 sensor 值的 NPZ，而本診斷的觀測是由前向算子從 DNS 算出來的，
    # 不讀該 NPZ；硬套會引入無謂依賴並可能無故失敗。故取它的兩個實質保障——
    # `_resolve_data_path`（不自己 join 資料根）與 shape 驗證——但不取 NPZ 耦合。
    # 座標讀法與 `fourdvar_identifiability` 逐字相同，確保與 oracle 的 sensor 集可比。
    sensor_path = _resolve_data_path(sensor_json)
    if not sensor_path.exists():
        raise FileNotFoundError(f"sensor JSON not found: {sensor_path}")
    meta = json.load(open(sensor_path))
    coords = np.asarray(meta["selected_coordinates"], np.float64)
    K = int(meta["K"])
    if coords.shape != (K, 2):
        raise ValueError(f"sensor coordinates shape {coords.shape} != (K={K}, 2)")
    forward = make_forward(N, dt, steps_per_frame, nu, k_f, A, _sensor_indices(coords, N))

    def F(w0):
        return forward(w0, n_frames).reshape(-1)

    return F, omega0, N, K, steps_per_frame


def gate_adjoint(F, w0, N, rng) -> dict:
    """關卡 1：jvp 與 vjp 互為伴隨。不過此關，後續數字全部無意義。"""
    v = jnp.asarray(rng.standard_normal((N, N)))
    out, jv = jax.jvp(F, (jnp.asarray(w0),), (v,))
    u = jnp.asarray(rng.standard_normal(out.shape))
    (jtu,) = jax.vjp(F, jnp.asarray(w0))[1](u)
    lhs, rhs = float(jnp.vdot(jv, u)), float(jnp.vdot(v, jtu))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    return {"lhs": lhs, "rhs": rhs, "rel_err": rel, "pass": rel < 1e-10}


def gate_linearization(F, w0, B_mid: np.ndarray, omega_mid: np.ndarray, rng, eps_list) -> dict:
    """關卡 2：線性化在**我們實際要探測的擾動**上是否成立。

    2026-08-29 修正（job 5390 gate2 失敗後）。舊版用**白噪聲**方向，實測該方向有
    65.5% 能量在 k>85（dealias mask 當場清零）、33.4% 在 16<k≤85（黏性在 t=5 內
    以 e^{-500} 量級消滅），只有 1.16% 落在 k≤16。分母 ‖lin‖ 因此由 ~1% 的能量決定，
    相對誤差被除法放大成捨入噪聲——那是**儀器缺陷**，與線性化是否成立無關。
    該缺陷可獨立於失敗結果驗證（純頻譜計算，見事前登錄的修正段）。

    兩個子測試：
      (a) 有限差分 U 曲線，方向取自 **mid-band 基底**的隨機單位組合。
          判準不是「單調下降」——有限差分的捨入誤差必然在小 ε 反轉，那是數值常識，
          不是物理。改判「掃描中的**最小**相對誤差夠小」，並輸出整條曲線供人判讀。
      (b) **物理幅度測試（真正承重的那個）**：擾動取真實場自己的 mid-band 內容，
          比較非線性響應與線性響應。問的不是「導數對不對」（gate1 的伴隨檢查已
          回答），而是「線性化能不能描述我們在乎的那個幅度的擾動」。
    """
    w0j = jnp.asarray(w0)
    base = F(w0j)

    # (a) 有限差分 U 曲線，方向在 mid-band 內
    c = rng.standard_normal(B_mid.shape[0])
    d = np.tensordot(c / np.linalg.norm(c), B_mid, axes=(0, 0))
    d = d / np.linalg.norm(d)
    dj = jnp.asarray(d)
    _, lin = jax.jvp(F, (w0j,), (dj,))
    rows = []
    for eps in eps_list:
        fd = (F(w0j + eps * dj) - base) / eps
        rows.append({"eps": eps,
                     "rel_err": float(jnp.linalg.norm(fd - lin) / jnp.linalg.norm(lin))})
    fd_min = min(r["rel_err"] for r in rows)

    # (b) 物理幅度：擾動 = 真實場自己的 mid-band 內容
    dm = jnp.asarray(omega_mid)
    _, lin_m = jax.jvp(F, (w0j,), (dm,))
    nl_m = F(w0j + dm) - base
    phys = float(jnp.linalg.norm(nl_m - lin_m) / jnp.linalg.norm(lin_m))

    return {"fd_rows": rows, "fd_min_rel_err": fd_min,
            "physical_amp_rel_err": phys,
            "omega_mid_norm_frac": float(np.linalg.norm(omega_mid) / np.linalg.norm(w0)),
            "pass": fd_min < 1e-6 and phys < 0.2}


def band_spectrum(F, w0, B: np.ndarray, chunk: int) -> np.ndarray:
    """顯式建構 Φ·Bᵀ（每列一個基底方向）並回傳奇異值。"""
    w0j = jnp.asarray(w0)

    def col(v):
        return jax.jvp(F, (w0j,), (v,))[1]

    batched = jax.jit(jax.vmap(col))
    outs = []
    for s in range(0, B.shape[0], chunk):
        outs.append(np.asarray(batched(jnp.asarray(B[s:s + chunk]))))
        print(f"    [{min(s + chunk, B.shape[0])}/{B.shape[0]}]", flush=True)
    M = np.concatenate(outs, axis=0)          # [n_dof, n_obs]
    return np.linalg.svd(M, compute_uv=False)


def effective_rank(sig: np.ndarray, ref: float, etas) -> dict:
    return {f"{e:g}": int((sig > e * ref).sum()) for e in etas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", required=True)
    ap.add_argument("--sensor-json", required=True)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--nu", type=float, default=1e-4)
    ap.add_argument("--k_f", type=int, default=2)
    ap.add_argument("--A", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=2.5e-4)
    ap.add_argument("--mid-hi", type=float, default=_HIGH_BAND_LO_DEFAULT)
    ap.add_argument("--chunk", type=int, default=25, help="vmap 批次（記憶體與速度的取捨）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", choices=["adjoint", "linearization", "spectrum", "all"],
                    default="all", help="驗證階梯逐關；不過關不得讀後續數字")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    F, omega0, N, K, spf = _load(args.dns, args.sensor_json, args.n_frames,
                                 args.dt, args.nu, args.k_f, args.A)
    kr = radial_wavenumber_grid(N)
    k_s = sensor_band_edge(K)
    rng = np.random.default_rng(args.seed)
    n_obs = 2 * K * args.n_frames
    print(f"[cfg] N={N} K={K} n_frames={args.n_frames} steps/frame={spf} "
          f"k_s={k_s:.4f} mid_hi={args.mid_hi} n_obs={n_obs}", flush=True)

    out: dict = {"N": N, "K": K, "n_frames": args.n_frames, "k_s": k_s,
                 "mid_hi": args.mid_hi, "n_obs": n_obs, "seed": args.seed}

    if args.gate in ("adjoint", "all"):
        g1 = gate_adjoint(F, omega0, N, rng)
        out["gate_adjoint"] = g1
        print(f"[gate1 adjoint] rel_err={g1['rel_err']:.3e} pass={g1['pass']}", flush=True)
        if not g1["pass"]:
            raise SystemExit("gate1 FAILED：jvp/vjp 非伴隨，後續數字無意義。")

    if args.gate in ("linearization", "all"):
        B_mid = band_real_basis(N, kr, k_s, args.mid_hi)
        omega_mid = band_filter(omega0[None], kr, k_s, args.mid_hi)[0]
        g2 = gate_linearization(F, omega0, B_mid, omega_mid, rng,
                                [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        out["gate_linearization"] = g2
        for r in g2["fd_rows"]:
            print(f"[gate2 fd ] eps={r['eps']:.0e} rel_err={r['rel_err']:.3e}", flush=True)
        print(f"[gate2 fd ] min_rel_err={g2['fd_min_rel_err']:.3e}", flush=True)
        print(f"[gate2 phys] 真實 mid-band 擾動（‖δ‖/‖ω0‖={g2['omega_mid_norm_frac']:.3e}）"
              f" 非線性 vs 線性 rel_err={g2['physical_amp_rel_err']:.3e}", flush=True)
        print(f"[gate2] pass={g2['pass']}", flush=True)
        if not g2["pass"]:
            raise SystemExit("gate2 FAILED：線性化在此窗長不成立，EXP-513 的前提不滿足。")

    if args.gate in ("spectrum", "all"):
        for name, lo, hi in (("low", -1.0, k_s), ("mid", k_s, args.mid_hi)):
            B = band_real_basis(N, kr, lo, hi)
            print(f"[{name}] band=({lo:.2f},{hi:.2f}] 實自由度={B.shape[0]}", flush=True)
            sig = band_spectrum(F, omega0, B, args.chunk)
            out[f"sigma_{name}"] = sig.tolist()
            out[f"n_dof_{name}"] = int(B.shape[0])
        ref = float(out["sigma_low"][0])
        etas = [1e-2, 1e-3, 1e-4]
        for name in ("low", "mid"):
            sig = np.asarray(out[f"sigma_{name}"])
            out[f"eff_rank_{name}"] = effective_rank(sig, ref, etas)
            print(f"[{name}] sigma_1={sig[0]:.6e}  sigma_min={sig[-1]:.6e}  "
                  f"eff_rank={out[f'eff_rank_{name}']}  (ref=sigma_1^low)", flush=True)
        out["sigma1_mid_over_low"] = float(out["sigma_mid"][0]) / ref
        print(f"[ratio] sigma_1^mid / sigma_1^low = {out['sigma1_mid_over_low']:.6e}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
