#!/usr/bin/env python3
"""diag_observability.py — 部署 sensors 的線性可觀測性與資訊量診斷（DNS-free）。

What:
    對「實際部署」的固定 sensor 組（預設 les_qr K=100, t0-20），量化 velocity-only
    線性可觀測性與資訊量，用兩種 DNS-free basis 對照：
      A. LES-POD basis（data-driven 替身；POD modes 來自 LES 全場，非 DNS）
      B. divergence-free Fourier basis（解析、零資料；stream-function 構造）
    對每個 basis 的 sensing operator CΦ 報告兩層指標：
      幾何層（observability_metrics）：rank(tol)、κ=σ_max/σ_min、奇異值譜、
        null-space fraction、effective rank（spectral entropy η=exp(-Σ p log p)）。
      資訊量層（information_metrics）：D/A/E-optimality（log det / Σσ⁻² / σ_min²）、
        以及 POD 能量加權的 mutual information I(a;y)（linear-Gaussian 觀測模型）。

Theory（五層論證，本腳本只覆蓋可由 DNS-free 線性理論驗證的中間三層）:
    1. Nyquist            — 均勻網格 anti-aliasing 的保守 spectral reference（非 sparse 下界）。
    2. POD effective rank — flow snapshots 的有效自由度（r_95/r_99/r_99.9，cumulative energy）。
    3. QR-pivot placement — sensor 位置依 dominant POD subspace 而選，非任意。
    4. κ(CΦ_r) + 資訊量   — 量化這組位置對 dominant subspace 的觀測穩定度與資訊量。
    5. nonlinear K-sweep  — 真正的 sensor-count 充分性，須用完整 DeepONet/CfC 模型實測。
    關鍵隔離：第 2-4 層是 linear POD sparse-sensing theory，只「診斷與背書 sensor
    placement」，不是非線性 DeepONet/CfC reconstruction 的形式保證；count 充分性由
    第 5 層的 nonlinear K-sweep error plateau 提供。本腳本輸出第 2-4 層的數字。

Why:
    既有 pi-lnn 有 POD recon error 與 div-free null-space，但都沒報 sensing operator
    的 condition number κ 與資訊量，且 POD 那支用 DNS-POD（oracle）、非部署 sensors。
    本腳本補這個 delta：κ + 資訊量 + LES-POD（DNS-free）+ 部署 sensors，直接背書
    sec:placement「placement depends on leading-POD conditioning」目前無數字的斷言。

    範圍：本檔只做 velocity-only 線性可觀測性。pressure（p-in-sensor）的可觀測性
    是非線性的（∇²p = -∂_i u_j ∂_j u_i 二次於速度），需 locally-linearized Jacobian
    J_p(ā)，留待後續以 JAX autodiff 實作（見 TODO）。

Convention:
    場 array convention 對齊 pi-lnn DNS/LES：u[t, axis_1=x, axis_2=y]，domain [0,L)²。
    sensor coord (cx, cy) → ix=argmin|cx-xg|, iy=argmin|cy-yg| → 取 u[:, ix, iy]。

div-free Fourier sampling matrix（A_div）構造 port 自
    ../pi-lnn/scripts/under_determined_proof_divfree.py（stream-function 基底），
    額外帶入 2π 物理因子（對 κ/rank/null 不變，因 κ 為比值）。

Usage（真實 LES，依需求在本機或 lab-server 跑）:
    PYTHONPATH=. \\
      uv run python scripts/diag_observability.py \\
        --sensor-json data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path

_DEFAULT_LES = _resolve_data_path("data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy")


# ── grid / sensor 對位 ────────────────────────────────────────────────────────

def grid_axes(N: int, L: float = 1.0) -> np.ndarray:
    """均勻週期 grid 座標 [0, L)。x 與 y 共用（方域）。"""
    return np.arange(N, dtype=np.float64) * (L / N)


def coords_to_xy_indices(coords: np.ndarray, N: int, L: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """sensor 物理座標 (K,2)=(x,y) → grid index (ix, iy)。

    對齊 u[t, axis_1=x, axis_2=y]：第 0 欄是 x→ix（axis_1），第 1 欄是 y→iy（axis_2）。
    """
    g = grid_axes(N, L)
    ix = np.argmin(np.abs(coords[:, 0:1] - g[None, :]), axis=1)
    iy = np.argmin(np.abs(coords[:, 1:2] - g[None, :]), axis=1)
    return ix.astype(int), iy.astype(int)


# ── LES-POD basis ─────────────────────────────────────────────────────────────

def pod_basis(u: np.ndarray, v: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """leading-r POD modes（method-of-snapshots SVD）。

    Args:
        u, v: (T, N, N) 場時序。
        r: 保留模態數。
    Returns:
        Phi:   (2N², r) 正交 POD modes（u 段疊 v 段）。
        svals: (min(2N²,T),) 全奇異值（供 energy 累積判讀）。
    """
    T = u.shape[0]
    X = np.concatenate([u.reshape(T, -1).T, v.reshape(T, -1).T], axis=0)  # (2N², T)
    X = X - X.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :r], S


def effective_rank_thresholds(svals: np.ndarray, thresholds=(0.95, 0.99, 0.999)) -> dict:
    """POD 有效自由度：最小 r 使累積能量 Σσ_i²|_{i≤r} / Σσ² ≥ 門檻（理論第 2 層）。"""
    s2 = np.asarray(svals, dtype=np.float64) ** 2
    cum = np.cumsum(s2) / s2.sum()
    out = {}
    for thr in thresholds:
        idx = int(np.searchsorted(cum, thr))  # cum 單調遞增；first idx with cum≥thr
        out[thr] = (idx + 1) if idx < cum.size else None
    return out


def time_window_mask(time: np.ndarray, t_spinup: float = 0.0, t_max: float | None = None) -> np.ndarray:
    """共同時間窗 [t_spinup, t_max] 的布林遮罩（t_max=None → 不設上界）。

    用於在同窗對比不同資料源（如 LES t0-50 vs DNS t0-20），移除窗長混淆。
    """
    time = np.asarray(time, dtype=np.float64)
    mask = time >= t_spinup
    if t_max is not None:
        mask = mask & (time <= t_max)
    return mask


def split_time_halves(time: np.ndarray, t_spinup: float = 0.0, t_max: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """把窗 [t_spinup, t_max] 從時間中點切兩半，回傳 (mask_first, mask_second)。

    供同源 realization 對照：量「同一資料源兩個子窗的 leading-r POD 子空間差異」，作為
    principal-angle 的 realization-noise 基準線——cross-source 角度須相對此基準判讀，
    否則會把 realization variance 誤讀為 surrogate gap。
    """
    time = np.asarray(time, dtype=np.float64)
    hi = float(time.max()) if t_max is None else t_max
    mid = 0.5 * (t_spinup + hi)
    first = (time >= t_spinup) & (time < mid)
    second = (time >= mid) & (time <= hi)
    return first, second


def principal_angles(Phi_a: np.ndarray, Phi_b: np.ndarray) -> dict:
    """兩正交基所張子空間的 principal angles（subspace 對齊度，量化 surrogate gap）。

    σ_i = svd(Φ_aᵀ Φ_b) = cos θ_i ⇒ θ_i = arccos σ_i。對 POD 的 sign/子空間內旋轉
    簡併不變（比較子空間本身、非逐模態），故適合量 LES-POD ↔ DNS-POD 落差。
    Φ_a (D, r_a)、Φ_b (D, r_b) 需各自欄正交、同 ambient 維度 D；角數 = min(r_a, r_b)。
    """
    s = np.linalg.svd(Phi_a.T @ Phi_b, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)  # svd 降序；數值上可能微越界 [0,1]
    theta = np.degrees(np.arccos(s))
    return {
        "n_angles": int(s.size),
        "angles_deg": theta.tolist(),
        "theta_max_deg": float(theta.max()) if theta.size else 0.0,
        "theta_median_deg": float(np.median(theta)) if theta.size else 0.0,
        "theta_p95_deg": float(np.percentile(theta, 95)) if theta.size else 0.0,
        "mean_cos2": float((s ** 2).mean()) if s.size else 0.0,  # 子空間能量重疊（1=完全對齊）
    }


def transfer_verdict(r: int, excess_deg: float, rank_floor: int, excess_tol_deg: float = 5.0) -> str:
    """principal-angle transfer 的判讀（含 energetic-rank 護欄）。

    rank_floor = 兩源 leading energetic rank 的較小者。r 超過它 → 比較進入 noise floor
    （某源已無足夠能量模態），principal angle 不再反映子空間落差，標為不可靠；否則以
    excess（cross − 同源 realization 基準）判：≤tol → 無額外 surrogate gap，>tol → 真 gap。
    """
    if r > rank_floor:
        return "unreliable (r>energetic rank)"
    if excess_deg <= excess_tol_deg:
        return "no excess gap"
    return "excess surrogate gap"


def pod_observation_operator(Phi: np.ndarray, ix: np.ndarray, iy: np.ndarray, N: int) -> np.ndarray:
    """CΦ：把 POD modes 投影到 sensor 點的 (u,v)。回傳 (2K, r)。"""
    r = Phi.shape[1]
    Phi_u = Phi[: N * N].reshape(N, N, r)
    Phi_v = Phi[N * N:].reshape(N, N, r)
    return np.concatenate([Phi_u[ix, iy, :], Phi_v[ix, iy, :]], axis=0)  # (2K, r)


# ── divergence-free Fourier basis ─────────────────────────────────────────────

def divfree_modes(kmax: int) -> list[tuple[int, int]]:
    """k_x²+k_y² ≤ kmax² 的整數波向量（排除原點，含 ±q 全集）。"""
    modes = []
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky == 0:
                continue
            if kx * kx + ky * ky <= kmax * kmax:
                modes.append((kx, ky))
    return modes


def divfree_half_modes(kmax: int) -> list[tuple[int, int]]:
    """每組共軛對 (q, -q) 取一個代表（半平面 ky>0 或 ky==0 & kx>0）。

    真實 stream function ψ 滿足 ψ_{-q}=conj(ψ_q)，故獨立實 DoF = 2×|half modes|
    = |full modes|，而非把全集 ±q 的 re/im 都當獨立（會 double-count）。
    """
    half = []
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky == 0:
                continue
            if kx * kx + ky * ky > kmax * kmax:
                continue
            if ky > 0 or (ky == 0 and kx > 0):
                half.append((kx, ky))
    return half


def divfree_observation_operator(coords: np.ndarray, kmax: int, L: float = 1.0
                                 ) -> tuple[np.ndarray, list]:
    """stream-function div-free Fourier sensing matrix C_u。回傳 (A [2K, 2H], half_modes)。

    real ψ = Σ_{q∈half} [α_q cos φ_q + β_q sin φ_q]，φ_q=2π(k_x x+k_y y)；
    u_x=∂ψ/∂y, u_y=-∂ψ/∂x ⇒ ∇·u≡0。每個 half-mode 2 個實 DoF (α,β)，總 2H 實 DoF。
    """
    half = divfree_half_modes(kmax)
    H = len(half)
    K = coords.shape[0]
    A = np.zeros((2 * K, 2 * H), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for s, (xs, ys) in enumerate(coords):
        for q, (kx, ky) in enumerate(half):
            phase = two_pi * (kx * xs + ky * ys)
            cos_p, sin_p = np.cos(phase), np.sin(phase)
            # u_x = ∂ψ/∂y = Σ 2π k_y (-α sinφ + β cosφ)
            A[2 * s, 2 * q] = -two_pi * ky * sin_p       # ∂u_x/∂α
            A[2 * s, 2 * q + 1] = two_pi * ky * cos_p    # ∂u_x/∂β
            # u_y = -∂ψ/∂x = Σ 2π k_x (α sinφ - β cosφ)
            A[2 * s + 1, 2 * q] = two_pi * kx * sin_p    # ∂u_y/∂α
            A[2 * s + 1, 2 * q + 1] = -two_pi * kx * cos_p  # ∂u_y/∂β
    return A, half


def velocity_field_from_psi(psi_ab: np.ndarray, half_modes: list, N: int, L: float = 1.0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """stream-function 實係數 (α,β) per half-mode → (u_x, u_y) 全場 (N,N)。

    供 div-free 性質驗證與後續 J_p。係數排列與 divfree_observation_operator 一致。
    """
    g = grid_axes(N, L)
    X, Y = np.meshgrid(g, g, indexing="ij")  # axis_1=x, axis_2=y
    ux = np.zeros((N, N))
    uy = np.zeros((N, N))
    two_pi = 2.0 * np.pi
    for q, (kx, ky) in enumerate(half_modes):
        a, b = psi_ab[2 * q], psi_ab[2 * q + 1]
        phase = two_pi * (kx * X + ky * Y)
        common = -a * np.sin(phase) + b * np.cos(phase)
        ux += two_pi * ky * common
        uy += -two_pi * kx * common
    return ux, uy


# ── observability metrics ─────────────────────────────────────────────────────

def observability_metrics(C: np.ndarray, tol: float = 1e-10) -> dict:
    """sensing operator C (n_obs, n_dof) 的線性可觀測性指標。

    null_fraction: n_dof 中無法被 sensors 觀測的維度比例 = (n_dof - rank)/n_dof。
    """
    n_obs, n_dof = C.shape
    s = np.linalg.svd(C, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    smin_full = float(s[-1]) if s.size else 0.0
    rank = int((s > tol * smax).sum()) if smax > 0 else 0
    nz = s[s > tol * smax]
    smin_nz = float(nz[-1]) if nz.size else 0.0
    # cond: 真 condition number σ_max/σ_min。underdetermined（rank<n_dof，存在 null space）→ inf。
    # 注意 wide matrix 的 svd(compute_uv=False) 只回 min(m,n) 個 σ，s[-1] 是最小「非零」σ 而非 0，
    # 故不能只靠 smin_full>0 判滿秩，須顯式比對 rank>=n_dof。
    cond = float(smax / smin_full) if (rank >= n_dof and smin_full > 0) else float("inf")
    # cond_observable: 限制在可觀測子空間內的條件數（永遠有限，量化可觀測模態的病態程度）
    cond_obs = float(smax / smin_nz) if smin_nz > 0 else float("inf")
    p = (s ** 2) / (s ** 2).sum() if s.size and (s ** 2).sum() > 0 else np.array([1.0])
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return {
        "n_obs": int(n_obs),
        "n_dof": int(n_dof),
        "rank": rank,
        "cond": cond,
        "cond_observable": cond_obs,
        "sigma_max": smax,
        "sigma_min_nonzero": smin_nz,
        "null_fraction": float((n_dof - rank) / n_dof) if n_dof else 0.0,
        "effective_rank": eff_rank,
        "singular_values": s.tolist(),
    }


def information_metrics(C: np.ndarray, mode_energy: np.ndarray | None = None,
                        snr: float = 1e3, tol: float = 1e-10) -> dict:
    """sensing operator C=CΦ_r 的資訊量理論指標（linear-Gaussian 觀測模型）。

    把 observability_metrics 的條件數診斷升級為完整資訊量層。觀測模型：
        a ~ N(0, Σ_a),  Σ_a = diag(mode_energy)（POD 能量為先驗；None→白先驗 I）
        y = C a + ε,     ε ~ N(0, σ_ε² I),  σ_ε² = σ̃_max² / snr（以最強感測模態定噪）
    snr 定義為「最強感測模態的訊噪比」，使 MI 對 mode_energy 的任意單位尺度不變。

    幾何層（無先驗，量化 r 模態被觀測的病態程度，承襲 κ）：
        dopt_logdet    : D-optimality，log det(CᵀC)|observable = Σ 2·log σ_i（資訊體積）
        aopt_trace_inv : A-optimality，Σ 1/σ_i²（後驗平均變異；越小越好）
        eopt_min_eig   : E-optimality，σ_min²（最壞可觀測方向；越大越好）
    Bayesian/資訊論層（含能量先驗）：
        mutual_info_nats/bits : I(a;y)=½Σ log(1 + snr·(σ̃_i/σ̃_max)²)，σ̃=svd(C·diag√energy)

    D/A/E 限制在可觀測子空間（σ_i>tol·σ_max），故 rank 不足時仍有限；不可觀測方向
    （null space）對資訊量零貢獻，由能量加權的 MI 自然吸收（dead mode → log(1+0)≈0）。
    """
    s = np.linalg.svd(C, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    obs = s[s > tol * smax] if smax > 0 else np.array([])
    dopt = float(2.0 * np.log(obs).sum()) if obs.size else 0.0
    aopt = float((1.0 / obs ** 2).sum()) if obs.size else 0.0
    eopt = float(obs[-1] ** 2) if obs.size else 0.0

    # energy-weighted mutual information（先驗能量併入感測算子後再取奇異值）
    W = C if mode_energy is None else C * np.sqrt(np.asarray(mode_energy, dtype=np.float64))[None, :]
    sw = np.linalg.svd(W, compute_uv=False)
    sw_max2 = float(sw[0] ** 2) if sw.size else 0.0
    if sw_max2 > 0:
        mi_nats = float(0.5 * np.log1p(snr * (sw ** 2) / sw_max2).sum())
    else:
        mi_nats = 0.0
    return {
        "dopt_logdet": dopt,
        "aopt_trace_inv": aopt,
        "eopt_min_eig": eopt,
        "mutual_info_nats": mi_nats,
        "mutual_info_bits": float(mi_nats / np.log(2.0)),
        "snr": float(snr),
    }


# ── driver ────────────────────────────────────────────────────────────────────

def _load_les(les_path: Path, t_spinup: float, time_stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.load(les_path, allow_pickle=True).item()
    t = np.asarray(raw["time"], dtype=np.float64)
    mask = t >= t_spinup
    u = np.asarray(raw["u"], dtype=np.float64)[mask][::time_stride]
    v = np.asarray(raw["v"], dtype=np.float64)[mask][::time_stride]
    N = u.shape[-1]
    return u, v, N


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor-json", default="data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json")
    ap.add_argument("--les", default=str(_DEFAULT_LES), help="LES 全場 .npy（POD modes 來源；DNS-free）")
    ap.add_argument("--kmax", type=int, default=16, help="div-free Fourier 截斷 |k|≤kmax")
    ap.add_argument("--r-list", default="4,8,16,32,64,100,150,200", help="POD 模態數掃描")
    ap.add_argument("--snr", type=float, default=1e3, help="mutual information 的最強感測模態訊噪比")
    ap.add_argument("--t-spinup", type=float, default=0.0, help="LES spin-up 排除（秒）")
    ap.add_argument("--time-stride", type=int, default=1, help="LES 時間抽樣間隔")
    ap.add_argument("--out", default="artifacts/diag/observability.json")
    args = ap.parse_args()

    meta = json.load(open(args.sensor_json))
    coords = np.asarray(meta["selected_coordinates"], dtype=np.float64)
    K = coords.shape[0]
    print(f"[sensors] {Path(args.sensor_json).name}  K={K}")

    report: dict = {"sensor_json": args.sensor_json, "K": K, "kmax": args.kmax}

    # ── Basis B: div-free Fourier（解析、零資料；先跑，最輕）──
    A_u, half = divfree_observation_operator(coords, args.kmax)
    m_fourier = observability_metrics(A_u)
    i_fourier = information_metrics(A_u, mode_energy=None, snr=args.snr)  # 解析 basis 無能量先驗→白
    n_full = len(divfree_modes(args.kmax))
    report["fourier_divfree"] = {**{k: v for k, v in m_fourier.items() if k != "singular_values"},
                                 **i_fourier, "full_disk_modes": n_full, "real_dof": A_u.shape[1],
                                 "singular_values": m_fourier["singular_values"]}
    print(f"\n=== Basis B: divergence-free Fourier (|k|<={args.kmax}) ===")
    print(f"  disk modes={n_full} (conj-symmetric) → {A_u.shape[1]} real DoF;  2K={2*K} obs")
    print(f"  rank={m_fourier['rank']}  κ={m_fourier['cond']:.3e}  "
          f"null_frac={m_fourier['null_fraction']*100:.2f}%  eff_rank={m_fourier['effective_rank']:.2f}")
    print(f"  MI={i_fourier['mutual_info_bits']:.2f} bits  D-opt={i_fourier['dopt_logdet']:.2f}  "
          f"A-opt={i_fourier['aopt_trace_inv']:.3e}  E-opt={i_fourier['eopt_min_eig']:.3e}  (snr={args.snr:g})")

    # ── Basis A: LES-POD（DNS-free 替身）──
    r_list = [int(x) for x in args.r_list.split(",")]
    les_path = Path(args.les)
    print(f"\n=== Basis A: LES-POD ({les_path.name}) ===")
    if not les_path.exists():
        print(f"  [skip] LES 全場不存在: {les_path}\n         （設 PILNJAX_DATA_ROOT 或 --les 指向 LES .npy；本機/lab-server 皆可）")
        report["les_pod"] = {"skipped": True, "reason": f"not found: {les_path}"}
    else:
        u, v, N = _load_les(les_path, args.t_spinup, args.time_stride)
        ix, iy = coords_to_xy_indices(coords, N)
        r_max = min(max(r_list), u.shape[0], 2 * N * N)
        Phi_full, svals = pod_basis(u, v, r_max)
        cum = np.cumsum(svals ** 2) / (svals ** 2).sum()
        r_thr = effective_rank_thresholds(svals, (0.95, 0.99, 0.999))  # 理論第 2 層：有效自由度
        print(f"  LES u{u.shape}  N={N}  POD r_max={r_max}  (top-{r_max} var={cum[r_max-1]*100:.2f}%)")
        print(f"  effective rank: r_95={r_thr[0.95]}  r_99={r_thr[0.99]}  r_99.9={r_thr[0.999]} "
              f"(over full POD spectrum, len={svals.size})")
        print(f"  {'r':>5} {'rank':>5} {'κ':>12} {'null%':>7} {'eff_r':>7} {'POD%':>7} "
              f"{'MI_bits':>9} {'D-opt':>9}  (snr={args.snr:g})")
        per_r = []
        for r in r_list:
            if r > r_max:
                continue
            C = pod_observation_operator(Phi_full[:, :r], ix, iy, N)
            m = observability_metrics(C)
            # 先驗能量 = leading-r POD eigenvalues（σ_i²），併入 MI 的能量加權
            info = information_metrics(C, mode_energy=svals[:r] ** 2, snr=args.snr)
            per_r.append({"r": r, **{k: vv for k, vv in m.items() if k != "singular_values"},
                          **info, "pod_var": float(cum[r - 1])})
            print(f"  {r:>5} {m['rank']:>5} {m['cond']:>12.3e} {m['null_fraction']*100:>6.2f}% "
                  f"{m['effective_rank']:>7.2f} {cum[r-1]*100:>6.2f}% "
                  f"{info['mutual_info_bits']:>9.2f} {info['dopt_logdet']:>9.2f}")
        report["les_pod"] = {"les_file": str(les_path), "N": N, "T_used": int(u.shape[0]),
                             "snr": float(args.snr),
                             "effective_rank": {str(k): v for k, v in r_thr.items()},
                             "per_r": per_r}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
