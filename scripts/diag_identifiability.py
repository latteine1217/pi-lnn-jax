#!/usr/bin/env python3
"""diag_identifiability.py — mid-band 可識別性診斷的 substrate（Ticket 01）。

What:
    以 **sensor Nyquist sampling band edge** k_s=√(K/π) 為界，對重建速度場做保相位的
    shell-wise rel-L2 分解（low k≤k_s / mid k_s–mid_hi / high k>mid_hi）。這是後續
    identifiability 分析（Ticket 02 band-limited floor、Ticket 03 NS-oracle）的共用底座：
    先把「當前模型在哪個 sensor-band 重建得好/壞」量成一張表。

    與 metric_artifact 的差別：那裡的 band_rel_err 是 **k_η（dissipation）band 的能譜
    （scalar E(k)）相對誤差**；本檔要的是 **sensor-band（√(K/π)）的 velocity 場 rel-L2**
    ——保相位（FFT band-mask → iFFT → field rel-L2），與論文 primary 的 velocity rel-L2
    同語意，只是切成 band。兩者 wavenumber convention 一致（fftfreq(N,d=1/N)，見
    compute_energy_spectrum）。

Why:
    velocity 15% 的誤差主要來自 mid/high band（本 session 實測 mid≈57%）。要談「能不能
    加強 mid-band」，先得有一個可重現、可測試的 sensor-band 分解入口，而非散在 bash 裡的
    一次性計算。

Convention:
    場 array u[t, axis_1=x, axis_2=y]，方域 [0,L)²（對齊 pi-lnn DNS/LES 與 diag_observability）。
    wavenumber |k| = √(kx²+ky²)，kx=fftfreq(N,d=1/N) → integer cycles-per-domain。
    聚合 = per-frame rel-L2 再對時間取算術平均（對齊 metric_artifact 的 metrics_mean）。

Usage（本機診斷，讀既有 eval 產物）:
    uv run python scripts/diag_identifiability.py \\
        --fields artifacts/kolmogorov/main5_s42/final_eval/fields.npz --K 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import load_dns_from_path

_HIGH_BAND_LO_DEFAULT = 16.0


# ── sensor-band 幾何 ──────────────────────────────────────────────────────────

def sensor_band_edge(K: int) -> float:
    """sensor Nyquist sampling band edge k_s = √(K/π)（integer cycles-per-domain）。

    K 個點感測器在方域上的等效均勻取樣間距對應的最高可解析波數；K=100→5.6419、
    K=400→11.28。這是「sensor 能直接約束的波數上界」，mid-band 從此開始。
    """
    if K <= 0:
        raise ValueError(f"K 必須為正整數，收到 {K}")
    return float(np.sqrt(K / np.pi))


def radial_wavenumber_grid(N: int) -> np.ndarray:
    """radial |k| grid，convention 對齊 metric_artifact.compute_energy_spectrum。"""
    kx = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2)


def band_filter(field_txy: np.ndarray, kr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """保相位 band 濾波：保留 lo < |k| ≤ hi 的 2D FFT modes，iFFT 回實空間。

    field: [T, N, N] 實數；對 (axis_1, axis_2) 做 2D FFT。回傳同形狀實場。
    """
    F = np.fft.fft2(field_txy, axes=(1, 2))
    F = F * ((kr > lo) & (kr <= hi))[None, :, :]
    return np.fft.ifft2(F, axes=(1, 2)).real


def _per_frame_rel_l2(a_pred: np.ndarray, a_dns: np.ndarray, eps: float) -> float:
    """單分量 per-frame rel-L2 的時間平均：mean_t ‖Δa(t)‖ / ‖a_dns(t)‖。"""
    num = np.sqrt(np.sum((a_pred - a_dns) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(a_dns ** 2, axis=(1, 2)))
    return float(np.mean(num / (den + eps)))


def _per_frame_vec_rel_l2_series(up, vp, ud, vd, eps: float) -> np.ndarray:
    """vector (u,v) 每幀 rel-L2 [T]（未做時間平均），與 metric_artifact 的 uv_rel_err 同定義。"""
    num = np.sqrt(np.sum((up - ud) ** 2, axis=(1, 2)) + np.sum((vp - vd) ** 2, axis=(1, 2)))
    den = np.sqrt(np.sum(ud ** 2, axis=(1, 2)) + np.sum(vd ** 2, axis=(1, 2)))
    return num / (den + eps)


def _per_frame_vec_rel_l2(up, vp, ud, vd, eps: float) -> float:
    """vector (u,v) per-frame rel-L2 的時間平均。"""
    return float(np.mean(_per_frame_vec_rel_l2_series(up, vp, ud, vd, eps)))


def velocity_band_rel_l2(
    u_pred: np.ndarray, v_pred: np.ndarray, u_dns: np.ndarray, v_dns: np.ndarray,
    K: int, mid_hi: float = _HIGH_BAND_LO_DEFAULT, eps: float = 1e-12,
) -> dict:
    """sensor-band 分解的 velocity rel-L2（保相位，per-frame time-mean）。

    band 邊界：low [0, k_s]、mid (k_s, mid_hi]、high (mid_hi, ∞)，k_s=√(K/π)。
    每個 band 回 u_rel / v_rel / uv_rel（vector）。這是 Ticket 01 的核心 substrate。
    """
    if not (u_pred.shape == v_pred.shape == u_dns.shape == v_dns.shape):
        raise ValueError(
            f"場形狀不一致：u_pred{u_pred.shape} v_pred{v_pred.shape} "
            f"u_dns{u_dns.shape} v_dns{v_dns.shape}"
        )
    if u_pred.ndim != 3 or u_pred.shape[1] != u_pred.shape[2]:
        raise ValueError(f"需 [T,N,N] 方域場，收到 {u_pred.shape}")
    N = u_pred.shape[-1]
    kr = radial_wavenumber_grid(N)
    k_s = sensor_band_edge(K)
    hi_cap = float(kr.max()) + 1.0
    # low band 含 DC（mean flow）：lo=-1 使 mask (kr>lo) 涵蓋 kr=0。mid/high 為半開 (lo,hi]。
    bands = {"low": (-1.0, k_s), "mid": (k_s, mid_hi), "high": (mid_hi, hi_cap)}
    out: dict = {"K": int(K), "k_sensor_edge": k_s, "mid_hi": float(mid_hi), "bands": {}}
    for name, (lo, hi) in bands.items():
        up = band_filter(u_pred, kr, lo, hi); vp = band_filter(v_pred, kr, lo, hi)
        ud = band_filter(u_dns, kr, lo, hi); vd = band_filter(v_dns, kr, lo, hi)
        out["bands"][name] = {
            "k_lo": max(0.0, float(lo)),
            "k_hi": (float("inf") if hi == hi_cap else float(hi)),
            "u_rel": _per_frame_rel_l2(up, ud, eps),
            "v_rel": _per_frame_rel_l2(vp, vd, eps),
            "uv_rel": _per_frame_vec_rel_l2(up, vp, ud, vd, eps),
        }
    # 全場（不分 band）vector rel-L2，作為 sanity 對照（應對得上 metrics 的 uv_rel_err）
    out["full_uv_rel"] = _per_frame_vec_rel_l2(u_pred, v_pred, u_dns, v_dns, eps)
    return out


def band_limited_floor(
    u_dns: np.ndarray, v_dns: np.ndarray, K: int, eps: float = 1e-12, per_frame: bool = False,
):
    """無-PDE band-limited information floor（velocity vector rel-L2）。

    把 DNS 低通到 sensor band edge k_s=√(K/π)（含 DC），
    floor = ‖DNS − lowpass(DNS)‖ / ‖DNS‖ = unresolved band（k>k_s）的相對能量。
    這是「完美重建 resolved band、完全放棄 unresolved」的下界：任何**無 PDE / 無額外
    先驗**的 K-sensor 方法都無法低於它。模型若穩健低於此，即在 sensor band 外放了正確
    內容——physics 超解析的必要條件（充分性需 physics-off 對照排除 decoder inductive bias）。
    per_frame=True 回 [T] 供逐幀對照；否則回時間平均純量。
    """
    if u_dns.shape != v_dns.shape or u_dns.ndim != 3 or u_dns.shape[1] != u_dns.shape[2]:
        raise ValueError(f"需一致的 [T,N,N] 方域場，收到 u{u_dns.shape} v{v_dns.shape}")
    N = u_dns.shape[-1]
    kr = radial_wavenumber_grid(N)
    k_s = sensor_band_edge(K)
    u_lp = band_filter(u_dns, kr, -1.0, k_s)
    v_lp = band_filter(v_dns, kr, -1.0, k_s)
    series = _per_frame_vec_rel_l2_series(u_lp, v_lp, u_dns, v_dns, eps)
    return series if per_frame else float(series.mean())


def reconstruct_uv_rel_l2(
    per_t_u_rel: np.ndarray, per_t_v_rel: np.ndarray,
    u_dns_norm_sq: np.ndarray, v_dns_norm_sq: np.ndarray,
) -> float:
    """從既有 metrics.json 的 per-t u_rel/v_rel + DNS 分量 norm² 重組 total vector uv_rel。

    給無 fields.npz 的舊 campaign 用（無法做 band 分解，但能補 total uv_rel）。
    公式：uv(t)=√((u_rel²·‖u_dns‖² + v_rel²·‖v_dns‖²)/(‖u_dns‖²+‖v_dns‖²))，再 time-mean。
    本 session 已驗此重組對原生 uv_rel_err 精確到 <1e-9。
    """
    per_t_u_rel = np.asarray(per_t_u_rel); per_t_v_rel = np.asarray(per_t_v_rel)
    un2 = np.asarray(u_dns_norm_sq); vn2 = np.asarray(v_dns_norm_sq)
    if not (len(per_t_u_rel) == len(per_t_v_rel) == len(un2) == len(vn2)):
        raise ValueError(
            f"長度不一致：u_rel{len(per_t_u_rel)} v_rel{len(per_t_v_rel)} "
            f"un2{len(un2)} vn2{len(vn2)}（不得靠 zip 靜默截短）"
        )
    uv_t = np.sqrt((per_t_u_rel ** 2 * un2 + per_t_v_rel ** 2 * vn2) / (un2 + vn2))
    return float(np.mean(uv_t))


def mid_band_linear_observability(coords: np.ndarray, K: int, mid_hi: float = _HIGH_BAND_LO_DEFAULT) -> dict:
    """mid-band div-free modes 的**線性**可觀測性（leakage-free，不依賴特定場）。

    複用 diag_observability 的解析 div-free Fourier operator（零資料，故無 leakage），
    量 sensor sampling operator 對 mid-band（k_s<|k|≤mid_hi）div-free modes 的 rank / null。
    null_fraction 高 = K 個 sensor 無法**線性**決定 mid-band（欠定）→ linear 可識別性 floor。

    ⚠️ 這是 Ticket 03 的 projection oracle：它是**保守**估計——只量 linear 可觀測性。
    nonlinear NS 動力學可把 mid-band 耦合到 low-band 補資訊（模型實測 mid-band 已優於此
    linear 下界），故本函式**不能下 floor 結論**；floor 須 nonlinear 4D-Var oracle（Ticket 04）。
    """
    from scripts.diag_observability import divfree_observation_operator, observability_metrics
    coords = np.asarray(coords, dtype=np.float64)
    k_s = sensor_band_edge(K)
    A, half = divfree_observation_operator(coords, kmax=int(np.ceil(mid_hi)))
    half = np.asarray(half)
    kmag = np.hypot(half[:, 0], half[:, 1])
    mid = (kmag > k_s) & (kmag <= mid_hi)
    if not mid.any():
        raise ValueError(f"mid shell ({k_s:.2f},{mid_hi}] 內無 div-free mode")
    cols = np.concatenate([[2 * q, 2 * q + 1] for q in np.where(mid)[0]])
    m = observability_metrics(A[:, cols])
    return {
        "K": int(K), "k_sensor_edge": k_s, "n_obs": int(m["n_obs"]),
        "mid_dof": int(m["n_dof"]), "rank": int(m["rank"]),
        "null_fraction": float(m["null_fraction"]),
    }


# ── I/O 與對齊（fail-fast，不套預設、不寬鬆對齊）─────────────────────────────

def load_fields_and_dns(fields_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """讀 eval 產物 fields.npz（u_pred/v_pred/t/dns_path）+ 對齊的 DNS 場。

    fail-fast：缺鍵、缺 DNS、時間軸無法整除對齊、對齊後 t 不吻合 → raise 點名。
    """
    fields_path = Path(fields_path)
    if not fields_path.exists():
        raise FileNotFoundError(f"fields.npz 不存在：{fields_path}")
    f = np.load(fields_path, allow_pickle=True)
    for key in ("u_pred", "v_pred", "t", "dns_path"):
        if key not in f.files:
            raise KeyError(f"{fields_path} 缺鍵 {key!r}（有 {list(f.files)}）")
    u_pred = np.asarray(f["u_pred"], dtype=np.float64)
    v_pred = np.asarray(f["v_pred"], dtype=np.float64)
    t_fields = np.asarray(f["t"], dtype=np.float64)
    dns_path = str(f["dns_path"])

    u_full, v_full, t_full = load_dns_from_path(dns_path, time_stride=1)
    u_full = np.asarray(u_full, dtype=np.float64); v_full = np.asarray(v_full, dtype=np.float64)
    t_full = np.asarray(t_full, dtype=np.float64)
    # DNS 存的時刻比 pred 密（si100 給 201，pred 101）；下採樣 stride 使幀數吻合。
    # 註：201 幀 [::2] = 101 幀是正確對齊，但 201 不被 101 整除——故以「下採樣後長度
    # 吻合」判定，不用整除；stride 或相位錯一律由後續 t 逐點對齊擋下（fail-fast）。
    if len(t_full) < len(t_fields):
        raise ValueError(f"DNS 幀數 {len(t_full)} 少於 pred 幀數 {len(t_fields)}，無法對齊")
    stride = round(len(t_full) / len(t_fields))
    if stride < 1 or len(t_full[::stride]) != len(t_fields):
        raise ValueError(
            f"DNS 幀數 {len(t_full)} 無法以整數 stride 下採樣到 pred 幀數 {len(t_fields)}"
        )
    u_dns = u_full[::stride]; v_dns = v_full[::stride]; t_dns = t_full[::stride]
    if not np.allclose(t_dns, t_fields, atol=1e-4):
        raise ValueError(
            f"時間軸對齊後不吻合（stride={stride}）：max|Δt|={np.max(np.abs(t_dns - t_fields)):.2e}"
        )
    return u_pred, v_pred, u_dns, v_dns


def main() -> None:
    ap = argparse.ArgumentParser(description="mid-band sensor-band velocity rel-L2 分解")
    ap.add_argument("--fields", required=True, help="eval 產物 fields.npz（含 u_pred/v_pred/t/dns_path）")
    ap.add_argument("--K", type=int, required=True, help="sensor 數（決定 band edge √(K/π)）")
    ap.add_argument("--mid-hi", type=float, default=_HIGH_BAND_LO_DEFAULT, help="mid/high 分界波數")
    ap.add_argument("--out", default=None, help="輸出 json 路徑（可選）")
    args = ap.parse_args()

    u_pred, v_pred, u_dns, v_dns = load_fields_and_dns(args.fields)
    res = velocity_band_rel_l2(u_pred, v_pred, u_dns, v_dns, args.K, mid_hi=args.mid_hi)

    print(f"=== sensor-band velocity rel-L2 分解（{Path(args.fields).parent.parent.name}）===")
    print(f"K={res['K']}  k_sensor_edge=√(K/π)={res['k_sensor_edge']:.4f}  mid_hi={res['mid_hi']:.1f}")
    print(f"full-field uv_rel = {100*res['full_uv_rel']:.2f}%")
    print(f"{'band':6s} {'k_range':>14s} {'u_rel':>8s} {'v_rel':>8s} {'uv_rel':>8s}")
    for name in ("low", "mid", "high"):
        b = res["bands"][name]
        krange = f"({b['k_lo']:.2f},{b['k_hi']:.2f}]" if np.isfinite(b["k_hi"]) else f"({b['k_lo']:.2f},∞)"
        print(f"{name:6s} {krange:>14s} {100*b['u_rel']:7.2f}% {100*b['v_rel']:7.2f}% {100*b['uv_rel']:7.2f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
