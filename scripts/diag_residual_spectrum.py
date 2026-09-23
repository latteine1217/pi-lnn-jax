"""真值場的 PDE residual 頻譜地板（階段 1：不需要 checkpoint）。

Why: 模型殘差譜 R(k)=|R̂(k)|² 在高 k 上升是 **null prediction**，不是發現——
線性化後 R ≈ 𝓛·δu，而 NS 算符的 symbol 為 i k·U + νk²，隨 k 單調上升；
離散化誤差與浮點捨入同樣集中在高 k。要判讀模型的 R(k)，必須先知道
「把**真值**代進同一條殘差路徑會得到什麼」。

本腳本量四個地板，逐一對應一個會偽造出「高 k 殘差高」的來源：

  F1  譜微分 / fp64      資料本身的物理-數值不自洽（真正的地板）
  F2  中央差分 / fp64    F2−F1 = 空間離散化誤差的貢獻
  F3  時間 stride×2      F3−F1 = 時間離散化誤差的貢獻
  F4  譜微分 / fp32      F4−F1 = 浮點捨入的貢獻（評估路徑實際吃 float32）

另輸出**各項自身量級譜**（|ω_t|²、|u·∇ω|²、|ν∇²ω|² …），供無量綱化：
單看 R(k) 分不出「物理沒學到」與「算符本來就放大高頻」，
R(k)/Σ_terms(k)（各項互相抵消得多乾淨）才是無量綱、可跨頻帶比較的量。

殘差有兩式，刻意都算：
  mom  含壓力，與 `physics.py:ns_residuals`（訓練 loss 實際用的）同式
  vort 渦量傳輸，消去壓力，與 `physics_field.vorticity_transport_residual` 同式

頻帶與殼層 bin 沿用 `metric_artifact.compute_energy_spectrum` 的定義，
確保與既有 `band_rel_err_*` / `gamma_k` 落在同一組 k 上（否則兩套頻帶無法對齊）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path
from pi_lnn_jax.metric_artifact import compute_energy_spectrum, compute_vorticity
from pi_lnn_jax.physics_field import vorticity_transport_residual


# ── 殼層 bin：與 compute_energy_spectrum 等價，但用 bincount（逐 frame 呼叫 ~180 次
#    mask 太慢）。等價性由 _assert_shell_matches_repo 在每次執行時實測驗證。──────
def _shell_index(N: int) -> tuple[np.ndarray, int]:
    kx = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    n_bins = int(np.ceil(K.max())) + 1
    # (K>=ki-0.5)&(K<ki+0.5) ⟺ floor(K+0.5)；K 為整數平方和的平方根，
    # 不可能恰為半整數，故無 tie-breaking 歧義。
    return np.floor(K + 0.5).astype(np.int64), n_bins


def _shell_power(field: np.ndarray, idx: np.ndarray, n_bins: int) -> np.ndarray:
    """單一實場的逐殼層功率 Σ|F̂|²（歸一化同 compute_energy_spectrum）。"""
    N = field.shape[-1]
    P = np.abs(np.fft.fft2(field) / (N * N)) ** 2
    return np.bincount(idx.ravel(), weights=P.ravel(), minlength=n_bins)


def _assert_shell_matches_repo(u: np.ndarray, v: np.ndarray,
                               idx: np.ndarray, n_bins: int) -> None:
    """bincount 版殼層必須與 repo 的 compute_energy_spectrum 逐 bin 吻合。"""
    k_ref, E_ref = compute_energy_spectrum(u, v)
    E_mine = 0.5 * (_shell_power(u, idx, n_bins) + _shell_power(v, idx, n_bins))
    if len(E_ref) != n_bins:
        raise AssertionError(f"shell 數不符: repo={len(E_ref)} mine={n_bins}")
    if not np.allclose(E_ref, E_mine, rtol=1e-12, atol=1e-30):
        raise AssertionError(
            f"殼層 bin 與 repo 不等價，max|Δ|={np.abs(E_ref - E_mine).max():.3e}")
    # Parseval：ΣE_k == 0.5·mean(u²+v²)
    lhs, rhs = E_mine.sum(), 0.5 * np.mean(u ** 2 + v ** 2)
    if not np.isclose(lhs, rhs, rtol=1e-10):
        raise AssertionError(f"Parseval 破缺: ΣE_k={lhs:.6e} vs 0.5⟨u²+v²⟩={rhs:.6e}")


# ── 微分算子：譜（週期域上機器精度）與中央差分（既有 physics_field 用的）─────────
def _spec_d(f: np.ndarray, axis: int, L: float) -> np.ndarray:
    N = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    k[N // 2] = 0.0            # Nyquist 的一階導數置零（標準做法，避免 aliasing）
    shape = [1] * f.ndim
    shape[axis] = N
    return np.real(np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(f, axis=axis), axis=axis))


def _spec_lap(f: np.ndarray, L: float) -> np.ndarray:
    N = f.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    K2 = k[:, None] ** 2 + k[None, :] ** 2
    return np.real(np.fft.ifft2(-K2 * np.fft.fft2(f, axes=(-2, -1)), axes=(-2, -1)))


def _fd_d(f: np.ndarray, axis: int, h: float) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)


def _fd_lap(f: np.ndarray, h: float) -> np.ndarray:
    out = np.zeros_like(f)
    for ax in (-2, -1):
        out = out + (np.roll(f, -1, axis=ax) - 2.0 * f + np.roll(f, 1, axis=ax)) / h ** 2
    return out


class Ops:
    """把「用哪套空間微分」變成一個可切換的物件，讓 F1/F2 走完全相同的殘差程式碼。"""

    def __init__(self, mode: str, L: float, N: int):
        self.mode = mode
        self.L, self.h = L, L / N

    def dx(self, f):
        return _spec_d(f, -2, self.L) if self.mode == "spectral" else _fd_d(f, -2, self.h)

    def dy(self, f):
        return _spec_d(f, -1, self.L) if self.mode == "spectral" else _fd_d(f, -1, self.h)

    def lap(self, f):
        return _spec_lap(f, self.L) if self.mode == "spectral" else _fd_lap(f, self.h)


def _time_deriv(f: np.ndarray, t: np.ndarray, order: int) -> tuple[np.ndarray, slice]:
    """時間導數 + 有效內部點 slice。階數是**可量的變因**而非細節：存檔間隔
    dt=0.025 遠粗於積分步長，ω_t 的截斷誤差是殘差地板的主要來源之一，
    要能用 order=2 vs 4 把它的量級量出來。"""
    dt = np.diff(t)
    # 容差對到 float32：DNS 時間軸經 loader 轉 float32，0.05 的捨入讓逐格 dt 有
    # ~1e-5 的相對離散（實測 9.5e-6）。1e-9 會把它誤判成不等距，而真正的不等距
    # （sensor_time_independent 的 gap）是量級差異，1e-4 仍擋得住。
    # h 取平均而非 dt[0]：後者若剛好落在捨入偏大的一格，會變成系統性偏差。
    spread = float((dt.max() - dt.min()) / dt.mean())
    if spread > 1e-4:
        raise ValueError(
            f"時間軸非均勻（相對離散 {spread:.3e} > 1e-4），中央差分不適用")
    h = float(np.mean(dt))
    if order == 2:
        return (f[2:] - f[:-2]) / (2.0 * h), slice(1, -1)
    if order == 4:
        return (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * h), slice(2, -2)
    raise ValueError(f"time_order 必須是 2 或 4，得到 {order}")


# ── 殘差：與 physics.py:ns_residuals / physics_field 同式（符號逐項對齊）────────
def residual_terms(u, v, p, t, y, nu, A, k_f, ops: Ops,
                   time_order: int = 2, with_mom: bool = True):
    """回傳 {方程: {項名: [T',N,N]}}；殘差 = 各項之和（項已含正負號）。

    分項回傳而非只回總和，是因為無量綱化需要各項自身的量級譜。
    所有項共用同一組內部時間點（由 time_order 決定），確保各 floor 可比。
    """
    omega_full = ops.dx(v) - ops.dy(u)
    om_t, sl = _time_deriv(omega_full, t, time_order)
    u_t, _ = _time_deriv(u, t, time_order)
    v_t, _ = _time_deriv(v, t, time_order)
    u, v, p, omega = u[sl], v[sl], p[sl], omega_full[sl]
    f_x = A * np.sin(2.0 * np.pi * k_f * y)[None, None, :]          # y 沿 axis -1

    mom_u = {
        "u_t":   u_t,
        "adv":   u * ops.dx(u) + v * ops.dy(u),
        "p_x":   ops.dx(p),
        "visc": -nu * ops.lap(u),
        "force": -np.broadcast_to(f_x, u.shape),
    }
    mom_v = {
        "v_t":   v_t,
        "adv":   u * ops.dx(v) + v * ops.dy(v),
        "p_y":   ops.dy(p),
        "visc": -nu * ops.lap(v),
    }
    cont = {"u_x": ops.dx(u), "v_y": ops.dy(v)}

    curl_f = -A * (2.0 * np.pi * k_f) * np.cos(2.0 * np.pi * k_f * y)[None, None, :]
    vort = {
        "omega_t": om_t,
        "adv":     u * ops.dx(omega) + v * ops.dy(omega),
        "visc":   -nu * ops.lap(omega),
        "force":  -np.broadcast_to(curl_f, u.shape),
    }
    out = {"cont": cont, "vort": vort}
    if with_mom:
        out.update({"mom_u": mom_u, "mom_v": mom_v})
    return out


def spectra_of(terms: dict[str, np.ndarray], idx, n_bins) -> dict[str, list]:
    """殘差總和譜 + 各項自身譜（皆對時間平均）。"""
    T = next(iter(terms.values())).shape[0]
    R = sum(terms.values())
    acc = {"residual": np.zeros(n_bins)}
    acc.update({f"term::{k}": np.zeros(n_bins) for k in terms})
    for i in range(T):
        acc["residual"] += _shell_power(R[i], idx, n_bins)
        for k, arr in terms.items():
            acc[f"term::{k}"] += _shell_power(arr[i], idx, n_bins)
    out = {k: (a / T) for k, a in acc.items()}
    # 抵消品質：殘差相對於「各項自身量級之和」——無量綱、可跨頻帶比較
    denom = sum(v for k, v in out.items() if k.startswith("term::"))
    out["cancellation"] = out["residual"] / np.maximum(denom, 1e-300)
    out["rms_real"] = np.array([float(np.sqrt(np.mean(R ** 2)))])   # Parseval 交叉檢查用
    return {k: v.tolist() for k, v in out.items()}


BANDS = {"low": (0, 5), "mid": (5, 16), "high": (16, 10 ** 9)}   # 論文 fig:band_err caption 定義


def band_summary(k_arr: np.ndarray, spec: list) -> dict:
    s = np.asarray(spec)
    tot = s.sum()
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (k_arr > lo) & (k_arr <= hi) if lo > 0 else (k_arr <= hi)
        out[name] = {"sum": float(s[sel].sum()),
                     "frac": float(s[sel].sum() / tot) if tot > 0 else float("nan")}
    return out


def gate_pressure_consistency(u, v, p, ops: Ops, thresh: float = 0.9) -> float:
    """p 必須與 (u,v) 滿足壓力 Poisson 方程，否則含壓力的動量殘差沒有意義。

    Why 這是閘門而非警告：不一致的 p 會讓 mom 殘差被一個與物理無關的量主導
    （實測 corr(∇²p, rhs) = −0.41、best-fit −1.70），而譜圖看起來依然「合理」。
    這種錯誤不會自己現形，只能靠事前斷言擋下。
    """
    rhs = -(ops.dx(u) ** 2 + ops.dy(v) ** 2 + 2.0 * ops.dy(u) * ops.dx(v))
    lp = ops.lap(p)
    corr = float(np.sum(lp * rhs) / np.sqrt(np.sum(lp ** 2) * np.sum(rhs ** 2) + 1e-300))
    if corr < thresh:
        raise AssertionError(
            f"壓力場與速度場不自洽: corr(∇²p, Poisson rhs) = {corr:+.4f} < {thresh}。\n"
            f"  含壓力的動量殘差在這份資料上無法定義地板。\n"
            f"  用 --skip-mom 只走渦量傳輸式（消去 p），或改用一致的 DNS 檔。")
    return corr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns-path", required=True)
    ap.add_argument("--re", type=float, required=True, help="nu = 1/Re")
    ap.add_argument("--kolmogorov-A", type=float, required=True)
    ap.add_argument("--kolmogorov-kf", type=float, required=True)
    ap.add_argument("--domain-length", type=float, required=True)
    ap.add_argument("--t-start", type=int, default=0)
    ap.add_argument("--t-end", type=int, default=-1, help="-1 = 到最後一個 frame")
    ap.add_argument("--skip-mom", action="store_true",
                    help="只算渦量傳輸式與連續方程（資料的 p 與 u,v 不自洽時必須開）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = _resolve_data_path(args.dns_path)
    if not path.exists():
        raise FileNotFoundError(f"DNS npy 不存在: {path}")
    obj = np.load(path, allow_pickle=True).item()
    need = ("u", "v", "time", "y") if args.skip_mom else ("u", "v", "p", "time", "y")
    for key in need:
        if key not in obj:
            raise KeyError(f"DNS npy 缺 '{key}': {path}")

    t_end = len(obj["time"]) if args.t_end < 0 else args.t_end
    sl = slice(args.t_start, t_end)
    # 刻意直讀 fp64（loader 會轉 float32）——F1 要量的是資料本身的地板，
    # 捨入的貢獻由 F4 單獨量。
    u = np.ascontiguousarray(obj["u"][sl], dtype=np.float64)
    v = np.ascontiguousarray(obj["v"][sl], dtype=np.float64)
    p = (np.zeros_like(u) if args.skip_mom
         else np.ascontiguousarray(obj["p"][sl], dtype=np.float64))
    t = np.asarray(obj["time"][sl], dtype=np.float64)
    y = np.asarray(obj["y"], dtype=np.float64)
    T, Nx, N = u.shape
    if Nx != N:
        raise ValueError(f"非方域 ({Nx},{N}) 不支援")
    if T < 6:
        raise ValueError(f"時間 frame 數 {T} < 6，4 階時間差分無意義")

    L, nu = args.domain_length, 1.0 / args.re
    # y 軸慣例必須與 forcing 一致：physics.py 用 A·sin(2π k_f y)，y=j·L/N。
    y_expected = np.arange(N) * (L / N)
    if not np.allclose(y, y_expected, atol=1e-9):
        raise AssertionError(
            f"y 軸與 forcing 假設不符: y[:3]={y[:3]} 期望={y_expected[:3]}（domain_length 錯？）")

    idx, n_bins = _shell_index(N)
    _assert_shell_matches_repo(u[0], v[0], idx, n_bins)
    k_arr = np.arange(n_bins)
    ops_spec = Ops("spectral", L, N)

    print("=== 真值殘差頻譜地板 ===")
    print(f"data      : {path}")
    print(f"frames    : [{args.t_start}, {t_end}) → T={T}, dt={np.diff(t)[0]:.6g}, N={N}")
    print(f"physics   : Re={args.re:g} (nu={nu:.3e})  A={args.kolmogorov_A}  "
          f"k_f={args.kolmogorov_kf}  L={L}")
    src = obj.get("config", {})
    if isinstance(src, dict) and "source_N" in src:
        print(f"provenance: 由 N={src['source_N']} 降採樣 stride={src.get('downsample_stride')} "
              f"而來（source={Path(str(src.get('source_file',''))).name}）")

    if args.skip_mom:
        print("\n[gate] --skip-mom：跳過含壓力的動量方程")
        corr_p = None
    else:
        corr_p = gate_pressure_consistency(u[:4], v[:4], p[:4], ops_spec)
        print(f"\n[gate] 壓力自洽 corr(∇²p, rhs) = {corr_p:+.4f} OK")

    floors: dict[str, dict] = {}

    def run(tag, uu, vv, pp, tt, mode, order):
        ops = Ops(mode, L, N)
        terms = residual_terms(uu, vv, pp, tt, y, nu, args.kolmogorov_A,
                               args.kolmogorov_kf, ops,
                               time_order=order, with_mom=not args.skip_mom)
        floors[tag] = {eq: spectra_of(tm, idx, n_bins) for eq, tm in terms.items()}
        for eq, sp in floors[tag].items():
            lhs, rhs = float(np.sum(sp["residual"])), float(sp["rms_real"][0]) ** 2
            if not np.isclose(lhs, rhs, rtol=1e-8, atol=1e-30):
                raise AssertionError(f"[{tag}/{eq}] Parseval 破缺: {lhs:.6e} vs {rhs:.6e}")
        print(f"  [{tag}] " + "  ".join(
            f"{eq} rms={floors[tag][eq]['rms_real'][0]:.4e}" for eq in floors[tag]))

    print("\n--- 計算地板（每個 floor 只改一個變因）---")
    run("F1_spectral_fp64_t2", u, v, p, t, "spectral", 2)
    run("F2_findiff_fp64_t2", u, v, p, t, "findiff", 2)
    run("F3_spectral_stride2_t2", u[::2], v[::2], p[::2], t[::2], "spectral", 2)
    run("F4_spectral_fp32_t2",
        u.astype(np.float32).astype(np.float64),
        v.astype(np.float32).astype(np.float64),
        p.astype(np.float32).astype(np.float64), t, "spectral", 2)
    run("F5_spectral_fp64_t4", u, v, p, t, "spectral", 4)

    # 與 repo 既有實作對拍：F2 的 vort 殘差必須逐點等於 physics_field 的 |R_omega|。
    # 這道閘門擋的是軸序 / 符號 / forcing 寫錯——寫錯了譜也會很漂亮。
    ops_fd = Ops("findiff", L, N)
    ref = vorticity_transport_residual(u, v, t, nu, args.kolmogorov_A,
                                       args.kolmogorov_kf, L)[1:-1]
    mine = np.abs(sum(residual_terms(u, v, p, t, y, nu, args.kolmogorov_A,
                                     args.kolmogorov_kf, ops_fd, 2, False)["vort"].values()))
    rel = np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-30)
    if rel > 1e-10:
        raise AssertionError(f"vort 殘差與 physics_field 不吻合，rel={rel:.3e}")
    print(f"\n  [gate] vort 殘差 vs physics_field.vorticity_transport_residual: "
          f"rel_max={rel:.2e} OK")

    E_k = np.mean([0.5 * (_shell_power(u[i], idx, n_bins) + _shell_power(v[i], idx, n_bins))
                   for i in range(T)], axis=0)

    base = floors["F1_spectral_fp64_t2"]
    print("\n--- 地板的頻帶分佈（F1）：能量 vs 殘差 vs 抵消品質 ---")
    print(f"  {'band':<7}{'E frac':>9}" + "".join(
        f"{eq+' Rfrac':>13}{eq+' cancel':>14}" for eq in base))
    for b, (lo, hi) in BANDS.items():
        sel = (k_arr > lo) & (k_arr <= hi) if lo > 0 else (k_arr <= hi)
        row = f"  {b:<7}{band_summary(k_arr, E_k.tolist())[b]['frac']:>8.2%}"
        for eq in base:
            c = np.asarray(base[eq]["cancellation"])
            row += (f"{band_summary(k_arr, base[eq]['residual'])[b]['frac']:>12.2%}"
                    f"{np.nanmean(c[sel]):>14.3e}")
        print(row)

    print("\n--- 地板來源分解（rms(R)，每列只改一個變因）---")
    eqs = list(base.keys())
    print(f"  {'floor':<26}" + "".join(f"{e:>14}" for e in eqs))
    for tag in floors:
        print(f"  {tag:<26}" + "".join(
            f"{floors[tag][e]['rms_real'][0]:>14.4e}" for e in eqs))

    out = {
        "provenance": {
            "dns_path": str(path), "file_bytes": path.stat().st_size,
            "t_start": args.t_start, "t_end": t_end, "T": T,
            "dt": float(np.diff(t)[0]), "N": N,
            "re": args.re, "nu": nu, "kolmogorov_A": args.kolmogorov_A,
            "kolmogorov_kf": args.kolmogorov_kf, "domain_length": L,
            "skip_mom": args.skip_mom, "pressure_corr": corr_p,
            "source_N": src.get("source_N") if isinstance(src, dict) else None,
            "downsample_stride": src.get("downsample_stride") if isinstance(src, dict) else None,
            "bands": {k: list(v) for k, v in BANDS.items()},
            "note": "階段1=真值地板，無模型。F1 為地板本身；F2..F5 各減 F1 為單一來源貢獻。",
        },
        "k": k_arr.tolist(),
        "E_k": E_k.tolist(),
        "floors": floors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
