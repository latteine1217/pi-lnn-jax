#!/usr/bin/env python3
"""DNS 速度場的 POD 譜，對照感測取樣圓盤的 Fourier 帶寬。

What:
    對 DNS 快照矩陣做 economy SVD 取 POD 能量譜，並用**同一份場**算殼層能譜，
    然後回答一個問題：要達到「K 支感測器的取樣圓盤所含的能量分率」，
    需要幾個 POD 模態？

Why:
    空間軸（uv_rel_err vs K）在 K≈250 之後變平。兩個競爭解釋：
      (a) 能譜陡峭 —— 圓盤外已經沒什麼能量可撿，平緩是 E(k) 的性質；
      (b) 低維結構 —— 流場的有效秩遠小於圓盤內的 Fourier 模數，
          算子若在利用它，早該在更小的 K 就飽和。
    判別觀測是 `pod_modes_for_disc_energy / K`：
      ≈1 → 兩者同一件事，(b) 不成立；
      ≪1 → 流場確實低維，而曲線卻跟著 Fourier 模數走，代表那段壓縮沒被用到。

    ⚠️ POD 秩上限是快照數。若 r_99 逼近 n_frames，量到的是取樣不足而不是流場的秩，
    本腳本會明確標記 `rank_limited_by_snapshots`，不得當成流場性質讀。

Usage:
    uv run python scripts/diag_dns_pod_rank.py \
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
        --time-stride 1 --out artifacts/diag/dns_pod_rank.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import load_dns_from_path

# 空間軸掃過的預算；k_s = sqrt(K/pi) 是稿子 chapter01 事前登錄的取樣 Nyquist
SWEEP_K = (50, 100, 200, 400, 800)
ENERGY_LEVELS = (0.90, 0.99, 0.999, 0.9999)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dns", required=True, help="DNS .npy 路徑（相對路徑走 PILNJAX_DATA_ROOT）")
    p.add_argument("--time-stride", type=int, required=True,
                   help="時間下採樣。**必填無預設**：POD 秩直接受快照數限制，"
                        "吃預設等於讓取樣密度靜默決定結論")
    p.add_argument("--out", required=True, help="JSON 產物落點")
    p.add_argument("--subtract-mean", choices=("both", "yes", "no"), default="both",
                   help="POD 取脈動（減時間平均）或原場。Kolmogorov flow 有強迫出來的"
                        "平均剖面，兩者的秩差很多，預設兩個都報")
    return p.parse_args()


def shell_energy_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """逐幀的整數殼層動能譜，回傳時間平均 [n_k]。

    與 evaluate 落在 `metrics_per_t[].E_dns_k` 的量同定義：E(k) = (1/2)|û|²
    在 |k| 四捨五入到整數的殼上求和，故 ΣE(k) = 空間平均動能。
    """
    n_t, ny, nx = u.shape
    if ny != nx:
        raise ValueError(f"本診斷假設方形網格，收到 {ny}x{nx}")
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kmag = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    kbin = np.rint(kmag).astype(np.int32)
    n_k = int(kbin.max()) + 1
    acc = np.zeros(n_k, dtype=np.float64)
    for t in range(n_t):
        uh = np.fft.fft2(u[t]) / (nx * ny)
        vh = np.fft.fft2(v[t]) / (nx * ny)
        e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
        acc += np.bincount(kbin.ravel(), weights=e.ravel(), minlength=n_k)
    return acc / n_t


def pod_energy_fractions(u: np.ndarray, v: np.ndarray, subtract_mean: bool) -> dict:
    """economy SVD 的累積能量分率。快照沿時間、自由度沿 (2, ny, nx) 攤平。"""
    n_t = u.shape[0]
    x = np.concatenate([u.reshape(n_t, -1), v.reshape(n_t, -1)], axis=1).astype(np.float64)
    if subtract_mean:
        x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    e = s ** 2
    tot = float(e.sum())
    if tot <= 0.0:
        raise ValueError("快照矩陣能量為零——輸入不對")
    cum = np.cumsum(e) / tot
    return {
        "n_snapshots": int(n_t),
        "singular_values": s.tolist(),
        "cum_energy": cum.tolist(),
        "rank_at": {f"{lv}": int(np.searchsorted(cum, lv) + 1) for lv in ENERGY_LEVELS},
    }


def main() -> None:
    a = parse_args()
    u, v, t = load_dns_from_path(a.dns, time_stride=a.time_stride)
    print(f"[dns] {a.dns}  stride={a.time_stride}  frames={u.shape[0]}  grid={u.shape[1]}x{u.shape[2]}"
          f"  t=[{t[0]:.3f}, {t[-1]:.3f}]")

    ek = shell_energy_spectrum(u, v)
    tot_e = float(ek.sum())
    cum_k = np.cumsum(ek) / tot_e
    print(f"[spectrum] ΣE(k) = {tot_e:.6f}  →  U_rms = sqrt(2ΣE) = {np.sqrt(2 * tot_e):.4f} m/s")
    for lv in ENERGY_LEVELS:
        print(f"           {lv*100:7.3f}% 能量在 k <= {int(np.searchsorted(cum_k, lv))}")

    modes = ["yes", "no"] if a.subtract_mean == "both" else [a.subtract_mean]
    pod = {}
    for m in modes:
        pod[m] = pod_energy_fractions(u, v, subtract_mean=(m == "yes"))
        r = pod[m]["rank_at"]
        print(f"[pod  ] subtract_mean={m:3s}  " +
              "  ".join(f"r_{lv*100:g}={r[str(lv)]}" for lv in ENERGY_LEVELS))

    # ── 判別觀測：達到「取樣圓盤所含能量」需要幾個 POD 模態 ──────────────
    per_k = []
    for K in SWEEP_K:
        ks = float(np.sqrt(K / np.pi))
        # 圓盤內 = 完整落在半徑 k_s 內的整數殼層（與台帳用的截斷慣例一致）
        k_in = int(np.floor(ks))
        frac_in = float(cum_k[k_in])
        row = {"K": K, "k_s": ks, "shells_kept": k_in,
               "disc_energy_fraction": frac_in,
               "fourier_truncation_floor_pct": float(np.sqrt(1.0 - frac_in) * 100.0)}
        for m in modes:
            cum = np.asarray(pod[m]["cum_energy"])
            n_snap = pod[m]["n_snapshots"]
            if frac_in > cum[-1]:
                row[f"pod_modes_for_disc_energy_{m}"] = None
                row[f"rank_limited_by_snapshots_{m}"] = True
            else:
                r = int(np.searchsorted(cum, frac_in) + 1)
                row[f"pod_modes_for_disc_energy_{m}"] = r
                row[f"compression_{m}"] = K / r
                row[f"rank_limited_by_snapshots_{m}"] = bool(r >= n_snap)
        per_k.append(row)

    print()
    hdr = f"{'K':>5}{'k_s':>7}{'殼層':>5}{'圓盤能量%':>11}{'Fourier 下界%':>14}"
    for m in modes:
        hdr += f"{'POD模態('+m+')':>15}{'壓縮×':>8}"
    print(hdr)
    for r in per_k:
        line = (f"{r['K']:>5}{r['k_s']:>7.2f}{r['shells_kept']:>5}"
                f"{r['disc_energy_fraction']*100:>11.4f}{r['fourier_truncation_floor_pct']:>14.2f}")
        for m in modes:
            rm = r.get(f"pod_modes_for_disc_energy_{m}")
            line += f"{'n/a' if rm is None else rm:>15}"
            line += f"{'—' if rm is None else format(r[f'compression_{m}'], '.1f'):>8}"
        print(line)

    warn = [r["K"] for r in per_k
            if any(r.get(f"rank_limited_by_snapshots_{m}") for m in modes)]
    if warn:
        print(f"\n[WARN] K={warn} 的 POD 模態數已觸及快照數上限 "
              f"({[pod[m]['n_snapshots'] for m in modes]})——那是取樣不足，不是流場的秩。"
              f"降低 --time-stride 或換更長的軌跡才能解析。")

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        rev = None
    payload = {
        "producer": "scripts/diag_dns_pod_rank.py",
        "provenance": {"code_revision": rev, "dns": str(a.dns),
                       "time_stride": a.time_stride, "n_frames": int(u.shape[0]),
                       "grid": list(u.shape[1:]), "t_range": [float(t[0]), float(t[-1])]},
        "spectrum": {"E_k": ek.tolist(), "total_energy": tot_e,
                     "u_rms_from_spectrum": float(np.sqrt(2 * tot_e)),
                     "k_at": {f"{lv}": int(np.searchsorted(cum_k, lv)) for lv in ENERGY_LEVELS}},
        "pod": pod,
        "per_budget": per_k,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
