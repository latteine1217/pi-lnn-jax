#!/usr/bin/env python3
"""K 支感測器對 DNS POD 子空間的可觀測性：條件數與 oracle 重建下界。

What:
    用 DNS 自己的 POD 模態 Ψ_r 與感測點取樣算子 C 組出觀測矩陣 Θ_r = C Ψ_r，
    掃 r 報三件事：
      1. κ(Θ_r) —— 係數反演的病態程度；
      2. oracle gappy 重建誤差 —— 拿**真**基底 + 無雜訊感測值做最小平方，
         逐幀重建整場後量 uv 相對 L2（與主指標同定義）；
      3. 同一條在感測雜訊下的放大。

Why:
    `diag_dns_pod_rank` 量到 K=200 的 400 個純量量測對上「載 99.99% 能量的 92 個
    POD 模態」仍是 4.3× 超定，而實測誤差 8.63% 只等價於解出 28 個模態。
    那是**計數論證**，不是可解性論證——Θ_r 可能病態。本腳本把計數換成量測：
    若 oracle 重建在 r≈90 附近就逼近 1%，「感測器數不是瓶頸」成立；
    若 κ 在 r 還很小時就爆掉，瓶頸其實在佈點的可觀測性，那是完全不同的結論。

    ⚠️ 這是 **oracle**：Ψ_r 與時間平均都由 DNS 導出，sensor-only 方法拿不到
    （本專案已量過 DNS-free 的替代基底不堪用）。它給的是「若能取得真基底」的
    上界，用來判斷資訊在不在，**不是可部署的重建方法**。

Usage:
    uv run python scripts/diag_pod_observability.py \
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
        --sensors data/kolmogorov_sensors/re10000/sensors_qrpivot_K200_....json \
        --time-stride 1 --noise 0.0,0.05 --out artifacts/diag/pod_observability_K200.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path, load_dns_from_path

RANKS = (5, 10, 20, 28, 40, 53, 70, 92, 120, 150, 180)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dns", required=True)
    p.add_argument("--sensors", required=True, help="sensor set JSON（只取 selected_coordinates）")
    p.add_argument("--time-stride", type=int, required=True,
                   help="必填無預設：POD 秩受快照數限制，吃預設等於讓取樣密度靜默決定結論")
    p.add_argument("--noise", default="0.0",
                   help="逗號分隔的感測相對雜訊水準（σ 相對於該通道的 rms）")
    p.add_argument("--out", required=True)
    return p.parse_args()


def sensor_grid_indices(coords: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """[K,2] 的 [0,1) 座標 → 網格索引 (iy, ix)。非格點直接拋錯，不四捨五入吞掉。"""
    scaled = coords.astype(np.float64) * n
    idx = np.rint(scaled)
    off = np.abs(scaled - idx).max()
    if off > 1e-6:
        raise ValueError(f"sensor 座標不落在 {n}² 網格上（最大偏移 {off:.3e}）——"
                         f"本診斷用整數索引取樣，不做內插")
    idx = idx.astype(np.int64) % n
    # coords 的慣例是 (x, y)；場的索引是 [y, x]
    return idx[:, 1], idx[:, 0]


def main() -> None:
    a = parse_args()
    noises = [float(s) for s in a.noise.split(",") if s.strip()]

    u, v, t = load_dns_from_path(a.dns, time_stride=a.time_stride)
    n_t, ny, nx = u.shape
    if ny != nx:
        raise ValueError(f"本診斷假設方形網格，收到 {ny}x{nx}")
    meta = json.loads(_resolve_data_path(a.sensors).read_text())
    K = int(meta["K"])
    coords = np.asarray(meta["selected_coordinates"], dtype=np.float64)
    if coords.shape != (K, 2):
        raise ValueError(f"selected_coordinates {coords.shape} 與 K={K} 不符")
    iy, ix = sensor_grid_indices(coords, nx)
    print(f"[dns] frames={n_t} grid={ny}x{nx}  [sensors] {Path(a.sensors).name}  K={K}")

    # ── 快照矩陣與 POD（脈動）──────────────────────────────────────────
    x = np.concatenate([u.reshape(n_t, -1), v.reshape(n_t, -1)], axis=1).astype(np.float64)
    xm = x.mean(axis=0)
    xf = x - xm
    _, s, vt = np.linalg.svd(xf, full_matrices=False)
    cum = np.cumsum(s ** 2) / float((s ** 2).sum())
    print(f"[pod] r_99={int(np.searchsorted(cum,0.99))+1}  r_99.9={int(np.searchsorted(cum,0.999))+1}"
          f"  r_99.99={int(np.searchsorted(cum,0.9999))+1}   (快照上限 {n_t})")

    # 感測列：u 區塊在前半、v 區塊在後半
    flat = iy * nx + ix
    rows = np.concatenate([flat, flat + ny * nx])          # [2K]
    ref_norm = np.linalg.norm(x, axis=1)                   # 逐幀全場範數（含平均）
    rms = np.array([np.sqrt((u ** 2).mean()), np.sqrt((v ** 2).mean())])
    rng = np.random.default_rng(0)

    ranks = [r for r in RANKS if r <= min(len(s), 2 * K)]
    results = []
    print(f"\n{'r':>5}{'能量%':>10}{'截斷下界%':>11}{'κ(Θ)':>10}{'σ_min/σ_max':>13}"
          + "".join(f"{'oracle@'+format(nz,'g'):>13}" for nz in noises))
    for r in ranks:
        psi = vt[:r]                                        # [r, 2*Ngrid]，列正交
        theta = psi[:, rows].T                              # [2K, r]
        sv = np.linalg.svd(theta, compute_uv=False)
        kappa = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        floor = float(np.sqrt(max(1.0 - cum[r - 1], 0.0)) * 100.0)
        row = {"r": r, "energy": float(cum[r - 1]), "truncation_floor_pct": floor,
               "kappa": kappa, "sigma_min": float(sv[-1]), "sigma_max": float(sv[0])}
        pinv = np.linalg.pinv(theta)
        for nz in noises:
            errs = []
            for i in range(n_t):
                y = xf[i, rows].copy()
                if nz > 0:
                    sig = np.repeat(rms, K) * nz
                    y = y + rng.normal(scale=sig)
                rec = xm + psi.T @ (pinv @ y)
                errs.append(np.linalg.norm(rec - x[i]) / ref_norm[i])
            row[f"oracle_uv_rel_err_noise{nz:g}"] = float(np.mean(errs) * 100.0)
        results.append(row)
        print(f"{r:>5}{cum[r-1]*100:10.4f}{floor:11.3f}{kappa:10.2f}{sv[-1]/sv[0]:13.4f}"
              + "".join(f"{row['oracle_uv_rel_err_noise'+format(nz,'g')]:13.3f}" for nz in noises))

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        rev = None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "producer": "scripts/diag_pod_observability.py",
        "provenance": {"code_revision": rev, "dns": str(a.dns), "sensors": str(a.sensors),
                       "time_stride": a.time_stride, "n_frames": int(n_t),
                       "grid": [int(ny), int(nx)], "K": K, "noise_levels": noises,
                       "oracle_basis": "DNS POD (fluctuations); NOT sensor-only deployable"},
        "pod_cum_energy": cum.tolist(),
        "rows": results,
    }, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
