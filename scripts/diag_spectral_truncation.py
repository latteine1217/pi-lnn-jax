#!/usr/bin/env python
"""appendix06 的 spectral-truncation 下界 — 完美重建到 k_cut 時的殘餘誤差。

Why this exists:
    該表原本無生成碼（`git log -S` 只命中 .tex 本身），且**未載明時間平均窗**——
    全時段與 t>=1 兩種讀法給出不同的低-k_cut 值，無從對照。它承擔的角色是把
    B3 的 KE 與 vorticity 誤差映射到「有效截斷波數」，所以窗的定義直接決定
    appendix06 的 bandwidth attribution 結論。本腳本補上生成端並強制載明窗。

Definition:
    對每個 k_cut，把 DNS 場低通到 |k| <= k_cut（cyclic 波數）——k_cut 是**保留圓盤的
    半徑**，與 sqrt(K/pi) 數模的意義相同；這**不是** eq:radial_spectrum 的殼層編號，
    後者對應 |k| < k_cut + 0.5，兩者恰差半個殼層（k_cut=4 時 14.20% vs 7.77%，近兩倍）。
    選半徑的理由：本表的值要拿去和 k_max^sensor = sqrt(K/pi) 比，該量是圓盤半徑。
    再與未濾波場比較：

        KE loss   = |KE(filtered) - KE(full)| / KE(full)
        omega err = ||omega(filtered) - omega(full)||_2 / ||omega(full)||_2

    omega 用中央差分（與 metric_artifact.compute_vorticity 同算子），不是譜式 k^2 E(k)：
    要被反演的量測值本身就是有限差分算出來的，兩側算子必須一致才可比。

    兩者皆逐幀計算後對時間取算術平均——與 eq:rel_l2 / eq:ke_mape 的聚合順序一致
    （先逐幀成比值，再平均），而非先聚合再成比值。

    這是「完美重建」的下界：任何只恢復 k <= k_cut 的重建，其誤差不可能低於此。

Usage:
    uv run python scripts/diag_spectral_truncation.py \
        --dns-path data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
        --window all --output artifacts/diag/spectral_truncation_re10000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def lowpass(field: np.ndarray, kcut: float) -> np.ndarray:
    """把 [T,N,N] 場低通到 |k| <= kcut（cyclic 整數波數）。"""
    N = field.shape[-1]
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    mask = (np.sqrt(KX**2 + KY**2) <= kcut)
    return np.real(np.fft.ifft2(np.fft.fft2(field, axes=(-2, -1)) * mask, axes=(-2, -1)))


def vorticity(u: np.ndarray, v: np.ndarray, L: float = 1.0) -> np.ndarray:
    """omega = d_x v - d_y u，週期中央差分（與 metric_artifact.compute_vorticity 同慣例）。"""
    N = u.shape[-1]
    d = L / N
    dv_dx = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2 * d)
    du_dy = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * d)
    return dv_dx - du_dy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns-path", required=True)
    ap.add_argument("--k-cuts", default="4,5,6,8,16")
    ap.add_argument("--window", default="all", choices=["all", "post_spinup"],
                    help="all = 全部 201 幀；post_spinup = t >= 1 s（與 enstrophy 診斷同窗）")
    ap.add_argument("--output", default="artifacts/diag/spectral_truncation.json")
    a = ap.parse_args()

    raw = np.load(a.dns_path, allow_pickle=True)
    d = raw.item() if raw.dtype == object else raw
    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    t = np.asarray(d["time"], dtype=np.float64)

    sel = np.ones(t.size, dtype=bool) if a.window == "all" else (t >= 1.0)
    u, v, t_sel = u[sel], v[sel], t[sel]

    ke_full = 0.5 * (u**2 + v**2).mean(axis=(1, 2))          # 逐幀
    w_full = vorticity(u, v)
    w_norm = np.linalg.norm(w_full.reshape(w_full.shape[0], -1), axis=1)

    kcuts = [float(x) for x in a.k_cuts.split(",")]
    rows = []
    for kc in kcuts:
        uf, vf = lowpass(u, kc), lowpass(v, kc)
        ke_f = 0.5 * (uf**2 + vf**2).mean(axis=(1, 2))
        ke_loss = np.abs(ke_f - ke_full) / ke_full                       # 逐幀比值
        wf = vorticity(uf, vf)
        dw = (wf - w_full).reshape(wf.shape[0], -1)
        w_err = np.linalg.norm(dw, axis=1) / w_norm
        rows.append({
            "k_cut": kc,
            "ke_loss_pct": float(100 * ke_loss.mean()),
            "ke_loss_sd_pct": float(100 * ke_loss.std(ddof=1)),
            "omega_rel_l2_pct": float(100 * w_err.mean()),
            "omega_rel_l2_sd_pct": float(100 * w_err.std(ddof=1)),
        })

    out = {
        "provenance": {
            "script": "scripts/diag_spectral_truncation.py",
            "dns_path": str(Path(a.dns_path).resolve()),
            "grid": int(u.shape[-1]),
            "window": a.window,
            "n_frames": int(t_sel.size),
            "t_range": [float(t_sel[0]), float(t_sel[-1])],
            "wavenumber_convention": "cyclic, |k| <= k_cut on the FFT grid",
            "aggregation": "per-frame ratio, then arithmetic time mean",
        },
        "rows": rows,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"[trunc] window={a.window}  frames={t_sel.size}  t in [{t_sel[0]:.2f}, {t_sel[-1]:.2f}]  grid={u.shape[-1]}²")
    print(f"  {'k_cut':>6}  {'KE loss %':>12}  {'omega rel-L2 %':>16}")
    for r in rows:
        print(f"  {r['k_cut']:6.0f}  {r['ke_loss_pct']:12.2f}  {r['omega_rel_l2_pct']:16.2f}")
    print(f"[out] {op}")


if __name__ == "__main__":
    main()
