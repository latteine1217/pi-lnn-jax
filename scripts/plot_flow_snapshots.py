#!/usr/bin/env python3
"""渦度場快照面板：看這條軌跡在不同時刻長什麼樣。

What:
    在指定時刻畫渦度場，每格標註該時刻的 ω_rms 與 enstrophy。可選疊上 sensor 位置。

Why:
    這條軌跡的 enstrophy 跨越 144 → 5.4 → 67（衰減到谷底再被 forcing 推起來），
    純看數字很難理解「評估窗長什麼樣」。面板圖讓「評估窗落在最安靜的凹陷處」
    這件事一眼可見。

色階選擇（判讀前提）：
    **每格各自normalize 到 ±3σ**，因為跨時刻的 ω 幅度差一個數量級以上，共用色階
    會讓低 enstrophy 的格子全白、看不到結構。要比幅度看標註的 ω_rms，不要比顏色深淺。
    `--shared-scale` 可切成共用色階（看幅度、犧牲結構）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo 內 scripts/journal_style.py
from journal_style import save_figure, setup_style

from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ω = ∂v/∂x − ∂u/∂y（譜空間；axis0=x, axis1=y，對齊 DNS convention）。"""
    N = u.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return np.fft.ifft2(1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)).real


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", required=True)
    ap.add_argument("--times", nargs="+", type=float, required=True)
    ap.add_argument("--sensor-json", default=None, help="疊 sensor 位置（只畫在第一格）")
    ap.add_argument("--shared-scale", action="store_true",
                    help="共用色階（看幅度）；預設每格 ±3σ（看結構）")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="flow_snapshots")
    args = ap.parse_args()

    u_all, v_all, t_all = load_dns_from_path(args.dns, time_stride=1)
    t_all = np.asarray(t_all, dtype=float)

    panels = []
    for want in args.times:
        i = int(np.argmin(np.abs(t_all - want)))
        if abs(float(t_all[i]) - want) > 0.5 * float(np.diff(t_all).mean()):
            raise ValueError(
                f"DNS 時間軸上沒有接近 t={want} 的幀（最近 {float(t_all[i])}）；不寬鬆對齊")
        u, v = np.asarray(u_all[i], float), np.asarray(v_all[i], float)
        w = vorticity(u, v)
        panels.append({"t": float(t_all[i]), "w": w,
                       "rms": float(np.sqrt(np.mean(w ** 2))),
                       "enstrophy": float(0.5 * np.mean(w ** 2)),
                       "ke": float(0.5 * np.mean(u ** 2 + v ** 2))})

    pos = None
    if args.sensor_json:
        pos = np.asarray(load_sensors_from_path(args.sensor_json, time_stride=1)["sensor_pos"])

    setup_style()
    import matplotlib.pyplot as plt
    ncols = min(args.ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 2.25 * nrows),
                             squeeze=False)
    vmax_shared = max(3.0 * p["rms"] for p in panels)

    for ax, p in zip(axes.ravel(), panels):
        vmax = vmax_shared if args.shared_scale else 3.0 * p["rms"]
        ax.imshow(p["w"].T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(rf"$t={p['t']:.0f}$   $Z={p['enstrophy']:.1f}$", fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        if pos is not None and p is panels[0]:
            ax.scatter(pos[:, 0], pos[:, 1], s=1.2, c="k", alpha=0.7, linewidths=0)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.tight_layout(pad=0.4)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_dir / args.stem))

    print(f"{'t':>7} {'enstrophy':>10} {'KE':>9} {'omega_rms':>10}")
    for p in panels:
        print(f"{p['t']:>7.1f} {p['enstrophy']:>10.2f} {p['ke']:>9.4f} {p['rms']:>10.2f}")
    print(f"[out] {out_dir / (args.stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
