#!/usr/bin/env python3
"""LES 場上的 QR-pivot sensor 佈點（thesis fig:les_sensor_placement）。

What:
    placement LES（`N=256`, `t_end=50` s）最後一格的渦量場，疊上 sensor json 裡
    的 K 個 QR-pivot 座標。

Why 這支腳本存在:
    此圖原本沒有生成腳本（`docs/paper-figure-provenance.md` §3）。2026-09-17 的
    投稿前稽核記了一條：舊檔嵌了「LES vorticity ω at t = 50 with K=100 sensor
    placement」這行標題，與 caption 講同一件事，而 appendix02／TMLR 引用的四張
    圖只有這張帶內嵌標題——是 chartjunk；那行的 `t = 50` 另外缺單位。兩者都只能
    重產才修得掉。

    因此本檔**不畫標題**：圖說由 caption 負責。markers 的意義留在圖例裡（那是
    key，不是重複的標題）。

⚠ omega 的取用來源（2026-09-17 重產時查到的資料不一致）:
    `kolmogorov_les_Re10000_N256_T50_standalone.npy` 內，`u`／`v` **恰為其自身
    `omega` 所蘊含速度的相反數**（由 omega 反解 psi 再取 u,v，corr = −1.000000）。
    同一支生成器的 `..._T5_dns_init_FIXED.npy` 與 DNS 檔都是 +1.000000；T50 檔
    的日期（2026-05-17）早於生成器修正（`generate_kolmogorov_les.py.bak`、
    `_FIXED` 檔皆 2026-06-17）。

    舊圖畫的是**檔內 `omega`**，本檔預設沿用（`--omega-source stored`），不在重產
    時悄悄翻號——那是改動已發表圖的內容。`--omega-source velocity` 可改畫
    `∂v/∂x − ∂u/∂y`（即佈點腳本 `generate_sensors_qrpivot_from_les.py` 實際用的
    那一個）。兩者只差一個全域符號。

    **佈點本身不受影響**：該腳本五個特徵全部由 u,v 導出（含它自己算的 omega），
    全域符號對 POD/QR pivot 無作用。受影響的只有這張圖的色階方向。

記憶體:
    LES T50 全場是 2.4 GB 的 pickled dict，`np.load(...).item()` 會整個讀進來。
    取完最後一格立刻釋放，峰值約 2.5 GB。

Usage:
    PILNJAX_DATA_ROOT=<大檔池> uv run python scripts/plot_les_sensor_placement.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, save_figure  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pi_lnn_jax.data import _resolve_data_path  # noqa: E402
from pi_lnn_jax.metric_artifact import compute_vorticity  # noqa: E402

LES_DEFAULT = "data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy"
SENSORS_DEFAULT = ("data/kolmogorov_sensors/re10000/"
                   "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json")
LEGEND_PT = 8.5


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", default=LES_DEFAULT)
    ap.add_argument("--sensors", default=SENSORS_DEFAULT,
                    help="sensor json；其 les_source_file 必須與 --les 相符")
    ap.add_argument("--omega-source", choices=("stored", "velocity"), default="stored",
                    help="畫檔內 omega（沿用舊圖）或由 u,v 導出；見 module docstring")
    ap.add_argument("--width-in", type=float, default=0.7 * 425.2 / 72.0,
                    help="原生寬度（預設 = thesis 的 0.7\\linewidth，縮放係數 1）")
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "architecture"))
    args = ap.parse_args()

    meta_path = _resolve_data_path(args.sensors)
    if not meta_path.exists():
        raise FileNotFoundError(f"sensor json 不存在: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if Path(meta["les_source_file"]).name != Path(args.les).name:
        raise ValueError(
            "sensor json 的 les_source_file 與 --les 不是同一個檔：\n"
            f"  json: {meta['les_source_file']}\n  --les: {args.les}\n"
            "佈點畫在別的場上就不是這個佈點的圖了。"
        )
    coords = np.asarray(meta["selected_coordinates"], dtype=float)
    if coords.shape != (meta["K"], 2):
        raise ValueError(f"座標形狀 {coords.shape} 與 K={meta['K']} 不符")

    les_path = _resolve_data_path(args.les)
    if not les_path.exists():
        raise FileNotFoundError(f"LES .npy 不存在: {les_path}")
    obj = np.load(les_path, allow_pickle=True).item()
    t = np.asarray(obj["time"], dtype=float)
    u = np.asarray(obj["u"][-1], dtype=np.float64)
    v = np.asarray(obj["v"][-1], dtype=np.float64)
    w_stored = np.asarray(obj["omega"][-1], dtype=np.float64)
    cfg = dict(obj["config"])
    del obj

    w_vel = compute_vorticity(u, v, domain_length=float(cfg.get("L", 1.0)))
    agree = float(np.corrcoef(w_vel.ravel(), w_stored.ravel())[0, 1])
    if agree < 0.99:
        print("=" * 72)
        print("[WARN] 檔內 omega 與 ∂v/∂x−∂u/∂y 不一致 "
              f"(corr = {agree:+.6f})：{les_path}")
        print(f"[WARN] 本圖畫的是 --omega-source={args.omega_source}；"
              "兩者只差全域符號，見 module docstring。")
        print("=" * 72)
    w = w_stored if args.omega_source == "stored" else w_vel
    lim = float(np.abs(w).max())
    print(f"[check] LES 末格 t={t[-1]:.4f} s（T_end={cfg.get('T_end')}），"
          f"N={cfg.get('N')}, |omega|max={lim:.3f} 1/s, omega_rms={np.sqrt((w**2).mean()):.3f}")
    print(f"[check] K={meta['K']} 個座標，x∈[{coords[:,0].min():.3f},{coords[:,0].max():.3f}] "
          f"y∈[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]")

    setup_style("thesis")
    fig, ax = plt.subplots(figsize=(args.width_in, args.width_in * 0.865))
    ax.grid(False)
    im = ax.imshow(w.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                   vmin=-lim, vmax=lim, rasterized=True)
    ax.scatter(coords[:, 0], coords[:, 1], s=14, facecolors="white",
               edgecolors="black", linewidths=0.6, zorder=3,
               label=f"$K={meta['K']}$ sensors (QR-pivot on LES)")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=LEGEND_PT, framealpha=0.9,
              handletextpad=0.3, borderpad=0.35)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\omega$ (1/s)")

    for p in save_figure(fig, str(Path(args.out) / "les_T50_vorticity_with_sensors")):
        print(f"[write] {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
