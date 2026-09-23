#!/usr/bin/env python3
"""量測 body 邊界那一圈格點對「導數型」指標的污染量級（technical-debt TD-24 的先量後決）。

問題：非週期路徑的導數走 `np.gradient`，body mask 是**算完之後**才套的
（`metric_artifact.compute_divergence_l2` / `compute_vorticity`）。緊鄰 body 的
那一圈流體格點，其中央差分 stencil 會跨進 body 內部；事後套 mask 保留的正是
這一圈。受影響的是 `div_*_l2` / `omega_rel_err` / `enstrophy_rel_err`。

本腳本**只量參考場（DNS）自己**：不需要 checkpoint、不需要模型。DNS 場在 body
內部是 no-slip 的零速，跨界 stencil 的假梯度因此是 O(u_wall_adjacent/Δ)，量級
與真實近壁剪切同階——所以「這一圈到底把 RMS 抬高多少」是可以在參考場上直接讀出來的
上界指標。pred 側的比值型指標（`omega_rel_err` 的分子分母同時受影響）要完整判定
仍需重跑一次 cylinder eval；本腳本的用途是決定「值不值得重跑」。

用法：
    uv run python scripts/diag_mask_erosion.py --npz data/cylinder_v1.npz
    uv run python scripts/diag_mask_erosion.py --synthetic   # 本機自檢，不需資料

輸出的判讀：`delta` 欄是「侵蝕一格後的值 / 原值 - 1」。接近 0 表示那一圈可忽略，
記一條 known-limitation 收工；顯著非零表示現有 cylinder 的 ω / enstrophy / div
數字帶著系統性污染，需要決定是否改程式並重跑。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from pi_lnn_jax.metric_artifact import compute_divergence_l2, compute_vorticity

#: 4-鄰域十字。np.gradient 的中央差分只跨 ±1 格，所以侵蝕一格即足夠；
#: 用十字而非 3x3 方塊是因為 x 與 y 的差分互不相干，對角格點不進任一 stencil。
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def erode(mask: np.ndarray) -> np.ndarray:
    """導數型指標該用的遮罩：比場遮罩內縮一格。

    `border_value=1` 讓**域邊界**不被侵蝕——那裡 `np.gradient` 自己會退到單邊差分，
    取的值全在域內，沒有跨界污染。會污染的只有 body 邊界。
    """
    return binary_erosion(mask, structure=_CROSS, border_value=1)


def _frame_measures(u, v, mask, dns_x, dns_y, Lx, Ly) -> dict[str, float]:
    div = compute_divergence_l2(u, v, periodic=False, dns_x=dns_x, dns_y=dns_y,
                                Lx=Lx, Ly=Ly, mask=mask)
    omega = compute_vorticity(u, v, periodic=False, dns_x=dns_x, dns_y=dns_y,
                              Lx=Lx, Ly=Ly)
    sel = mask.astype(bool)
    return {
        "div_rms": div,
        "omega_rms": float(np.sqrt(np.mean((omega ** 2)[sel]))),
        "enstrophy": 0.5 * float(np.mean((omega ** 2)[sel])),
    }


def report(u_seq, v_seq, mask, dns_x, dns_y, Lx, Ly, label: str) -> None:
    mask = mask.astype(bool)
    mask_e = erode(mask)
    n, n_e = int(mask.sum()), int(mask_e.sum())
    print(f"=== {label} ===")
    print(f"frames={len(u_seq)}  grid={mask.shape}  "
          f"mask cells {n} → {n_e}（侵蝕掉 {n - n_e}，佔 {100 * (n - n_e) / n:.2f}%）")
    rows = {k: ([], []) for k in ("div_rms", "omega_rms", "enstrophy")}
    for u, v in zip(u_seq, v_seq, strict=True):
        a = _frame_measures(u, v, mask, dns_x, dns_y, Lx, Ly)
        b = _frame_measures(u, v, mask_e, dns_x, dns_y, Lx, Ly)
        for k in rows:
            rows[k][0].append(a[k])
            rows[k][1].append(b[k])
    print(f"{'metric':<12}{'as-is':>14}{'eroded':>14}{'delta':>12}")
    for k, (raw, ero) in rows.items():
        r, e = float(np.mean(raw)), float(np.mean(ero))
        print(f"{k:<12}{r:>14.6g}{e:>14.6g}{(e / r - 1.0) * 100:>11.2f}%")
    print()


def _synthetic() -> None:
    """解析場自檢：無散度渦流 + 圓形 body（body 內強制 0，模擬 no-slip）。

    存在的理由是這支腳本在沒有 cylinder 資料的機器上也必須被實際執行過一次——
    未跑過的診斷腳本不是證據。這裡的數字不代表真實 cylinder 的量級。
    """
    H, W = 128, 256
    dns_x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    dns_y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    gx, gy = np.meshgrid(dns_x, dns_y)
    u = np.sin(2 * np.pi * gy) * np.cos(2 * np.pi * gx)
    v = -np.cos(2 * np.pi * gy) * np.sin(2 * np.pi * gx)
    cen, rad = (0.25, 0.5), 0.06
    mask = np.sqrt((gx - cen[0]) ** 2 + (gy - cen[1]) ** 2) - rad > 0
    u = np.where(mask, u, 0.0)
    v = np.where(mask, v, 0.0)
    report([u], [v], mask, dns_x, dns_y, 0.325, 0.178, "synthetic self-check")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, help="cylinder dump（dump_cylinder_v1.py 產出）")
    p.add_argument("--synthetic", action="store_true", help="解析場自檢，不需資料")
    args = p.parse_args()
    if not args.npz and not args.synthetic:
        p.error("需要 --npz 或 --synthetic")
    if args.synthetic:
        _synthetic()
    if not args.npz:
        return
    if not args.npz.exists():
        raise FileNotFoundError(f"cylinder dump 不存在: {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    need = ("dns_u", "dns_v", "dns_x", "dns_y", "Lx", "Ly",
            "body_center", "body_radius")
    missing = [k for k in need if k not in d]
    if missing:
        raise KeyError(f"{args.npz} 缺欄位 {missing}；本診斷不從其他欄位推導幾何")
    dns_x, dns_y = np.asarray(d["dns_x"]), np.asarray(d["dns_y"])
    gx, gy = np.meshgrid(dns_x, dns_y)
    cen, rad = np.asarray(d["body_center"]), float(d["body_radius"])
    # 與 pipeline/cylinder/run.py 的 body_mask 同式（normalized 座標的圓）。
    mask = np.sqrt((gx - cen[0]) ** 2 + (gy - cen[1]) ** 2) - rad > 0
    report(np.asarray(d["dns_u"]), np.asarray(d["dns_v"]), mask,
           dns_x, dns_y, float(d["Lx"]), float(d["Ly"]), f"DNS reference — {args.npz}")


if __name__ == "__main__":
    main()
