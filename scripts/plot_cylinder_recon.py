"""畫 cylinder 重建場圖：DNS / Reconstruction / |Error| × (u, v, ω)。

讀 train_cylinder_v1.py --dump_fields 存的 npz（第一個 val 時刻的場）。
動機：取代之前 inline 臨時繪圖，固化成可複用、可重現的腳本。

座標 gx/gy 為等向 normalized grid（body 在此係正圓 → set_aspect('equal')）；
DNS 與 Reconstruction 每欄共用對稱色階（同色=同值，誠實視覺比較），|Error| 自帶色階。

用法：
  uv run python scripts/plot_cylinder_recon.py \
    --npz data/cylinder_re1781_recon.npz \
    --out docs/figures/cylinder_recon_re1781_best.png \
    --title "Re=1781 cylinder wake reconstruction (boundary K400 + FED128)"
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _masked(field, fluid):
    """body 內設 NaN，渲染為 set_bad 顏色。"""
    out = np.array(field, dtype=float)
    out[fluid < 0.5] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Cylinder wake reconstruction")
    ap.add_argument("--sensors", action="store_true", default=True,
                    help="DNS 列疊 sensor 位置")
    ap.add_argument("--no-sensors", dest="sensors", action="store_false")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    d = np.load(args.npz)
    gx, gy = d["gx"], d["gy"]
    fluid = d["mask"]
    bc, br = d["body_center"], float(d["body_radius"])
    u_rel = float(d["u_rel"])
    v_rel = float(d["v_rel"])

    cols = [
        ("u", d["dns_u"], d["pred_u"]),
        ("v", d["dns_v"], d["pred_v"]),
        ("$\\omega$", d["dns_w"], d["pred_w"]),
    ]
    row_labels = ["DNS", "Reconstruction", "|Error|"]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.2), constrained_layout=True)
    field_cmap = plt.get_cmap("RdBu_r").copy()
    field_cmap.set_bad("0.85")
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("0.85")

    x0, x1 = float(gx.min()), float(gx.max())
    y0, y1 = float(gy.min()), float(gy.max())
    # body 在資料中以 [0,1]² 正規化框的單一半徑圓表示 → 用 equal aspect 讓 body 呈正圓
    # （normalized 視圖；square-pixel 會把 body 拉成橢圓，視覺上像失真的 cylinder）。
    in_bbox = None
    if args.sensors and "sensor_pos" in d:
        sp = np.asarray(d["sensor_pos"])
        in_bbox = ((sp[:, 0] >= x0) & (sp[:, 0] <= x1) &
                   (sp[:, 1] >= y0) & (sp[:, 1] <= y1))

    for j, (name, dns, pred) in enumerate(cols):
        dns_m, pred_m = _masked(dns, fluid), _masked(pred, fluid)
        err_m = _masked(np.abs(pred - dns), fluid)
        vmax = float(np.nanmax(np.abs([np.nanmin(dns_m), np.nanmax(dns_m),
                                       np.nanmin(pred_m), np.nanmax(pred_m)])))
        vmax = vmax if vmax > 0 else 1.0
        emax = float(np.nanmax(err_m))
        emax = emax if emax > 0 else 1.0

        panels = [
            (dns_m, field_cmap, -vmax, vmax),
            (pred_m, field_cmap, -vmax, vmax),
            (err_m, err_cmap, 0.0, emax),
        ]
        for i, (data, cmap, vmin, vx) in enumerate(panels):
            ax = axes[i, j]
            im = ax.pcolormesh(gx, gy, data, cmap=cmap, vmin=vmin, vmax=vx,
                               shading="auto", rasterized=True)
            ax.add_patch(Circle((bc[0], bc[1]), br, facecolor="0.4",
                                 edgecolor="k", lw=0.8, zorder=5))
            ax.set_aspect("equal")
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0 and in_bbox is not None and in_bbox.any():
                ax.scatter(sp[in_bbox, 0], sp[in_bbox, 1], s=4, c="lime",
                           edgecolors="k", linewidths=0.2, zorder=6)
            if i == 0:
                ax.set_title(name, fontsize=14)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=13)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="4%", pad=0.04)
            fig.colorbar(im, cax=cax)

    fig.suptitle(f"{args.title}\nu rel-L2 = {u_rel:.3f}   v rel-L2 = {v_rel:.3f}"
                 f"   (DNS & Reconstruction share per-column scale; "
                 f"green = {int(in_bbox.sum()) if in_bbox is not None else 0} sensors)",
                 fontsize=13)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}  (u_rel={u_rel:.3f} v_rel={v_rel:.3f})")


if __name__ == "__main__":
    main()
