"""3-stage cylinder sensor placement：全域均勻 + 邊界層 bias + wake QR。

複用 generate_sensors_qrpivot_cylinder 的 load/body/snapshot/farthest/QR。
動機：hybrid(uniform+QR) 缺 body/wall 邊界層覆蓋；加 boundary stage 在
body 表面附近 + 上下 wall 附近 farthest-point 加密（邊界層梯度大、resolution 關鍵）。

Stage A 全域均勻（含前側 freestream）→ Stage B 邊界層 bias → Stage C wake QR。
輸出 schema 兼容 CylinderDataset（json: selected_coordinates/sensor_*；npz: t/u/v[K,T]）。

用法（2026-08-21 起不需 pi-lnn env）：
  uv run python scripts/gen_sensors_boundary.py \
    --shards <arrow> --K 400 --n-uniform 120 --n-boundary 120 --bl-thresh 0.03 \
    --out data/cylinder_sensors
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common.cylinder_sensors import (  # noqa: E402
    cylinder_geometry, load_qrpivot_module, write_sensor_set,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--n-uniform", type=int, default=120, help="全域均勻 farthest-point（含前側）")
    ap.add_argument("--n-boundary", type=int, default=120, help="邊界層 bias（body+wall 附近）")
    ap.add_argument("--bl-thresh", type=float, default=0.03, help="邊界層厚度（物理單位）")
    ap.add_argument("--time-stride", type=int, default=20)
    ap.add_argument("--body-threshold", type=float, default=1e-4)
    ap.add_argument("--gen-dir", default="scripts", help="generate_sensors_qrpivot_cylinder.py 所在")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qr = load_qrpivot_module(args.gen_dir)
    shards = [qr.load_shard(Path(p)) for p in args.shards]
    geom = cylinder_geometry(shards, args.body_threshold, qr.detect_cylinder_mask)
    farthest_point_sampling = qr.farthest_point_sampling
    coords_grid, fluid_indices = geom.coords_grid, geom.fluid_indices
    bc, br = geom.body_center, geom.body_radius

    # 邊界層 mask（fluid order）：body 表面附近 OR 上下 wall 附近
    d_body = np.sqrt((geom.fx - bc[0]) ** 2 + (geom.fy - bc[1]) ** 2) - br
    ylo, yhi = float(geom.y2d.min()), float(geom.y2d.max())
    wall_d = np.minimum(geom.fy - ylo, yhi - geom.fy)
    bl_idx = np.argwhere((d_body < args.bl_thresh) | (wall_d < args.bl_thresh)).ravel()
    n_near_body = int((d_body < args.bl_thresh).sum())
    n_near_wall = int((wall_d < args.bl_thresh).sum())
    print(f"邊界層 fluid 點: {len(bl_idx)}（body<{args.bl_thresh}: {n_near_body}, wall<{args.bl_thresh}: {n_near_wall}）")

    # Stage A: 全域均勻
    print(f"Stage A: 全域 farthest-point {args.n_uniform}")
    uni = farthest_point_sampling(coords_grid, args.n_uniform, seed=0)
    # Stage B: 邊界層 farthest-point
    nb = min(args.n_boundary, len(bl_idx))
    print(f"Stage B: 邊界層 farthest-point {nb}")
    bl_local = farthest_point_sampling(coords_grid[bl_idx], nb, seed=1)
    bnd = bl_idx[bl_local]
    chosen = np.unique(np.concatenate([uni, bnd]))
    # Stage C: QR 在剩餘（wake informative）
    n_qr = args.K - len(chosen)
    print(f"Stage C: QR pivot {n_qr}（剩餘 {len(fluid_indices) - len(chosen)} fluid）")
    A = qr.build_snapshot_matrix(shards, args.time_stride, geom.fluid_mask)
    remain = np.setdiff1d(np.arange(len(fluid_indices)), chosen)
    qr_local = qr.qr_pivot_sensors(A[:, remain], n_qr)
    qr_pick = remain[qr_local]
    sensor_fluid_idx = np.sort(np.concatenate([chosen, qr_pick]))[:args.K]

    re_tag = f"Re{shards[0]['Re']:.0f}"
    write_sensor_set(
        out_dir=Path(args.out),
        base=f"sensors_bnd{args.n_uniform}u{nb}b_K{args.K}_cylinder_{re_tag}",
        geom=geom, sensor_fluid_idx=sensor_fluid_idx, shard=shards[0],
        K=args.K, body_threshold=args.body_threshold,
        method="uniform_boundary_qr", time_stride=args.time_stride,
        extra={
            "n_uniform": int(args.n_uniform), "n_boundary": int(nb), "n_qr": int(n_qr),
        },
    )


if __name__ == "__main__":
    main()
