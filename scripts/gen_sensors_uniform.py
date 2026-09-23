"""完全均勻（farthest-point quasi-uniform）cylinder sensor placement。

複用 generate_sensors_qrpivot_cylinder 的 load/body/farthest-point。
動機：測 cross-attention 基於 sensor 間 relative position 的假設——均勻分布（一致間距、
規律相對位置）是否比 wake / boundary bias placement 更易重建，且符合 Nyquist 均勻取樣假設。
純 farthest-point on full fluid（最大化最小間距 = blue-noise 準均勻），無 boundary / wake bias。

輸出 schema 兼容 CylinderDataset / dump_cylinder_v1.py（json: selected_coordinates/sensor_*；
npz: t/u/v[K,T]）。
用法（2026-08-21 起 qrpivot / Arrow I/O 已在本專案，不需 pi-lnn env）：
  uv run python scripts/gen_sensors_uniform.py \\
    --shards <arrow> --K 400 --out data/cylinder_sensors
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
    ap.add_argument("--time-stride", type=int, default=20)
    ap.add_argument("--body-threshold", type=float, default=1e-4)
    ap.add_argument("--gen-dir", default="scripts", help="generate_sensors_qrpivot_cylinder.py 所在")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qr = load_qrpivot_module(args.gen_dir)
    shards = [qr.load_shard(Path(p)) for p in args.shards]
    geom = cylinder_geometry(shards, args.body_threshold, qr.detect_cylinder_mask)

    # 純 farthest-point 準均勻（最大化最小間距，全 fluid，無 bias）
    sensor_fluid_idx = np.sort(
        qr.farthest_point_sampling(geom.coords_grid, args.K, seed=args.seed)
    )[:args.K]
    print(f"farthest-point uniform K={len(sensor_fluid_idx)}")

    re_tag = f"Re{shards[0]['Re']:.0f}"
    write_sensor_set(
        out_dir=Path(args.out),
        base=f"sensors_uniform_K{args.K}_cylinder_{re_tag}",
        geom=geom, sensor_fluid_idx=sensor_fluid_idx, shard=shards[0],
        K=args.K, body_threshold=args.body_threshold,
        method="farthest_point_uniform", time_stride=args.time_stride,
    )


if __name__ == "__main__":
    main()
