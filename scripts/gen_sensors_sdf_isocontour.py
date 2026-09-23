"""SDF 等值線 cylinder sensor placement：沿 body signed-distance 的線性間距等值線均勻布點。

複用 generate_sensors_qrpivot_cylinder 的 load/body/farthest-point。
動機：wake-only QR 集中尾流、上游 freestream 無 sensor → over-energy。改用沿 body SDF
（d_body=√((x-cx)²+(y-cy)²)-R，像勢能場等位線）的 m 條線性間距等值線（同心等距環）布點，
每條 farthest-point 均勻取點，達全域覆蓋 + 幾何感知。

Stage: 計算 SDF → m 條線性 iso-levels（body 近場→遠場）→ 每條 farthest-point 取 K/m
       → 不足 K 以全域 farthest 補足。
輸出 schema 兼容 CylinderDataset（json: selected_coordinates/sensor_*；npz: t/u/v[K,T]）。

用法（2026-08-21 起 qrpivot / Arrow I/O 已在本專案，不需 pi-lnn env）：
  uv run python scripts/gen_sensors_sdf_isocontour.py \\
    --shards <arrow> --K 400 --n-levels 8 --out data/cylinder_sensors
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
    ap.add_argument("--n-levels", type=int, default=8, help="SDF 等值線條數（線性間距）")
    ap.add_argument("--band-frac", type=float, default=0.6,
                    help="每條等值線帶寬 = band_frac × level 間距（過小易空帶，過大易重疊）")
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

    # SDF（fluid order）：到 body 的 signed distance（d=0 body 表面，d↑ 遠離）
    d_body = np.sqrt((geom.fx - bc[0]) ** 2 + (geom.fy - bc[1]) ** 2) - br
    d_min, d_max = float(d_body.min()), float(d_body.max())
    m = args.n_levels
    # 線性間距 iso-levels：去掉兩端極值（避免貼壁/貼域界的退化帶）
    levels = np.linspace(d_min, d_max, m + 2)[1:-1]
    spacing = (d_max - d_min) / (m + 1)
    band = args.band_frac * spacing
    per_level = max(1, args.K // m)
    print(f"SDF d∈[{d_min:.4f},{d_max:.4f}] {m} linear levels spacing={spacing:.4f} "
          f"band={band:.4f} per_level≈{per_level}")

    chosen = []
    for li, L in enumerate(levels):
        band_idx = np.argwhere(np.abs(d_body - L) < band).ravel()
        n_pick = min(per_level, len(band_idx))
        if n_pick == 0:
            print(f"  level {li} d={L:.4f}: 空帶，跳過")
            continue
        local = farthest_point_sampling(coords_grid[band_idx], n_pick, seed=li)
        chosen.append(band_idx[local])
        print(f"  level {li} d={L:.4f}: {len(band_idx)} 候選 → 取 {n_pick}")
    sensor_fluid_idx = np.unique(np.concatenate(chosen)) if chosen else np.array([], int)
    # 不足 K → 全域 farthest 補足（保證 K 一致，公平對比 boundary K400）
    if len(sensor_fluid_idx) < args.K:
        remain = np.setdiff1d(np.arange(len(fluid_indices)), sensor_fluid_idx)
        need = args.K - len(sensor_fluid_idx)
        fill = farthest_point_sampling(coords_grid[remain], need, seed=99)
        sensor_fluid_idx = np.concatenate([sensor_fluid_idx, remain[fill]])
        print(f"  不足 K，全域 farthest 補 {need}")
    sensor_fluid_idx = np.sort(sensor_fluid_idx)[:args.K]

    re_tag = f"Re{shards[0]['Re']:.0f}"
    write_sensor_set(
        out_dir=Path(args.out),
        base=f"sensors_sdf_iso{m}_K{args.K}_cylinder_{re_tag}",
        geom=geom, sensor_fluid_idx=sensor_fluid_idx, shard=shards[0],
        K=args.K, body_threshold=args.body_threshold,
        method="sdf_isocontour_linear", time_stride=args.time_stride,
        extra={"n_levels": int(m), "band": float(band)},
    )


if __name__ == "__main__":
    main()
