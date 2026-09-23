"""Dump cylinder data for pi-lnn-jax v1。

用法（2026-08-21 起 CylinderDataset / Arrow I/O 已移植進本專案，不再借 pi-lnn env）：
  uv run python scripts/legacy/dump_cylinder_v1.py --out data/cylinder_v1.npz

輸出 .npz（pi-lnn-jax 讀）：
  sensors:  sensor_vals[K,T,C], sensor_pos[K,2], sensor_time[T],
            obs_mean[C], obs_std[C], train_t_idx, val_t_idx
  geometry: body_center[2], body_radius（圓擬合，給解析 SDF）, Lx, Ly
  DNS eval: dns_u/dns_v[T_eval,H,W], dns_x[W], dns_y[H], dns_t_idx
  meta:     re_value
"""
import argparse
import sys
import numpy as np

ARROW = "/home/junyi/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical/data-00000-of-00092.arrow"
RE = 10031.0
SUB = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=100,
                    help="sensor 數（對應 sensors_qrpivot_K{K}_cylinder_Re10031 檔）")
    ap.add_argument("--sensor_stem", default=None,
                    help="sensor 檔名 stem（不含 .json/_values.npz）；指定則 override --K 的 qrpivot 構造，"
                         "用於不同 placement，如 sensors_hybrid95downstream_K95_cylinder_Re10031")
    ap.add_argument("--n-eval-times", type=int, default=30, help="dump 幾個 DNS 時刻供 field eval")
    ap.add_argument("--arrow", default=ARROW, help="DNS arrow shard（不同 Re 用不同 shard，如 Re1781=data-00020）")
    ap.add_argument("--re", type=float, default=RE, help="Re value（對應 arrow 的 Re）")
    args = ap.parse_args()
    arrow, re_val = args.arrow, args.re

    stem = args.sensor_stem or f"sensors_qrpivot_K{args.K}_cylinder_Re10031"
    sensor_json = f"data/cylinder_sensors/{stem}.json"
    sensor_npz = f"data/cylinder_sensors/{stem}_values.npz"
    print(f"[0] sensor placement: {stem}")

    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _common.cylinder_dataset import CylinderDataset
    from _common.cylinder_arrow import load_arrow_fields

    print(f"[1] CylinderDataset 載入 sensors (K={args.K}) + body + 正規化 ...")
    ds = CylinderDataset(sensor_json, sensor_npz, arrow, re_value=re_val, sensor_subsample=SUB)
    print(f"    sensor_vals={ds.sensor_vals.shape} sensor_pos={ds.sensor_pos.shape} "
          f"sensor_time={ds.sensor_time.shape}")
    print(f"    Lx={ds.Lx:.4f} Ly={ds.Ly:.4f}  obs={ds.observed_channel_names}")

    # ── 圓擬合（解析 SDF：center=mean, R=sqrt(2·mean d²) 等面積估計）──
    body = ds.body_xy                              # [N_body,2] normalized
    center = body.mean(axis=0)
    d2 = np.sum((body - center) ** 2, axis=1)
    radius = float(np.sqrt(2.0 * d2.mean()))       # 等面積半徑（uniform disk: mean d²=R²/2）
    print(f"[2] body 圓擬合: center=({center[0]:.4f},{center[1]:.4f}) R={radius:.4f} "
          f"(N_body={len(body)})")

    # ── DNS 全場（subsample T 對齊 sensor_time）──
    print("[3] 載入 DNS 全場 (arrow) ...")
    f = load_arrow_fields(arrow)
    T_full, H, W = f["T"], f["H"], f["W"]
    t_idx_sub = np.arange(0, T_full, SUB)          # 與 dataset 對齊
    u_sub = f["u"][t_idx_sub]                      # [T_sub,H,W]
    v_sub = f["v"][t_idx_sub]
    # normalized grid 軸
    x_lo, x_hi, y_lo, y_hi = ds.x_lo, ds.x_hi, ds.y_lo, ds.y_hi
    dns_x = ((f["x"][0, :] - x_lo) / (x_hi - x_lo)).astype(np.float32)   # [W]
    dns_y = ((f["y"][:, 0] - y_lo) / (y_hi - y_lo)).astype(np.float32)   # [H]
    print(f"    grid T_sub={u_sub.shape[0]} H={H} W={W}")

    # eval 用 val 時刻（held-out），最多 n_eval_times 個
    val_idx = np.sort(ds.val_t_idx)
    eval_t = val_idx[:: max(1, len(val_idx) // args.n_eval_times)][:args.n_eval_times]
    print(f"[4] DNS eval 取 {len(eval_t)} 個 val 時刻")

    out = dict(
        sensor_vals=ds.sensor_vals.astype(np.float32),
        sensor_pos=ds.sensor_pos.astype(np.float32),
        sensor_time=ds.sensor_time.astype(np.float32),
        obs_mean=ds.observed_channel_mean.astype(np.float32),
        obs_std=ds.observed_channel_std.astype(np.float32),
        train_t_idx=ds.train_t_idx.astype(np.int32),
        val_t_idx=ds.val_t_idx.astype(np.int32),
        body_center=center.astype(np.float32),
        body_radius=np.float32(radius),
        body_xy=body.astype(np.float32),               # 填充 disk 點（BC no-slip 取樣源）
        bc_inflow_u=np.float32(ds.bc_inflow_u),         # inlet u（DNS 量測，BC 用）
        Lx=np.float32(ds.Lx), Ly=np.float32(ds.Ly),
        re_value=np.float32(re_val),
        dns_u=u_sub[eval_t].astype(np.float32),     # [n_eval,H,W] 物理單位
        dns_v=v_sub[eval_t].astype(np.float32),
        dns_x=dns_x, dns_y=dns_y,
        dns_t_idx=eval_t.astype(np.int32),
    )
    np.savez_compressed(args.out, **out)
    sz = sum(a.nbytes for a in out.values()) / 1e6
    print(f"[5] 已存 {args.out}  (~{sz:.1f} MB raw, compressed 更小)")


if __name__ == "__main__":
    main()
