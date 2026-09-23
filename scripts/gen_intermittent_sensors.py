#!/usr/bin/env python3
"""從乾淨 sensor set 產生「間歇可用」的副本（thesis §7.2 intermittent availability）。

What:
    隨機丟棄一定比例的時間 frame，輸出時間軸不等距的 sensor set。每個 dropout
    rate 一個固定 mask（thesis 的協定），並在 JSON metadata 記下實際的 gap 統計
    （最長 gap、保留幀數），供與論文表格的 gap-bin 對照。

Why 這個實驗不需要動任何 pipeline code:
    CfC 分支以實際時間差前進狀態（models.py: dts = diff(sensor_time)），decoder
    以 sum(sensor_time <= t_q) - 1 對齊 query 時間並算出 dt_to_query——後者正是
    論文所謂的 staleness。不等距時間軸因此是原生支援的輸入，不是特例。
    這也是這個實驗的重點：vanilla 分支只能讀「最近一筆」並在 gap 中保持輸入
    不變，CfC 則把 gap 長度當成積分步長，兩者的差異就是連續時間的價值。

Why 丟 frame 而不是丟 sensor:
    這裡丟的是**時間**（封包遺失、感測器離線），與空間 sensor dropout
    （exp_drop：遮蔽部分 sensor 位置）是不同的失效模式，兩者不可互相替代。

Usage:
    uv run python scripts/gen_intermittent_sensors.py \
        --source data/kolmogorov_sensors/re10000/sensors_....json \
        --rates 0.3,0.5,0.7,0.9 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="乾淨 sensor JSON 路徑")
    p.add_argument("--rates", default="0.3,0.5,0.7,0.9",
                   help="要丟棄的 frame 比例，逗號分隔")
    p.add_argument("--seed", type=int, default=42,
                   help="mask 的 RNG seed（每個 rate 一個固定 mask）")
    p.add_argument("--pre-stride", type=int, default=1,
                   help="丟 frame 前先做的等距下採樣。要與主線 baseline 可比就給"
                        "主線的 time_stride（EXP-245 為 2，201→101 frames），"
                        "否則 intermittent 的 clean 對照會落在不同的時間解析度上。"
                        "產出的 config 必須設 time_strides=[1]，不然會被再採樣一次")
    p.add_argument("--out-dir", default=None, help="輸出目錄，預設與 source 同目錄")
    return p.parse_args()


def _gap_stats(kept_idx: np.ndarray, n_total: int) -> dict:
    """以 frame 為單位的 gap 統計；gap = 兩次保留之間隔了幾個 frame。"""
    gaps = np.diff(kept_idx) - 1
    return {
        "n_kept": int(kept_idx.size),
        "n_total": int(n_total),
        "longest_gap_frames": int(gaps.max()) if gaps.size else 0,
        "mean_gap_frames": float(gaps.mean()) if gaps.size else 0.0,
        "first_kept_frame": int(kept_idx[0]),
    }


def main() -> int:
    args = parse_args()
    src_json = Path(args.source)
    if not src_json.is_file():
        raise FileNotFoundError(f"來源 sensor JSON 不存在: {src_json}")
    meta = json.loads(src_json.read_text())

    npz_name = meta.get("dns_values_npz")
    if not npz_name:
        raise ValueError(f"{src_json} 缺 dns_values_npz 欄位")
    src_npz = src_json.parent / Path(npz_name).name
    if not src_npz.is_file():
        raise FileNotFoundError(f"觀測值 npz 不存在: {src_npz}")

    data = np.load(src_npz)
    u, v, t = data["u"], data["v"], data["time"]   # u,v: [K, T]; t: [T]
    if args.pre_stride < 1:
        raise ValueError(f"--pre-stride 必須 >= 1，收到 {args.pre_stride}")
    if args.pre_stride > 1:
        u, v, t = u[:, ::args.pre_stride], v[:, ::args.pre_stride], t[::args.pre_stride]
        print(f"  pre-stride {args.pre_stride} → {t.size} frames（對齊主線 baseline）")
    n_total = t.size

    out_dir = Path(args.out_dir) if args.out_dir else src_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_json.stem

    for rate in (float(x) for x in args.rates.split(",")):
        if not (0.0 < rate < 1.0):
            raise ValueError(f"dropout rate 必須在 (0,1)，收到 {rate}")
        n_keep = int(round(n_total * (1.0 - rate)))
        if n_keep < 2:
            raise ValueError(
                f"rate={rate} 只留下 {n_keep} 幀，CfC 至少需要 2 幀才有 Δt"
            )
        rng = np.random.default_rng(args.seed)
        kept = np.sort(rng.choice(n_total, size=n_keep, replace=False))

        tag = f"gap{round(rate * 100):02d}"
        out_npz = out_dir / f"{stem}_{tag}_dns_values.npz"
        out_json = out_dir / f"{stem}_{tag}.json"
        np.savez(out_npz, u=u[:, kept], v=v[:, kept], time=t[kept])

        stats = _gap_stats(kept, n_total)
        new_meta = dict(meta)
        new_meta["dns_values_npz"] = out_npz.name
        new_meta["intermittent_dropout_rate"] = rate
        new_meta["intermittent_seed"] = args.seed
        new_meta["clean_source"] = src_json.name
        new_meta["pre_stride"] = int(args.pre_stride)
        new_meta["time_steps"] = int(n_keep)
        new_meta.update(stats)
        out_json.write_text(json.dumps(new_meta, indent=2))

        print(f"  {out_json.name}  kept {stats['n_kept']}/{n_total}  "
              f"longest gap {stats['longest_gap_frames']} frames")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
