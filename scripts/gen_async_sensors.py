"""gen_async_sensors.py — 由既有 sensor 組產生「非同步不規則取樣」版本。

每個 sensor 各自隨機保留 n_keep 幀（其餘填 NaN），而不是全部 sensor 同時缺幀（那是間歇取樣，
`gen_sensors_qrpivot_les.py` 的 gap 檔）。n_keep 預設取與 gap 檔相同的保留幀數，兩種取樣每個 sensor 的
觀測量相同，差別只在缺值是否同步。`--n-keep` 直接指定保留幀數，用於 `--pre-stride 1` 的細時間網格
（觀測時刻不全落在 stride 2 的網格上，觀測量仍與粗網格版相同）。保證每一幀至少一個 sensor 有觀測（否則該幀的 token attention 無鍵可讀）。

讀取端：`data.load_sensors_from_path` 遇到 NaN 只用觀測值算正規化常數；模型須設 `async_sensors = true`
（NaN＝沒有觀測），訓練端 `check_async_sensors` 會在兩者不一致時報錯。

Usage:
    uv run python scripts/gen_async_sensors.py \
        --src data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50vort.json \
        --pre-stride 2 --drop-rate 0.7 --seed 42 \
        --out data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50vort_async70.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="完整時間序列的 sensor JSON")
    ap.add_argument("--pre-stride", type=int, required=True, help="先對完整序列做的時間 stride（與 gap 檔相同）")
    keep_arg = ap.add_mutually_exclusive_group(required=True)
    keep_arg.add_argument("--drop-rate", type=float, help="每個 sensor 丟掉的幀比例")
    keep_arg.add_argument("--n-keep", type=int,
                          help="每個 sensor 保留的幀數；在較細的時間網格上維持與粗網格相同的觀測量時用")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="輸出 JSON；NPZ 寫在同目錄的 <stem>_dns_values.npz")
    ap.add_argument("--max-tries", type=int, default=1000)
    args = ap.parse_args()

    src = _resolve_data_path(args.src)
    meta = json.loads(src.read_text())
    npz_path = src.parent / Path(meta["dns_values_npz"]).name
    if not npz_path.is_file():
        raise SystemExit(f"找不到 {npz_path}")
    z = np.load(npz_path)
    channels = meta.get("channels", ["u", "v"])
    K = int(meta["K"])
    time = np.asarray(z["time"])[:: args.pre_stride]
    T = time.size
    n_keep = args.n_keep if args.n_keep is not None else int(round((1.0 - args.drop_rate) * T))
    if not 1 <= n_keep < T:
        raise SystemExit(f"保留幀數 {n_keep} 不合理（T={T}）")

    rng = np.random.default_rng(args.seed)
    for attempt in range(1, args.max_tries + 1):
        keep = np.zeros((K, T), dtype=bool)
        for k in range(K):
            keep[k, rng.choice(T, size=n_keep, replace=False)] = True
        if keep.any(axis=0).all():
            break
    else:
        raise SystemExit(f"{args.max_tries} 次抽樣都有完全沒有觀測的幀")

    out_json = Path(args.out)
    out_npz = out_json.with_name(out_json.stem + "_dns_values.npz")
    for p in (out_json, out_npz):
        if p.exists():
            raise SystemExit(f"{p} 已存在，不覆寫")
    arrays = {"time": time}
    for c in channels:
        v = np.asarray(z[c], dtype=np.float32)[:, :: args.pre_stride]
        if v.shape != (K, T):
            raise SystemExit(f"channel {c} 形狀 {v.shape} ≠ ({K}, {T})")
        arrays[c] = np.where(keep, v, np.nan).astype(np.float32)
    np.savez(out_npz, **arrays)

    per_frame = keep.sum(axis=0)
    gaps = [np.diff(np.flatnonzero(keep[k])) for k in range(K)]
    meta_out = dict(meta)
    meta_out.update({
        "dns_values_npz": out_npz.name,
        "time_stride": 1, "pre_stride": args.pre_stride, "time_steps": T, "sensor_time_points": T,
        "async_drop_rate": 1.0 - n_keep / T, "async_seed": args.seed, "async_n_keep_per_sensor": n_keep,
        "async_sampling_attempts": attempt,
        "async_min_sensors_per_frame": int(per_frame.min()),
        "async_mean_sensors_per_frame": float(per_frame.mean()),
        "async_longest_gap_frames": int(max(g.max() for g in gaps if g.size)),
        "async_source_json": str(args.src),
        "generator": "scripts/gen_async_sensors.py",
    })
    out_json.write_text(json.dumps(meta_out, indent=1))
    print(f"{out_json}：T={T}、K={K}、每 sensor 保留 {n_keep} 幀；每幀觀測 sensor 數 "
          f"min {per_frame.min()} / 平均 {per_frame.mean():.1f}；最長 gap {meta_out['async_longest_gap_frames']} 幀")


if __name__ == "__main__":
    main()
