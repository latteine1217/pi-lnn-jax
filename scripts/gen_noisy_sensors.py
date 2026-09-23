#!/usr/bin/env python3
"""從乾淨 sensor set 產生帶量測噪音的副本（thesis §4.5.4 noise robustness）。

What:
    讀一份既有 sensor JSON + values npz，對 u/v 各自加 per-channel 高斯噪音
    （σ_c = noise_frac × 該 channel 的 std），寫出獨立的一組 JSON + npz。
    每個 (noise level, seed) 一份，檔名帶兩者。

Why 預先生成而不是訓練時注入:
    量測噪音是**資料的屬性**，不是訓練過程的隨機性。訓練端若要注入，就得在
    建構期消耗 RNG，而 `tests/test_pipeline_assembly.py` 的邊界哨兵明確禁止
    ——建構期碰 RNG 會讓 clean 與 noisy 兩種跑法的 RNG 流錯位，連 clean
    baseline 都不再 bit-identical。噪音落在資料層則訓練端零改動，且 noisy
    觀測本身成為可檢視、可重現、可比對的一份資料集。

    噪音 realization 隨 seed 變（不是所有 seed 共用一份）：thesis 報的是每個
    noise level n=5 seeds，若噪音固定，那 5 個 seed 只反映初始化變異，會低估
    觀測噪音的實際影響。

Usage:
    uv run python scripts/gen_noisy_sensors.py \
        --source data/kolmogorov_sensors/re10000_fps/sensors_....json \
        --noise 0.01,0.03,0.05,0.10 --seeds 42,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import apply_sensor_noise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="乾淨 sensor JSON 路徑")
    p.add_argument("--noise", default="0.01,0.03,0.05,0.10",
                   help="噪音比例（相對各 channel std），逗號分隔")
    p.add_argument("--seeds", default="42,1,2,3,4", help="seed 清單，逗號分隔")
    p.add_argument("--out-dir", default=None,
                   help="輸出目錄，預設與 source 同目錄")
    return p.parse_args()


def _noise_tag(frac: float) -> str:
    """0.01 → n01、0.10 → n10；檔名不出現小數點，避免與副檔名解析打架。"""
    return f"n{round(frac * 100):02d}"


def main() -> int:
    args = parse_args()
    src_json = Path(args.source)
    if not src_json.is_file():
        raise FileNotFoundError(f"來源 sensor JSON 不存在: {src_json}")
    meta = json.loads(src_json.read_text())

    npz_name = meta.get("dns_values_npz")
    if not npz_name:
        raise ValueError(f"{src_json} 缺 dns_values_npz 欄位，無法定位觀測值")
    src_npz = src_json.parent / Path(npz_name).name
    if not src_npz.is_file():
        raise FileNotFoundError(f"觀測值 npz 不存在: {src_npz}")

    data = np.load(src_npz)
    for key in ("u", "v", "time"):
        if key not in data.files:
            raise ValueError(f"{src_npz} 缺欄位 {key!r}（有 {data.files}）")
    u, v, t = data["u"], data["v"], data["time"]

    out_dir = Path(args.out_dir) if args.out_dir else src_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.noise.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    stem = src_json.stem

    written = 0
    for frac in fracs:
        for seed in seeds:
            tag = f"{_noise_tag(frac)}_s{seed}"
            # u/v 疊成 (K, T, 2) 一起處理，per-channel σ 才各自取自己的 std
            stacked = np.stack([u, v], axis=-1)
            noisy = apply_sensor_noise(stacked, frac, seed=seed)
            nu, nv = noisy[..., 0], noisy[..., 1]

            out_npz = out_dir / f"{stem}_{tag}_dns_values.npz"
            out_json = out_dir / f"{stem}_{tag}.json"
            np.savez(out_npz, u=nu.astype(u.dtype), v=nv.astype(v.dtype), time=t)

            new_meta = dict(meta)
            new_meta["dns_values_npz"] = out_npz.name
            new_meta["noise_fraction"] = frac
            new_meta["noise_seed"] = seed
            new_meta["clean_source"] = src_json.name
            out_json.write_text(json.dumps(new_meta, indent=2))

            realized = float(((nu - u).std() / (u.std() + 1e-12)))
            print(f"  {out_json.name}  sigma_u/std_u = {realized:.4f} (target {frac})")
            written += 2

    print(f"[out] 寫入 {written} 個檔案 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
