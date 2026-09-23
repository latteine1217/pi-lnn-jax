"""Stride-downsample a Kolmogorov DNS .npy onto a coarser grid.

What: 對 dict-schema 的 DNS 檔做空間 stride 取樣（512 -> 128 for stride 4），
      輸出相同 schema，並在 config 記錄 downsample_stride / source_N / source_file。
Why:  主線資料集（Re=1e3 N=128 ds4、Re=1e4 N=256 ds4）都是「高解析生成 + stride 降採樣」，
      降採樣工具原先在已不存在的外部 repo。此處重建之，使 cross-Re 資料集同源可重跑。

diagnostics 保留來源解析度的值（KE / enstrophy / divergence 在細網格上算才準），
不重算，以免把降採樣的截斷誤差偽裝成物理量。
"""

import argparse
from pathlib import Path

import numpy as np

FIELDS = ("u", "v", "omega", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source DNS .npy (dict schema)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.src, allow_pickle=True).item()
    s = args.stride
    src_N = int(d["config"]["N"])
    if src_N % s != 0:
        raise ValueError(f"stride {s} does not divide source N {src_N}")

    out = dict(d)
    for f in FIELDS:
        if f in d:
            out[f] = np.ascontiguousarray(d[f][:, ::s, ::s])
    out["x"] = np.ascontiguousarray(d["x"][::s])
    out["y"] = np.ascontiguousarray(d["y"][::s])

    cfg = dict(d["config"])
    cfg["N"] = src_N // s
    cfg["downsample_stride"] = s
    cfg["source_N"] = src_N
    cfg["source_file"] = str(Path(args.src).resolve())
    out["config"] = cfg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)

    print(f"{Path(args.src).name}  N={src_N} -> {cfg['N']}  "
          f"shape={out['u'].shape}  -> {args.out}")


if __name__ == "__main__":
    main()
