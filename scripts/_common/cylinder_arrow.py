"""cylinder_arrow.py — RealPDEBench cylinder Arrow shard 的全場讀取。

What: 讀單一 Arrow shard，回傳 u/v/p 全場與 grid metadata（x/y/t）。

Why:  dump 腳本要把 Arrow 轉成本專案的 npz 契約，需要全場而非 sensor 時序；
      `qrpivot_cylinder.load_shard` 讀的是 sensor 選點用的欄位集（含 vo），
      兩者 schema 不同，刻意不合併。

Provenance: 自 pi-lnn `scripts/evaluate_cylinder.py:load_arrow_fields` 逐字移植
    （2026-08-21），以消除本專案對 pi-lnn repo 的執行期依賴。演算法未改動。
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa


def load_arrow_fields(shard_path: str) -> dict:
    """讀取 Arrow shard，回傳 u/v/p 全場 + grid metadata。"""
    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {n: batch.column(n)[0].as_py() for n in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    xH, xW = row["x_shape_h"], row["x_shape_w"]
    t_len  = row["t_shape"]
    # CRIT-2: fail-fast 對 grid shape mismatch（u/v/p shape vs x/y shape 必須相同）。
    # Why: 之前若 (xH, xW) != (H, W)，evaluator 會在後段 SDF assert 才炸（已 reshape 過 →
    #      不可恢復狀態）。在這裡早 raise 訊息更清楚。
    if (xH, xW) != (H, W):
        raise ValueError(
            f"Arrow shard grid shape mismatch: u/v/p shape ({H},{W}) "
            f"vs x/y shape ({xH},{xW}). 同一 shard 必須一致。"
        )
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)
    p = np.frombuffer(row["p"], dtype=np.float32).reshape(T, H, W)
    x = np.frombuffer(row["x"], dtype=np.float64).reshape(xH, xW)
    y = np.frombuffer(row["y"], dtype=np.float64).reshape(xH, xW)
    t = np.frombuffer(row["t"], dtype=np.float64)[:t_len]
    return dict(u=u, v=v, p=p, x=x, y=y, t=t, T=T, H=H, W=W)
