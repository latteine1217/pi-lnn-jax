"""cylinder sensor placement 腳本的共用件。

What: 三支 placement 腳本（uniform / boundary / sdf_isocontour）共用的
      「shard → body/fluid 幾何」推導與「sensor set → JSON+NPZ」輸出，
      加上匯入 pi-lnn 端 qrpivot 模組的單一入口。

Why: 這兩段原本在三支腳本各複製一份（原始碼註解自承「與 boundary 版一致」、
     「schema 同 boundary/sdf」）。輸出 schema 是下游 CylinderDataset 的契約，
     三份各自演化就會產出彼此不相容的 sensor set，而且錯配是靜默的。

Note: Arrow I/O（`load_shard`，需 pyarrow）住在 `_common.qrpivot_cylinder`。
      該模組原屬 pi-lnn repo，2026-08-21 移植進本專案後，這三支腳本不再需要
      pi-lnn env，直接用本專案的 .venv 執行即可。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np


class CylinderGeometry(NamedTuple):
    """一組 shard 推導出的 body / fluid 幾何。

    fluid_indices 為 flat index（row-major，`flat = i * W + j`）；
    coords_grid 為對應的 (row, col)，供 farthest-point sampling 用。
    fx / fy 為流體格點的物理座標，供 SDF / 邊界層距離計算用。
    """
    x2d: np.ndarray
    y2d: np.ndarray
    H: int
    W: int
    body_mask: np.ndarray
    fluid_mask: np.ndarray
    fluid_indices: np.ndarray
    coords_grid: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    body_center: np.ndarray
    body_radius: float


def load_qrpivot_module(gen_dir: str | None = None) -> Any:
    """回傳 qrpivot 模組（`load_shard` / `detect_cylinder_mask` /
    `farthest_point_sampling` / `build_snapshot_matrix` / `qr_pivot_sensors`）。

    預設用本專案內建的 `_common.qrpivot_cylinder`（2026-08-21 從 pi-lnn 移植）。
    `gen_dir` 是外部同名模組的 override：目錄內有 `generate_sensors_qrpivot_cylinder.py`
    就用它，否則落回內建。兩個呼叫端靠這條——既有 sbatch 傳 `--gen-dir scripts`
    （移植後那裡已無該檔，於是落回內建），測試用它注入 stub 以免依賴真實 Arrow 資料。
    """
    if gen_dir:
        cand = Path(gen_dir) / "generate_sensors_qrpivot_cylinder.py"
        if cand.is_file():
            # 依路徑載入而非 `import <名稱>`：後者會把 override 留在 sys.modules，
            # 讓同一 process 內的後續呼叫拿到上一個 override（測試曾因此互相污染）。
            spec = importlib.util.spec_from_file_location("_qrpivot_override", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[qrpivot] 使用外部模組 override：{cand}")
            return mod
    from . import qrpivot_cylinder as mod
    return mod


def cylinder_geometry(
    shards: list[dict],
    body_threshold: float,
    detect_body_mask: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> CylinderGeometry:
    """從一組 shard 推導 body / fluid 幾何。

    body 取所有 shard 的**交集**：任一 shard 判為流體的格點就不算 body
    （body 靜止不動，某個 shard 測到流動代表那裡不是固體）。

    `detect_body_mask` 由呼叫端傳入（通常是 qrpivot 模組的
    `detect_cylinder_mask`），讓本模組不必依賴 pi-lnn，也讓測試能用合成資料。
    """
    if not shards:
        raise ValueError("shards 為空，無法推導幾何")

    x2d, y2d = shards[0]["x"], shards[0]["y"]
    H, W = x2d.shape

    body_mask = np.ones((H, W), bool)
    for s in shards:
        body_mask &= detect_body_mask(s["u"], s["v"], body_threshold)
    if not body_mask.any():
        raise ValueError(
            f"body_threshold={body_threshold} 下未偵測到任何 body 格點；"
            "門檻過嚴或 shard 不含 cylinder"
        )

    fluid_mask = ~body_mask
    fluid_indices = np.argwhere(fluid_mask.reshape(-1)).ravel()
    coords_grid = np.stack([fluid_indices // W, fluid_indices % W], axis=1)
    fx = x2d.reshape(-1)[fluid_indices]
    fy = y2d.reshape(-1)[fluid_indices]

    # body 質心與等效半徑（由二階矩推得，非假設圓形）
    bflat = np.argwhere(body_mask.reshape(-1)).ravel()
    bx, by = x2d.reshape(-1)[bflat], y2d.reshape(-1)[bflat]
    body_center = np.array([bx.mean(), by.mean()])
    body_radius = float(
        np.sqrt(2.0 * np.mean((bx - body_center[0]) ** 2 + (by - body_center[1]) ** 2))
    )

    print(f"Fluid {len(fluid_indices)} / {H * W}  body {int(body_mask.sum())}")
    print(f"body center=({body_center[0]:.4f},{body_center[1]:.4f}) R={body_radius:.4f}")

    return CylinderGeometry(
        x2d=x2d, y2d=y2d, H=H, W=W,
        body_mask=body_mask, fluid_mask=fluid_mask,
        fluid_indices=fluid_indices, coords_grid=coords_grid,
        fx=fx, fy=fy, body_center=body_center, body_radius=body_radius,
    )


def write_sensor_set(
    out_dir: Path,
    base: str,
    *,
    geom: CylinderGeometry,
    sensor_fluid_idx: np.ndarray,
    shard: dict,
    K: int,
    body_threshold: float,
    method: str,
    time_stride: int,
    extra: dict | None = None,
) -> tuple[Path, Path]:
    """輸出 `<base>.json` + `<base>_values.npz`，回傳兩者路徑。

    schema 是下游 CylinderDataset 的契約：
      json — selected_coordinates [K,2]、sensor_i/sensor_j/sensor_flat、meta
      npz  — t [T]、u [K,T]、v [K,T]

    `sensor_fluid_idx` 是在 **fluid 子集**內的索引（farthest-point / QR 的輸出），
    這裡負責映射回 grid 的 (i, j)。`extra` 併入 json 供各 placement 法記錄自己的參數。
    """
    W = geom.W
    sensor_flat = geom.fluid_indices[sensor_fluid_idx]
    sensor_i = (sensor_flat // W).astype(int)
    sensor_j = (sensor_flat % W).astype(int)
    sensor_x = geom.x2d[sensor_i, sensor_j]
    sensor_y = geom.y2d[sensor_i, sensor_j]
    coords_xy = np.stack([sensor_x, sensor_y], axis=1)
    print(
        f"Sensor x∈[{sensor_x.min():.4f},{sensor_x.max():.4f}] "
        f"y∈[{sensor_y.min():.4f},{sensor_y.max():.4f}] K={len(sensor_fluid_idx)}"
    )

    # [T,H,W] → 取樣點 → 轉置成 CylinderDataset 要的 [K,T]
    u_s = shard["u"][:, sensor_i, sensor_j].T.astype(np.float32)
    v_s = shard["v"][:, sensor_i, sensor_j].T.astype(np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base}.json"
    npz_path = out_dir / f"{base}_values.npz"

    payload = {
        "K": int(K),
        "domain": [
            float(geom.x2d.min()), float(geom.x2d.max()),
            float(geom.y2d.min()), float(geom.y2d.max()),
        ],
        "grid": [int(geom.H), int(geom.W)],
        "n_fluid_cells": int(geom.fluid_mask.sum()),
        "n_body_cells": int(geom.body_mask.sum()),
        "body_threshold": float(body_threshold),
        "method": method,
        "Re_list": [float(shard["Re"])],
        "time_stride_qr": int(time_stride),
        "selected_coordinates": coords_xy.tolist(),
        "sensor_i": sensor_i.tolist(),
        "sensor_j": sensor_j.tolist(),
        "sensor_flat": sensor_flat.tolist(),
        "values_npz": str(npz_path),
    }
    if extra:
        payload.update(extra)

    json_path.write_text(json.dumps(payload, indent=2))
    np.savez(npz_path, t=shard["t"], u=u_s, v=v_s)
    print(f"Saved: {json_path}\nSaved: {npz_path} (u shape {u_s.shape})")
    return json_path, npz_path
