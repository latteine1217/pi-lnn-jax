#!/usr/bin/env python3
"""qrpivot_cylinder.py — QR Pivoting 從 RealPDEBench Cylinder Arrow 資料選取最優 sensor 位置。

Provenance: 自 pi-lnn `scripts/generate_sensors_qrpivot_cylinder.py` 逐字移植（2026-08-21），
    以消除本專案對 pi-lnn repo 的執行期依賴。原檔無 torch，移植未改動任何演算法。

What:
    從一個或多個 RealPDEBench cylinder Arrow shard 載入速度場，
    排除 cylinder body 格點後，執行 column-pivoted QR 分解，
    選出 K 個在流體域中最有資訊的空間位置，輸出 JSON（位置）與 NPZ（u, v 時序）。

Why:
    Kolmogorov 版本假設週期域可用 spectral gradient；
    cylinder wake 為非週期非均勻格，需改用 FD gradient。
    vorticity vo 已在 Arrow 中直接可用，不需另行計算。
    cylinder body（零速度格）必須從候選位置排除。

Algorithm:
    1. 讀取 N_traj 個 Arrow shard，各取 T_sub = T // time_stride 幀
    2. 偵測 cylinder body mask（所有時間步的速度量級中位數 < threshold）
    3. 建構 snapshot matrix A ∈ ℝ^{(N_feat × T_sub × N_traj) × N_fluid}
       特徵：u, v, vo, |∇u|_fd, |∇v|_fd
    4. Gram matrix G = A A^T → top-K 左奇異向量（Kolmogorov 法）
    5. U_k = A^T V_k / σ_k → QR with column pivoting → K fluid-domain indices
    6. 取回各 shard 在 sensor 位置的完整時序

Usage:
    uv run python scripts/generate_sensors_qrpivot_cylinder.py \\
        --shards  /path/to/data-00000-of-00092.arrow \\
                  /path/to/data-00001-of-00092.arrow \\
        --K 100 \\
        --time-stride 20 \\
        --out data/cylinder_sensors

Outputs:
    sensors_qrpivot_K{K}_cylinder.json
    sensors_qrpivot_K{K}_cylinder_values.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.linalg import qr

N_FEATURES = 4  # u, v, |∇u|_fd, |∇v|_fd  (vo 在 numerical shard 為 None)


# ── 資料讀取 ─────────────────────────────────────────────────────────────────

def load_shard(path: Path) -> dict:
    """從 Arrow IPC stream 讀取一個 cylinder shard。

    Returns dict with keys:
        sim_id, Re, u, v, p, vo, x, y, t
        u/v/p/vo: float32 ndarray [T, H, W]
        x/y: float64 ndarray [H, W]
        t: float64 ndarray [T]
    """
    with open(path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()

    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    t_len = row["t_shape"]
    x_H, x_W = row["x_shape_h"], row["x_shape_w"]

    return {
        "sim_id": row["sim_id"],
        "Re": float(row["sim_id"].replace(".h5", "")),
        "u": np.frombuffer(row["u"],  dtype=np.float32).reshape(T, H, W),
        "v": np.frombuffer(row["v"],  dtype=np.float32).reshape(T, H, W),
        "p": np.frombuffer(row["p"],  dtype=np.float32).reshape(T, H, W),
        "vo": (np.frombuffer(row["vo"], dtype=np.float32).reshape(T, H, W)
               if row["vo"] is not None else None),
        "x": np.frombuffer(row["x"],  dtype=np.float64).reshape(x_H, x_W),
        "y": np.frombuffer(row["y"],  dtype=np.float64).reshape(x_H, x_W),
        "t": np.frombuffer(row["t"],  dtype=np.float64)[:t_len],
    }


# ── 幾何工具 ─────────────────────────────────────────────────────────────────

def detect_cylinder_mask(u: np.ndarray, v: np.ndarray,
                         threshold: float = 1e-4) -> np.ndarray:
    """偵測 cylinder body：各時間步速度量級中位數 < threshold 的格點。

    Why 中位數：某些 shard 的初始幾幀可能有數值雜訊，中位數比均值穩健。
    Returns bool mask [H, W]，True = cylinder interior（排除）。
    """
    # 每隔 100 幀取樣避免 memory 問題
    idx = np.arange(0, u.shape[0], max(1, u.shape[0] // 40))
    mag = np.median(np.abs(u[idx]) + np.abs(v[idx]), axis=0)
    return mag < threshold


def fd_gradient_magnitude(field: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """有限差分梯度量級 |∇f|，適用於非均勻 tensor-product 格。

    Why FD 而非 spectral：cylinder wake 為非週期域，spectral gradient 會引入 Gibbs 振盪。
    假設 tensor-product 格：x 只沿 W 方向（axis=1）變化，y 只沿 H 方向（axis=0）變化。
    """
    x_1d = x2d[0, :]   # [W] 沿列方向的 x 座標
    y_1d = y2d[:, 0]   # [H] 沿行方向的 y 座標
    dfdx = np.gradient(field, x_1d, axis=1)
    dfdy = np.gradient(field, y_1d, axis=0)
    return np.sqrt(dfdx**2 + dfdy**2).astype(np.float32)


# ── Snapshot matrix ───────────────────────────────────────────────────────────

def normalize_rows(A: np.ndarray) -> None:
    """In-place row normalisation（零均值單位標準差）。"""
    A -= A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    A /= std


def build_snapshot_matrix(shards: list[dict], time_stride: int,
                           fluid_mask: np.ndarray) -> np.ndarray:
    """建構 snapshot matrix A ∈ ℝ^{(N_feat × T_sub × N_traj) × N_fluid}。

    fluid_mask: bool [H, W]，True = 有效流體格點（cylinder exterior）。
    """
    H, W = fluid_mask.shape
    n_fluid = fluid_mask.sum()
    flat_fluid = fluid_mask.reshape(-1)  # [H*W]

    x2d = shards[0]["x"]
    y2d = shards[0]["y"]

    rows_list = []
    for si, shard in enumerate(shards):
        u_all, v_all, vo_all = shard["u"], shard["v"], shard["vo"]
        T = u_all.shape[0]
        t_idx = np.arange(0, T, time_stride)
        T_sub = len(t_idx)
        print(f"  Shard {si+1}/{len(shards)}  Re={shard['Re']:.0f}  "
              f"T={T} → {T_sub} frames (stride={time_stride})")

        block = np.empty((N_FEATURES * T_sub, n_fluid), dtype=np.float32)
        for bi, ti in enumerate(t_idx):
            u = u_all[ti]; v = v_all[ti]
            grad_u = fd_gradient_magnitude(u, x2d, y2d)
            grad_v = fd_gradient_magnitude(v, x2d, y2d)

            features = np.stack([
                u.reshape(-1)[flat_fluid],
                v.reshape(-1)[flat_fluid],
                grad_u.reshape(-1)[flat_fluid],
                grad_v.reshape(-1)[flat_fluid],
            ], axis=0)  # [N_FEATURES, N_fluid]

            block[bi * N_FEATURES : (bi + 1) * N_FEATURES] = features

        rows_list.append(block)

    A = np.concatenate(rows_list, axis=0)  # [total_rows, N_fluid]
    print(f"Snapshot matrix: {A.shape}  [{A.nbytes / 1e6:.0f} MB]")
    normalize_rows(A)
    return A


# ── QR pivot ─────────────────────────────────────────────────────────────────

def qr_pivot_sensors(A: np.ndarray, K: int) -> np.ndarray:
    """Gram matrix → truncated SVD → QR with column pivoting → top-K indices in fluid domain."""
    print(f"Computing Gram matrix G = A A^T ({A.shape[0]}×{A.shape[0]}) ...")
    G = (A @ A.T).astype(np.float64)

    eigenvalues, V = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, V = eigenvalues[order], V[:, order]

    pos_ev = eigenvalues[eigenvalues > 0]
    explained = eigenvalues[:K].sum() / pos_ev.sum() if pos_ev.size > 0 else 0.0
    print(f"Top-{K} modes explain {explained:.1%} of variance")

    V_k = V[:, :K].astype(np.float32)
    sigma_k = np.sqrt(np.maximum(eigenvalues[:K], 0)).astype(np.float32)

    print(f"Computing U_k ({A.shape[1]} × {K}) ...")
    U_k = (A.T @ V_k) / sigma_k[None, :]

    print(f"QR with column pivoting on U_k^T ({K} × {A.shape[1]}) ...")
    _, _, piv = qr(U_k.T.astype(np.float64), pivoting=True)

    return np.sort(piv[:K])  # fluid-domain indices, sorted


def farthest_point_sampling(coords: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling 在 fluid coords 上選 n 個 maximally spread 點。

    Why: 純 QR-pivot 會把 sensor 集中在「資訊最豐富」的 wake 區，但流場其他位置
         （inflow upstream, top/bottom freestream）完全沒 sensor，model 在這些區
         域只能靠 BC + physics 約束（弱）。Hybrid uniform + QR 給 model 全域 anchor
         + wake 高頻細節雙重保險。

    Args:
        coords: [N, 2] integer (row, col) grid coords of candidate fluid cells.
        n: 要選的點數
        seed: 從 fluid 中心起點開始（用 seed 控制 ties）

    Returns:
        selected: [n] indices into coords (corresponds to fluid_indices order).
    """
    N = coords.shape[0]
    if n > N:
        raise ValueError(f"n={n} > available fluid points {N}")
    coords_f = coords.astype(np.float64)
    # 從 fluid 中心點開始（reproducibility 友好；seed 暫保留作後續擴充）
    center = coords_f.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(coords_f - center, axis=1)))
    selected = [first]
    dists = np.linalg.norm(coords_f - coords_f[first], axis=1)
    for _ in range(n - 1):
        nxt = int(dists.argmax())
        selected.append(nxt)
        new_d = np.linalg.norm(coords_f - coords_f[nxt], axis=1)
        dists = np.minimum(dists, new_d)
    return np.array(selected, dtype=np.int64)


def hybrid_uniform_qr_sensors(
    A: np.ndarray,
    fluid_mask: np.ndarray,
    K: int,
    n_uniform: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hybrid sensor placement: n_uniform 個 farthest-point + (K-n_uniform) 個 QR pivot.

    Args:
        A: snapshot matrix [rows, n_fluid]（已 row-normalized）
        fluid_mask: bool [H, W]，True = 有效流體
        K: 總 sensor 數
        n_uniform: 第一階段均勻分佈的 sensor 數（剩 K-n_uniform 由 QR 選）
        seed: farthest-point sampling seed

    Returns:
        sensor_fluid_idx_sorted: [K] fluid-domain indices, sorted
        uniform_picks_in_fluid:  [n_uniform] uniform 階段選的 indices（in fluid order）
        qr_picks_in_fluid:       [K-n_uniform] QR 階段選的 indices（in fluid order）
    """
    H, W = fluid_mask.shape
    fluid_indices = np.argwhere(fluid_mask.reshape(-1)).ravel()
    coords = np.stack([fluid_indices // W, fluid_indices % W], axis=1)  # [N_fluid, 2]

    if n_uniform <= 0:
        # 純 QR pipeline（向後相容）
        qr_picks = qr_pivot_sensors(A, K)
        return qr_picks, np.array([], dtype=np.int64), qr_picks
    if n_uniform >= K:
        raise ValueError(f"n_uniform={n_uniform} >= K={K}；至少留 1 個給 QR")

    # Stage 1: farthest-point sampling 在整個 fluid 域選 n_uniform 個 spatially-spread 點
    print(f"Stage 1: farthest-point sampling for {n_uniform} uniform sensors ...")
    uniform_picks = farthest_point_sampling(coords, n_uniform, seed=seed)

    # Stage 2: QR pivot 在剩下 fluid columns 選 K - n_uniform 個
    n_qr = K - n_uniform
    remaining_mask = np.ones(len(fluid_indices), dtype=bool)
    remaining_mask[uniform_picks] = False
    A_remain = A[:, remaining_mask]
    print(f"Stage 2: QR pivot on remaining {A_remain.shape[1]} fluid cells "
          f"for {n_qr} sensors ...")
    qr_picks_in_remain = qr_pivot_sensors(A_remain, n_qr)
    # 把「remaining 子空間 index」轉回「原 fluid 域 index」
    remaining_idx_in_fluid = np.argwhere(remaining_mask).ravel()
    qr_picks_in_fluid = remaining_idx_in_fluid[qr_picks_in_remain]

    # 合併 + sort（duplicate-check：uniform/QR 不該重疊，因 QR 在 remain 子集做）
    all_picks = np.concatenate([uniform_picks, qr_picks_in_fluid])
    if len(np.unique(all_picks)) != len(all_picks):
        raise RuntimeError("uniform 跟 QR 階段選到重複點（logic bug）")
    return np.sort(all_picks), uniform_picks, qr_picks_in_fluid


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QR-pivot optimal sensor placement for cylinder wake (RealPDEBench Arrow)"
    )
    parser.add_argument("--shards", nargs="+", required=True,
                        help="Arrow shard 路徑（一個或多個）")
    parser.add_argument("--K", type=int, default=100, help="Sensor 總數")
    parser.add_argument("--n-uniform", type=int, default=0,
                        help="第一階段 farthest-point uniform sampling 的 sensor 數；"
                             "剩下的 (K - n_uniform) 由 QR pivot 在剩餘 fluid 點上選。"
                             " 0 = 純 QR（向後相容預設）。"
                             " 例：--K 100 --n-uniform 20 → 20 個全場均勻 + 80 個 QR")
    parser.add_argument("--time-stride", type=int, default=20,
                        help="時間取樣步距（預設 20：3990 幀 → 200 幀）")
    parser.add_argument("--body-threshold", type=float, default=1e-4,
                        help="Cylinder body 偵測速度閾值")
    parser.add_argument("--out", required=True, help="輸出目錄")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="若設定，改用 uniform random placement（從 fluid 域均勻隨機抽 K 點），"
                             "而非 QR-pivot。用於 placement ablation（CEXP-041）。"
                             "body 排除 / axis convention / 輸出 schema 與 QR 完全相同。")
    args = parser.parse_args()

    K = args.K
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入資料 ──────────────────────────────────────────────────────────────
    print(f"Loading {len(args.shards)} shard(s) ...")
    shards = []
    for path in args.shards:
        print(f"  {path}")
        shards.append(load_shard(Path(path)))
        print(f"    Re={shards[-1]['Re']:.0f}  shape={shards[-1]['u'].shape}")

    x2d = shards[0]["x"]
    y2d = shards[0]["y"]
    H, W = x2d.shape

    # ── Cylinder body mask ────────────────────────────────────────────────────
    print("Detecting cylinder body mask ...")
    # 聯集各 shard 的 body mask（取交集更保守；取聯集更安全）
    body_mask = np.ones((H, W), dtype=bool)
    for shard in shards:
        body_mask &= detect_cylinder_mask(shard["u"], shard["v"], args.body_threshold)
    fluid_mask = ~body_mask
    n_body = body_mask.sum()
    n_fluid = fluid_mask.sum()
    print(f"  Cylinder body: {n_body} cells  Fluid domain: {n_fluid} / {H*W} cells")

    fluid_indices = np.argwhere(fluid_mask.reshape(-1)).ravel()  # [N_fluid]

    if args.random_seed is not None:
        # Random placement ablation：均勻隨機從 fluid 域抽 K 點（不建 snapshot matrix）。
        # body 已排除（fluid_indices 來自 fluid_mask），axis/schema 與 QR 路徑完全相同。
        print(f"Random placement (seed={args.random_seed}): {K} sensors from {len(fluid_indices)} fluid cells ...")
        _rng = np.random.default_rng(args.random_seed)
        sensor_fluid_idx = _rng.choice(len(fluid_indices), size=K, replace=False)
        uniform_picks_in_fluid = sensor_fluid_idx
        qr_picks_in_fluid = np.array([], dtype=np.int64)
    else:
        # ── Snapshot matrix + QR pivot ────────────────────────────────────────────
        print("Building snapshot matrix ...")
        A = build_snapshot_matrix(shards, args.time_stride, fluid_mask)
        sensor_fluid_idx, uniform_picks_in_fluid, qr_picks_in_fluid = hybrid_uniform_qr_sensors(
            A, fluid_mask, K=K, n_uniform=int(args.n_uniform),
        )
        del A

    # 轉回 (H, W) 格點 flat index
    sensor_flat = fluid_indices[sensor_fluid_idx]  # [K] 在 H×W 的 flat index
    sensor_i = (sensor_flat // W).astype(int)      # 行 index
    sensor_j = (sensor_flat % W).astype(int)       # 列 index

    # 物理座標
    sensor_x = x2d[sensor_i, sensor_j]
    sensor_y = y2d[sensor_i, sensor_j]
    coords_xy = np.stack([sensor_x, sensor_y], axis=1)  # [K, 2]
    print(f"Sensor x ∈ [{sensor_x.min():.4f}, {sensor_x.max():.4f}]")
    print(f"Sensor y ∈ [{sensor_y.min():.4f}, {sensor_y.max():.4f}]")

    # ── 提取時序 ──────────────────────────────────────────────────────────────
    print("Extracting sensor time series ...")
    # 每個 shard 各自輸出（用第一個 shard 作代表；多 shard 情況可擴充）
    shard = shards[0]
    u_sensors = shard["u"][:, sensor_i, sensor_j].T.astype(np.float32)  # [K, T]
    v_sensors = shard["v"][:, sensor_i, sensor_j].T.astype(np.float32)  # [K, T]
    t_out = shard["t"]

    # ── 儲存 ─────────────────────────────────────────────────────────────────
    re_tag = f"Re{shards[0]['Re']:.0f}"
    if len(shards) > 1:
        re_tag = f"Re{shards[0]['Re']:.0f}-{shards[-1]['Re']:.0f}"

    # 檔名加入 hybrid 標記避免覆寫舊 sensor 集
    method_tag = "qrpivot" if int(args.n_uniform) <= 0 else f"hybrid{int(args.n_uniform)}qr{K-int(args.n_uniform)}"
    base = f"sensors_{method_tag}_K{K}_cylinder_{re_tag}"
    json_path = out_dir / f"{base}.json"
    npz_path  = out_dir / f"{base}_values.npz"

    # 把 uniform / QR 階段的 (i, j) 也記錄，方便後續視覺化區分兩類 sensor
    uniform_flat = fluid_indices[uniform_picks_in_fluid] if len(uniform_picks_in_fluid) > 0 else np.array([], dtype=np.int64)
    qr_flat      = fluid_indices[qr_picks_in_fluid]
    payload = {
        "K": K,
        "n_uniform": int(args.n_uniform),
        "n_qr": K - int(args.n_uniform),
        "domain": "cylinder_wake",
        "grid": f"{H}x{W}",
        "n_fluid_cells": int(n_fluid),
        "n_body_cells": int(n_body),
        "body_threshold": args.body_threshold,
        "method": "qr_pivoting" if int(args.n_uniform) <= 0 else "hybrid_uniform_qr",
        "features": ["u", "v", "grad_u_mag_fd", "grad_v_mag_fd"],
        "time_stride_qr": args.time_stride,
        "Re_list": [s["Re"] for s in shards],
        "source_shards": [str(p) for p in args.shards],
        "selected_coordinates": coords_xy.tolist(),   # [[x, y], ...]
        "sensor_i": sensor_i.tolist(),
        "sensor_j": sensor_j.tolist(),
        "sensor_flat": sensor_flat.tolist(),
        "uniform_sensor_flat": uniform_flat.tolist(),  # 給 evaluator / plot 區分兩類
        "qr_sensor_flat":      qr_flat.tolist(),
        "values_npz": str(npz_path),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

    np.savez(
        npz_path,
        t=t_out,
        u=u_sensors,   # [K, T]
        v=v_sensors,
        x=sensor_x,    # [K]
        y=sensor_y,
    )
    print(f"Saved: {npz_path}  (shape: {u_sensors.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
