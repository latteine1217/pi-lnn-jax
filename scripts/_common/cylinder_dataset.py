# scripts/_common/cylinder_dataset.py
"""CylinderDataset: 載入 cylinder wake sensor 時序 + Arrow 格點，提供稀疏資料採樣。

What: 從 QR pivot JSON/NPZ（感測器）與 RealPDEBench Arrow shard（格點與全場）
      建立訓練資料結構。API 與 KolmogorovDataset 完全相容，
      使 picon_kolmogorov.py 的 training loop 無需修改即可支援 cylinder case。

Why:  cylinder wake 為非週期非均勻格，座標 (x, y) 需正規化至 [0,1]²；
      物理 collocation 點須限制在流體域（排除 cylinder body）。
      Arrow shard 只在建構時讀格點 metadata，全場 evaluation 由 evaluator 直接讀。

Provenance: 自 pi-lnn `src/cylinder_dataset.py` 移植（2026-08-21），以消除本專案對
    pi-lnn repo 的執行期依賴。唯一刪除的是 torch autograd 路徑（`_sdf_grid_torch` 與
    `query_body_distance_torch`）——那條給 PyTorch hard-BC 用，本專案走 JAX，且既有
    呼叫端（dump 腳本）只讀 numpy 欄位。其餘逐字保留。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

# cylinder Re 範圍 3000–11000 的正規化常數
RE_MEAN: float = 7000.0
RE_STD:  float = 2500.0

BODY_THRESHOLD: float = 1e-4   # 速度量級低於此值視為 cylinder body


def _load_arrow_grid(shard_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """從 Arrow shard 讀取格點座標與時間軸，不讀完整場資料節省記憶體。

    Returns:
        x2d [H, W] float64, y2d [H, W] float64, t [T] float64
    """
    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()

    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T    = row["shape_t"]
    H, W = row["shape_h"], row["shape_w"]
    xH, xW = row["x_shape_h"], row["x_shape_w"]
    t_len = row["t_shape"]

    x2d = np.frombuffer(row["x"], dtype=np.float64).reshape(xH, xW)
    y2d = np.frombuffer(row["y"], dtype=np.float64).reshape(xH, xW)
    t   = np.frombuffer(row["t"], dtype=np.float64)[:t_len]
    return x2d, y2d, t


def _detect_body(shard_path: Path, H: int, W: int) -> np.ndarray:
    """偵測 cylinder body：在多個時刻速度量級中位數 < BODY_THRESHOLD 的格點。

    體速度永遠為 0（Dirichlet no-slip BC），取中位數對 transient 初始幀穩健。
    """
    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T = row["shape_t"]
    u_all = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    v_all = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)

    # 每隔 100 幀取樣，避免讀完整 T
    idx = np.arange(0, T, max(1, T // 40))
    mag = np.median(np.abs(u_all[idx]) + np.abs(v_all[idx]), axis=0)
    return mag < BODY_THRESHOLD   # [H, W] bool


def _measure_inflow_u(shard_path: Path, H: int, W: int) -> float:
    """量測 inlet 邊界（x=x_min, 即 W=0 那一列）的 u 平均值，做 BC loss 錨點。

    Why: bc_inflow_u 之前 hardcode 0.33，multi-Re 訓練時各 Re 的 inflow 速度
         不同會導致錨點錯誤。直接從 DNS shard 量測，可自動因 Re 而異。
         用中位數抗 transient 早期擾動。
    """
    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        batch = reader.read_next_batch()
    row = {name: batch.column(name)[0].as_py() for name in batch.schema.names}
    T = row["shape_t"]
    u_all = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    # u_all[:, :, 0] 是 x_min 那一列：shape [T, H]
    return float(np.median(u_all[:, :, 0]))


@dataclass
class CylinderDataset:
    """Cylinder wake sensor dataset with normalised coordinates.

    Public API 與 KolmogorovDataset 完全相容：
        sensor_vals [K, T, C], sensor_pos [K, 2], sensor_time [T]
        observed_channel_names, observed_channel_mean/std
        re_value, re_norm, dt_phys
        train_t_idx, val_t_idx
        sample_sensor_batch(), sample_physics_points()

    Cylinder-specific（非 Kolmogorov 對應欄位）:
        fluid_xy [N_fluid, 2] — 正規化的流體域 collocation 候選點
        body_xy  [N_body,  2] — 正規化的 cylinder body 點，BC loss no-slip 用
        x_lo/x_hi/y_lo/y_hi   — 物理座標範圍（正規化用）
        Lx/Ly                 — 物理域長度（米），= x_hi-x_lo / y_hi-y_lo；
                                NS residual chain-rule 把 normalized 梯度轉物理梯度用
    """

    sensor_vals:  np.ndarray
    observed_channel_names: tuple
    observed_channel_mean: np.ndarray
    observed_channel_std:  np.ndarray
    sensor_pos:   np.ndarray
    sensor_time:  np.ndarray
    re_value:     float
    re_norm:      float
    dt_phys:      float
    train_t_idx:  np.ndarray
    val_t_idx:    np.ndarray
    fluid_xy:     np.ndarray   # [N_fluid, 2] 正規化座標
    body_xy:      np.ndarray   # [N_body,  2] 正規化座標（cylinder body 內部點）
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    Lx:   float                # 物理 x 域長度（米）= x_hi - x_lo
    Ly:   float                # 物理 y 域長度（米）= y_hi - y_lo
    bc_inflow_u: float         # 從 DNS 量測的 inlet u 平均（m/s）；BC loss 錨點

    # KolmogorovDataset 對應欄位（訓練 loop 相容）
    dns_x: np.ndarray   # 1D 正規化 x 節點（domain bound 用）
    dns_y: np.ndarray   # 1D 正規化 y 節點
    dns_time: np.ndarray

    def __init__(
        self,
        sensor_json:  str | Path,
        sensor_npz:   str | Path,
        arrow_shard:  str | Path,
        re_value:     float,
        observed_channel_names: tuple | list | None = None,
        train_ratio:  float = 0.8,
        seed:         int   = 42,
        sensor_subsample: int = 1,
    ) -> None:
        # sensor_subsample: 每隔幾幀取一幀，用於壓縮 T=3990 → T/stride
        # Why: CfC 每步需遍歷完整 sensor timeline；T=3990 比 Kolmogorov T=201 慢 20x。
        #      stride=20 → T≈200，與 Kolmogorov 計算量相當。
        rng = np.random.default_rng(seed)
        arrow_path = Path(arrow_shard)

        # ── 感測器位置（物理座標，後正規化）────────────────────────────
        with open(sensor_json, encoding="utf-8") as f:
            meta = json.load(f)
        coords_raw = np.array(meta["selected_coordinates"], dtype=np.float64)  # [K, 2]

        # ── 格點座標與時間 ──────────────────────────────────────────────
        x2d, y2d, t_full = _load_arrow_grid(arrow_path)
        H, W = x2d.shape

        # 物理座標範圍（用於正規化）
        self.x_lo = float(x2d.min()); self.x_hi = float(x2d.max())
        self.y_lo = float(y2d.min()); self.y_hi = float(y2d.max())
        # 物理域長度（米）：NS residual chain-rule 用 du/dx_phys = du/dx_norm / Lx。
        self.Lx = self.x_hi - self.x_lo
        self.Ly = self.y_hi - self.y_lo

        def norm_x(x): return (np.asarray(x) - self.x_lo) / (self.x_hi - self.x_lo)
        def norm_y(y): return (np.asarray(y) - self.y_lo) / (self.y_hi - self.y_lo)

        # 正規化感測器座標 → [0, 1]²
        sensor_pos_norm = np.stack(
            [norm_x(coords_raw[:, 0]), norm_y(coords_raw[:, 1])], axis=1
        ).astype(np.float32)
        self.sensor_pos = sensor_pos_norm

        # ── 感測器時序 ───────────────────────────────────────────────────
        npz = np.load(sensor_npz, allow_pickle=True)
        requested = tuple(observed_channel_names) if observed_channel_names else ("u", "v")
        obs_names, obs_fields = [], []
        for key in requested:
            if key in npz:
                obs_names.append(key)
                obs_fields.append(npz[key].astype(np.float32))  # [K, T]
        if not obs_fields:
            raise ValueError(f"NPZ {sensor_npz} 不含 {requested}")

        t_raw = npz["t"].astype(np.float32)               # [T_full]
        stride = max(1, int(sensor_subsample))
        t_idx_sub = np.arange(0, len(t_raw), stride)
        self.sensor_time = t_raw[t_idx_sub]              # [T_sub]
        self.observed_channel_names = tuple(obs_names)
        raw_full = np.stack(obs_fields, axis=-1)         # [K, T_full, C]
        raw = raw_full[:, t_idx_sub, :]                  # [K, T_sub, C]

        # ── Train/Val 切割（先切，避免 normalize 統計 leak val 部分）────
        T_sub = len(self.sensor_time)
        _idx = np.arange(T_sub)
        rng.shuffle(_idx)
        _n_train = int(T_sub * train_ratio)
        self.train_t_idx = _idx[:_n_train]
        self.val_t_idx   = _idx[_n_train:]

        # mean/std 僅用 train 區段，避免將未來 val 統計洩漏進 normalize 因子。
        # 應用時對全部 raw（含 val）使用相同 train stats — 標準 ML 做法。
        raw_train = raw[:, self.train_t_idx, :]
        self.observed_channel_mean = raw_train.mean(axis=(0, 1)).astype(np.float32)
        self.observed_channel_std  = np.maximum(
            raw_train.std(axis=(0, 1)), 1e-6
        ).astype(np.float32)
        self.sensor_vals = (
            (raw - self.observed_channel_mean[None, None, :])
            / self.observed_channel_std[None, None, :]
        ).astype(np.float32)
        self.dt_phys = float(self.sensor_time[1] - self.sensor_time[0])

        # ── Cylinder body mask + 流體域 collocation 點 ────────────────────
        body_mask = _detect_body(arrow_path, H, W)     # [H, W] bool
        fluid_mask = ~body_mask                         # [H, W] bool
        # Inlet u 量測：從 DNS shard 直接讀 x=x_min 那一列。
        # Why: 取代 config 的 hardcode bc_inflow_u=0.33，multi-Re 自動因 Re 而異。
        self.bc_inflow_u = _measure_inflow_u(arrow_path, H, W)

        x_flat = x2d.reshape(-1)
        y_flat = y2d.reshape(-1)
        ff = fluid_mask.reshape(-1)
        bf = body_mask.reshape(-1)
        fluid_x_norm = norm_x(x_flat[ff]).astype(np.float32)
        fluid_y_norm = norm_y(y_flat[ff]).astype(np.float32)
        self.fluid_xy = np.stack([fluid_x_norm, fluid_y_norm], axis=1)  # [N_fluid, 2]
        # body 點：no-slip BC 採樣源（u=v=0 在 body 內部恆成立，不限於表面）。
        body_x_norm = norm_x(x_flat[bf]).astype(np.float32)
        body_y_norm = norm_y(y_flat[bf]).astype(np.float32)
        self.body_xy = np.stack([body_x_norm, body_y_norm], axis=1)     # [N_body, 2]

        # ── Body-distance SDF（signed distance function）──────────────────
        # Why: 用於 hard body BC 約束（Sukumar 2022 風格）：
        #        u(x,y,t) = (φ(x,y) / scale) · NN(x,y,t)
        #      其中 φ 在 body 邊界 = 0，fluid 內 > 0 → 強制 body 內 u=0（no-slip 物理保證）。
        #      φ 必須對 xy 可微，否則 NS residual 算 ∂u/∂x 會缺 chain rule（cylinder_007/008
        #      的 detach numpy lookup bug 已驗證）。所以提供：
        #        - self._sdf_grid       : np.ndarray [H, W]
        # 實作：對 body_mask 做 distance_transform_edt → 對每個 cell 給到最近 body cell
        #      的 Euclidean distance（pixel 單位），再除 max(H-1, W-1) 得 normalized 座標距離。
        from scipy.ndimage import distance_transform_edt
        sdf_grid_pix = distance_transform_edt(~body_mask)   # [H, W]，pixel 單位
        self._sdf_grid = (sdf_grid_pix.astype(np.float32) / max(H - 1, W - 1))   # numpy [H, W]
        self._sdf_H = int(H)
        self._sdf_W = int(W)
        # Hard BC scale：fluid 區的最大 distance，用於 gate normalization
        # gate = (φ / scale).clamp(0, 1) → fluid 中心 (φ=max) gate ≈ 1
        self.bc_distance_scale = float(self._sdf_grid.max())

        # KolmogorovDataset 相容欄位：用正規化的唯一 x/y 值
        self.dns_x    = np.unique(norm_x(x2d[0, :])).astype(np.float32)   # 沿 W 方向
        self.dns_y    = np.unique(norm_y(y2d[:, 0])).astype(np.float32)   # 沿 H 方向
        self.dns_time = self.sensor_time

        # ── Re 正規化 ────────────────────────────────────────────────────
        self.re_value = float(re_value)
        self.re_norm  = float((re_value - RE_MEAN) / RE_STD)
        # train/val 切割已在 sensor_vals 正規化前完成，避免統計 leak。

    def query_body_distance(self, xy_normalized: np.ndarray) -> np.ndarray:
        """Bilinear interpolate body distance field at given normalized xy points (numpy).

        Args:
            xy_normalized: [N, 2] in [0, 1]² normalized coordinates.

        Returns:
            distances: [N] distances to body boundary in normalized units (0 = on body).
        """
        if not hasattr(self, "_sdf_grid"):
            raise AttributeError("SDF grid 未計算（cylinder dataset 才有）")
        H, W = self._sdf_H, self._sdf_W
        col = np.clip(xy_normalized[:, 0] * (W - 1), 0.0, float(W - 1))
        row = np.clip(xy_normalized[:, 1] * (H - 1), 0.0, float(H - 1))
        c0 = np.minimum(col.astype(np.int64), W - 2)
        r0 = np.minimum(row.astype(np.int64), H - 2)
        c1 = c0 + 1
        r1 = r0 + 1
        wc = col - c0
        wr = row - r0
        d00 = self._sdf_grid[r0, c0]
        d01 = self._sdf_grid[r0, c1]
        d10 = self._sdf_grid[r1, c0]
        d11 = self._sdf_grid[r1, c1]
        d0 = d00 * (1 - wc) + d01 * wc
        d1 = d10 * (1 - wc) + d11 * wc
        return (d0 * (1 - wr) + d1 * wr).astype(np.float32)

    def sample_sensor_batch(
        self, rng: np.random.Generator, n: int, t_max: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """採樣 n 個 sensor 觀測 (xy_norm, t, c_obs, ref_value)。"""
        t_pool = np.arange(len(self.sensor_time), dtype=np.int32)
        if t_max is not None:
            t_pool = t_pool[self.sensor_time[t_pool] <= t_max]
            if len(t_pool) == 0:
                t_pool = np.array([0], dtype=np.int32)

        t_idx      = rng.choice(t_pool, size=n, replace=True)
        sensor_idx = rng.integers(0, self.sensor_pos.shape[0], size=n)
        c_obs      = rng.integers(0, len(self.observed_channel_names), size=n).astype(np.int32)

        xy  = self.sensor_pos[sensor_idx].astype(np.float32)
        t_q = self.sensor_time[t_idx].astype(np.float32)
        ref = self.sensor_vals[sensor_idx, t_idx, c_obs].astype(np.float32)
        return xy, t_q, c_obs, ref

    def sample_physics_points(
        self,
        rng: np.random.Generator,
        n: int,
        t_max: float | None = None,
        strategy: str = "random",
    ) -> tuple[np.ndarray, np.ndarray]:
        """採樣物理殘差點 (xy_norm, t)，僅限流體域（cylinder body 已排除）。

        strategy 選項：
          "random"     — 從 fluid_xy 均勻隨機採樣（baseline）
          "chebyshev"  — 退化為 "random"
          "body_aware" — 70% 均勻採樣 + 30% 偏向 body 邊界（distance < threshold）；
                         用於強化 boundary layer 學習（Re=10031 cylinder shedding）
        """
        t_start = float(self.sensor_time[0])
        t_end   = float(self.sensor_time[-1]) if t_max is None else min(float(t_max), float(self.sensor_time[-1]))

        if strategy == "body_aware" and hasattr(self, "_sdf_grid"):
            # 30% 偏 body 邊界，70% uniform
            n_near = int(round(n * 0.3))
            n_uniform = n - n_near
            # 預計算 fluid_xy 的 SDF（一次性，cache 後重用）
            if not hasattr(self, "_fluid_sdf"):
                self._fluid_sdf = self.query_body_distance(self.fluid_xy)
            # 取 distance < median 的 fluid 點作為「near-body pool」
            _thresh = float(np.median(self._fluid_sdf))
            _near_idx = np.where(self._fluid_sdf < _thresh)[0]
            if len(_near_idx) == 0:
                # fallback：全 uniform
                chosen = rng.integers(0, len(self.fluid_xy), size=n)
            else:
                chosen_near = _near_idx[rng.integers(0, len(_near_idx), size=n_near)]
                chosen_uniform = rng.integers(0, len(self.fluid_xy), size=n_uniform)
                chosen = np.concatenate([chosen_near, chosen_uniform])
            xy = self.fluid_xy[chosen].astype(np.float32)
        else:
            chosen = rng.integers(0, len(self.fluid_xy), size=n)
            xy = self.fluid_xy[chosen].astype(np.float32)

        t  = rng.uniform(t_start, t_end, n).astype(np.float32)
        return xy, t
