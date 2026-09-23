"""Phase A classical sparse-reconstruction baselines（training-free，CPU 可跑）。

設計（對齊 knowledge/overview/current-status.md §2）:
  統一介面 → 輸出 (u_pred, v_pred) 形狀 [T, N, N]，可直接餵 pi_lnn_jax.evaluate.compute_metrics，
  與 PI-CON 共用 metric / sensors / DNS，公平比較 by construction。

座標約定（與 pi_lnn_jax.data 一致）:
  - DNS 場 convention：field[t, ix, iy]，(x=ix/N, y=iy/N)，週期單位方域 [0,1)^2。
  - sensor_pos [K, 2] 為 (x, y) ∈ [0,1]^2；sensor_vals [T, K, 2] 為 (u, v)。

本檔只含「不需訓練」的 baseline：
  - InterpBaseline：逐 snapshot scatter→grid 內插（floor baseline，對應 thin-plate-spline）。
  - GappyPOD：POD basis（由 train-Re DNS 建）+ 感測點 least-squares 擬合係數（線性子空間強 baseline）。
深度監督式 baseline（Voronoi-CNN / SHRED）屬 Phase B，需 GPU，另置 baselines_deep.py。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def _sensor_grid_indices(sensor_pos: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """sensor_pos [K,2] (x,y)∈[0,1]^2 → 最近格點 (ix, iy)，週期 wrap。"""
    idx = np.round(np.asarray(sensor_pos, dtype=np.float64) * n).astype(int) % n
    return idx[:, 0], idx[:, 1]


class InterpBaseline:
    """逐 snapshot 散點→網格內插。

    periodic=True（Kolmogorov 預設）: 用 3x3 週期鏡像補齊感測點，使整個 query grid
    落在凸包內並近似週期性；linear 內插殘留的 NaN 以 nearest 回填。
    periodic=False: 純 scipy.griddata（用於非週期/測試 affine 場）。
    """

    def __init__(self, method: str = "linear", periodic: bool = True):
        if method not in ("linear", "cubic", "nearest"):
            raise ValueError(f"unsupported interp method: {method}")
        self.method = method
        self.periodic = periodic

    def reconstruct(self, sensor_vals, sensor_pos, grid_shape) -> tuple[np.ndarray, np.ndarray]:
        sensor_vals = np.asarray(sensor_vals, dtype=np.float64)  # [T,K,2]
        sensor_pos = np.asarray(sensor_pos, dtype=np.float64)    # [K,2]
        N = int(grid_shape[0])
        if grid_shape[1] != N:
            raise ValueError("InterpBaseline 目前僅支援方域 (N, N)")
        T = sensor_vals.shape[0]

        g = np.arange(N) / N
        XX, YY = np.meshgrid(g, g, indexing="ij")  # field[ix,iy] at (g[ix], g[iy])
        query = np.stack([XX.ravel(), YY.ravel()], axis=1)  # [N*N, 2]

        u_pred = np.empty((T, N, N), dtype=np.float32)
        v_pred = np.empty((T, N, N), dtype=np.float32)
        for t in range(T):
            for c, out in ((0, u_pred), (1, v_pred)):
                pts, vals = self._maybe_tile(sensor_pos, sensor_vals[t, :, c])
                field = griddata(pts, vals, query, method=self.method)
                if np.isnan(field).any():
                    # 凸包外殘留 → nearest 回填（不影響內部 affine 精確性）
                    fallback = griddata(pts, vals, query, method="nearest")
                    field = np.where(np.isnan(field), fallback, field)
                out[t] = field.reshape(N, N).astype(np.float32)
        return u_pred, v_pred

    def _maybe_tile(self, pos: np.ndarray, vals: np.ndarray):
        """週期模式下用 3x3 鏡像複製感測點（值週期相同），覆蓋 [-1,2]^2。"""
        if not self.periodic:
            return pos, vals
        tiled_pos, tiled_vals = [], []
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                tiled_pos.append(pos + np.array([dx, dy]))
                tiled_vals.append(vals)
        return np.concatenate(tiled_pos, axis=0), np.concatenate(tiled_vals, axis=0)


class GappyPOD:
    """Gappy-POD / least-squares POD 重建（Everson & Sirovich）。

    fit: 由 train-Re DNS 場建 POD basis（[u; v] 聯合 SVD）。
    reconstruct: 對每個時間，於感測點解 min ||Phi_s a - y_s||，再 a → 全場。

    leakage 防線：basis 只能用 train-Re DNS（呼叫端負責，勿傳入 held-out Re 場）。
    """

    def __init__(self, n_modes: int | None = None, energy: float = 0.99):
        self.n_modes = n_modes
        self.energy = energy
        self.N: int | None = None
        self.modes: np.ndarray | None = None  # [2N^2, r]
        self.mean: np.ndarray | None = None   # [2N^2, 1]

    def fit(self, train_u, train_v) -> "GappyPOD":
        train_u = np.asarray(train_u, dtype=np.float64)  # [M,N,N]
        train_v = np.asarray(train_v, dtype=np.float64)
        if train_u.shape != train_v.shape or train_u.ndim != 3 or train_u.shape[1] != train_u.shape[2]:
            raise ValueError(f"train fields 需 [M,N,N] 方域，得 {train_u.shape}")
        M, N, _ = train_u.shape
        self.N = N
        # snapshot 矩陣 A [D=2N^2, M]，每欄一個 [u_flat; v_flat] snapshot（row-major）
        stacked = np.stack([train_u.reshape(M, -1), train_v.reshape(M, -1)], axis=1)  # [M,2,N^2]
        A = stacked.reshape(M, -1).T  # [2N^2, M]
        self.mean = A.mean(axis=1, keepdims=True)
        U, S, _ = np.linalg.svd(A - self.mean, full_matrices=False)  # U [D, min(D,M)]
        r = self._select_r(S)
        self.modes = U[:, :r]
        return self

    def _select_r(self, S: np.ndarray) -> int:
        if self.n_modes is not None:
            return int(min(self.n_modes, len(S)))
        total = float(np.sum(S ** 2))
        if total <= 0:
            return 1
        cum = np.cumsum(S ** 2) / total
        return int(np.searchsorted(cum, self.energy) + 1)

    def reconstruct(self, sensor_vals, sensor_pos) -> tuple[np.ndarray, np.ndarray]:
        if self.modes is None or self.N is None:
            raise RuntimeError("GappyPOD 未 fit；先呼叫 .fit(train_u, train_v)")
        sensor_vals = np.asarray(sensor_vals, dtype=np.float64)  # [T,K,2]
        N = self.N
        ix, iy = _sensor_grid_indices(sensor_pos, N)
        flat = ix * N + iy
        rows = np.concatenate([flat, N * N + flat])  # u 段 + v 段
        Phi_s = self.modes[rows, :]          # [2K, r]
        mean_s = self.mean[rows, 0]          # [2K]

        T = sensor_vals.shape[0]
        u_pred = np.empty((T, N, N), dtype=np.float32)
        v_pred = np.empty((T, N, N), dtype=np.float32)
        for t in range(T):
            y = np.concatenate([sensor_vals[t, :, 0], sensor_vals[t, :, 1]])  # [2K]
            a, *_ = np.linalg.lstsq(Phi_s, y - mean_s, rcond=None)
            full = (self.modes @ a) + self.mean[:, 0]  # [2N^2]
            u_pred[t] = full[: N * N].reshape(N, N).astype(np.float32)
            v_pred[t] = full[N * N:].reshape(N, N).astype(np.float32)
        return u_pred, v_pred


class GappyDMD(GappyPOD):
    """Gappy 重建，基底改用 DMD 模態（其餘與 GappyPOD 完全相同）。

    存在的理由是一個受控對照：sensor 最小平方投影、秩選擇、leakage 防線
    都繼承自 GappyPOD，**唯一**的差別是基底從 POD（能量最優的靜態子空間）
    換成 exact DMD（帶特徵值的動態模態）。兩者若表現相近，代表在該 sensor
    預算與時窗下，動態資訊對重建沒有額外貢獻——那是關於問題的判讀，因此
    「只換基底」必須是結構上的事實，不能是敘述上的宣稱。

    DMD 模態是複數而投影需要實基底：一對共軛模態以 [Re, Im] 兩個實向量
    表示，張成的實子空間與該共軛對相同；丟掉虛部會把旋轉分量一併丟掉。

    leakage 防線與 GappyPOD 相同：basis 只能用 train-Re DNS。
    """

    @staticmethod
    def _real_modes(Phi: np.ndarray, w: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """複 DMD 模態 → 實基底。實特徵值取一欄，共軛對取 [Re, Im] 兩欄。"""
        cols: list[np.ndarray] = []
        skip: set[int] = set()
        for j in range(len(w)):
            if j in skip:
                continue
            if abs(w[j].imag) <= tol:
                cols.append(Phi[:, j].real)
                continue
            for k in range(j + 1, len(w)):
                if k not in skip and abs(w[k] - np.conj(w[j])) <= tol * max(1.0, abs(w[j])):
                    skip.add(k)
                    break
            cols.append(Phi[:, j].real)
            cols.append(Phi[:, j].imag)
        return np.column_stack(cols)

    def fit(self, train_u, train_v) -> "GappyDMD":
        train_u = np.asarray(train_u, dtype=np.float64)
        train_v = np.asarray(train_v, dtype=np.float64)
        if train_u.shape != train_v.shape or train_u.ndim != 3 or train_u.shape[1] != train_u.shape[2]:
            raise ValueError(f"train fields 需 [M,N,N] 方域，得 {train_u.shape}")
        M, N, _ = train_u.shape
        if M < 2:
            raise ValueError(
                f"DMD 需要至少 2 個 snapshot 才能定義動態，收到 M={M}；"
                "不退化成 POD——那會讓兩個 baseline 在資料不足時悄悄變成同一個")
        self.N = N
        stacked = np.stack([train_u.reshape(M, -1), train_v.reshape(M, -1)], axis=1)
        A = stacked.reshape(M, -1).T                      # [2N^2, M]
        self.mean = A.mean(axis=1, keepdims=True)
        Ac = A - self.mean
        X, Y = Ac[:, :-1], Ac[:, 1:]

        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        r = min(self._select_r(S), S.size)
        Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
        # 低階算子 Atilde = Ur* Y Vr Sr^-1，其特徵對給出 DMD 模態
        Atilde = Ur.conj().T @ Y @ Vr / Sr
        w, W = np.linalg.eig(Atilde)
        Phi = Y @ Vr @ (W / Sr[:, None])                  # exact DMD（Tu 2014）
        self.modes = self._real_modes(Phi, w)
        return self
