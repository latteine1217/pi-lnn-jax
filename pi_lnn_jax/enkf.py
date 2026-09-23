"""Ensemble Kalman Filter（stochastic / perturbed-observations）用於稀疏感測重建。

與本專案其他重建法的定位差異要講清楚：EnKF **知道精確的 PDE 離散化**
（forward model 就是 solver 本身），而 PI-CON 只有 residual 形式的物理約束、
GappyPOD/DMD 只有資料建的線性子空間。所以它不是「同一條件下的更強方法」，
是「拿到更多資訊的方法」——比較時必須連同這個不對等一起報告。

實作選 stochastic EnKF（擾動觀測）而非 ETKF：觀測數 O(10^2) 時
(HPH^T + R) 的求逆很便宜，而擾動觀測版的更新式與局域化的結合最直觀
（Schur 乘積直接逐元素乘在 PH^T 與 HPH^T 上），少一層可能出錯的變換。
**兩者都要乘**——只乘其中一個時濾波器會在稀疏觀測下發散，見
`_build_obs_localization` 的 docstring 與 `tests/test_enkf.py` 的回歸測試。

高維下 localization 與 inflation 是能不能跑的前提，不是調參選項：
N_e=O(10) 對上 O(10^4) 維狀態，取樣協方差必然退化成低秩且遠場出現偽相關。
兩者的參數會影響結果，故應掃描並報告敏感度，而不是挑一組好看的。
"""
from __future__ import annotations

import numpy as np


def gaspari_cohn(r: np.ndarray, c: float) -> np.ndarray:
    """Gaspari–Cohn 局域化函數（Gaspari & Cohn 1999 eq. 4.10）。

    在 r=0 為 1、r>=2c 為 0，且處處非負——後者是關鍵：用高斯截斷之類的
    近似會讓局域化後的協方差失去半正定性，濾波器可能發散。
    """
    r = np.abs(np.asarray(r, dtype=np.float64)) / float(c)
    out = np.zeros_like(r)

    m = r <= 1.0
    x = r[m]
    out[m] = (((-0.25 * x + 0.5) * x + 0.625) * x - 5.0 / 3.0) * x**2 + 1.0

    m = (r > 1.0) & (r < 2.0)
    x = r[m]
    out[m] = ((((x / 12.0 - 0.5) * x + 0.625) * x + 5.0 / 3.0) * x - 5.0) * x \
        + 4.0 - 2.0 / (3.0 * x)
    return np.clip(out, 0.0, 1.0)


class EnKF:
    """稀疏 (u,v) 點觀測下的 stochastic EnKF。狀態為實空間渦度 [N, N]。"""

    def __init__(self, solver, obs_ij: np.ndarray, obs_std: float,
                 inflation: float = 1.02, loc_radius: float = 0.3,
                 seed: int = 0):
        self.solver = solver
        self.obs_ij = np.asarray(obs_ij, dtype=int)      # [K, 2] 網格索引
        self.obs_std = float(obs_std)
        self.inflation = float(inflation)
        self.loc_radius = float(loc_radius)
        self.rng = np.random.default_rng(seed)
        self.K = self.obs_ij.shape[0]
        self._loc = self._build_localization()
        self._loc_obs = self._build_obs_localization()

    # ── 觀測算子 ─────────────────────────────────────────────────────
    def observe(self, w: np.ndarray) -> np.ndarray:
        """渦度 → 觀測向量 [u(K); v(K)]。

        一律經 solver.velocity，不自行重算 Poisson 反解：兩份實作會悄悄分岔，
        而分岔的後果是同化把狀態往錯的方向拉，且看起來仍然收斂。
        """
        u, v = self.solver.velocity(w)
        i, j = self.obs_ij[:, 0], self.obs_ij[:, 1]
        return np.concatenate([u[i, j], v[i, j]])

    # ── 局域化 ───────────────────────────────────────────────────────
    def _cell_centres(self) -> np.ndarray:
        """格點中心座標 [N]。狀態點與觀測點共用同一組，故相對距離自洽。"""
        N, L = self.solver.N, self.solver.L
        return (np.arange(N) + 0.5) * L / N

    def _periodic_gc(self, ax, ay, bx, by) -> np.ndarray:
        """兩組點在週期方域上的 Gaspari--Cohn 權重 [len(a), len(b)]。"""
        L = self.solver.L

        def per(d):                                          # 週期最短距離
            d = np.abs(d)
            return np.minimum(d, L - d)

        dx = per(ax[:, None] - bx[None, :])
        dy = per(ay[:, None] - by[None, :])
        return gaspari_cohn(np.sqrt(dx**2 + dy**2), self.loc_radius)

    def _obs_coords(self) -> tuple[np.ndarray, np.ndarray]:
        g = self._cell_centres()
        return g[self.obs_ij[:, 0]], g[self.obs_ij[:, 1]]

    def _build_localization(self) -> np.ndarray:
        """[n_state, 2K] 的狀態--觀測局域化權重。"""
        g = self._cell_centres()
        GX, GY = np.meshgrid(g, g, indexing="ij")
        ox, oy = self._obs_coords()
        rho = self._periodic_gc(GX.ravel(), GY.ravel(), ox, oy)
        return np.concatenate([rho, rho], axis=1)            # u 與 v 兩段共用

    def _build_obs_localization(self) -> np.ndarray:
        """[2K, 2K] 的觀測--觀測局域化權重。

        Why 這個必須存在：Schur-product 局域化是作用在**完整**協方差 P 上
        （Houtekamer & Mitchell 2001），而 P 被局域化蘊含 PH^T 與 HPH^T
        **兩者**都被局域化。只乘 PH^T 會讓增益
        K = (ρ∘PH^T)(HPH^T + R)^{-1} 不對應任何合法協方差的 Kalman gain：
        分析增量不再被 ensemble 子空間束縛，每次同化都注入能量。

        這不是理論顧慮。實測（N=128、100 obs ＝ 0.6% 覆蓋、Ne=64）：只局域化
        PH^T 時相對誤差 0.31 → 1.00 → 12.7 → NaN；兩者都局域化則
        0.08 → 0.06 → 0.05 → 0.04 穩定收斂。觀測覆蓋率越低差異越劇烈——
        6% 覆蓋時幾乎看不出來，這正是舊版孿生測試漏掉它的原因。

        PSD 不受破壞：Gaspari--Cohn 是正定相關函數，2x2 的全 1 分塊結構
        ＝ kron(ones(2,2), ρ)，兩者皆半正定，Schur 乘積定理保證乘積半正定。
        """
        ox, oy = self._obs_coords()
        rho = self._periodic_gc(ox, oy, ox, oy)              # [K, K]
        return np.block([[rho, rho], [rho, rho]])            # u/v 兩段共用同一空間權重

    # ── 預測與分析 ───────────────────────────────────────────────────
    def forecast(self, ens: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
        """把每個 member 往前推 n_steps。"""
        return np.stack([self.solver.integrate(w, dt=dt, n_steps=n_steps) for w in ens])

    def analyze(self, ens: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """單次分析步；ens [N_e, N, N]，y_obs [2K]。"""
        Ne, N, _ = ens.shape
        X = ens.reshape(Ne, -1).T                            # [n_state, Ne]
        mean = X.mean(axis=1, keepdims=True)
        # inflation 在分析前施加於偏差，補償取樣造成的 spread 低估
        X = mean + self.inflation * (X - mean)

        Y = np.stack([self.observe(w) for w in
                      X.T.reshape(Ne, N, N)], axis=1)        # [2K, Ne]
        Xp = X - X.mean(axis=1, keepdims=True)
        Yp = Y - Y.mean(axis=1, keepdims=True)

        # 兩者都局域化——理由與實測見 `_build_obs_localization` 的 docstring。
        # 只乘其中一個會讓增益不再是合法的 Kalman gain。
        PHt = (Xp @ Yp.T) / (Ne - 1) * self._loc
        HPHt = (Yp @ Yp.T) / (Ne - 1) * self._loc_obs
        R = np.eye(2 * self.K) * max(self.obs_std, 1e-8) ** 2
        # 擾動觀測：每個 member 配一份獨立噪音，否則分析 spread 會被系統性低估
        pert = self.rng.normal(0.0, max(self.obs_std, 1e-8), size=(2 * self.K, Ne))
        innov = (y_obs[:, None] + pert) - Y

        gain_rhs = np.linalg.solve(HPHt + R, innov)          # [2K, Ne]
        return (X + PHt @ gain_rhs).T.reshape(Ne, N, N)
