"""最小 2D Kolmogorov 譜方法 solver（渦度形式），供資料同化當 forward model。

用途邊界：這**不是** DNS 生成器。專案的參考 DNS 由 pi-lnn 的
`tools/dns_generator/` 產生（速度形式、含統計對齊與輸出格式），本模組只提供
EnKF 預測步需要的東西——把渦度場往前推。刻意不共用那份程式碼：那裡大半的
邏輯（初始統計對齊、快照輸出）對 forward 傳播是負擔，而移植它會讓「參考
資料怎麼生成」和「同化怎麼推進」綁在一起。

方程（週期方域，L×L）:
    dw/dt = -J(psi, w) + nu * lap(w) + f
    lap(psi) = -w,     f = -k_f * A * cos(k_f * y * 2pi/L)

f 的形式對應速度形式的 Kolmogorov forcing `A sin(k_f y)` 作用於 x-momentum：
取 curl 後多一個 -k_f 因子。符號寫反不會讓模擬爆掉，只會讓流場方向相反，
故由單元測試以「forcing 應驅動 <u> 而非 <v>」釘住。

時間積分用 integrating-factor RK4：黏性項解析積分，非線性項 RK4。對 EnKF
的用途（每個同化區間推進數十到數百步）精度足夠，且比 ETDRK4 少一組需要
數值穩定化的係數——那些係數在小 nu*k^2*dt 下要用圍道積分求，是靜默失準的
常見來源。
"""
from __future__ import annotations

import numpy as np


class KolmogorovSolver:
    """渦度形式 2D Kolmogorov 流 solver。狀態為實空間渦度 w[N, N]。"""

    def __init__(self, N: int, nu: float, L: float = 1.0, k_f: int = 2,
                 forcing_amplitude: float = 1.0, nonlinear: bool = True):
        if N % 2 != 0:
            raise ValueError(f"N 需為偶數（FFT 對稱性），收到 {N}")
        self.N, self.nu, self.L = N, float(nu), float(L)
        self.k_f, self.A = int(k_f), float(forcing_amplitude)
        self.nonlinear = bool(nonlinear)

        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)      # 物理波數
        self.KX, self.KY = np.meshgrid(k1, k1, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        # Poisson 反解的 k=0：渦度的平均值不決定速度，該模態置零而非發散
        self.K2_inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)

        # 2/3 dealiasing：非線性項每次乘積後套用
        kmax = np.max(np.abs(k1)) * (2.0 / 3.0)
        self.dealias = (np.abs(self.KX) <= kmax) & (np.abs(self.KY) <= kmax)

        y = np.arange(N) * L / N
        # 渦度形式的 forcing：curl of (A sin(k_f*2pi*y/L), 0) = -k_f*2pi/L * A cos(...)
        kf_phys = 2.0 * np.pi * self.k_f / L
        self.f_hat = np.fft.fft2(
            -kf_phys * self.A * np.cos(kf_phys * y)[None, :] * np.ones((N, 1)))

    # ── 診斷/輔助 ────────────────────────────────────────────────────
    def velocity(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """渦度 → 速度（u, v）。psi 由 Poisson 解出，u = d psi/dy, v = -d psi/dx。"""
        psi_hat = np.fft.fft2(w) * self.K2_inv
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        return u, v

    def curl(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """速度 → 渦度 dv/dx - du/dy（velocity 的逆，供自洽性檢查）。"""
        uh, vh = np.fft.fft2(u), np.fft.fft2(v)
        return np.real(np.fft.ifft2(1j * self.KX * vh - 1j * self.KY * uh))

    def low_pass(self, w: np.ndarray, k_cut: int) -> np.ndarray:
        """保留 |k| <= k_cut（以整數波數計）的成分。"""
        kk = np.sqrt(self.K2) * self.L / (2.0 * np.pi)
        return np.real(np.fft.ifft2(np.fft.fft2(w) * (kk <= k_cut)))

    # ── 右手邊與時間積分 ─────────────────────────────────────────────
    def _nonlinear_hat(self, w_hat: np.ndarray) -> np.ndarray:
        """-J(psi, w) 的譜表示，帶 2/3 dealiasing。"""
        if not self.nonlinear:
            return np.zeros_like(w_hat)
        psi_hat = w_hat * self.K2_inv
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        wx = np.real(np.fft.ifft2(1j * self.KX * w_hat))
        wy = np.real(np.fft.ifft2(1j * self.KY * w_hat))
        return -np.fft.fft2(u * wx + v * wy) * self.dealias

    def _rhs_hat(self, w_hat: np.ndarray) -> np.ndarray:
        """非黏性部分（非線性 + forcing）；黏性由 integrating factor 處理。"""
        return self._nonlinear_hat(w_hat) + self.f_hat

    def step(self, w_hat: np.ndarray, dt: float) -> np.ndarray:
        """integrating-factor RK4 前進一步（輸入輸出皆為譜空間）。"""
        E_half = np.exp(-self.nu * self.K2 * dt / 2.0)
        E_full = np.exp(-self.nu * self.K2 * dt)
        a = self._rhs_hat(w_hat)
        b = self._rhs_hat(E_half * (w_hat + 0.5 * dt * a))
        c = self._rhs_hat(E_half * w_hat + 0.5 * dt * b)
        d = self._rhs_hat(E_full * w_hat + dt * E_half * c)
        return (E_full * w_hat
                + dt / 6.0 * (E_full * a + 2.0 * E_half * (b + c) + d))

    def integrate(self, w: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
        """實空間渦度前進 n_steps；回傳實空間渦度。"""
        w_hat = np.fft.fft2(np.asarray(w, dtype=np.float64))
        for _ in range(int(n_steps)):
            w_hat = self.step(w_hat, dt)
        return np.real(np.fft.ifft2(w_hat))
