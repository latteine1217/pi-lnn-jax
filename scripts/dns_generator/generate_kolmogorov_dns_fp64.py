"""
Kolmogorov Flow DNS 資料生成器 (統一版本)
==========================================

自動檢測並使用最佳計算後端：
- GPU 可用：PyTorch (MPS/CUDA)
- 僅 CPU：NumPy

使用 pseudo-spectral 方法求解 2D Kolmogorov flow。

作者：PINNs-MVP 團隊
日期：2025-11-22
版本：2.0-Unified
"""

import numpy as np
import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
import logging
import time
import math

# 檢測 PyTorch 是否可用
try:
    import torch
    TORCH_AVAILABLE = True
    torch.set_float32_matmul_precision("high")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def select_backend() -> Tuple[str, Optional[object]]:
    """
    自動選擇最佳計算後端
    
    Returns:
        backend: 'torch-mps', 'torch-cuda', 'torch-cpu', 'numpy'
        device: torch.device 或 None
    """
    if not TORCH_AVAILABLE:
        logging.info("🔧 PyTorch 不可用，使用 NumPy (CPU)")
        return 'numpy', None
    
    # 優先 MPS (Apple Metal)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f"🚀 使用 PyTorch + MPS (Apple Metal GPU)")
        return 'torch-mps', device
    
    # 次選 CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"🚀 使用 PyTorch + CUDA (NVIDIA GPU)")
        return 'torch-cuda', device
    
    # CPU PyTorch
    if TORCH_AVAILABLE:
        device = torch.device('cpu')
        logging.info(f"🔧 使用 PyTorch (CPU)")
        return 'torch-cpu', device
    
    # 回退到 NumPy
    logging.info("🔧 使用 NumPy (CPU)")
    return 'numpy', None


class KolmogorovFlowDNS:
    """
    2D Kolmogorov Flow DNS 求解器（自動選擇 GPU/CPU）
    """

    def __init__(
        self,
        N: int = 1024,
        L: float = 1.0,
        nu: float = 1e-5,
        A: float = 0.1,
        k_f: int = 2,
        dt: float = 0.001,
        dealias: bool = True,
        dealias_mode: str = "3/2",
        backend: Optional[str] = None,
        project_after_step: bool = True,
        enforce_zero_mean: bool = True,
        seed: int = 42,
        target_initial_ke: float = 0.16145706176757812,
        target_initial_forcing_coeff: float = 0.04649721458554268,
        target_initial_spectrum_bins: int = 8,
        integrator: str = 'rk4',
        ic_mode: str = 'band_limited_random',
        ic_k_cutoff: Optional[float] = None,
        drag_alpha: float = 0.0,
    ):
        """
        Args:
            N: 網格點數（每方向）
            L: 域長度
            nu: 黏滯係數
            A: 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            dealias: 是否啟用去混疊
            dealias_mode: 去混疊方式 ('2/3' 或 '3/2')
            backend: 手動指定後端 ('torch-mps', 'torch-cuda', 'numpy', None=auto)
            integrator: 時間積分方案 ('rk4' 或 'etdrk4')
            ic_mode: IC 生成模式
                - 'band_limited_random' (default): real-space rng → FFT → low-pass
                  （既有行為，與所有訓練資料 byte-aligned）
                - 'spectral_seeded': 對每個 integer Fourier mode 用 deterministic sub-RNG
                  （Grid independence test 用；保證不同 N 在 |k|<=k_cutoff 共同 modes 一致）
                  此模式會 skip _align_initial_statistics 以保 N-invariance
            ic_k_cutoff: spectral_seeded mode 的 |k| cutoff；None 則用 max(8.0, 4*k_f)（同 band_limited 預設）
        """
        # 選擇後端
        if backend is None:
            self.backend, self.device = select_backend()
        else:
            if backend.startswith('torch-'):
                if not TORCH_AVAILABLE:
                    raise RuntimeError(f"PyTorch 不可用，無法使用 {backend}")
                device_name = backend.split('-')[1]
                self.device = torch.device(device_name)
            else:
                self.device = None
            self.backend = backend
        
        self.use_torch = self.backend.startswith('torch')
        
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.dealias = dealias
        self.dealias_mode = dealias_mode
        self.project_after_step = project_after_step
        self.enforce_zero_mean = enforce_zero_mean
        self.seed = seed
        self.target_initial_ke = target_initial_ke
        self.target_initial_forcing_coeff = target_initial_forcing_coeff
        self.target_initial_spectrum_bins = target_initial_spectrum_bins
        self.integrator = integrator
        # 線性（Ekman/Rayleigh）阻尼係數。2D 逆級聯把能量送回 k_min，該尺度唯一的
        # 移除機制是黏性本身（1/(nu k_min^2)=253 於 nu=1e-4, L=1），無阻尼時能量在
        # 最低模態累積成 condensate、評估窗永遠不在統計穩態上。-alpha*u 是不隨 k 變的
        # 匯，與黏性同為對角線性算子，故一併進 ETDRK4 的 L，不增加時間步限制。
        # 預設 0.0 = 既有行為：alpha=0 時 -(nu*k2 + 0.0) 與原本的 -nu*k2 逐位元相同
        # （fp32/fp64 皆驗過，含零的符號位）。目前沒有測試釘住這點。
        self.drag_alpha = float(drag_alpha)
        if self.drag_alpha < 0.0:
            raise ValueError(f'drag_alpha 須 >= 0，收到 {drag_alpha}')

        # IC mode 設定（Grid independence test 用 spectral_seeded）
        self.ic_mode = ic_mode
        if self.ic_mode not in ("band_limited_random", "spectral_seeded"):
            raise ValueError(
                f"Unknown ic_mode: {self.ic_mode}. "
                f"Must be 'band_limited_random' or 'spectral_seeded'."
            )
        self.ic_k_cutoff = (
            float(ic_k_cutoff) if ic_k_cutoff is not None else max(8.0, 4.0 * float(self.k_f))
        )

        # 可重現亂數初始化
        self.rng = np.random.default_rng(self.seed)
        if self.use_torch:
            torch.manual_seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.seed)

        # 空間網格
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if self.use_torch:
            self.X = torch.tensor(X, dtype=torch.float64, device=self.device)
            self.Y = torch.tensor(Y, dtype=torch.float64, device=self.device)
        else:
            self.X = X
            self.Y = Y

        # 波數網格
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        kx, ky = np.meshgrid(k, k, indexing='ij')
        k2 = kx**2 + ky**2
        
        if self.use_torch:
            self.kx = torch.tensor(kx, dtype=torch.float64, device=self.device)
            self.ky = torch.tensor(ky, dtype=torch.float64, device=self.device)
            self.k2 = torch.tensor(k2, dtype=torch.float64, device=self.device)
            self.ikx = 1j * self.kx
            self.iky = 1j * self.ky
            self.k2_safe = self.k2.clone()
            self.k2_safe[0, 0] = 1.0
        else:
            self.kx = kx
            self.ky = ky
            self.k2 = k2
            self.ikx = 1j * self.kx
            self.iky = 1j * self.ky
            self.k2_safe = self.k2.copy()
            self.k2_safe[0, 0] = 1.0

        if self.dealias and self.dealias_mode not in {"2/3", "3/2"}:
            raise ValueError(f"Unknown dealias_mode: {self.dealias_mode}")

        if self.dealias and self.dealias_mode == "3/2":
            self.pad_N = int(3 * self.N / 2)
            self.pad_scale = (self.pad_N / self.N) ** 2
            self.pad_scale_inv = (self.N / self.pad_N) ** 2

            k_pad = 2 * np.pi * np.fft.fftfreq(self.pad_N, d=self.L / self.pad_N)
            kx_pad, ky_pad = np.meshgrid(k_pad, k_pad, indexing='ij')

            if self.use_torch:
                self.pad_kx = torch.tensor(kx_pad, dtype=torch.float64, device=self.device)
                self.pad_ky = torch.tensor(ky_pad, dtype=torch.float64, device=self.device)
            else:
                self.pad_kx = kx_pad
                self.pad_ky = ky_pad
        else:
            self.pad_N = None
            self.pad_scale = 1.0
            self.pad_scale_inv = 1.0
            self.pad_kx = None
            self.pad_ky = None

        # 去混疊遮罩
        if dealias and self.dealias_mode == "2/3":
            k_max = k.max() * 2 / 3
            mask = np.sqrt(kx**2 + ky**2) <= k_max
            if self.use_torch:
                self.dealias_mask = torch.tensor(mask, dtype=torch.float64, device=self.device)
            else:
                self.dealias_mask = mask.astype(float)
        else:
            if self.use_torch:
                self.dealias_mask = torch.ones_like(self.k2)
            else:
                self.dealias_mask = np.ones_like(k2)

        # 預先計算能譜分箱（使用無因次整數模態 |m|）
        # 索引慣例（下游讀 initial_spectrum_achieved 時常搞錯）：
        #   bin edges 從 0.5 起 → spectrum[0] 對應 **k=1**，不是 k=0；
        #   k=0（平均模態）刻意不入箱，spectrum_valid 會把它濾掉。
        # 涵蓋範圍：nbins = N//2，只到軸向 Nyquist，**不含** Fourier 方盒的對角
        # corner（|k|max = (N/2)√2）。這裡的譜只用於 IC 低波數匹配與診斷，不用來
        # 檢查 Parseval；要 Parseval 精確的譜請用 metric_artifact.compute_energy_spectrum。
        k_mode = np.fft.fftfreq(self.N, d=1.0 / self.N)
        kx_mode, ky_mode = np.meshgrid(k_mode, k_mode, indexing='ij')
        k_radius = np.sqrt(kx_mode**2 + ky_mode**2)
        self.spectrum_nbins = self.N // 2
        self.spectrum_bin_edges = np.arange(0.5, self.spectrum_nbins + 1.5, 1.0)
        self.spectrum_k = 0.5 * (self.spectrum_bin_edges[:-1] + self.spectrum_bin_edges[1:])
        self.spectrum_bin_index = np.digitize(k_radius.ravel(), self.spectrum_bin_edges) - 1
        self.spectrum_valid = (
            (self.spectrum_bin_index >= 0) & (self.spectrum_bin_index < len(self.spectrum_k))
        )
        self.low_k_target_spectrum = np.array([
            8.497518494299e-04,
            6.975045434551e-03,
            2.264935849117e-02,
            6.049940390321e-02,
            3.254797171623e-02,
            2.788478953810e-02,
            8.010594852487e-03,
            1.667450231230e-03,
        ], dtype=np.float64)

        # 物理波數（在 L=1 的文獻設定下，k_f=2 對應 forcing = A*sin(4πy)）
        k_phys = k_f * 2 * np.pi / L
        self.k_phys = k_phys

        # 強迫項（實空間，Kolmogorov forcing 僅作用於 x-momentum）
        if self.use_torch:
            self.forcing_x = A * torch.sin(k_phys * self.Y)
            self.forcing_y = torch.zeros_like(self.forcing_x)
        else:
            self.forcing_x = A * np.sin(k_phys * Y)
            self.forcing_y = np.zeros_like(self.forcing_x)

        # 預先計算頻譜空間強迫項 (避免在 compute_rhs 重複計算)
        self.forcing_U_hat = self.fft2(self.forcing_x)
        self.forcing_V_hat = self.fft2(self.forcing_y)

        # 初始化場 — 兩種 IC 模式
        init_amplitude = 1.0
        if self.ic_mode == "band_limited_random":
            # 既有 default: real-space rng → FFT → low-pass
            U_init_np, init_k_cutoff = self._make_band_limited_random_field(amplitude=init_amplitude)
            V_init_np, _ = self._make_band_limited_random_field(amplitude=init_amplitude, k_cutoff=init_k_cutoff)
            self.init_k_cutoff = init_k_cutoff

            if self.use_torch:
                U_init = torch.tensor(U_init_np, dtype=torch.float64, device=self.device)
                V_init = torch.tensor(V_init_np, dtype=torch.float64, device=self.device)
                self.U_hat = self.fft2(U_init) * self.dealias_mask
                self.V_hat = self.fft2(V_init) * self.dealias_mask
            else:
                U_init = U_init_np
                V_init = V_init_np
                self.U_hat = np.fft.fft2(U_init) * self.dealias_mask
                self.V_hat = np.fft.fft2(V_init) * self.dealias_mask

            logging.info(
                f"✅ 初始化 band-limited 隨機場：amplitude = {init_amplitude:.6f}, "
                f"seed = {self.seed}, k_cutoff = {init_k_cutoff:.1f}"
            )
        elif self.ic_mode == "spectral_seeded":
            # Grid independence test 用：mode-indexed SeedSequence 保證 N-invariance
            U_hat_np, V_hat_np = self._make_spectral_ic(self.seed, self.ic_k_cutoff)
            self.init_k_cutoff = float(self.ic_k_cutoff)
            if self.use_torch:
                self.U_hat = torch.tensor(U_hat_np, dtype=torch.complex128, device=self.device) * self.dealias_mask
                self.V_hat = torch.tensor(V_hat_np, dtype=torch.complex128, device=self.device) * self.dealias_mask
            else:
                self.U_hat = U_hat_np * self.dealias_mask
                self.V_hat = V_hat_np * self.dealias_mask
            logging.info(
                f"✅ 初始化 spectral-seeded IC：seed = {self.seed}, "
                f"k_cutoff = {self.ic_k_cutoff:.1f}（mode-indexed SeedSequence, N-invariant）"
            )
        else:
            raise ValueError(f"Unknown ic_mode: {self.ic_mode}")

        # 初始化後立即投影，確保 ∇·u = 0
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)
        if self.enforce_zero_mean:
            self.U_hat, self.V_hat = self._zero_mean_hat(self.U_hat, self.V_hat)

        # 以 JAXPI transient 為目標對齊初始統計量（優先對齊 KE 與 forcing mode）
        self._align_initial_statistics()
        self._setup_etd_coefficients()
        logging.info(f"⚙️  時間積分方案：{self.integrator}")

    def to_numpy(self, tensor):
        """將 Torch tensor 轉為 NumPy（用於保存）"""
        if self.use_torch:
            return tensor.cpu().numpy()
        return tensor

    def _compute_ke_from_fields(self, U, V):
        if self.use_torch:
            return float((0.5 * torch.mean(U**2 + V**2)).item())
        return float(0.5 * np.mean(U**2 + V**2))

    def _compute_forcing_mode_coeff(self, U):
        Y_np = self.to_numpy(self.Y) if self.use_torch else self.Y
        phi = np.sin(self.k_phys * Y_np)
        U_np = self.to_numpy(U) if self.use_torch else U
        phi_norm = float(np.mean(phi**2))
        return float(np.mean(U_np * phi) / max(phi_norm, 1e-12))

    def _match_initial_low_k_spectrum(self):
        """調整低波數 shell 能量，對齊 JAXPI initial spectrum 的前幾個 bins。"""
        nbins = min(self.target_initial_spectrum_bins, len(self.low_k_target_spectrum))
        if nbins <= 0:
            return

        U_hat_np = self.to_numpy(self.U_hat) if self.use_torch else self.U_hat.copy()
        V_hat_np = self.to_numpy(self.V_hat) if self.use_torch else self.V_hat.copy()
        e2d = 0.5 * (np.abs(U_hat_np) ** 2 + np.abs(V_hat_np) ** 2) / (self.N ** 4)

        for shell in range(nbins):
            mask_flat = self.spectrum_valid & (self.spectrum_bin_index == shell)
            if not np.any(mask_flat):
                continue
            mask = mask_flat.reshape(self.N, self.N)
            current_shell_energy = float(e2d[mask].sum())
            target_shell_energy = float(self.low_k_target_spectrum[shell])
            if current_shell_energy <= 0 or target_shell_energy <= 0:
                continue
            scale = math.sqrt(target_shell_energy / current_shell_energy)
            U_hat_np[mask] *= scale
            V_hat_np[mask] *= scale

        if self.use_torch:
            self.U_hat = torch.tensor(U_hat_np, dtype=self.U_hat.dtype, device=self.device)
            self.V_hat = torch.tensor(V_hat_np, dtype=self.V_hat.dtype, device=self.device)
        else:
            self.U_hat = U_hat_np
            self.V_hat = V_hat_np

        self.U_hat = self.U_hat * self.dealias_mask
        self.V_hat = self.V_hat * self.dealias_mask
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)
        if self.enforce_zero_mean:
            self.U_hat, self.V_hat = self._zero_mean_hat(self.U_hat, self.V_hat)

    def _align_initial_statistics(self, n_iter: int = 5):
        """調整初始化場，使其較接近目標初始 KE、forcing-mode coefficient 與低波數頻譜。"""
        # spectral_seeded mode: 跳過 alignment 內的 KE/forcing-coeff iterative loop（保 N-invariance），
        # 但仍對 post-projection KE 做 **單一 multiplicative rescale** 到 target_initial_ke
        # （此 rescale factor 對所有 N 完全一致，Parseval 保證 → N-invariance 維持）
        if self.ic_mode == "spectral_seeded":
            # Post-projection KE rescale (compressible part 已被 _project_hat 移除)
            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)
            current_ke = self._compute_ke_from_fields(U, V)
            if current_ke > 1e-30:
                scale = math.sqrt(self.target_initial_ke / current_ke)
                self.U_hat = self.U_hat * scale
                self.V_hat = self.V_hat * scale

            # Record final stats (post-rescale)
            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)
            omega, _, _ = self.compute_omega_div(
                U_hat=self.U_hat, V_hat=self.V_hat, return_numpy=True
            )
            init_spectrum = self.compute_energy_spectrum(U, V)
            self.initial_ke = self._compute_ke_from_fields(U, V)
            self.initial_forcing_coeff = self._compute_forcing_mode_coeff(U)
            self.initial_enstrophy = float(0.5 * np.mean(omega ** 2))
            self.initial_spectrum_achieved = init_spectrum[: self.target_initial_spectrum_bins].astype(np.float64)
            logging.info(
                f"📌 spectral_seeded IC mode: 跳過 iterative alignment 以保 N-invariance, "
                f"only single KE rescale to target. "
                f"KE={self.initial_ke:.6f}, forcing_coeff={self.initial_forcing_coeff:.6f}, "
                f"enstrophy={self.initial_enstrophy:.6f}"
            )
            return

        # band_limited_random mode (既有 default 行為)
        for _ in range(n_iter):
            self._match_initial_low_k_spectrum()

            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)
            current_ke = self._compute_ke_from_fields(U, V)
            if current_ke > 0:
                scale = math.sqrt(self.target_initial_ke / current_ke)
                self.U_hat *= scale
                self.V_hat *= scale

            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)
            current_coeff = self._compute_forcing_mode_coeff(U)
            delta_coeff = self.target_initial_forcing_coeff - current_coeff

            if abs(delta_coeff) > 1e-10:
                if self.use_torch:
                    U = U + delta_coeff * torch.sin(self.k_phys * self.Y)
                else:
                    U = U + delta_coeff * np.sin(self.k_phys * self.Y)
                self.U_hat = self.fft2(U) * self.dealias_mask
                self.V_hat = self.fft2(V) * self.dealias_mask
                self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)
                if self.enforce_zero_mean:
                    self.U_hat, self.V_hat = self._zero_mean_hat(self.U_hat, self.V_hat)

        U = self.ifft2(self.U_hat)
        V = self.ifft2(self.V_hat)
        omega, _, _ = self.compute_omega_div(U_hat=self.U_hat, V_hat=self.V_hat, return_numpy=True)
        init_spectrum = self.compute_energy_spectrum(U, V)
        self.initial_ke = self._compute_ke_from_fields(U, V)
        self.initial_forcing_coeff = self._compute_forcing_mode_coeff(U)
        self.initial_enstrophy = float(0.5 * np.mean(omega**2))
        self.initial_spectrum_achieved = init_spectrum[: self.target_initial_spectrum_bins].astype(np.float64)

        logging.info(
            f"📌 初始統計量對齊後：KE={self.initial_ke:.6f}, "
            f"forcing_coeff={self.initial_forcing_coeff:.6f}, enstrophy={self.initial_enstrophy:.6f}, "
            f"low-k spectrum={np.array2string(self.initial_spectrum_achieved, precision=3, separator=', ')}"
        )

    def fft2(self, field):
        """2D FFT"""
        if self.use_torch:
            return torch.fft.fft2(field)
        return np.fft.fft2(field)

    def ifft2(self, field_hat):
        """2D IFFT"""
        if self.use_torch:
            return torch.fft.ifft2(field_hat).real
        return np.fft.ifft2(field_hat).real

    def fft2_batch(self, fields):
        """批次 2D FFT"""
        if self.use_torch:
            return torch.fft.fft2(fields, dim=(-2, -1))
        return np.fft.fft2(fields, axes=(-2, -1))

    def ifft2_batch(self, fields_hat):
        """批次 2D IFFT"""
        if self.use_torch:
            return torch.fft.ifft2(fields_hat, dim=(-2, -1)).real
        return np.fft.ifft2(fields_hat, axes=(-2, -1)).real

    def _make_spectral_ic(self, seed: int, k_cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        What: 對每個 integer Fourier mode (kx_int, ky_int) 在 |k| <= k_cutoff 圓盤內,
              用 SeedSequence(seed, spawn_key=(kx_int+10000, ky_int+10000)) 派生 sub-RNG 生 complex normal,
              套上同形 spectral filter exp(-(|k|/k_cutoff)^8), Hermitian conjugate 對稱保 real field。
              最後 hat *= N² 補回 NumPy FFT convention scaling, 讓實空間 amplitude 與 N 無關。

        Why:  保證對任意 N1, N2 >= 2*ceil(k_cutoff)+1, 不同 N 在共同 grid points 的實空間 field 一致：
              ifft(hat_N2)[::stride, ::stride] ≡ ifft(hat_N1) where stride = N2 / N1
              這是 Grid Independence Test 的 IC 同一性必要條件
              （Gemini 2026-05-24 brainstorm 警告："不同 N 呼叫 rng.standard_normal((N,N)) 不是 GI test"）。

        Returns:
            (U_hat, V_hat) — both complex128, shape (N, N), Hermitian symmetric, (0,0) mode = 0
            未做 Leray projection 與 dealias，由 caller 處理。
        """
        N = self.N
        U_hat = np.zeros((N, N), dtype=np.complex128)
        V_hat = np.zeros((N, N), dtype=np.complex128)

        k_cutoff_int = int(np.ceil(k_cutoff))
        if 2 * k_cutoff_int + 1 > N:
            raise ValueError(
                f"N={N} too small to contain k_cutoff={k_cutoff} "
                f"(need N >= {2 * k_cutoff_int + 1})"
            )

        # Iterate over integer modes (kx_int, ky_int) ∈ [-k_cutoff_int, +k_cutoff_int]²
        # 只對 Hermitian half-space 生 modes（lexicographic ordering），conjugate 填對位避免雙寫
        for kx_int in range(-k_cutoff_int, k_cutoff_int + 1):
            for ky_int in range(-k_cutoff_int, k_cutoff_int + 1):
                k_mag = math.sqrt(kx_int ** 2 + ky_int ** 2)
                if k_mag > k_cutoff or k_mag == 0:
                    continue
                conj_kx, conj_ky = -kx_int, -ky_int
                if (kx_int, ky_int) < (conj_kx, conj_ky):
                    continue  # 由 conjugate position 填

                # Deterministic sub-RNG by signed mode index
                sub_ss = np.random.SeedSequence(
                    seed, spawn_key=(kx_int + 10000, ky_int + 10000)
                )
                sub_rng = np.random.default_rng(sub_ss)

                # Spectral filter（與既有 band_limited 同形狀, k=k_cutoff 時 ≈ e^{-1}）
                filt = math.exp(-(k_mag / k_cutoff) ** 8)

                # 4 個 complex normal coefficients
                re_u, im_u, re_v, im_v = sub_rng.standard_normal(4) * filt

                # 主位置
                ix = kx_int % N
                iy = ky_int % N
                # 自共軛 mode (Nyquist 邊界): k ≡ -k mod N. 必須 Hermitian-consistent (im=0).
                # 對 k_cutoff < N/2 通常不觸發，但 future-proof.
                ix_c = conj_kx % N
                iy_c = conj_ky % N
                is_self_conjugate = (ix_c == ix) and (iy_c == iy)
                if is_self_conjugate:
                    U_hat[ix, iy] = re_u + 0j  # im=0 強制
                    V_hat[ix, iy] = re_v + 0j
                else:
                    U_hat[ix, iy] = re_u + 1j * im_u
                    V_hat[ix, iy] = re_v + 1j * im_v
                    # Conjugate 對應位置
                    U_hat[ix_c, iy_c] = re_u - 1j * im_u
                    V_hat[ix_c, iy_c] = re_v - 1j * im_v

        # FFT amplitude convention: NumPy fft2 沒 normalize, ifft2 除 N².
        # 為了讓實空間 field 在 common grid points 跨 N 一致, hat 乘 N² 補回。
        # (key invariant: ifft(hat)[::stride, ::stride] N-invariant)
        U_hat *= float(N * N)
        V_hat *= float(N * N)

        # (0, 0) mode 強制 0（zero-mean enforcement）
        U_hat[0, 0] = 0.0
        V_hat[0, 0] = 0.0

        # 注意: KE rescale 到 target_initial_ke 由 _align_initial_statistics 在 post-projection 完成
        # （這裡若 rescale，後續 _project_hat 會把 compressible part 移除導致 KE 折半，
        #  搬到 post-projection 做 rescale 才精準）
        return U_hat, V_hat

    def _make_band_limited_random_field(self, amplitude: float = 1.0, k_cutoff: Optional[float] = None):
        """生成經頻譜限頻的隨機實空間場，避免 raw white noise 的高波數主導。"""
        if k_cutoff is None:
            k_cutoff = max(8.0, 4.0 * float(self.k_f))

        k_mode = np.fft.fftfreq(self.N, d=1.0 / self.N)
        kx_mode, ky_mode = np.meshgrid(k_mode, k_mode, indexing='ij')
        k_radius = np.sqrt(kx_mode**2 + ky_mode**2)

        # 平滑低通濾波：保留低/中波數隨機模態，抑制高波數 white-noise tail
        spectral_filter = np.exp(- (k_radius / max(k_cutoff, 1e-12)) ** 8)
        spectral_filter[k_radius > 1.5 * k_cutoff] = 0.0

        field = amplitude * self.rng.standard_normal((self.N, self.N))
        field_hat = np.fft.fft2(field) * spectral_filter
        field_bl = np.fft.ifft2(field_hat).real

        return field_bl, float(k_cutoff)

    def pad_hat(self, field_hat):
        """頻譜零填充到 3/2 尺寸"""
        pad = (self.pad_N - self.N) // 2
        if self.use_torch:
            field_shift = torch.fft.fftshift(field_hat)
            padded = torch.zeros((self.pad_N, self.pad_N), dtype=field_hat.dtype, device=field_hat.device)
            padded[pad:pad + self.N, pad:pad + self.N] = field_shift
            return torch.fft.ifftshift(padded)

        field_shift = np.fft.fftshift(field_hat)
        padded = np.zeros((self.pad_N, self.pad_N), dtype=field_hat.dtype)
        padded[pad:pad + self.N, pad:pad + self.N] = field_shift
        return np.fft.ifftshift(padded)

    def truncate_hat(self, field_hat_pad):
        """頻譜裁切回原始尺寸"""
        pad = (self.pad_N - self.N) // 2
        if self.use_torch:
            field_shift = torch.fft.fftshift(field_hat_pad)
            truncated = field_shift[pad:pad + self.N, pad:pad + self.N]
            return torch.fft.ifftshift(truncated)

        field_shift = np.fft.fftshift(field_hat_pad)
        truncated = field_shift[pad:pad + self.N, pad:pad + self.N]
        return np.fft.ifftshift(truncated)
    
    def _project_hat(self, U_hat, V_hat):
        """
        對頻譜空間的速度場進行投影，強制滿足不可壓縮條件 ∇·u = 0。
        僅做 Leray projection；是否強制零平均流由 _zero_mean_hat / enforce_zero_mean 控制。
        """
        k_dot_u = self.kx * U_hat + self.ky * V_hat
        correction = k_dot_u / self.k2_safe
        U_hat_proj = U_hat - self.kx * correction
        V_hat_proj = V_hat - self.ky * correction

        return U_hat_proj, V_hat_proj

    def _zero_mean_hat(self, U_hat, V_hat):
        """將速度場的空間平均（(0,0) 模態）設為 0。"""
        U_hat_zero = U_hat.clone() if self.use_torch else U_hat.copy()
        V_hat_zero = V_hat.clone() if self.use_torch else V_hat.copy()
        U_hat_zero[0, 0] = 0.0 + 0.0j
        V_hat_zero[0, 0] = 0.0 + 0.0j
        return U_hat_zero, V_hat_zero

    def project_incompressible(self):
        """
        對速度場進行譜空間投影，強制滿足不可壓縮條件 ∇·u = 0。
        如有需要，再額外強制零平均流。
        """
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)
        if self.enforce_zero_mean:
            self.U_hat, self.V_hat = self._zero_mean_hat(self.U_hat, self.V_hat)

    def compute_omega_hat(self, U_hat=None, V_hat=None):
        """計算渦度頻譜 ω_hat"""
        if U_hat is None or V_hat is None:
            U_hat = self.U_hat
            V_hat = self.V_hat

        return self.ikx * V_hat - self.iky * U_hat

    def compute_omega_div(self, U_hat=None, V_hat=None, return_numpy: bool = True):
        """計算渦度與散度 (ω, ∇·u)"""
        if U_hat is None or V_hat is None:
            U_hat = self.U_hat
            V_hat = self.V_hat

        omega_hat = self.compute_omega_hat(U_hat, V_hat)
        div_hat = self.ikx * U_hat + self.iky * V_hat

        omega = self.ifft2(omega_hat)
        div_field = self.ifft2(div_hat)

        if self.use_torch and return_numpy:
            omega = self.to_numpy(omega)
            div_field = self.to_numpy(div_field)

        # 快取，供壓力/診斷使用
        self.last_omega_hat = omega_hat
        self.last_div_hat = div_hat

        return omega, div_field, omega_hat

    def compute_energy_spectrum(self, U, V):
        """計算 2D 速度場的徑向平均 kinetic energy spectrum。"""
        U_np = self.to_numpy(U) if self.use_torch else U
        V_np = self.to_numpy(V) if self.use_torch else V

        U_hat = np.fft.fft2(U_np)
        V_hat = np.fft.fft2(V_np)
        e2d = 0.5 * (np.abs(U_hat) ** 2 + np.abs(V_hat) ** 2) / (self.N ** 4)

        spectrum = np.zeros(len(self.spectrum_k), dtype=np.float64)
        flat_e = e2d.ravel()
        np.add.at(spectrum, self.spectrum_bin_index[self.spectrum_valid], flat_e[self.spectrum_valid])
        return spectrum

    def _compute_convection_hat(self, U_hat, V_hat):
        """
        What: 計算非線性對流項的頻譜表示。
        Why: 讓 RK4 與 ETDRK4 共用同一條對流與去混疊路徑，避免兩份數值核心分叉。
        """
        omega_hat = self.compute_omega_hat(U_hat, V_hat)

        if self.dealias and self.dealias_mode == "3/2":
            if self.use_torch:
                U_hat_pad = self.pad_hat(U_hat)
                V_hat_pad = self.pad_hat(V_hat)
                omega_hat_pad = 1j * (self.pad_kx * V_hat_pad - self.pad_ky * U_hat_pad)
                fields_hat = torch.stack([U_hat_pad, V_hat_pad, omega_hat_pad], dim=0)
            else:
                U_hat_pad = self.pad_hat(U_hat)
                V_hat_pad = self.pad_hat(V_hat)
                omega_hat_pad = 1j * (self.pad_kx * V_hat_pad - self.pad_ky * U_hat_pad)
                fields_hat = np.stack([U_hat_pad, V_hat_pad, omega_hat_pad], axis=0)

            U_pad, V_pad, omega_pad = self.ifft2_batch(fields_hat)
            U_pad *= self.pad_scale
            V_pad *= self.pad_scale
            omega_pad *= self.pad_scale

            conv_U = omega_pad * V_pad
            conv_V = -omega_pad * U_pad

            if self.use_torch:
                conv_hat_pad = self.fft2_batch(torch.stack([conv_U, conv_V], dim=0)) * self.pad_scale_inv
            else:
                conv_hat_pad = self.fft2_batch(np.stack([conv_U, conv_V], axis=0)) * self.pad_scale_inv

            conv_U_hat = self.truncate_hat(conv_hat_pad[0])
            conv_V_hat = self.truncate_hat(conv_hat_pad[1])
        else:
            if self.use_torch:
                fields_hat = torch.stack([U_hat, V_hat, omega_hat], dim=0)
            else:
                fields_hat = np.stack([U_hat, V_hat, omega_hat], axis=0)

            U, V, omega = self.ifft2_batch(fields_hat)

            # 對流項 (Rotational form): -ω×u = (ω * v, -ω * u)
            conv_U = omega * V
            conv_V = -omega * U

            # 轉回頻譜空間並去混疊
            if self.use_torch:
                conv_hat = self.fft2_batch(torch.stack([conv_U, conv_V], dim=0))
            else:
                conv_hat = self.fft2_batch(np.stack([conv_U, conv_V], axis=0))

            conv_hat = conv_hat * self.dealias_mask
            conv_U_hat, conv_V_hat = conv_hat[0], conv_hat[1]

        return conv_U_hat, conv_V_hat, omega_hat

    def _compute_nl_rhs(self, U_hat, V_hat):
        """
        What: 計算不含擴散的 RHS。
        Why: ETDRK4 會精確積分線性擴散項，只需要對流與強迫作為顯式 stage 驅動。
        """
        conv_U_hat, conv_V_hat, _ = self._compute_convection_hat(U_hat, V_hat)
        rhs_U_star = conv_U_hat + self.forcing_U_hat
        rhs_V_star = conv_V_hat + self.forcing_V_hat
        rhs_U, rhs_V = self._project_hat(rhs_U_star, rhs_V_star)
        return rhs_U, rhs_V

    def compute_rhs(self, U_hat, V_hat):
        """
        What: 計算包含擴散的完整 RHS。
        Why: 保留既有 RK4 路徑，確保舊流程仍可用於對照與回歸測試。
        """
        conv_U_hat, conv_V_hat, omega_hat = self._compute_convection_hat(U_hat, V_hat)

        _lin = self.nu * self.k2 + self.drag_alpha
        diff_U_hat = -_lin * U_hat
        diff_V_hat = -_lin * V_hat

        rhs_U_star = conv_U_hat + diff_U_hat + self.forcing_U_hat
        rhs_V_star = conv_V_hat + diff_V_hat + self.forcing_V_hat

        rhs_U, rhs_V = self._project_hat(rhs_U_star, rhs_V_star)

        return rhs_U, rhs_V, omega_hat

    def _setup_etd_coefficients(self):
        """
        What: 預計算 ETDRK4 所需係數。
        Why: 線性擴散項在頻譜空間是對角算子，可先把積分因子與 phi 函數係數算好，
             之後每步直接重用，降低高 Re 下的擴散 CFL 壓力。
        """
        if self.use_torch:
            L = -(self.nu * self.k2 + self.drag_alpha)
            Lh = L * self.dt
            Lh2 = Lh * 0.5

            def phi1(z):
                small = torch.abs(z) < 1e-6
                safe = torch.where(small, torch.ones_like(z), z)
                return torch.where(small, 1.0 + z / 2 + z ** 2 / 6, torch.expm1(safe) / safe)

            def phi2(z):
                small = torch.abs(z) < 1e-6
                safe = torch.where(small, torch.ones_like(z), z)
                return torch.where(small, 0.5 + z / 6 + z ** 2 / 24, (torch.expm1(safe) - safe) / safe ** 2)

            def phi3(z):
                small = torch.abs(z) < 1e-6
                safe = torch.where(small, torch.ones_like(z), z)
                return torch.where(
                    small,
                    1.0 / 6 + z / 24 + z ** 2 / 120,
                    (torch.expm1(safe) - safe - safe ** 2 / 2) / safe ** 3,
                )

            self.etd_E = torch.exp(Lh)
            self.etd_E2 = torch.exp(Lh2)
            self.etd_phi1_half = (self.dt / 2) * phi1(Lh2)
            self.etd_phi1_full = self.dt * phi1(Lh)
            p1, p2, p3 = phi1(Lh), phi2(Lh), phi3(Lh)
        else:
            L = -(self.nu * self.k2 + self.drag_alpha)
            Lh = L * self.dt
            Lh2 = Lh * 0.5

            def phi1(z):
                small = np.abs(z) < 1e-10
                safe = np.where(small, np.ones_like(z), z)
                return np.where(small, 1.0 + z / 2 + z ** 2 / 6, np.expm1(safe) / safe)

            def phi2(z):
                small = np.abs(z) < 1e-10
                safe = np.where(small, np.ones_like(z), z)
                return np.where(small, 0.5 + z / 6 + z ** 2 / 24, (np.expm1(safe) - safe) / safe ** 2)

            def phi3(z):
                small = np.abs(z) < 1e-10
                safe = np.where(small, np.ones_like(z), z)
                return np.where(
                    small,
                    1.0 / 6 + z / 24 + z ** 2 / 120,
                    (np.expm1(safe) - safe - safe ** 2 / 2) / safe ** 3,
                )

            self.etd_E = np.exp(Lh)
            self.etd_E2 = np.exp(Lh2)
            self.etd_phi1_half = (self.dt / 2) * phi1(Lh2)
            self.etd_phi1_full = self.dt * phi1(Lh)
            p1, p2, p3 = phi1(Lh), phi2(Lh), phi3(Lh)

        self.etd_coef_k1 = self.dt * (p1 - 3 * p2 + 4 * p3)
        self.etd_coef_k23 = self.dt * (2 * p2 - 4 * p3)
        self.etd_coef_k4 = self.dt * (-p2 + 4 * p3)

    def step_etdrk4(self):
        """
        What: 以 Cox-Matthews ETDRK4 積分一步。
        Why: 將線性擴散項精確積分，降低高 Re 下 dt 受擴散 CFL 限制的程度。
        """
        E = self.etd_E
        E2 = self.etd_E2
        U0, V0 = self.U_hat, self.V_hat

        k1_U, k1_V = self._compute_nl_rhs(U0, V0)

        a_U = E2 * U0 + self.etd_phi1_half * k1_U
        a_V = E2 * V0 + self.etd_phi1_half * k1_V
        k2_U, k2_V = self._compute_nl_rhs(a_U, a_V)

        b_U = E2 * U0 + self.etd_phi1_half * k2_U
        b_V = E2 * V0 + self.etd_phi1_half * k2_V
        k3_U, k3_V = self._compute_nl_rhs(b_U, b_V)

        c_U = E * U0 + self.etd_phi1_full * (2 * k3_U - k1_U)
        c_V = E * V0 + self.etd_phi1_full * (2 * k3_V - k1_V)
        k4_U, k4_V = self._compute_nl_rhs(c_U, c_V)

        self.U_hat = (
            E * U0
            + self.etd_coef_k1 * k1_U
            + self.etd_coef_k23 * (k2_U + k3_U)
            + self.etd_coef_k4 * k4_U
        )
        self.V_hat = (
            E * V0
            + self.etd_coef_k1 * k1_V
            + self.etd_coef_k23 * (k2_V + k3_V)
            + self.etd_coef_k4 * k4_V
        )

        if self.project_after_step:
            self.project_incompressible()

    def step_rk4(self):
        """RK4 時間步進"""
        U0, V0 = self.U_hat, self.V_hat

        # Stage 1
        k1_U, k1_V, _ = self.compute_rhs(U0, V0)

        # Stage 2
        k2_U, k2_V, _ = self.compute_rhs(
            U0 + 0.5 * self.dt * k1_U,
            V0 + 0.5 * self.dt * k1_V
        )

        # Stage 3
        k3_U, k3_V, _ = self.compute_rhs(
            U0 + 0.5 * self.dt * k2_U,
            V0 + 0.5 * self.dt * k2_V
        )

        # Stage 4
        k4_U, k4_V, _ = self.compute_rhs(
            U0 + self.dt * k3_U,
            V0 + self.dt * k3_V
        )

        # 更新
        self.U_hat = U0 + (self.dt / 6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        self.V_hat = V0 + (self.dt / 6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)

        # 額外保險：確保時間步進後的狀態也是嚴格無散度的
        if self.project_after_step:
            self.project_incompressible()

    def compute_pressure_from_state(self, U, V, omega=None, omega_hat=None):
        """計算壓力場 (用於輸出)，重用 omega/div 形式"""
        if self.use_torch and torch.is_tensor(U):
            if omega is None:
                if omega_hat is None:
                    omega_hat = getattr(self, 'last_omega_hat', None)
                    if omega_hat is None:
                        omega_hat = self.ikx * self.V_hat - self.iky * self.U_hat
                omega = torch.fft.ifft2(omega_hat).real

            omega_v = omega * V
            omega_u = omega * U
            speed2 = U**2 + V**2

            rhs_hat = -(self.ikx * torch.fft.fft2(omega_v) + self.iky * torch.fft.fft2(-omega_u))
            rhs_hat += 0.5 * self.k2 * torch.fft.fft2(speed2)

            P_hat = -rhs_hat / self.k2_safe
            P_hat[0, 0] = 0.0

            return torch.fft.ifft2(P_hat).real

        ikx = self.to_numpy(self.ikx) if self.use_torch else self.ikx
        iky = self.to_numpy(self.iky) if self.use_torch else self.iky
        k2 = self.to_numpy(self.k2) if self.use_torch else self.k2
        k2_safe = self.to_numpy(self.k2_safe) if self.use_torch else self.k2_safe

        if omega is None:
            if omega_hat is None:
                omega_hat = getattr(self, 'last_omega_hat', None)
                if omega_hat is None:
                    if self.use_torch:
                        omega_hat = ikx * self.to_numpy(self.V_hat) - iky * self.to_numpy(self.U_hat)
                    else:
                        omega_hat = ikx * self.V_hat - iky * self.U_hat
            else:
                omega_hat = self.to_numpy(omega_hat) if self.use_torch else omega_hat
            omega = np.fft.ifft2(omega_hat).real

        omega_v = omega * V
        omega_u = omega * U
        speed2 = U**2 + V**2

        rhs_hat = -(ikx * np.fft.fft2(omega_v) + iky * np.fft.fft2(-omega_u))
        rhs_hat += 0.5 * k2 * np.fft.fft2(speed2)

        P_hat = -rhs_hat / k2_safe
        P_hat[0, 0] = 0.0

        return np.fft.ifft2(P_hat).real

    def compute_pressure(self, U=None, V=None, omega=None, omega_hat=None):
        """計算壓力場 (用於輸出)"""
        if U is None or V is None:
            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)

        return self.compute_pressure_from_state(U, V, omega, omega_hat)

    def add_perturbation(self, method: str = 'random', amplitude: float = 0.1):
        """添加擾動"""
        if method == 'random':
            if self.use_torch:
                pert_U = amplitude * torch.randn_like(self.X)
                pert_V = amplitude * torch.randn_like(self.X)
            else:
                pert_U = amplitude * self.rng.standard_normal(self.X.shape)
                pert_V = amplitude * self.rng.standard_normal(self.X.shape)

            pert_U_hat = self.fft2(pert_U) * self.dealias_mask
            pert_V_hat = self.fft2(pert_V) * self.dealias_mask

        elif method == 'unstable_mode':
            # 不穩定模態（k_x=3, k_y=4）
            k_x, k_y = 3, self.k_f
            kx_phys = k_x * 2*np.pi / self.L
            ky_phys = k_y * 2*np.pi / self.L
            
            if self.use_torch:
                phase_U = amplitude * torch.cos(kx_phys * self.X + ky_phys * self.Y)
                phase_V = amplitude * torch.sin(kx_phys * self.X + ky_phys * self.Y)
            else:
                phase_U = amplitude * np.cos(kx_phys * self.X + ky_phys * self.Y)
                phase_V = amplitude * np.sin(kx_phys * self.X + ky_phys * self.Y)

            pert_U_hat = self.fft2(phase_U)
            pert_V_hat = self.fft2(phase_V)

        else:
            raise ValueError(f"Unknown perturbation method: {method}")

        self.U_hat += pert_U_hat
        self.V_hat += pert_V_hat

        # 擾動後立即投影，確保不可壓縮性
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)
        if self.enforce_zero_mean:
            self.U_hat, self.V_hat = self._zero_mean_hat(self.U_hat, self.V_hat)

        logging.info(f"✅ 添加擾動：method={method}, amplitude={amplitude}")

    def run(
        self,
        T_end: float = 20.0,
        save_interval: int = 100,
        output_file: str = "kolmogorov_dns.h5",
        perturbation_times: list = None,
        perturbation_method: str = 'random',
        perturbation_amplitude: float = None,
    ):
        """執行模擬"""
        n_steps = int(T_end / self.dt)
        
        # 預設擾動設定（避免落在數值噪聲）
        if perturbation_times is None:
            perturbation_times = []
            logging.info("ℹ️  未指定擾動時間，預設不注入擾動")

        # 自動設置擾動振幅（針對 Re=100 使用溫和擾動）
        if perturbation_amplitude is None:
            U0 = self.ifft2(self.U_hat)
            V0 = self.ifft2(self.V_hat)

            if self.use_torch:
                U0_val = U0.max().item()
                V0_val = V0.max().item()
            else:
                U0_val = U0.max()
                V0_val = V0.max()

            base_velocity = max(U0_val, V0_val, 0.1)  # 取 U/V 中較大者，最小 0.1
            
            # 關鍵調整：使用溫和擾動（0.5×），避免產生過大梯度導致散度誤差
            # 對於 Re=100 附近的臨界雷諾數，小擾動足以激發不穩定性
            perturbation_amplitude = min(0.8, 0.5 * base_velocity)  # 降低到 0.5× 且上限 0.8
            logging.info(f"📊 初始速度 U0={U0_val:.4f}, V0={V0_val:.4f}, 自動設置擾動振幅={perturbation_amplitude:.4f} (0.5×max(U0,V0), 上限0.8)")

        # 準備輸出 (NPY)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        n_snapshots = n_steps // save_interval + 1
        u_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float64)
        v_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float64)
        p_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float64)
        omega_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float64)
        time_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        ke_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        enstrophy_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        div_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        spectrum_snapshots = np.zeros((n_snapshots, len(self.spectrum_k)), dtype=np.float64)

        # 運行
        logging.info(f"▶️  開始 DNS 模擬：T_end={T_end}, n_steps={n_steps}")

        idx = 0
        t_start = time.time()
        for step in range(n_steps + 1):
            t = step * self.dt

            # 擾動
            if perturbation_times and any(abs(t - tp) < 0.5*self.dt for tp in perturbation_times):
                self.add_perturbation(perturbation_method, perturbation_amplitude)

            # 保存
            if step % save_interval == 0:
                U = self.ifft2(self.U_hat)
                V = self.ifft2(self.V_hat)

                # 計算診斷指標 (使用譜導數以獲得高精度)
                if self.use_torch:
                    omega, div_field, omega_hat = self.compute_omega_div(return_numpy=False)
                    P = self.compute_pressure_from_state(U, V, omega, omega_hat)

                    u_snapshots[idx] = self.to_numpy(U)
                    v_snapshots[idx] = self.to_numpy(V)
                    p_snapshots[idx] = self.to_numpy(P)
                    omega_snapshots[idx] = self.to_numpy(omega)

                    KE = 0.5 * torch.mean(U**2 + V**2).item()
                    enstrophy = 0.5 * torch.mean(omega**2).item()
                    div_err = torch.abs(div_field).max().item()
                    spectrum = self.compute_energy_spectrum(U, V)
                else:
                    omega, div_field, omega_hat = self.compute_omega_div()
                    P = self.compute_pressure_from_state(U, V, omega, omega_hat)

                    u_snapshots[idx] = U
                    v_snapshots[idx] = V
                    p_snapshots[idx] = P
                    omega_snapshots[idx] = omega

                    KE = 0.5 * np.mean(U**2 + V**2)
                    enstrophy = 0.5 * np.mean(omega**2)
                    div_err = np.abs(div_field).max()
                    spectrum = self.compute_energy_spectrum(U, V)

                time_snapshots[idx] = t

                ke_snapshots[idx] = KE
                enstrophy_snapshots[idx] = enstrophy
                div_snapshots[idx] = div_err
                spectrum_snapshots[idx] = spectrum

                if step % (save_interval * 10) == 0:
                    elapsed = time.time() - t_start
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_steps - step) / speed if speed > 0 else 0
                    logging.info(
                        f"  Step {step:6d}/{n_steps} | t={t:.3f} | "
                        f"KE={KE:.4e} | Enstrophy={enstrophy:.4e} | "
                        f"Div_err={div_err:.4e} | Speed={speed:.1f} steps/s | ETA={eta:.0f}s"
                    )

                idx += 1

            # 步進
            if step < n_steps:
                if self.integrator == 'etdrk4':
                    self.step_etdrk4()
                else:
                    self.step_rk4()

        output_data = {
            'x': np.linspace(0, self.L, self.N, endpoint=False),
            'y': np.linspace(0, self.L, self.N, endpoint=False),
            'u': u_snapshots,
            'v': v_snapshots,
            'p': p_snapshots,
            'omega': omega_snapshots,
            'time': time_snapshots,
            'diagnostics': {
                'kinetic_energy': ke_snapshots,
                'enstrophy': enstrophy_snapshots,
                'divergence_error': div_snapshots,
                'energy_spectrum': spectrum_snapshots,
                'spectrum_wavenumbers': self.spectrum_k.astype(np.float64),
            },
            'config': {
                'N': self.N,
                'L': self.L,
                'nu': self.nu,
                'A': self.A,
                'k_f': self.k_f,
                'dt': self.dt,
                'T_end': T_end,
                'dealias_mode': self.dealias_mode,
                'backend': self.backend,
                'device': str(self.device) if self.device else 'numpy',
                'integrator': self.integrator,
                'drag_alpha': self.drag_alpha,
                'save_interval': save_interval,
                'perturbation_times': perturbation_times,
                'perturbation_method': perturbation_method,
                'perturbation_amplitude': perturbation_amplitude,
                'seed': self.seed,
                'enforce_zero_mean': self.enforce_zero_mean,
                'ic_mode': self.ic_mode,
                'ic_k_cutoff': self.ic_k_cutoff,
                'init_mode': self.ic_mode,  # legacy alias for backward compat
                'init_k_cutoff': self.init_k_cutoff,
                'target_initial_ke': self.target_initial_ke,
                'target_initial_forcing_coeff': self.target_initial_forcing_coeff,
                'target_initial_spectrum_bins': self.target_initial_spectrum_bins,
                'target_initial_spectrum': self.low_k_target_spectrum[: self.target_initial_spectrum_bins].astype(np.float64),
                'initial_ke_achieved': self.initial_ke,
                'initial_forcing_coeff_achieved': self.initial_forcing_coeff,
                'initial_enstrophy_achieved': self.initial_enstrophy,
                'initial_spectrum_achieved': self.initial_spectrum_achieved.astype(np.float64),
            }
        }

        np.save(output_file, output_data, allow_pickle=True)

        total_time = time.time() - t_start
        logging.info(f"✅ 模擬完成！總耗時：{total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        logging.info(f"   平均速度：{n_steps/total_time:.1f} steps/s")
        logging.info(f"💾 結果已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow DNS 生成器 (自動選擇 GPU/CPU)')
    parser.add_argument('--N', type=int, default=1024, help='網格點數')
    parser.add_argument('--L', type=float, default=1.0, help='域長度（文獻對齊預設為 [0,1]）')
    parser.add_argument('--nu', type=float, default=1e-5, help='黏滯係數')
    parser.add_argument('--A', type=float, default=0.1, help='強迫振幅')
    parser.add_argument('--k_f', type=int, default=2, help='強迫波數（L=1 時對應 A*sin(4πy)）')
    parser.add_argument('--dt', type=float, default=0.001, help='時間步長')
    parser.add_argument('--T_end', type=float, default=20.0, help='結束時間')
    parser.add_argument('--save_interval', type=int, default=100, help='保存間隔')
    parser.add_argument('--output', type=str, default='data/kolmogorov_dns.npy', help='輸出文件 (.npy)')
    parser.add_argument('--perturbation_times', type=float, nargs='+', help='擾動時刻')
    parser.add_argument('--perturbation_method', type=str, default='random', 
                       choices=['random', 'unstable_mode'], help='擾動方法')
    parser.add_argument('--perturbation_amplitude', type=float, default=None, help='擾動振幅（默認自動推斷，Re~30-100 建議 1.0-1.5）')
    parser.add_argument('--backend', type=str, default=None,
                       choices=['torch-mps', 'torch-cuda', 'torch-cpu', 'numpy'],
                       help='手動指定後端（默認自動檢測）')
    parser.add_argument('--dealias-mode', type=str, default='3/2',
                       choices=['2/3', '3/2'],
                       help='去混疊方式')
    parser.add_argument('--no-project-after-step', action='store_true',
                       help='關閉時間步進後的額外投影')
    parser.add_argument('--seed', type=int, default=42, help='隨機初始化與隨機擾動的亂數種子')
    parser.add_argument('--drag-alpha', type=float, default=0.0,
                        help='線性 (Ekman) 阻尼係數 alpha；0 = 既有行為。'
                             '無阻尼時 2D 逆級聯會在最低模態堆成 condensate，'
                             '評估窗無法達到統計穩態')
    parser.add_argument('--no-enforce-zero-mean', action='store_true',
                       help='不要額外強制速度場的空間平均為 0')
    parser.add_argument('--target-initial-ke', type=float, default=0.16145706176757812,
                       help='目標初始 kinetic energy（預設對齊 JAXPI transient）')
    parser.add_argument('--target-initial-forcing-coeff', type=float, default=0.04649721458554268,
                       help='目標初始 u·sin(4πy) forcing mode coefficient（預設對齊 JAXPI transient）')
    parser.add_argument('--target-initial-spectrum-bins', type=int, default=8,
                       help='對齊初始低波數能譜的 shell 數量（預設 k=1~8）')
    parser.add_argument('--integrator', type=str, default='rk4',
                       choices=['rk4', 'etdrk4'],
                       help='時間積分方案（預設保持 rk4，相容舊流程）')
    parser.add_argument('--ic_mode', type=str, default='band_limited_random',
                       choices=['band_limited_random', 'spectral_seeded'],
                       help='IC 生成模式（spectral_seeded 用於 Grid Independence Test, 保 N-invariance）')
    parser.add_argument('--ic_k_cutoff', type=float, default=None,
                       help='spectral_seeded 的 |k| cutoff (default: max(8.0, 4*k_f))')

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("Kolmogorov Flow DNS 生成器 (統一版本)")
    logging.info("=" * 80)

    solver = KolmogorovFlowDNS(
        N=args.N,
        L=args.L,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
        dealias_mode=args.dealias_mode,
        backend=args.backend,
        project_after_step=not args.no_project_after_step,
        enforce_zero_mean=not args.no_enforce_zero_mean,
        seed=args.seed,
        target_initial_ke=args.target_initial_ke,
        target_initial_forcing_coeff=args.target_initial_forcing_coeff,
        target_initial_spectrum_bins=args.target_initial_spectrum_bins,
        integrator=args.integrator,
        ic_mode=args.ic_mode,
        ic_k_cutoff=args.ic_k_cutoff,
        drag_alpha=args.drag_alpha,
    )

    solver.run(
        T_end=args.T_end,
        save_interval=args.save_interval,
        output_file=args.output,
        perturbation_times=args.perturbation_times,
        perturbation_method=args.perturbation_method,
        perturbation_amplitude=args.perturbation_amplitude,
    )


if __name__ == '__main__':
    main()
