"""forcing_mode_coeff_u 的 isolated 單元測試（純 numpy，不載 model/DNS）。

驗證 Kolmogorov forcing mode 的複數 Fourier 係數投影。這個量與既有的
`kf_amp_ratio`（E_pred(k_f)/E_dns(k_f)，k-shell 能量比）**不是同一件事**：
本函式回傳的是 x-平均後 u(y) 在 forcing basis 上的複數係數，因此除振幅外
還帶相位——相位是能譜比看不到的診斷（能譜丟掉相位資訊）。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.evaluate import forcing_mode_coeff_u


def _grid(n: int, length: float = 1.0) -> np.ndarray:
    """單位胞週期格點（endpoint=False，與 DNS 場的 x/y 格一致）。"""
    return np.linspace(0.0, length, n, endpoint=False)


def test_pure_sine_amplitude_is_half():
    """u = A·sin(2π k_f y / L) → |coeff| = A/2，相位 = -π/2。

    sin 展開成 (e^{iθ}-e^{-iθ})/(2i)，只有 e^{iθ} 項與 basis e^{-iθ} 同調，
    故係數為 A/(2i) = -iA/2。
    """
    y = _grid(64)
    A, kf = 3.0, 4.0
    u = np.broadcast_to(A * np.sin(2 * np.pi * kf * y), (32, 64))

    amp, phase = forcing_mode_coeff_u(u, y, kf)

    assert amp == pytest.approx(A / 2)
    assert phase == pytest.approx(-np.pi / 2)


def test_pure_cosine_is_zero_phase():
    """u = A·cos(...) → 同樣 |coeff| = A/2，但相位 = 0。"""
    y = _grid(64)
    A, kf = 2.0, 4.0
    u = np.broadcast_to(A * np.cos(2 * np.pi * kf * y), (32, 64))

    amp, phase = forcing_mode_coeff_u(u, y, kf)

    assert amp == pytest.approx(A / 2)
    assert phase == pytest.approx(0.0)


def test_phase_shift_is_tracked():
    """輸入平移 φ → 回傳相位跟著平移，這是相位診斷的全部意義。"""
    y = _grid(64)
    kf, shift = 4.0, 0.7
    u = np.broadcast_to(np.cos(2 * np.pi * kf * y + shift), (32, 64))

    _, phase = forcing_mode_coeff_u(u, y, kf)

    assert phase == pytest.approx(shift)


def test_orthogonal_mode_gives_zero():
    """非 forcing 波數的成分投影為零（basis 正交）。"""
    y = _grid(64)
    u = np.broadcast_to(np.sin(2 * np.pi * 7.0 * y), (32, 64))

    amp, _ = forcing_mode_coeff_u(u, y, k_forcing=4.0)

    assert amp == pytest.approx(0.0, abs=1e-12)


def test_x_variation_is_averaged_out():
    """x 方向的成分被 x-平均消掉，只剩 u(y) 的 forcing 投影。

    u[x,y] 慣例（與 DNS 場 u[t,x,y] 一致）：axis 0 是 x。
    """
    x, y = _grid(32), _grid(64)
    kf = 4.0
    base = np.sin(2 * np.pi * kf * y)[None, :]
    contamination = np.sin(2 * np.pi * 3.0 * x)[:, None]  # x-平均為 0
    u = np.broadcast_to(base, (32, 64)) + contamination

    amp, phase = forcing_mode_coeff_u(u, y, kf)

    assert amp == pytest.approx(0.5)
    assert phase == pytest.approx(-np.pi / 2)


def test_domain_length_changes_basis_wavelength():
    """basis 波長必須跟著 domain_length 走，否則非單位域會 silent 偏。

    L=2 的域上，k_f=4 的模態波長是 L/k_f=0.5。若函式內部寫死 domain=1，
    投影 basis 會對不上，振幅塌掉——這正是要擋的失效模式。
    """
    L, kf, A = 2.0, 4.0, 3.0
    y = _grid(64, length=L)
    u = np.broadcast_to(A * np.sin(2 * np.pi * kf * y / L), (32, 64))

    amp, phase = forcing_mode_coeff_u(u, y, kf, domain_length=L)

    assert amp == pytest.approx(A / 2)
    assert phase == pytest.approx(-np.pi / 2)

    # 用錯的 domain_length 投影 → 振幅顯著塌掉（證明這個參數是承重的）
    wrong_amp, _ = forcing_mode_coeff_u(u, y, kf, domain_length=1.0)
    assert wrong_amp < 0.1 * amp
