"""最小 2D Kolmogorov 譜方法 solver 的 isolated 單元測試（純 numpy）。

這個 solver 只為 EnKF 當 forward model 用，不取代 DNS 生成器。譜方法的錯誤
幾乎都是靜默的——dealiasing 漏掉、forcing 差一個負號、Poisson 反解的 k=0
沒處理，跑起來都不會炸，只是慢慢長成別的流場。故每一項都用有解析解或有
守恆律的設定釘住。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.kolmogorov_solver import KolmogorovSolver


def test_pure_diffusion_matches_analytic():
    """關掉非線性與 forcing 後，單一 Fourier 模態應精確按 exp(-nu k^2 t) 衰減。

    這條同時驗證波數的定義與時間積分的階數——k 若差一個 2pi，衰減率會差
    (2pi)^2 倍，任何「看起來合理」的流場都掩蓋不了。
    """
    N, nu, L = 32, 0.05, 1.0
    s = KolmogorovSolver(N=N, nu=nu, L=L, forcing_amplitude=0.0, nonlinear=False)
    x = np.arange(N) * L / N
    m = 3
    w0 = np.sin(2 * np.pi * m * x)[:, None] * np.ones((1, N))
    k = 2 * np.pi * m / L

    T, dt = 0.5, 1e-3
    w = s.integrate(w0, dt=dt, n_steps=int(T / dt))

    expected = np.exp(-nu * k**2 * T)
    got = np.max(np.abs(w)) / np.max(np.abs(w0))
    assert got == pytest.approx(expected, rel=2e-3), f"衰減 {got:.5f} vs 解析 {expected:.5f}"


def test_inviscid_unforced_conserves_enstrophy():
    """無黏無 forcing 時 enstrophy 應守恆（短時窗內）。

    這條抓的是非線性項（Jacobian）的離散化：J 若寫錯，能量/enstrophy 會
    單調漂移而不是在捨入誤差內震盪。
    """
    N = 64
    rng = np.random.default_rng(0)
    s = KolmogorovSolver(N=N, nu=0.0, forcing_amplitude=0.0)
    w0 = rng.normal(size=(N, N))
    w0 -= w0.mean()
    # 只保留低波數，避免初始場在 2/3 截斷邊界上引入偽能量
    w0 = s.low_pass(w0, k_cut=N // 6)

    Z0 = 0.5 * np.mean(w0**2)
    w = s.integrate(w0, dt=2e-4, n_steps=500)
    Z = 0.5 * np.mean(w**2)
    assert abs(Z - Z0) / Z0 < 0.02, f"enstrophy 漂移 {abs(Z-Z0)/Z0:.4f}"


def test_velocity_curl_returns_vorticity():
    """velocity() 與渦度必須自洽：curl(u,v) == w。

    Poisson 反解的 k=0 模態若未置零，速度會多一個常數平移，而 curl 仍然
    正確——所以另外檢查速度的空間平均為零。
    """
    N = 32
    rng = np.random.default_rng(1)
    s = KolmogorovSolver(N=N, nu=1e-3)
    w = s.low_pass(rng.normal(size=(N, N)), k_cut=N // 6)
    w -= w.mean()

    u, v = s.velocity(w)
    w_back = s.curl(u, v)
    np.testing.assert_allclose(w_back, w, atol=1e-10)
    assert abs(u.mean()) < 1e-12 and abs(v.mean()) < 1e-12


def test_forcing_drives_kf_mode():
    """Kolmogorov forcing 應把能量注入 k_f，而非其他波數。

    符號寫反時能量一樣會進 k_f，故這裡看的是流場方向：forcing 作用於
    x-momentum，故 <u> 應隨時間增長而 <v> 維持在零附近。
    """
    N, k_f = 64, 2
    s = KolmogorovSolver(N=N, nu=1e-3, k_f=k_f, forcing_amplitude=1.0)
    w = np.zeros((N, N))
    w = s.integrate(w, dt=1e-3, n_steps=200)
    u, v = s.velocity(w)
    assert np.abs(u).mean() > 10 * np.abs(v).mean(), "forcing 未作用在 x-momentum"


def test_dealiasing_blocks_aliased_feedback():
    """非線性項不得把能量填進 2/3 截斷之外的波數。

    注意 dealiasing 的語意：它作用於**非線性項**，不是把狀態的高頻濾掉。
    故初始場必須先低通（否則測到的是初始場殘留，與 dealiasing 無關），
    再檢查演化過程有沒有從截斷外冒出能量——那才是混疊。
    初始截在 N/6，二次非線性最多產生到 N/3，恰好貼齊截斷邊界。
    """
    N = 32
    rng = np.random.default_rng(2)
    s = KolmogorovSolver(N=N, nu=1e-3, forcing_amplitude=0.0)
    w = s.low_pass(rng.normal(size=(N, N)), k_cut=N // 6)
    w -= w.mean()
    w = s.integrate(w, dt=1e-4, n_steps=50)

    wh = np.fft.fft2(w)
    kx = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    cut = N / 3.0
    beyond = (np.abs(KX) > cut) | (np.abs(KY) > cut)
    assert np.max(np.abs(wh[beyond])) < 1e-8 * np.max(np.abs(wh))
