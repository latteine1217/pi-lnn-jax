"""`eval_pressure_limit.reference_pressure` 的符號、尺度與物理一致性。

Why 兩層測試:
    Taylor–Green 解析解釘住符號與尺度（任意域長 L 下都要對，k 是角波數 2πn/L）；
    但解析解是穩態、無黏、無 forcing，證明不了「用在主線 DNS 上它就是動量方程裡的
    那個 p」。第二層直接拿本機 DNS 的相鄰幀估 ∂u/∂t，對動量方程的壓力項做最小
    平方擬合：正確的壓力係數必須是 +1。

    同一個擬合也釘住「DNS 檔的 `p` 欄位不可用」這個前提（生成器把 Π = p + ½|u|²
    的符號寫反）。哪天 DNS 重新生成、存檔 `p` 變成自洽的，那條測試會紅——那時
    `eval_pressure_limit` 不讀 `p` 的理由就要重新檢查，而不是默默繼續成立。
"""
from __future__ import annotations

import numpy as np
import pytest
from _paths import REPO_ROOT

from scripts.eval_pressure_limit import reference_pressure

DNS_PATH = REPO_ROOT / "data" / "dns" / "kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
#: 取三個不同時段的幀（初期、中段、後段），擬合結論不靠單一幀。
FRAMES = (50, 100, 150)


@pytest.mark.parametrize("L", [1.0, 2.0 * np.pi])
def test_taylor_green_sign_and_scale(L):
    """u = sin(ax)cos(ay), v = −cos(ax)sin(ay), a = 2π/L ⇒ p = (cos 2ax + cos 2ay)/4 + C。

    解析解的空間均值恰為 0，與函式的 k=0 → 0 慣例相同，故 C = 0 直接比。
    """
    N = 64
    a = 2.0 * np.pi / L
    x = np.arange(N) * (L / N)
    X, Y = np.meshgrid(x, x, indexing="ij")          # axis 0 = x, axis 1 = y
    u = np.sin(a * X) * np.cos(a * Y)
    v = -np.cos(a * X) * np.sin(a * Y)
    p_exact = (np.cos(2 * a * X) + np.cos(2 * a * Y)) / 4.0

    p = reference_pressure(u, v, L)
    assert np.linalg.norm(p - p_exact) / np.linalg.norm(p_exact) < 1e-10

    # [T, N, N] 逐幀批次與單幀結果相同（評估腳本以單幀呼叫，測試兩種形狀都成立）
    stacked = reference_pressure(np.stack([u, 2 * u]), np.stack([v, 2 * v]), L)
    assert np.allclose(stacked[0], p, rtol=0, atol=1e-14)
    assert np.allclose(stacked[1], 4 * p, rtol=0, atol=1e-13)   # p 對速度是二次


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        reference_pressure(np.zeros((8, 8)), np.zeros((8, 4)), 1.0)
    with pytest.raises(ValueError):
        reference_pressure(np.zeros((8, 8)), np.zeros((8, 8)), 0.0)


# ── 主線 DNS 上的動量方程一致性 ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def dns():
    if not DNS_PATH.exists():
        pytest.skip(f"本機無 DNS 檔：{DNS_PATH}")
    return np.load(DNS_PATH, allow_pickle=True).item()


def _pressure_fit(dns: dict, n: int, p: np.ndarray) -> tuple[float, float]:
    """動量殘差 R = ∂u/∂t + (u·∇)u − ν∇²u − f 應等於 −∇p。

    回傳 (c, ratio)：c 為 min‖R + c∇p‖ 的最小平方係數（正確壓力 ⇒ c = +1），
    ratio = ‖R + c∇p‖ / ‖R‖。空間導數用譜方法（與被測函式的 Poisson 解無關的另一條
    計算），∂u/∂t 用相鄰存檔幀的中央差分——存檔間隔 0.025 的截斷誤差是殘差地板的
    主要來源，所以 ratio 不會趨近 0。
    """
    cfg = dns["config"]
    L, nu, A, k_f = float(cfg["L"]), float(cfg["nu"]), float(cfg["A"]), float(cfg["k_f"])
    u_all, v_all, t = dns["u"], dns["v"], np.asarray(dns["time"], dtype=np.float64)
    dt = t[n + 1] - t[n - 1]
    u, v = np.asarray(u_all[n], dtype=np.float64), np.asarray(v_all[n], dtype=np.float64)
    N = u.shape[-1]
    k = 2.0 * np.pi / L * np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = k[:, None], k[None, :]

    def d(f, kk):
        return np.fft.ifft2(1j * kk * np.fft.fft2(f)).real

    def lap(f):
        return np.fft.ifft2(-(kx ** 2 + ky ** 2) * np.fft.fft2(f)).real

    y = np.arange(N) * (L / N)
    f_x = A * np.sin(2.0 * np.pi * k_f * y)[None, :]   # forcing 沿 axis 1 (y)，同 physics.py
    ru = (u_all[n + 1] - u_all[n - 1]) / dt + u * d(u, kx) + v * d(u, ky) - nu * lap(u) - f_x
    rv = (v_all[n + 1] - v_all[n - 1]) / dt + u * d(v, kx) + v * d(v, ky) - nu * lap(v)
    gx, gy = d(p, kx), d(p, ky)
    c = -(np.sum(ru * gx) + np.sum(rv * gy)) / (np.sum(gx ** 2) + np.sum(gy ** 2))
    ratio = np.sqrt(np.sum((ru + c * gx) ** 2 + (rv + c * gy) ** 2) / np.sum(ru ** 2 + rv ** 2))
    return float(c), float(ratio)


@pytest.mark.parametrize("n", FRAMES)
def test_poisson_pressure_closes_the_dns_momentum_equation(dns, n):
    """實測（本機 DNS）：c = 1.0000，ratio 0.08–0.13。"""
    L = float(dns["config"]["L"])
    p = reference_pressure(dns["u"][n], dns["v"][n], L)
    c, ratio = _pressure_fit(dns, n, p)
    assert abs(c - 1.0) < 0.01, f"frame {n}: 壓力係數 {c:.4f}，應為 +1（符號或尺度錯）"
    assert ratio < 0.2, f"frame {n}: 殘差只降到 {ratio:.3f}，Poisson 壓力沒有閉合動量方程"


@pytest.mark.parametrize("n", FRAMES)
def test_stored_dns_pressure_is_not_the_momentum_pressure(dns, n):
    """存檔 `p` 的擬合係數實測 −0.10～−0.13、與正確壓力的相對差約 2。

    這條紅了代表 DNS 已重新生成且 `p` 自洽——屆時重新評估 eval_pressure_limit
    的「不讀 p」設計，而不是直接刪掉這條測試。
    """
    L = float(dns["config"]["L"])
    c, _ = _pressure_fit(dns, n, np.asarray(dns["p"][n], dtype=np.float64))
    assert c < 0.5, f"frame {n}: 存檔 p 的係數 {c:.3f} 已接近 +1——DNS 的 p 可能已修正"
    p_ref = reference_pressure(dns["u"][n], dns["v"][n], L)
    p_st = dns["p"][n] - dns["p"][n].mean()
    assert np.linalg.norm(p_st - p_ref) / np.linalg.norm(p_ref) > 1.0
