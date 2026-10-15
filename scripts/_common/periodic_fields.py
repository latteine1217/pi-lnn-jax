"""週期方形域上逐幀的場運算：由速度解壓力、二階中央差分梯度。

陣列慣例一律是 [..., N, N]，最後兩軸依序為 x、y（本專案 DNS/LES 的 [T, x, y]）。
`eval_pressure_limit.py`（壓力評估）與 `gen_sensors_qrpivot_dns.py`（oracle 佈點特徵）
共用這裡的實作，確保兩處對「壓力」與「導數」的定義是同一個。
"""
from __future__ import annotations

import functools

import numpy as np


def reference_pressure(u: np.ndarray, v: np.ndarray, L: float) -> np.ndarray:
    """週期域不可壓 NS 的壓力：∇²p = −∂i∂j(u_i u_j)，空間均值為 0。

    Fourier 空間 p̂ = −k_i k_j F(u_i u_j) / |k|²，k = 2πn/L 為角波數，k = 0 模態設 0。
    形狀 [..., N, N]（最後兩軸 = x, y），故 [T, N, N] 逐幀一次算完。

    只有非線性項進源項：`physics.py` 的動量式其餘各項——∂u/∂t、ν∇²u、
    Kolmogorov forcing f_x = A sin(2πk_f y)、以及 opt-in 的線性阻尼 α·u——
    散度皆為 0（前兩者與阻尼項對無散度速度成立，forcing 只依賴 y 且只有 x 分量），
    取散度後不貢獻。

    Nyquist 列上的交叉項 k_x k_y 不滿足實場的 Hermitian 對稱；取 `.real` 等同把
    該處的奇數階導數設為 0（標準作法），對已解析的場影響可忽略。
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim < 2 or u.shape[-1] != u.shape[-2]:
        raise ValueError(f"u, v 須為同形狀的 [..., N, N]，收到 {u.shape} / {v.shape}")
    if not L > 0:
        raise ValueError(f"域長 L 須為正，收到 {L}")
    N = u.shape[-1]
    k = 2.0 * np.pi / L * np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = k[:, None], k[None, :]
    k2 = kx ** 2 + ky ** 2
    fft2 = functools.partial(np.fft.fft2, axes=(-2, -1))
    num = -(kx * kx * fft2(u * u) + 2.0 * kx * ky * fft2(u * v) + ky * ky * fft2(v * v))
    p_hat = np.zeros_like(num)
    nonzero = k2 > 0
    p_hat[..., nonzero] = num[..., nonzero] / k2[nonzero]
    return np.fft.ifft2(p_hat, axes=(-2, -1)).real


def periodic_gradient(f: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """週期二階中央差分 (∂f/∂x, ∂f/∂y)；axis −2 = x、axis −1 = y，dx = L/N。

    與 `metric_artifact.compute_vorticity(periodic=True)` 同一個模板，pred 與 ref
    走同一個算子，截斷誤差在相對誤差裡是共模的。
    """
    if f.ndim < 2 or f.shape[-1] != f.shape[-2]:
        raise ValueError(f"須為 [..., N, N]，收到 {f.shape}")
    h = L / f.shape[-1]
    fx = (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2.0 * h)
    fy = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * h)
    return fx, fy
