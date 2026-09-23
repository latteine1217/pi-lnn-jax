"""GappyDMD 的 isolated 單元測試（純 numpy）。

這個 baseline 存在的理由是「只換基底」的對照：與 GappyPOD 共用同一套
sensor 最小平方投影與同一套秩選擇，唯一差別是基底從 POD（能量最優的靜態
子空間）換成 DMD（帶特徵值的動態模態）。若兩者結果相近，代表在該 K 與
時窗下動態資訊對重建沒有額外貢獻——那是關於問題本身的判讀，不是關於
實作的判讀，所以「共用」必須是結構上的而非口頭上的。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.baselines import GappyDMD, GappyPOD


def _linear_flow(N=8, T=40, r=3, seed=0):
    """由固定線性算子生成的低秩軌跡：x_{k+1} = M x_k。

    DMD 的模型假設在此完全成立，故重建應該非常準——這是能證偽實作的最強
    測資（若 DMD 基底算錯，即使在完美線性系統上也回復不了）。
    """
    rng = np.random.default_rng(seed)
    D = 2 * N * N
    basis = np.linalg.qr(rng.normal(size=(D, r)))[0]        # [D, r] 正交
    lam = np.array([0.98, 0.95, 0.9])[:r]                    # 穩定實特徵值
    coeff = np.zeros((r, T))
    coeff[:, 0] = rng.normal(size=r) + 2.0
    for k in range(1, T):
        coeff[:, k] = lam * coeff[:, k - 1]
    A = basis @ coeff                                        # [D, T]
    u = A[: N * N].reshape(N, N, T).transpose(2, 0, 1)
    v = A[N * N:].reshape(N, N, T).transpose(2, 0, 1)
    return u, v


def _sensors(u, v, K=24, seed=1):
    rng = np.random.default_rng(seed)
    N = u.shape[1]
    idx = rng.choice(N * N, size=K, replace=False)
    ix, iy = idx // N, idx % N
    pos = np.stack([ix / N, iy / N], axis=1)
    vals = np.stack([u[:, ix, iy], v[:, ix, iy]], axis=-1)   # [T, K, 2]
    return vals, pos


def test_recovers_linear_flow():
    """線性系統上 DMD 假設成立 → 重建誤差應遠小於場的量級。"""
    u, v = _linear_flow()
    vals, pos = _sensors(u, v)
    m = GappyDMD(n_modes=3).fit(u, v)
    up, vp = m.reconstruct(vals, pos)

    err = np.linalg.norm(up - u) / np.linalg.norm(u)
    assert err < 0.05, f"線性流上的相對誤差 {err:.3f} 過大，DMD 基底可能算錯"


def test_shares_rank_selection_with_pod():
    """同一個 n_modes 下兩者的基底秩相同——秩選擇不可成為兩法差異的來源。"""
    u, v = _linear_flow()
    d = GappyDMD(n_modes=3).fit(u, v)
    p = GappyPOD(n_modes=3).fit(u, v)
    assert d.modes.shape[1] == p.modes.shape[1] == 3


def test_basis_is_real():
    """DMD 模態本質是複數；投影基底必須是實的，否則重建會帶虛部。"""
    u, v = _linear_flow()
    m = GappyDMD(n_modes=3).fit(u, v)
    assert np.isrealobj(m.modes), "基底含虛部——共軛對未正確轉為實子空間"


def test_conjugate_pairs_expand_to_real_span():
    """含振盪（複共軛特徵值）時，一對共軛模態張成二維實子空間。

    取 Re 與 Im 兩個實向量代表該對，才不會把旋轉分量丟掉。
    """
    rng = np.random.default_rng(2)
    N, T = 8, 40
    D = 2 * N * N
    B = np.linalg.qr(rng.normal(size=(D, 2)))[0]
    th = 0.3
    coeff = np.zeros((2, T))
    coeff[:, 0] = [1.0, 0.0]
    R = 0.97 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for k in range(1, T):
        coeff[:, k] = R @ coeff[:, k - 1]
    A = B @ coeff
    u = A[: N * N].reshape(N, N, T).transpose(2, 0, 1)
    v = A[N * N:].reshape(N, N, T).transpose(2, 0, 1)

    m = GappyDMD(n_modes=2).fit(u, v)
    vals, pos = _sensors(u, v)
    up, _ = m.reconstruct(vals, pos)
    err = np.linalg.norm(up - u) / np.linalg.norm(u)
    assert err < 0.10, f"振盪模態重建誤差 {err:.3f}；共軛對可能被丟掉一半"


def test_reconstruct_shape_contract():
    """與 GappyPOD 同一個回傳契約：(u_pred, v_pred) 皆 [T, N, N]。"""
    u, v = _linear_flow()
    vals, pos = _sensors(u, v)
    up, vp = GappyDMD(n_modes=3).fit(u, v).reconstruct(vals, pos)
    assert up.shape == u.shape and vp.shape == v.shape


def test_fit_requires_two_snapshots():
    """DMD 需要 snapshot pair；單張快照無法定義動態，硬失敗而非退化成 POD。"""
    u, v = _linear_flow(T=1)
    with pytest.raises(ValueError):
        GappyDMD(n_modes=2).fit(u, v)
