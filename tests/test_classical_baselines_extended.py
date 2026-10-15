"""`classical_baselines_fair.py --extended` 的三個變體與感測器交叉驗證。

這三個變體回應審稿人「古典基線沒有調到可得資訊」：圓盤 div-free LSQ、週期 RBF、
週期 IDW，超參數由 leave-sensors-out CV 選定。CV 若偷讀參考場、或選擇邏輯寫反，
表格只會安靜地多出看似合理的一列，所以釘住選擇在已知答案的場上選對、週期性真的成立。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.classical_baselines_fair import (
    _divfree_modes,
    _divfree_modes_disk,
    _periodic_images,
    divfree_trig_lsq,
    idw_periodic,
    select_by_sensor_cv,
    sensor_cv_error,
)


def _divfree_field(xy, k_list, seed=0):
    """單位週期方域上的已知無散度場：ψ = Σ a_k sin(2π k·x)，u = ∂ψ/∂y，v = −∂ψ/∂x。"""
    rng = np.random.default_rng(seed)
    u = np.zeros(len(xy)); v = np.zeros(len(xy))
    for (m, n) in k_list:
        a = rng.normal()
        th = 2 * np.pi * (m * xy[:, 0] + n * xy[:, 1])
        u += a * 2 * np.pi * n * np.cos(th)
        v -= a * 2 * np.pi * m * np.cos(th)
    return u, v


def test_disk_modes_are_the_half_plane_inside_the_radius():
    modes = _divfree_modes_disk(5)
    assert all(m * m + n * n <= 25 for m, n in modes)
    # 圓盤是方形的子集，且軸向 (5,0)、(0,5) 在內、角落 (5,5) 不在內
    assert set(modes) <= set(_divfree_modes(5))
    assert (5, 0) in modes and (0, 5) in modes and (5, 5) not in modes
    # 共軛不重複：(m,n) 與 (−m,−n) 不同時出現
    assert not any((-m, -n) in modes for m, n in modes)


def test_periodic_images_tile_nine_copies():
    pos = np.array([[0.1, 0.9]])
    tp, tv = _periodic_images(pos, np.array([3.0]))
    assert tp.shape == (9, 2) and np.all(tv == 3.0)
    assert any(np.allclose(p, [1.1, -0.1]) for p in tp)


def test_idw_periodic_uses_minimum_image_distance():
    """邊界兩側的點在週期域上是近鄰：查詢 x=0.99 應由 x=0.01 的感測器主導。"""
    pos = np.array([[0.01, 0.5], [0.5, 0.5]])
    vals = np.array([1.0, 0.0])
    out = idw_periodic(pos, vals, np.array([[0.99, 0.5]]), p=2.0)
    assert out[0] > 0.99


def test_sensor_cv_selects_radius_that_contains_the_field():
    """場只含 |k| ≤ 3 的模態：CV 應選 kc ≥ 3，且在 kc=2 時誤差明顯較大。"""
    rng = np.random.default_rng(1)
    K, T = 120, 3
    pos = rng.uniform(size=(K, 2))
    k_list = [(1, 0), (0, 2), (1, 2), (2, 2), (3, 0)]
    phys = np.stack([np.stack(_divfree_field(pos, k_list, seed=t), axis=1) for t in range(T)])

    def make(kc):
        m = _divfree_modes_disk(kc)
        return lambda ptr, utr, vtr, pte: divfree_trig_lsq(ptr, utr, vtr, pte, modes=m)

    best, curve = select_by_sensor_cv(make, [2, 3, 4], pos, phys, frames=range(T), folds=6)
    assert best >= 3
    assert curve[2] > 10 * curve[best]
    assert curve[3] < 1e-8


def test_sensor_cv_fits_on_training_sensors_only():
    """每折 fit 收到的感測器數 = K − 留出數，且誤差以留出點的真值為分母。"""
    rng = np.random.default_rng(2)
    K = 20
    pos = rng.uniform(size=(K, 2))
    phys = rng.normal(size=(1, K, 2))
    seen = []

    def fp(ptr, utr, vtr, pte):
        seen.append(len(ptr))
        assert np.all(np.isfinite(utr)) and np.all(np.isfinite(vtr))
        return np.zeros(len(pte)), np.zeros(len(pte))

    err = sensor_cv_error(fp, pos, phys, frames=[0], folds=4)
    assert err == pytest.approx(1.0)          # 預測全零 → 相對誤差恰為 1
    assert sorted(seen) == [15, 15, 15, 15]   # 每折留出 5 個


def test_sensor_cv_rejects_bad_fold_count():
    with pytest.raises(ValueError, match="folds"):
        sensor_cv_error(lambda *a: None, np.zeros((3, 2)), np.zeros((1, 3, 2)), [0], folds=5)
