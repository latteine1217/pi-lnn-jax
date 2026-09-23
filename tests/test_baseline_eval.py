"""baseline eval harness 純函式的單元測試（TDD: 先寫）。

這些是 eval 高風險邏輯（CLAUDE.md evalStrictness）：時間對齊須 fail-fast、
POD basis 須與 eval 快照不相交（leakage-free）、grid stride 須整除、denorm 須可逆。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.baseline_eval import (
    basis_indices_excluding,
    choose_grid_stride,
    denormalize_sensors,
    load_aligned_re,
    match_sensor_dns_times,
    select_pod_modes_by_validation,
)


def test_match_times_exact_subset():
    dns_t = (np.arange(20) * 0.01).astype(np.float32)
    sensor_time = dns_t[::4]  # 每 4 個取一
    idx = match_sensor_dns_times(sensor_time, dns_t)
    assert idx == [0, 4, 8, 12, 16]


def test_match_times_fail_when_no_dns_within_tolerance():
    dns_t = (np.arange(10) * 0.01).astype(np.float32)
    sensor_time = np.array([0.005], dtype=np.float32)  # 恰在兩 DNS 時間中間，gap 超 tol
    with pytest.raises(ValueError):
        match_sensor_dns_times(sensor_time, dns_t)


def test_match_times_fail_on_duplicate_mapping():
    dns_t = (np.arange(10) * 0.01).astype(np.float32)
    sensor_time = np.array([0.0, 0.002], dtype=np.float32)  # 兩個都最近到 idx0
    with pytest.raises(ValueError):
        match_sensor_dns_times(sensor_time, dns_t)


def test_basis_indices_are_disjoint_from_eval():
    basis = basis_indices_excluding(10, [0, 2, 4], min_basis=2)
    assert set(basis).isdisjoint({0, 2, 4})
    assert list(basis) == [1, 3, 5, 6, 7, 8, 9]


def test_basis_indices_fail_when_too_few():
    with pytest.raises(ValueError):
        basis_indices_excluding(5, [0, 1, 2, 3, 4], min_basis=2)


def test_choose_grid_stride_picks_finest_divisor_within_budget():
    # n=128, max_grid=40 → s=4（128/4=32≤40；s=2→64>40）
    assert choose_grid_stride(128, max_grid=40) == 4
    # n 本身已 ≤ budget → s=1
    assert choose_grid_stride(32, max_grid=40) == 1


def test_choose_grid_stride_fail_when_not_divisor():
    with pytest.raises(ValueError):
        choose_grid_stride(100, grid_stride=3)


def test_choose_grid_stride_explicit_divisor():
    assert choose_grid_stride(128, grid_stride=4) == 4
    assert choose_grid_stride(128) == 1


def test_denormalize_sensors_inverts_normalization():
    rng = np.random.default_rng(0)
    T, K = 5, 7
    phys = rng.normal(size=(T, K, 2))
    norm_stats = {"u_mean": 2.0, "u_std": 3.0, "v_mean": -1.0, "v_std": 0.5}
    norm = np.empty_like(phys)
    norm[..., 0] = (phys[..., 0] - norm_stats["u_mean"]) / norm_stats["u_std"]
    norm[..., 1] = (phys[..., 1] - norm_stats["v_mean"]) / norm_stats["v_std"]
    recovered = denormalize_sensors(norm, norm_stats)
    assert np.allclose(recovered, phys, atol=1e-6)


def _joint_low_rank(rng, N, r, M):
    bu = rng.standard_normal((r, N, N))
    bv = rng.standard_normal((r, N, N))
    c = rng.standard_normal((M, r))
    u = np.einsum("mr,rxy->mxy", c, bu)
    v = np.einsum("mr,rxy->mxy", c, bv)
    return u, v


def test_select_pod_modes_avoids_underfitting_and_uses_validation():
    rng = np.random.default_rng(0)
    N, r, B, K = 12, 8, 80, 30
    basis_u, basis_v = _joint_low_rank(rng, N, r, B)  # 聯合 rank-r 子空間
    idx = rng.choice(N * N, size=K, replace=False)
    ix, iy = idx // N, idx % N
    sensor_pos = np.stack([ix / N, iy / N], axis=1).astype(np.float32)

    best, curve = select_pod_modes_by_validation(
        basis_u, basis_v, sensor_pos, [2, 4, 8, 16], val_frac=0.3, seed=1)

    assert best >= r              # 不選 underfit（< 真實 rank）
    assert curve[best] < curve[2]  # 選到的 val 誤差優於 underfit 點
    assert best in curve           # 回傳值落在掃描格內


# ── load_aligned_re 的回傳契約 ────────────────────────────────────────────
#
# 它是三支 high-risk eval 腳本共用的 IO 編排。真實資料只在 lab-server 上有，
# 故此處把兩個 loader 換成合成資料，只驗回傳契約本身。

_ALIGNED_KEYS_BEFORE_TIMES = frozenset({
    "sensor_phys", "sensor_pos", "T", "K", "N", "s", "Nprime", "eval_idx",
    "dns_u_eval", "dns_v_eval", "dns_u_full", "dns_v_full",
})


def _patch_loaders(monkeypatch, *, n_dns=10, n_sensor=4, N=8, K=3):
    """把 load_aligned_re 內部 lazy import 的兩個 loader 換成合成資料。

    sensor 取 DNS 時間軸的每隔 2 幀，故 eval_idx 應為 [0,2,4,6][:n_sensor]。
    """
    dns_t = (np.arange(n_dns) * 0.25).astype(np.float64)
    dns_u = np.arange(n_dns * N * N, dtype=np.float64).reshape(n_dns, N, N)
    dns_v = dns_u + 0.5
    sensor_time = dns_t[::2][:n_sensor]

    def _sensors(_path, time_stride=1):
        return {
            "sensor_vals": np.zeros((n_sensor, K, 2)),
            "sensor_pos": np.zeros((K, 2)),
            "sensor_time": sensor_time,
            "norm_stats": {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0},
        }

    def _dns(_path, time_stride=1, return_p=False):
        return dns_u, dns_v, dns_t

    monkeypatch.setattr("pi_lnn_jax.data.load_sensors_from_path", _sensors)
    monkeypatch.setattr("pi_lnn_jax.data.load_dns_from_path", _dns)
    return dns_t


def test_load_aligned_re_reports_the_evaluation_times(monkeypatch):
    """dns_t_eval 是 EvaluationContext.times 的唯一來源，須逐值等於被選中的 DNS 時間。

    先前腳本手上只有 eval_idx（索引），要填 times 只能拿 index 充數——那會讓
    artifact 的時間軸與物理時間脫鉤。
    """
    dns_t = _patch_loaders(monkeypatch, n_sensor=4)

    out = load_aligned_re("s.json", "d.npy", sensor_time_stride=1, sensor_T=4,
                          grid_stride=1, max_grid=0)

    assert "dns_t_eval" in out
    np.testing.assert_array_equal(out["dns_t_eval"], dns_t[out["eval_idx"]])
    assert len(out["dns_t_eval"]) == out["dns_u_eval"].shape[0]


def test_load_aligned_re_keeps_every_existing_key(monkeypatch):
    """加欄位不得改動既有回傳面——三支 caller 逐鍵索引它。"""
    _patch_loaders(monkeypatch)

    out = load_aligned_re("s.json", "d.npy", sensor_time_stride=1, sensor_T=4,
                          grid_stride=1, max_grid=0)

    assert _ALIGNED_KEYS_BEFORE_TIMES <= set(out)
    assert set(out) - _ALIGNED_KEYS_BEFORE_TIMES == {"dns_t_eval"}
