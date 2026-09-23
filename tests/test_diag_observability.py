"""diag_observability velocity-only 可觀測性診斷的 isolated 單元測試。

全部用合成輸入（不載 LES/DNS），驗證：
  - axis convention（coord (x,y) → u[ix,iy]）
  - observability_metrics 的 rank/κ/null_fraction 在已知矩陣上正確
  - LES-POD operator 形狀 + POD modes 正交
  - div-free Fourier basis 真的 ∇·u≈0（spectral 導數）+ mode count
"""
from __future__ import annotations

import numpy as np
import pytest

import json
from pathlib import Path

from scripts.diag_observability import (
    coords_to_xy_indices,
    divfree_half_modes,
    divfree_modes,
    divfree_observation_operator,
    effective_rank_thresholds,
    information_metrics,
    observability_metrics,
    pod_basis,
    pod_observation_operator,
    principal_angles,
    split_time_halves,
    time_window_mask,
    transfer_verdict,
    velocity_field_from_psi,
)


# ── axis convention ───────────────────────────────────────────────────────────

def test_coords_to_xy_indices_axis_convention():
    """coord 第0欄=x→ix（axis_1），第1欄=y→iy（axis_2）。"""
    N = 4  # grid 座標 = [0, .25, .5, .75]
    coords = np.array([[0.5, 0.25], [0.0, 0.75], [0.74, 0.01]])
    ix, iy = coords_to_xy_indices(coords, N)
    np.testing.assert_array_equal(ix, [2, 0, 3])
    np.testing.assert_array_equal(iy, [1, 3, 0])


# ── observability_metrics ─────────────────────────────────────────────────────

def test_metrics_orthonormal_full_rank():
    """正交欄 → κ=1、rank=滿、null_fraction=0、eff_rank≈n_dof。"""
    n_obs, r = 10, 4
    C = np.zeros((n_obs, r))
    C[np.arange(r), np.arange(r)] = 1.0  # 4 個正交單位欄
    m = observability_metrics(C)
    assert m["rank"] == r
    assert m["cond"] == pytest.approx(1.0)
    assert m["null_fraction"] == pytest.approx(0.0)
    assert m["effective_rank"] == pytest.approx(r, abs=1e-6)


def test_metrics_rank_deficient_unobservable_mode():
    """重複欄 → rank 少 1、κ=inf、null_fraction=1/r（一個模態不可觀測）。"""
    C = np.array([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0]])  # 第3欄 = 第1欄 → rank 2, n_dof 3
    m = observability_metrics(C)
    assert m["rank"] == 2
    assert m["null_fraction"] == pytest.approx(1.0 / 3.0)
    assert np.isinf(m["cond"])


def test_metrics_condition_number_value():
    """對角 [3,1] → κ=3。"""
    m = observability_metrics(np.diag([3.0, 1.0]))
    assert m["cond"] == pytest.approx(3.0)


# ── LES-POD ───────────────────────────────────────────────────────────────────

def test_pod_basis_orthonormal_and_shape():
    rng = np.random.RandomState(0)
    T, N = 8, 6
    u = rng.randn(T, N, N)
    v = rng.randn(T, N, N)
    r = 5
    Phi, svals = pod_basis(u, v, r)
    assert Phi.shape == (2 * N * N, r)
    # POD modes 正交歸一
    gram = Phi.T @ Phi
    np.testing.assert_allclose(gram, np.eye(r), atol=1e-8)


def test_pod_observation_operator_picks_sensor_rows():
    """CΦ 應等於 modes 在 sensor (ix,iy) 的值（u 段疊 v 段）。"""
    N, r, K = 5, 3, 2
    Phi = np.arange(2 * N * N * r, dtype=float).reshape(2 * N * N, r)
    ix = np.array([1, 4])
    iy = np.array([2, 0])
    C = pod_observation_operator(Phi, ix, iy, N)
    assert C.shape == (2 * K, r)
    Phi_u = Phi[: N * N].reshape(N, N, r)
    Phi_v = Phi[N * N:].reshape(N, N, r)
    np.testing.assert_array_equal(C[:K], Phi_u[ix, iy, :])
    np.testing.assert_array_equal(C[K:], Phi_v[ix, iy, :])


# ── div-free Fourier ──────────────────────────────────────────────────────────

def test_divfree_mode_count_small():
    # |k|≤1 排除原點：(±1,0),(0,±1) = 4
    assert len(divfree_modes(1)) == 4
    # |k|≤2：上面4個 + (±1,±1)[4] + (±2,0),(0,±2)[4] = 12
    assert len(divfree_modes(2)) == 12


def test_divfree_half_modes_are_half():
    # 半平面恰為全集的一半（共軛對各取一）
    assert 2 * len(divfree_half_modes(4)) == len(divfree_modes(4))
    assert len(divfree_half_modes(1)) == 2  # (1,0),(0,1)


def test_divfree_operator_real_dof_count():
    coords = np.array([[0.1, 0.2], [0.7, 0.9], [0.5, 0.5]])
    A, half = divfree_observation_operator(coords, kmax=2)
    # real DoF = 2×half = full disk mode count（共軛對稱，無 double-count）
    assert A.shape == (2 * 3, 2 * len(half))
    assert A.shape[1] == len(divfree_modes(2))


def test_velocity_field_is_divergence_free():
    """real stream-function 構造的速度場 ∇·u ≈ 0（spectral 導數，到機器精度）。"""
    N = 32
    half = divfree_half_modes(4)
    rng = np.random.RandomState(1)
    psi_ab = rng.randn(2 * len(half))
    ux, uy = velocity_field_from_psi(psi_ab, half, N)
    kfreq = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi
    KX = kfreq[:, None]
    KY = kfreq[None, :]
    div_hat = 1j * KX * np.fft.fft2(ux) + 1j * KY * np.fft.fft2(uy)
    div = np.real(np.fft.ifft2(div_hat))
    rms = np.sqrt(np.mean(div ** 2))
    scale = np.sqrt(np.mean(ux ** 2 + uy ** 2)) + 1e-30
    assert rms / scale < 1e-10


# ── time_window_mask（共同時間窗）─────────────────────────────────────────────

def test_time_window_mask_bounds():
    t = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    m = time_window_mask(t, t_spinup=5.0, t_max=20.0)
    np.testing.assert_array_equal(m, [False, True, True, True, True, False])


def test_time_window_mask_open_upper():
    t = np.array([0.0, 5.0, 10.0])
    m = time_window_mask(t, t_spinup=0.0, t_max=None)
    np.testing.assert_array_equal(m, [True, True, True])


def test_split_time_halves_at_midpoint():
    """窗 [0,10] 從中點 5 切兩半。"""
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    a, b = split_time_halves(t, t_spinup=0.0, t_max=10.0)
    np.testing.assert_array_equal(a, [True, True, True, False, False, False])   # t<5
    np.testing.assert_array_equal(b, [False, False, False, True, True, True])   # 5≤t≤10


def test_split_time_halves_excludes_outside_window():
    """窗 [5,20] 中點 12.5；窗外（0,25）兩半皆排除。"""
    t = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    a, b = split_time_halves(t, t_spinup=5.0, t_max=20.0)
    np.testing.assert_array_equal(a, [False, True, True, False, False, False])
    np.testing.assert_array_equal(b, [False, False, False, True, True, False])


# ── transfer_verdict（principal-angle 判讀含 rank 護欄）────────────────────────

def test_transfer_verdict_reliable_no_excess():
    """r 在 energetic rank 內、excess≤tol → 無額外 gap。"""
    assert transfer_verdict(12, -2.6, rank_floor=18) == "no excess gap"


def test_transfer_verdict_reliable_excess():
    """r 在 rank 內、excess>tol → 真 excess gap。"""
    assert transfer_verdict(15, 8.0, rank_floor=18) == "excess surrogate gap"


def test_transfer_verdict_beyond_rank_floor():
    """r 超過 energetic rank → 不可靠（進入 noise floor），不論 excess。"""
    assert transfer_verdict(68, 7.1, rank_floor=18) == "unreliable (r>energetic rank)"


# ── principal_angles（子空間對齊度）───────────────────────────────────────────

def test_principal_angles_identical_subspace_zero():
    """同一子空間 → 所有 principal angle = 0°。"""
    rng = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rng.randn(20, 5))
    out = principal_angles(Q, Q)
    assert out["theta_max_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["mean_cos2"] == pytest.approx(1.0, abs=1e-9)


def test_principal_angles_orthogonal_subspace_90():
    """正交子空間 → 所有 angle = 90°。"""
    D = 8
    A = np.eye(D)[:, :2]   # e0,e1
    B = np.eye(D)[:, 2:4]  # e2,e3
    out = principal_angles(A, B)
    assert out["theta_max_deg"] == pytest.approx(90.0)
    assert out["theta_median_deg"] == pytest.approx(90.0)


def test_principal_angles_partial_overlap():
    """共享一維 → angle 集合 = {0°, 90°}。"""
    D = 8
    A = np.eye(D)[:, [0, 1]]  # e0,e1
    B = np.eye(D)[:, [0, 2]]  # e0,e2（共享 e0）
    out = principal_angles(A, B)
    angles = sorted(out["angles_deg"])
    assert angles[0] == pytest.approx(0.0, abs=1e-6)
    assert angles[1] == pytest.approx(90.0)


def test_principal_angles_known_30deg():
    """1D 子空間夾 30° → θ=30°。"""
    a = 30.0 * np.pi / 180.0
    A = np.array([[1.0], [0.0], [0.0], [0.0]])
    B = np.array([[np.cos(a)], [np.sin(a)], [0.0], [0.0]])
    out = principal_angles(A, B)
    assert out["theta_max_deg"] == pytest.approx(30.0)


# ── effective_rank_thresholds（POD 有效自由度）─────────────────────────────────

def test_effective_rank_thresholds_cumulative_energy():
    """svals²=[4,3,2,1]→cum=[.4,.7,.9,1.]：門檻=最小 r 使累積能量≥門檻。"""
    svals = np.sqrt(np.array([4.0, 3.0, 2.0, 1.0]))
    out = effective_rank_thresholds(svals, (0.5, 0.9, 0.999))
    assert out[0.5] == 2     # cum[1]=.7≥.5
    assert out[0.9] == 3     # cum[2]=.9≥.9
    assert out[0.999] == 4   # cum[3]=1.0≥.999


# ── information_metrics（資訊量理論層）─────────────────────────────────────────

def test_information_white_prior_orthonormal():
    """正交 C（σ_i=1）、白先驗、snr=10：D-opt=0、A-opt=r、E-opt=1、MI=½·r·log(1+snr)。"""
    n_obs, r, snr = 10, 4, 10.0
    C = np.zeros((n_obs, r))
    C[np.arange(r), np.arange(r)] = 1.0
    m = information_metrics(C, mode_energy=None, snr=snr)
    assert m["dopt_logdet"] == pytest.approx(0.0, abs=1e-9)       # Σ 2·log(1)=0
    assert m["aopt_trace_inv"] == pytest.approx(float(r))         # Σ 1/1²
    assert m["eopt_min_eig"] == pytest.approx(1.0)                # σ_min²
    mi_expected = 0.5 * r * np.log(1.0 + snr)                     # 每模態 log(1+snr)，σ̃_max=1
    assert m["mutual_info_nats"] == pytest.approx(mi_expected)
    assert m["mutual_info_bits"] == pytest.approx(mi_expected / np.log(2.0))


def test_mutual_info_monotonic_in_snr():
    """同一 C，MI 隨 snr 單調遞增（更小 noise → 更多資訊）。"""
    C = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    lo = information_metrics(C, snr=1.0)["mutual_info_nats"]
    hi = information_metrics(C, snr=100.0)["mutual_info_nats"]
    assert hi > lo


def test_information_energy_weighting_suppresses_dead_mode():
    """近零能量模態貢獻 MI≈0：energy=[1,1e-12] 的 MI ≈ 單模態 ½·log(1+snr) < 等能量。"""
    n_obs, snr = 4, 10.0
    C = np.zeros((n_obs, 2))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    mi_dead = information_metrics(C, mode_energy=np.array([1.0, 1e-12]), snr=snr)["mutual_info_nats"]
    mi_uniform = information_metrics(C, mode_energy=np.array([1.0, 1.0]), snr=snr)["mutual_info_nats"]
    assert mi_dead == pytest.approx(0.5 * np.log(1.0 + snr), abs=1e-6)
    assert mi_dead < mi_uniform


def test_information_rank_deficient_finite():
    """rank 不足時 D/A/E 仍在可觀測子空間上有限：C=diag([3,1,0])。"""
    C = np.diag([3.0, 1.0, 0.0])
    m = information_metrics(C, mode_energy=None, snr=1.0)
    assert m["dopt_logdet"] == pytest.approx(2.0 * np.log(3.0))   # 2(log3+log1)
    assert m["aopt_trace_inv"] == pytest.approx(1.0 / 9.0 + 1.0)  # 1/3²+1/1²
    assert m["eopt_min_eig"] == pytest.approx(1.0)                # 最小非零 σ²
    assert np.isfinite(m["dopt_logdet"]) and np.isfinite(m["aopt_trace_inv"])


def test_fourier_reproduces_paper_ceiling():
    """部署 K=100 sensors, |k|<=16 → rank=200, null≈74.9%（sec:ceiling oracle）。"""
    jp = Path("data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json")
    if not jp.exists():
        pytest.skip(f"sensor JSON 不存在: {jp}")
    coords = np.asarray(json.load(open(jp))["selected_coordinates"], dtype=np.float64)
    assert coords.shape[0] == 100
    A, half = divfree_observation_operator(coords, kmax=16)
    m = observability_metrics(A)
    assert A.shape[1] == 796          # 796 real DoF（共軛對稱）
    assert m["rank"] == 200           # = 2K，full row rank
    assert m["null_fraction"] == pytest.approx(0.749, abs=0.005)  # (796-200)/796
