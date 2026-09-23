"""EnKF 的 isolated 測試，含 twin experiment（純 numpy，小尺度）。

twin experiment = 真值與 ensemble 由**同一個** solver 生成，故 model error
為零。這是隔離「EnKF 機制本身正確嗎」的唯一方式：若 perfect-model 下都
收斂不了，就不必談 model error 的情形。反過來，twin 收斂**不**代表真實
設定可行——那要另外測。

高維 EnKF 的兩個標配（localization、inflation）在這裡是能不能跑的前提而非
調參選項：N_e=O(10) 對上 O(10^3) 維狀態，取樣協方差必然退化。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.enkf import EnKF, gaspari_cohn
from pi_lnn_jax.kolmogorov_solver import KolmogorovSolver


def test_gaspari_cohn_properties():
    """局域化函數：r=0 為 1、單調遞減、超過 2c 為 0、處處非負。"""
    c = 0.3
    r = np.linspace(0, 3 * c, 200)
    rho = gaspari_cohn(r, c)
    assert rho[0] == pytest.approx(1.0)
    assert np.all(rho >= -1e-12)
    assert np.all(rho[r > 2 * c] == 0.0)
    assert np.all(np.diff(rho) <= 1e-12), "應單調不增"


def _twin_setup(N=32, K=40, seed=0):
    """小尺度 twin：真值軌跡 + 稀疏 (u,v) 觀測 + 獨立的初始 ensemble。"""
    rng = np.random.default_rng(seed)
    solver = KolmogorovSolver(N=N, nu=5e-3, k_f=2, forcing_amplitude=1.0)
    w_true = solver.low_pass(rng.normal(size=(N, N)), k_cut=N // 6)
    w_true -= w_true.mean()
    w_true = solver.integrate(w_true, dt=2e-3, n_steps=300)   # 甩掉暫態

    idx = rng.choice(N * N, size=K, replace=False)
    obs_ij = np.stack([idx // N, idx % N], axis=1)
    return solver, w_true, obs_ij, rng


def test_observation_operator_matches_solver_velocity():
    """觀測算子必須與 solver.velocity 一致——各算一次就會悄悄分岔。"""
    solver, w, obs_ij, _ = _twin_setup()
    f = EnKF(solver, obs_ij, obs_std=0.0, inflation=1.0, loc_radius=0.3)
    u, v = solver.velocity(w)
    expected = np.concatenate([u[obs_ij[:, 0], obs_ij[:, 1]],
                               v[obs_ij[:, 0], obs_ij[:, 1]]])
    np.testing.assert_allclose(f.observe(w), expected, atol=1e-12)


def test_twin_experiment_beats_free_run():
    """同化後的分析誤差應明顯低於不同化的自由跑——這是 EnKF 的最低要求。"""
    solver, w_true, obs_ij, rng = _twin_setup()
    N, dt, n_between, n_cycles = solver.N, 2e-3, 25, 12
    obs_std = 0.02

    # 初始 ensemble：由真值以外的隨機場演化而來（不從真值擾動，避免洩漏）
    ens = np.stack([
        solver.integrate(
            solver.low_pass(rng.normal(size=(N, N)), k_cut=N // 6), dt=dt, n_steps=300)
        for _ in range(24)])
    free = ens.copy()

    f = EnKF(solver, obs_ij, obs_std=obs_std, inflation=1.05, loc_radius=0.35, seed=1)
    truth = w_true.copy()
    for _ in range(n_cycles):
        truth = solver.integrate(truth, dt=dt, n_steps=n_between)
        ens = f.forecast(ens, dt=dt, n_steps=n_between)
        free = f.forecast(free, dt=dt, n_steps=n_between)
        y = f.observe(truth) + rng.normal(0, obs_std, size=2 * len(obs_ij))
        ens = f.analyze(ens, y)

    err_da = np.sqrt(np.mean((ens.mean(0) - truth) ** 2))
    err_free = np.sqrt(np.mean((free.mean(0) - truth) ** 2))
    assert err_da < 0.5 * err_free, f"同化 RMSE {err_da:.4f} 未明顯優於自由跑 {err_free:.4f}"


def test_ensemble_does_not_collapse():
    """inflation 的職責：spread 不得塌到零，否則濾波器對後續觀測失聰。"""
    solver, w_true, obs_ij, rng = _twin_setup(seed=3)
    N, dt = solver.N, 2e-3
    ens = np.stack([
        solver.integrate(
            solver.low_pass(rng.normal(size=(N, N)), k_cut=N // 6), dt=dt, n_steps=300)
        for _ in range(16)])
    spread0 = ens.std(axis=0).mean()

    f = EnKF(solver, obs_ij, obs_std=0.02, inflation=1.08, loc_radius=0.35, seed=2)
    truth = w_true.copy()
    for _ in range(15):
        truth = solver.integrate(truth, dt=dt, n_steps=25)
        ens = f.forecast(ens, dt=dt, n_steps=25)
        ens = f.analyze(ens, f.observe(truth) + rng.normal(0, 0.02, size=2 * len(obs_ij)))

    spread = ens.std(axis=0).mean()
    assert spread > 0.02 * spread0, f"ensemble 已塌陷：spread {spread:.2e} vs 初始 {spread0:.2e}"


def test_analysis_shape_and_finiteness():
    solver, w_true, obs_ij, rng = _twin_setup(seed=5)
    ens = np.stack([solver.low_pass(rng.normal(size=(solver.N, solver.N)), k_cut=4)
                    for _ in range(8)])
    f = EnKF(solver, obs_ij, obs_std=0.05, inflation=1.02, loc_radius=0.3, seed=4)
    out = f.analyze(ens, f.observe(w_true))
    assert out.shape == ens.shape and np.all(np.isfinite(out))


def test_localization_applies_to_both_covariance_blocks():
    """Schur 乘積必須同時作用在 PH^T 與 HPH^T 上。

    Why 這個測試存在：舊版只把局域化乘在 PH^T 上、HPH^T 未乘。那使增益
    K = (ρ∘PH^T)(HPH^T + R)^{-1} 不對應任何合法協方差的 Kalman gain，
    每次分析都注入能量。本檔原有的孿生測試看不見它——那些用約 6% 的觀測
    覆蓋率，而真實設定是 0.6%，差一個數量級。

    這裡只釘住結構性質（矩陣存在、形狀、半正定），行為由
    `test_sparse_observation_filter_does_not_diverge` 釘住。
    """
    solver, _, obs_ij, _ = _twin_setup(seed=11)
    f = EnKF(solver, obs_ij, obs_std=0.02, inflation=1.02, loc_radius=0.3, seed=0)
    K = len(obs_ij)

    assert f._loc.shape == (solver.N ** 2, 2 * K)
    assert f._loc_obs.shape == (2 * K, 2 * K), "缺少觀測--觀測局域化矩陣"
    assert np.allclose(f._loc_obs, f._loc_obs.T), "局域化矩陣須對稱"
    assert np.all(f._loc_obs >= 0.0), "Gaspari--Cohn 權重不得為負（否則破壞半正定）"
    assert np.allclose(np.diag(f._loc_obs), 1.0), "自身距離為零 → 權重應為 1"
    # 半正定：Schur 乘積定理要求兩個因子都半正定才保得住 HPH^T 的半正定性
    assert np.linalg.eigvalsh(f._loc_obs).min() > -1e-10, "局域化矩陣非半正定"


def test_sparse_observation_filter_does_not_diverge():
    """真實稀疏度（約 0.6% 覆蓋）下濾波器必須收斂，而非發散。

    設定刻意貼近實際基線（`scripts/enkf_baseline.py`）的觀測密度：
    N=48 的 2304 格點只觀測 14 點 ＝ 0.61%，對照本檔其他孿生測試的約 6%。
    只局域化 PH^T 的舊實作在此設定下第一次分析就把相對誤差推到 7.6，
    第二次 781，第三次 NaN；正確實作單調下降。

    判準用「單調改善且維持有界」而非絕對門檻：後者會把某一組參數的調校
    結果寫死進測試。
    """
    N, dt, n_sub, Ne, K = 48, 2.5e-3, 40, 12, 14
    solver = KolmogorovSolver(N=N, nu=1e-3, L=1.0, k_f=2, forcing_amplitude=0.1)
    rng = np.random.default_rng(3)

    def spun_up(seed, n):
        g = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            w = solver.low_pass(g.normal(size=(N, N)), k_cut=8)
            out.append(solver.integrate(w / w.std(), dt=dt, n_steps=300))
        return np.stack(out)

    truth = spun_up(3, 1)[0]
    obs_ij = np.stack(np.unravel_index(rng.choice(N * N, K, replace=False), (N, N)), axis=1)
    assert K / (N * N) < 0.01, "本測試的重點是稀疏觀測；覆蓋率過高會失去鑑別力"

    tu, tv = solver.velocity(truth)
    obs_std = 0.10 * float(np.sqrt(np.mean(
        np.concatenate([tu[obs_ij[:, 0], obs_ij[:, 1]], tv[obs_ij[:, 0], obs_ij[:, 1]]]) ** 2)))

    f = EnKF(solver, obs_ij, obs_std=obs_std, inflation=1.02, loc_radius=0.3, seed=0)
    ens = spun_up(11, Ne)

    errs = []
    for t in range(5):
        if t > 0:
            truth = solver.integrate(truth, dt=dt, n_steps=n_sub)
            ens = f.forecast(ens, dt=dt, n_steps=n_sub)
        ens = f.analyze(ens, f.observe(truth))
        e = float(np.linalg.norm(ens.mean(axis=0) - truth) / np.linalg.norm(truth))
        assert np.isfinite(e), f"濾波器在第 {t} 次分析後發散（舊實作在此設定下必然如此）"
        errs.append(e)

    assert errs[-1] < errs[0], f"誤差未隨同化下降：{errs}"
    assert errs[-1] < 1.0, f"分析誤差未小於「預測全為零」的水準：{errs}"
