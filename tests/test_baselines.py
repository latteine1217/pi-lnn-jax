"""Phase A classical reconstruction baselines 的單元測試（TDD: 先寫）。

驗收目標（對齊 knowledge/overview/current-status.md §2.4）:
  - GappyPOD 對「已知低秩子空間內的場」可精確回復（K>=r 個感測點）。
  - InterpBaseline(linear) 對仿射場可精確重建（linear interpolation 對 affine 精確）。
  - baseline 輸出 (u_pred, v_pred) 形狀為 [T,N,N] 且可直接餵 pi_lnn_jax.evaluate.compute_metrics。
"""
from __future__ import annotations

import numpy as np

from pi_lnn_jax.baselines import GappyPOD, InterpBaseline


def _low_rank_dataset(rng, N, r, M):
    """產生共享係數的低秩 (u,v) 資料：snapshot = coeffs @ basis。"""
    basis_u = rng.standard_normal((r, N, N))
    basis_v = rng.standard_normal((r, N, N))
    coeffs = rng.standard_normal((M, r))
    u = np.einsum("mr,rxy->mxy", coeffs, basis_u)
    v = np.einsum("mr,rxy->mxy", coeffs, basis_v)
    return u, v, basis_u, basis_v


def test_gappy_pod_recovers_low_rank_field():
    rng = np.random.default_rng(0)
    N, r, M, K = 16, 4, 30, 12
    train_u, train_v, basis_u, basis_v = _low_rank_dataset(rng, N, r, M)

    # 一個落在同一子空間的 test snapshot
    c_test = rng.standard_normal((1, r))
    test_u = np.einsum("tr,rxy->txy", c_test, basis_u)  # [1,N,N]
    test_v = np.einsum("tr,rxy->txy", c_test, basis_v)

    # K 個相異 grid 感測點（K >= r）
    idx = rng.choice(N * N, size=K, replace=False)
    ix, iy = idx // N, idx % N
    sensor_pos = np.stack([ix / N, iy / N], axis=1).astype(np.float32)  # (x,y)
    sensor_vals = np.stack([test_u[0][ix, iy], test_v[0][ix, iy]], axis=1)[None]  # [1,K,2]

    model = GappyPOD(n_modes=r).fit(train_u, train_v)
    u_pred, v_pred = model.reconstruct(sensor_vals, sensor_pos)

    rel_u = np.linalg.norm(u_pred - test_u) / np.linalg.norm(test_u)
    rel_v = np.linalg.norm(v_pred - test_v) / np.linalg.norm(test_v)
    assert rel_u < 1e-4, f"u rel-err {rel_u:.2e}"
    assert rel_v < 1e-4, f"v rel-err {rel_v:.2e}"


def test_interp_reproduces_affine_field():
    N = 16
    rng = np.random.default_rng(1)
    gmax = (N - 1) / N
    # 4 個略在格點範圍外的角點 + 內部散點 → 整個 query grid 嚴格落在凸包內（無 NaN、affine 精確）
    corners = np.array([[-0.05, -0.05], [1.05, -0.05], [-0.05, 1.05], [1.05, 1.05]], float)
    interior = rng.uniform(0.0, gmax, size=(20, 2))
    sensor_pos = np.vstack([corners, interior]).astype(np.float32)

    X, Y = sensor_pos[:, 0], sensor_pos[:, 1]
    u = 2.0 * X + 3.0 * Y + 1.0
    v = -1.0 * X + 0.5 * Y
    sensor_vals = np.stack([u, v], axis=1)[None].astype(np.float32)  # [1,K,2]

    model = InterpBaseline(method="linear", periodic=False)
    u_pred, v_pred = model.reconstruct(sensor_vals, sensor_pos, grid_shape=(N, N))

    g = np.arange(N) / N
    XX, YY = np.meshgrid(g, g, indexing="ij")  # field[ix,iy] at (x=g[ix], y=g[iy])
    u_true = 2.0 * XX + 3.0 * YY + 1.0
    v_true = -1.0 * XX + 0.5 * YY

    assert not np.isnan(u_pred).any() and not np.isnan(v_pred).any()
    assert np.linalg.norm(u_pred[0] - u_true) / np.linalg.norm(u_true) < 1e-5
    assert np.linalg.norm(v_pred[0] - v_true) / np.linalg.norm(v_true) < 1e-5


def test_interp_periodic_produces_finite_field():
    """週期 tiling 路徑（production 預設）：對週期場全格點皆有限、無 NaN。"""
    N = 24
    g = np.arange(N) / N
    XX, YY = np.meshgrid(g, g, indexing="ij")
    rng = np.random.default_rng(3)
    K = 40
    idx = rng.choice(N * N, size=K, replace=False)
    ix, iy = idx // N, idx % N
    sensor_pos = np.stack([ix / N, iy / N], axis=1).astype(np.float32)
    u_field = np.sin(2 * np.pi * XX) * np.cos(2 * np.pi * YY)
    v_field = np.cos(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    sensor_vals = np.stack([u_field[ix, iy], v_field[ix, iy]], axis=1)[None].astype(np.float32)

    model = InterpBaseline(method="linear", periodic=True)
    u_pred, v_pred = model.reconstruct(sensor_vals, sensor_pos, grid_shape=(N, N))

    assert u_pred.shape == (1, N, N) and v_pred.shape == (1, N, N)
    assert np.isfinite(u_pred).all() and np.isfinite(v_pred).all()


def test_baseline_output_feeds_compute_metrics():
    from pi_lnn_jax.evaluate import compute_metrics

    rng = np.random.default_rng(2)
    N, r, M, K, T = 16, 4, 20, 10, 3
    train_u, train_v, basis_u, basis_v = _low_rank_dataset(rng, N, r, M)

    idx = rng.choice(N * N, size=K, replace=False)
    ix, iy = idx // N, idx % N
    sensor_pos = np.stack([ix / N, iy / N], axis=1).astype(np.float32)

    c_test = rng.standard_normal((T, r))
    test_u = np.einsum("tr,rxy->txy", c_test, basis_u)
    test_v = np.einsum("tr,rxy->txy", c_test, basis_v)
    sensor_vals = np.stack([test_u[:, ix, iy], test_v[:, ix, iy]], axis=2).astype(np.float32)  # [T,K,2]

    model = GappyPOD(n_modes=r).fit(train_u, train_v)
    u_pred, v_pred = model.reconstruct(sensor_vals, sensor_pos)

    assert u_pred.shape == (T, N, N) and v_pred.shape == (T, N, N)

    m = compute_metrics(
        u_pred[0], v_pred[0],
        test_u[0].astype(np.float32), test_v[0].astype(np.float32),
    )
    for key in ("u_rel_err", "ke_rel_err", "div_pred_l2", "low_band_rel_err"):
        assert key in m, f"missing metric key: {key}"
    assert m["u_rel_err"] < 1e-3, f"recovered u rel-err {m['u_rel_err']:.2e}"
