"""div-free trig LSQ 在近退化佈點下必須有界。

`scripts/classical_baselines_fair.py` 的 `--selftest` 用的是良態隨機佈點，抓不到
這個失效：舊解法 `A^T A + 絕對 ridge=1e-8` 在那裡與純 LSQ 沒有差別。實際跑論文表格
的 spacefill 佈點卻是規則晶格——farthest-point sampling 在週期域收斂到 1/16 棋盤
子晶格，使 kmax=5 帶內六個對角模態 {(3,3),(4,4),(5,5),(3,-3),(4,-4),(5,-5)} 線性
相依，設計矩陣 rank 116/122。絕對 ridge 不隨奇異譜縮放：近零奇異值落在它之上就被
「反轉」而非截斷，重建誤差衝到 1e4 量級。

本檔把舊解法原文抄在這裡當反面參照——證明斷言有牙齒，不會在無效的修正下靜默通過。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.classical_baselines_fair import (
    DEFAULT_TRIG_RCOND,
    _divfree_modes,
    _trig_design,
    divfree_trig_lsq,
    trig_design_diagnostics,
)

N_GRID = 64
KMAX = 5


def _legacy_ridge_solve(pos, u_s, v_s, grid_xy, kmax=KMAX, ridge=1e-8):
    """修正前的解法，逐行照抄——含它自己的 float32 相位（未 cast 座標）。"""
    modes = _divfree_modes(kmax)
    K = pos.shape[0]
    xs, ys = pos[:, 0], pos[:, 1]
    ncol = 2 * len(modes) + 2
    Au = np.zeros((K, ncol)); Av = np.zeros((K, ncol))
    for j, (m, n) in enumerate(modes):
        th = 2 * np.pi * (m * xs + n * ys)
        s, c = np.sin(th), np.cos(th)
        Au[:, 2 * j] = -4 * np.pi * n * s; Au[:, 2 * j + 1] = -4 * np.pi * n * c
        Av[:, 2 * j] = 4 * np.pi * m * s;  Av[:, 2 * j + 1] = 4 * np.pi * m * c
    Au[:, -2] = 1.0; Av[:, -1] = 1.0
    A = np.vstack([Au, Av])
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(ncol),
                           A.T @ np.concatenate([u_s, v_s]))
    G = _trig_design(grid_xy, modes)
    pred = G @ coef
    M = grid_xy.shape[0]
    return pred[:M], pred[M:]


def _synthetic_field(seed=0, k_in=KMAX, k_out=12, out_amp=0.3):
    """帶內無散度模態 ＋ 帶外內容。

    帶外內容是這個測試的承重面：它在感測值裡是模型無法表示的殘差，近退化方向就是
    靠它被放大。純帶內場即使佈點退化也擬得完美，測不出東西。
    """
    rng = np.random.default_rng(seed)
    coefs = [(m, n, *(rng.normal(size=2) * (1.0 if max(abs(m), abs(n)) <= k_in else out_amp)
                      / (m * m + n * n)))
             for (m, n) in _divfree_modes(k_out)]

    def evaluate(x, y):
        u = np.zeros_like(x, dtype=np.float64); v = np.zeros_like(x, dtype=np.float64)
        for m, n, a, b in coefs:
            th = 2 * np.pi * (m * x + n * y)
            psi = a * np.sin(th) + b * np.cos(th)
            u += -4 * np.pi * n * psi; v += 4 * np.pi * m * psi
        return u, v

    return evaluate


def _checkerboard_lattice(k=100, spacing=16, seed=0):
    """1/16 棋盤子晶格上取 K 點——farthest-point sampling 在週期域的收斂形態。"""
    pts = np.array([[(i + 0.5) / spacing, (j + 0.5) / spacing]
                    for i in range(spacing) for j in range(spacing) if (i + j) % 2 == 0])
    return pts[np.random.default_rng(seed).permutation(len(pts))[:k]]


@pytest.fixture(scope="module")
def bench():
    xs = np.linspace(0.0, 1.0, N_GRID, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel()], 1)
    evaluate = _synthetic_field()
    u_t, v_t = evaluate(X.ravel(), Y.ravel())
    scale = np.sqrt((u_t ** 2 + v_t ** 2).mean())

    def rel_err(u_rec, v_rec):
        return float(np.sqrt(((u_rec - u_t) ** 2 + (v_rec - v_t) ** 2).mean()) / scale)

    return grid, evaluate, rel_err


# 感測位置的 dtype 是上游 data.py 的實作細節（sensor_pos 是 float32），不該決定
# 結果：float32 相位誤差約 1e-7，恰好把精確退化的晶格推進最壞的近退化區間。
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_checkerboard_lattice_reconstruction_is_bounded(bench, dtype):
    grid, evaluate, rel_err = bench
    pos = _checkerboard_lattice().astype(dtype)
    u_s, v_s = evaluate(*np.asarray(pos, np.float64).T)

    diag = trig_design_diagnostics(pos, kmax=KMAX)
    assert diag["trig_rank"] < diag["trig_ncol"], "晶格佈點應被診斷為秩虧損"

    assert rel_err(*divfree_trig_lsq(pos, u_s, v_s, grid, KMAX)) < 2.0

    if dtype is np.float32:  # 舊解法只在近退化（非精確退化）下爆掉
        assert rel_err(*_legacy_ridge_solve(pos, u_s, v_s, grid)) > 1e3


def test_perturbed_lattice_reconstruction_is_bounded(bench):
    """精確退化不是最壞情況——「幾乎退化」才是，且 float64 也躲不掉。"""
    grid, evaluate, rel_err = bench
    rng = np.random.default_rng(7)
    pos = (_checkerboard_lattice() + 1e-7 * rng.standard_normal((100, 2))) % 1.0
    u_s, v_s = evaluate(*pos.T)

    assert rel_err(*divfree_trig_lsq(pos, u_s, v_s, grid, KMAX)) < 2.0
    assert rel_err(*_legacy_ridge_solve(pos, u_s, v_s, grid)) > 1e3


def test_wellposed_placement_matches_plain_least_squares(bench):
    """良態佈點下截斷不作用：既有論文數字（LES-QR 佈點）不因這個修正而移動。"""
    grid, evaluate, _ = bench
    pos = np.random.default_rng(3).random((100, 2))
    u_s, v_s = evaluate(*pos.T)

    assert trig_design_diagnostics(pos, kmax=KMAX)["trig_rank"] == 2 * len(
        _divfree_modes(KMAX)) + 2

    new = divfree_trig_lsq(pos, u_s, v_s, grid, KMAX)
    old = _legacy_ridge_solve(pos, u_s, v_s, grid)
    for a, b in zip(new, old):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-8 * np.abs(b).max())


def test_default_rcond_clears_both_regimes():
    """預設門檻必須夾在兩個實測區間之間：良態佈點的最小奇異值之下、近退化帶之上。"""
    well = np.linalg.svd(_trig_design(np.random.default_rng(3).random((100, 2)),
                                      _divfree_modes(KMAX)), compute_uv=False)
    assert DEFAULT_TRIG_RCOND < well[-1] / well[0] / 4

    rng = np.random.default_rng(7)
    for pert in (1e-9, 1e-7, 1e-5):
        s = np.linalg.svd(
            _trig_design((_checkerboard_lattice() + pert * rng.standard_normal((100, 2))) % 1.0,
                         _divfree_modes(KMAX)), compute_uv=False)
        assert s[-1] / s[0] < DEFAULT_TRIG_RCOND, f"擾動 {pert:.0e} 的近退化帶未被門檻涵蓋"
