"""LES 子空間投影地板：基底與排法必須與 gappy POD 那一列同源。

地板的用途是界定 gappy 那一列的下限（thesis app:fair_baselines）。只要兩者的
向量排法或基底來源有一點不同，這個比較就失去意義——而排法錯了不會 crash，
只會給出一個看起來合理的數字。本檔把「同源」釘成可執行的斷言。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.baselines import GappyPOD
from scripts.classical_baselines_fair import (
    project_reference_onto_basis,
    reference_time_mean,
)


def _library(seed=0, M=24, N=8):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(M, N, N)), rng.normal(size=(M, N, N))


def _rel_l2(u_p, v_p, u_r, v_r):
    num = np.sqrt(((u_p - u_r) ** 2 + (v_p - v_r) ** 2).sum())
    den = np.sqrt((u_r ** 2 + v_r ** 2).sum())
    return float(num / den)


def test_reference_time_mean_matches_the_gappy_stacking():
    """時均場的排法由 GappyPOD.fit 定義；本函式不得自成一格。"""
    u, v = _library()

    gp = GappyPOD(n_modes=3).fit(u, v)

    assert reference_time_mean(u, v) == pytest.approx(gp.mean, rel=0, abs=1e-12)


def test_a_field_inside_the_span_has_no_floor():
    """基底張成的子空間內的場，投影誤差是數值零——地板的定義如此。"""
    u, v = _library()
    gp = GappyPOD(n_modes=len(u) - 1).fit(u, v)

    # 用庫本身當「參考場」：M-1 個模態張成的子空間恰含所有 snapshot 的擾動。
    u_p, v_p = project_reference_onto_basis(gp.modes, gp.mean, u, v)

    assert _rel_l2(u_p, v_p, u, v) < 1e-10


def test_the_floor_falls_monotonically_with_rank_and_is_the_least_squares_optimum():
    """地板＝最小平方最優，故加模態只會降不會升。

    獨立參考用 lstsq 解係數（而非正交投影公式），兩條路必須同值——若基底不正交
    或排法錯位，兩者就會分岔。
    """
    u_lib, v_lib = _library(seed=1)
    u_ref, v_ref = _library(seed=2, M=5)
    gp = GappyPOD(n_modes=10).fit(u_lib, v_lib)

    errs = []
    for r in (2, 5, 10):
        u_p, v_p = project_reference_onto_basis(gp.modes[:, :r], gp.mean, u_ref, v_ref)
        errs.append(_rel_l2(u_p, v_p, u_ref, v_ref))

        T, N = u_ref.shape[0], u_ref.shape[1]
        X = np.stack([u_ref.reshape(T, -1), v_ref.reshape(T, -1)], 1).reshape(T, -1).T
        coef = np.linalg.lstsq(gp.modes[:, :r], X - gp.mean, rcond=None)[0]
        lsq = gp.mean + gp.modes[:, :r] @ coef
        assert u_p.ravel() == pytest.approx(lsq[: N * N].T.ravel(), rel=0, abs=1e-10)
        assert v_p.ravel() == pytest.approx(lsq[N * N:].T.ravel(), rel=0, abs=1e-10)

    assert errs[0] >= errs[1] >= errs[2]


def test_substituting_the_dns_time_mean_keeps_the_modes_and_changes_the_offset():
    """DNS 時均變體只換 mean：模態不動，且它是該基底下更好的偏移。"""
    u_lib, v_lib = _library(seed=3)
    u_ref, v_ref = _library(seed=4, M=6)
    u_ref += 0.7   # 讓兩個流場的平均量真的不同
    gp = GappyPOD(n_modes=4).fit(u_lib, v_lib)

    les = project_reference_onto_basis(gp.modes, gp.mean, u_ref, v_ref)
    dns = project_reference_onto_basis(
        gp.modes, reference_time_mean(u_ref, v_ref), u_ref, v_ref)

    assert _rel_l2(*dns, u_ref, v_ref) < _rel_l2(*les, u_ref, v_ref)


def test_a_basis_on_a_different_grid_is_rejected():
    """格點不符即失敗；形狀相容時靜默廣播會給出一個像樣但錯的地板。"""
    u_lib, v_lib = _library(N=8)
    gp = GappyPOD(n_modes=3).fit(u_lib, v_lib)
    u_ref, v_ref = _library(seed=9, M=3, N=16)

    with pytest.raises(ValueError, match="格點不符"):
        project_reference_onto_basis(gp.modes, gp.mean, u_ref, v_ref)
