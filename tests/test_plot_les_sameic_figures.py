"""same-IC LES 兩張圖的算式與前提檢查。

Why:
    2026-09-17 的投稿前稽核留了一條：四條圖層 MAJOR 的根因都在產圖碼而不在
    `.tex`。這兩張圖是同一輪補寫的腳本，指標算式與「這真的是 same-IC 的一對嗎」
    的前提檢查因此在這裡釘住——它們錯了不會 crash，只會給錯的可預報時長。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common.sameic_les_dns import load_sameic_pair  # noqa: E402
from plot_les_sameic_predictability import compute_series  # noqa: E402
from plot_dns_les_vorticity_compare import _index_of  # noqa: E402


def _field(seed: int, n: int = 32, nt: int = 4):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    kx, ky = np.meshgrid(x, x, indexing="ij")
    base = np.stack([np.sin(2 * np.pi * (kx + 0.3 * i)) * np.cos(2 * np.pi * ky)
                     for i in range(nt)])
    return base + 0.1 * rng.standard_normal(base.shape)


def _dump(tmp: Path, name: str, u, v, t):
    p = tmp / name
    np.save(p, {"u": u, "v": v, "time": t}, allow_pickle=True)
    return p


def test_identical_fields_give_zero_error_and_unit_ratios():
    u, v = _field(0), _field(1)
    s = compute_series(u, v, u, v, 5.0)
    assert np.allclose(s["rel_l2_full"], 0.0, atol=1e-10)
    assert np.allclose(s["rel_l2_low"], 0.0, atol=1e-10)
    assert np.allclose(s["rel_linf"], 0.0, atol=1e-10)
    assert np.allclose(s["corr_u"], 1.0, atol=1e-10)
    assert np.allclose(s["corr_omega"], 1.0, atol=1e-10)
    assert np.allclose(s["ke_ratio"], 1.0, atol=1e-10)
    assert np.allclose(s["enstrophy_ratio"], 1.0, atol=1e-10)


@pytest.mark.parametrize("c", [0.5, 1.5])
def test_uniform_scaling_has_closed_form(c):
    """LES = c·DNS：誤差是 |c−1|，相關仍為 1，能量比是 c²。"""
    u, v = _field(2), _field(3)
    s = compute_series(c * u, c * v, u, v, 5.0)
    assert np.allclose(s["rel_l2_full"], 100.0 * abs(c - 1.0))
    assert np.allclose(s["rel_l2_low"], 100.0 * abs(c - 1.0))
    assert np.allclose(s["rel_linf"], 100.0 * abs(c - 1.0))
    assert np.allclose(s["corr_u"], 1.0, atol=1e-10)
    assert np.allclose(s["ke_ratio"], c ** 2)
    assert np.allclose(s["enstrophy_ratio"], c ** 2)


def test_rel_linf_is_vector_magnitude_not_per_component():
    """誤差只放在 v 的單一格點上：rel-L^∞ 的分母必須是速度向量長度的最大值。"""
    u, v = _field(4), _field(5)
    lv = v.copy()
    lv[0, 3, 7] += 1.0
    s = compute_series(u, lv, u, v, 5.0)
    ref = np.sqrt(u[0] ** 2 + v[0] ** 2).max()
    assert s["rel_linf"][0] == pytest.approx(100.0 / ref)


def test_lowband_ignores_high_wavenumber_perturbation():
    """只動最高波數的一格：全場誤差非零，k≤5 的低波段誤差應為零。"""
    n = 32
    u, v = _field(6, n=n), _field(7, n=n)
    uh = np.fft.fft2(u, axes=(1, 2))
    uh[:, n // 2, n // 2] *= 3.0
    lu = np.real(np.fft.ifft2(uh, axes=(1, 2)))
    s = compute_series(lu, v, u, v, 5.0)
    assert (s["rel_l2_full"] > 1e-6).all()
    assert np.allclose(s["rel_l2_low"], 0.0, atol=1e-8)


def test_load_pair_rejects_mismatched_time_axis(tmp_path):
    u, v = _field(8), _field(9)
    a = _dump(tmp_path, "les.npy", u, v, np.arange(4) * 0.5)
    b = _dump(tmp_path, "dns.npy", u, v, np.arange(4) * 0.25)
    with pytest.raises(ValueError, match="時間軸"):
        load_sameic_pair(a, b)


def test_load_pair_rejects_sign_flipped_initial_condition(tmp_path):
    """兩支求解器的速度號約定相反時必須擋下，而不是在繪圖端自動翻號。"""
    u, v = _field(10), _field(11)
    t = np.arange(4) * 0.25
    a = _dump(tmp_path, "les.npy", -u, -v, t)
    b = _dump(tmp_path, "dns.npy", u, v, t)
    with pytest.raises(ValueError, match="same-IC"):
        load_sameic_pair(a, b)


def test_load_pair_accepts_matching_pair(tmp_path):
    u, v = _field(12), _field(13)
    t = np.arange(4) * 0.25
    lu = u.copy()
    lu[1:] += 0.01                      # t=0 相同，之後才分岔
    a = _dump(tmp_path, "les.npy", lu, v, t)
    b = _dump(tmp_path, "dns.npy", u, v, t)
    out = load_sameic_pair(a, b)
    assert out[0].dtype == np.float64
    assert np.allclose(out[4], t)


def test_index_of_refuses_nearest_neighbour_substitution():
    t = np.array([0.0, 0.25, 0.5])
    assert _index_of(t, 0.25) == 1
    with pytest.raises(ValueError, match="匹配時刻"):
        _index_of(t, 0.3)
