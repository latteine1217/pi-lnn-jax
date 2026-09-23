"""`scripts/diag_difficulty_curve.py` 的複雜度指標。

這條曲線會被用來「除掉難度成分」再解讀模型結果，所以指標本身要能在本機驗：
單模態場的有效模態數必須接近 1、寬頻場必須明顯更大，否則「流場變低維」這個
論證就沒有依據。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diag_difficulty_curve import flow_complexity


def _mode_field(N=64, kx=2, ky=0, amp=1.0):
    g = np.arange(N) / N
    X, Y = np.meshgrid(g, g, indexing="ij")
    phase = 2 * np.pi * (kx * X + ky * Y)
    return amp * np.sin(phase), amp * np.cos(phase)


def test_single_mode_has_one_effective_shell():
    u, v = _mode_field(kx=2)
    c = flow_complexity(u, v, k_sensor=5.0)
    assert c["effective_shells"] == pytest.approx(1.0, abs=0.05), c
    assert c["energy_above_k_sensor"] == pytest.approx(0.0, abs=1e-9)
    print("✓ single_mode_has_one_effective_shell")


def test_broadband_field_has_more_effective_shells():
    """鑑別力自檢：多模態場的有效模態數必須顯著大於 1。"""
    u = v = 0.0
    for k in (1, 2, 3, 4, 5, 6):
        a, b = _mode_field(kx=k, ky=k)
        u, v = u + a / k, v + b / k
    c = flow_complexity(np.asarray(u), np.asarray(v), k_sensor=5.0)
    assert c["effective_shells"] > 2.0, c
    assert c["energy_above_k_sensor"] > 0.0
    print(f"✓ broadband_field_has_more_effective_shells (PR={c['effective_shells']:.2f})")


def test_high_wavenumber_energy_is_counted_above_k_sensor():
    """k=8 的模態在 k_sensor=5 之上：高頻佔比必須是 1，不是 0。"""
    u, v = _mode_field(kx=8)
    c = flow_complexity(u, v, k_sensor=5.0)
    assert c["energy_above_k_sensor"] == pytest.approx(1.0, abs=1e-6), c
    print("✓ high_wavenumber_energy_is_counted_above_k_sensor")


def test_enstrophy_matches_analytic_single_mode():
    """ω 對單模態有解析解，拿它釘住譜微分沒寫錯。"""
    N, k = 64, 3
    u, v = _mode_field(N=N, kx=k, ky=0)          # u=sin, v=cos(2πkx)
    # ω = ∂v/∂x − ∂u/∂y = −2πk sin(2πkx) → Z = ½·mean(ω²) = ¼(2πk)²
    expected = 0.25 * (2 * np.pi * k) ** 2
    got = flow_complexity(u, v, k_sensor=5.0)["enstrophy"]
    assert got == pytest.approx(expected, rel=1e-6), (got, expected)
    print("✓ enstrophy_matches_analytic_single_mode")


def test_spectrum_satisfies_parseval():
    """能譜總和必須等於 KE —— 擋 FFT 正規化寫錯（週期域的 Parseval）。"""
    rng = np.random.default_rng(3)
    N = 64
    u = rng.normal(size=(N, N))
    v = rng.normal(size=(N, N))
    # **刻意不去均值**：改用權威能譜（含 DC bin）之後 Parseval 應精確成立。
    # 舊版抄本從 k=1 起算，測試靠事先去均值才過——那等於測不到 DC 缺口。
    assert abs(u.mean()) > 1e-3, "此測試需要非零 DC，否則看不到缺口"

    from pi_lnn_jax.metric_artifact import compute_energy_spectrum
    import diag_difficulty_curve as m

    _, spec = compute_energy_spectrum(u, v)
    ke = 0.5 * np.mean(u ** 2 + v ** 2)
    assert spec.sum() == pytest.approx(ke, rel=1e-12), (spec.sum(), ke)
    assert m.flow_complexity(u, v, k_sensor=5.64)["effective_shells"] > 1.0
    print("✓ spectrum_satisfies_parseval")


def test_sensor_index_wraps_periodically():
    """週期域上 x≈1⁻ 的 sensor 屬於 index 0，不是 N−1。"""
    N = 256
    pos = np.array([[0.0, 0.5], [1.0 - 1e-9, 0.25], [0.99999, 0.75]])
    idx = np.rint(pos * N).astype(int) % N
    assert idx[0, 0] == 0 and idx[1, 0] == 0 and idx[2, 0] == 0, idx
    clipped = np.clip(np.rint(pos * N).astype(int), 0, N - 1)
    assert clipped[1, 0] == N - 1, "鑑別力自檢：clip 版本確實會給出錯的索引"
    print("✓ sensor_index_wraps_periodically")
