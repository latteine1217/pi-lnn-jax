"""diag_residual_spectrum 的正確性測試。

核心是 Taylor-Green：它是 2D 不可壓 NS 的**精確解**（無 forcing），
渦量傳輸殘差與散度都應為 0。殘差診斷最危險的失效是「算出一條看起來合理的
譜，但式子的軸序/符號寫錯了」——那不會 crash，只會給錯結論。用有解析解的場
把整條路徑釘死，比對任何真實資料都可靠。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diag_residual_spectrum import (  # noqa: E402
    Ops, _shell_index, _shell_power, _assert_shell_matches_repo,
    residual_terms, spectra_of,
)


def taylor_green(N: int, t: np.ndarray, nu: float, L: float = 1.0):
    """u=-cos(2πx)sin(2πy)F(t), v=sin(2πx)cos(2πy)F(t), F=exp(-2(2π)²νt)。"""
    g = np.arange(N) * (L / N)
    X, Y = np.meshgrid(g, g, indexing="ij")
    F = np.exp(-2.0 * (2.0 * np.pi) ** 2 * nu * t)[:, None, None]
    u = -np.cos(2 * np.pi * X)[None] * np.sin(2 * np.pi * Y)[None] * F
    v = np.sin(2 * np.pi * X)[None] * np.cos(2 * np.pi * Y)[None] * F
    return u, v, g


@pytest.mark.parametrize("order", [2, 4])
def test_taylor_green_residual_is_zero(order):
    """精確解的渦量傳輸殘差與散度必須遠小於各項自身量級。"""
    N, nu, L = 32, 0.01, 1.0
    t = np.arange(9) * 1e-3
    u, v, y = taylor_green(N, t, nu, L)
    ops = Ops("spectral", L, N)
    terms = residual_terms(u, v, np.zeros_like(u), t, y, nu,
                           A=0.0, k_f=2.0, ops=ops, time_order=order, with_mom=False)
    for eq in ("vort", "cont"):
        R = sum(terms[eq].values())
        scale = max(float(np.sqrt(np.mean(x ** 2))) for x in terms[eq].values())
        assert scale > 0
        assert np.sqrt(np.mean(R ** 2)) / scale < 1e-6, (
            f"{eq} 殘差在精確解上不為零（order={order}）")


def test_advection_and_forcing_signs():
    """forcing 進的是渦量方程的 curl 項，符號寫反不會 crash 但結論全錯。

    對零速度場，vort 殘差只剩 −curl f，其 rms 必須等於 A·2π·k_f/√2。
    """
    N, nu, L, A, k_f = 32, 0.01, 1.0, 0.1, 2.0
    t = np.arange(9) * 1e-3
    z = np.zeros((len(t), N, N))
    ops = Ops("spectral", L, N)
    terms = residual_terms(z, z, z, t, np.arange(N) * (L / N), nu, A, k_f,
                           ops, time_order=2, with_mom=False)
    R = sum(terms["vort"].values())
    expected = A * 2.0 * np.pi * k_f / np.sqrt(2.0)   # rms of cos over a period
    assert np.isclose(np.sqrt(np.mean(R ** 2)), expected, rtol=1e-6)


def test_shell_binning_matches_metric_artifact():
    """bincount 版殼層與 repo 的 compute_energy_spectrum 必須逐 bin 等價。"""
    N = 32
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    idx, n_bins = _shell_index(N)
    _assert_shell_matches_repo(u, v, idx, n_bins)   # 內含 Parseval 檢查


def test_spectra_parseval():
    """Σ_k residual(k) 必須等於實空間 mean square。"""
    N = 32
    rng = np.random.default_rng(1)
    idx, n_bins = _shell_index(N)
    terms = {"a": rng.standard_normal((3, N, N)), "b": rng.standard_normal((3, N, N))}
    sp = spectra_of(terms, idx, n_bins)
    assert np.isclose(np.sum(sp["residual"]), float(sp["rms_real"][0]) ** 2, rtol=1e-10)


def test_findiff_underestimates_high_k():
    """中央差分的 symbol 是 sin(kh)/h，在高 k 系統性低估——這是判讀的前提。

    釘住它，避免有人把 physics_field 的 FD 版拿去做頻帶比較（會讓模型看起來更好）。
    """
    N, L = 64, 1.0
    g = np.arange(N) * (L / N)
    k = 20                                   # 高波數單模
    f = np.sin(2 * np.pi * k * g)[None, :, None] * np.ones((1, 1, N))
    d_spec = Ops("spectral", L, N).dx(f)
    d_fd = Ops("findiff", L, N).dx(f)
    ratio = np.sqrt(np.mean(d_fd ** 2)) / np.sqrt(np.mean(d_spec ** 2))
    assert ratio < 0.95, f"FD 未如預期低估高 k 導數（ratio={ratio:.4f}）"
    assert np.isclose(ratio, np.sin(2 * np.pi * k * L / N) / (2 * np.pi * k * L / N),
                      rtol=1e-6)


def test_time_axis_tolerance_matches_float32():
    """float32 捨入的時間軸要通過，結構性不等距要被擋。

    DNS 時間軸經 loader 轉 float32，dt=0.05 的捨入讓逐格 dt 有 ~1e-5 的相對離散
    （實測 9.5e-6，job 5527 因 rtol=1e-9 誤判而失敗）。而 sensor_time_independent
    的不等距是量級差異，必須仍然擋得住。
    """
    from diag_residual_spectrum import _time_deriv

    f = np.zeros((9, 4, 4))
    t32 = (np.arange(9) * 0.05).astype(np.float32).astype(np.float64)
    d = np.diff(t32)
    assert 0 < (d.max() - d.min()) / d.mean() < 1e-4, "此測試的前提（float32 捨入量級）已變"
    _time_deriv(f, t32, 2)                      # 不得 raise

    t_gap = np.array([0.0, 0.05, 0.10, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    with pytest.raises(ValueError, match="非均勻"):
        _time_deriv(f, t_gap, 2)


def test_band_aggregation_weighting_is_not_arithmetic_mean():
    """band 內的無量綱量必須用總和比，不是逐 shell 算術平均。

    k 殼層能量跨約 20 個數量級，算術平均讓空 shell 與主導 shell 等權。實測在
    真值 low band 上兩者差 84 倍（0.168 vs 0.0020），並會翻轉「哪個 band 相對
    地板最差」的排序。這裡用一個刻意的能量對比把該差異釘住。
    """
    # shell 0：能量微弱但殘差與項同量級 → 逐點比值 = 1（雜訊）
    # shell 1：主導 shell，比值 = 1e-3（真正的抵消品質）
    R = np.array([1e-3, 1.0])
    T = np.array([1e-3, 1e3])
    arithmetic = float(np.mean(R / T))          # ≈ 0.5，被雜訊 shell 主導
    total_ratio = float(R.sum() / T.sum())      # ≈ 1e-3，反映主導 shell
    assert arithmetic / total_ratio > 100, "此測試的前提（兩種聚合的量級差）已變"
    assert np.isclose(total_ratio, (1e-3 + 1.0) / (1e-3 + 1e3), rtol=1e-9)
    assert np.isclose(arithmetic, 0.5005, rtol=1e-3)


def test_headroom_closure_is_bounded_and_signed_correctly():
    """headroom 關閉率問的是「地板留的抵消空間被丟掉多少」，不是 cancel 的相對變化。

    job 5541 印出 −6751%，因為公式寫成 (cf−cp)/cf；cancel 越小越好，模型比地板差
    時該式為負且無界。正確式在 cp∈[cf,1] 上落在 [0,1]。
    """
    def closure(cf, cp):
        hr_f, hr_p = 1.0 - cf, 1.0 - cp
        return (hr_f - hr_p) / hr_f

    for cf, cp in [(2.022e-3, 1.385e-1), (1.075e-1, 3.176e-1), (6.790e-1, 9.819e-1)]:
        c = closure(cf, cp)
        assert 0.0 <= c <= 1.0, f"關閉率 {c} 落在 [0,1] 之外 (cf={cf}, cp={cp})"
    assert np.isclose(closure(0.5, 0.5), 0.0)      # 與地板相同 → 未丟失
    assert np.isclose(closure(0.5, 1.0), 1.0)      # 完全不抵消 → 全丟
