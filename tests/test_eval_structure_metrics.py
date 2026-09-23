"""結構/尺度指標的單元測試：vector-velocity rel-L2、shell-wise band 譜誤差、γ(k)。

設計理由直接編碼成測試：

  test_band_shellwise_catches_what_integrated_misses
      integrated band error（先把 band 內 E(k) 加總再比）允許 band 內跨 shell 抵消——
      這正是 KE MAPE 那個「先平均再比」的原罪往下一層再犯。shell-wise 版本不會。

  test_gamma_blind_to_pure_amplitude_scaling
      γ(k) 量的是相位/結構相干性，對純幅值縮放不敏感。一個把能量全部砍半的預測
      γ(k)≡1 卻譜誤差極大——所以 γ 與 band 譜誤差**必須配對解讀**，任一單獨都不能
      宣稱「結構正確」。

  test_lowpass_prediction_signature
      顯式低通預測（截斷 k>kc）的指標簽名：低頻 band 近乎完美、mid/high band 誤差
      趨近 100%。用來確認指標抓得到平滑偏差。
"""
import numpy as np
import pytest

from pi_lnn_jax.evaluate import (
    compute_energy_spectrum,
    compute_metrics,
    dissipation_wavenumber,
    spectral_coherence,
)


N = 64
KETA = 50.0          # 測試統一把 k_η 釘在 50 → band low=(0,5] mid=(5,20] high=(20,50]


def _nu_for_keta(u_d, v_d, target=KETA):
    """反推 ν 使 dissipation_wavenumber 恰好給出 target，便於精確控制 band 邊界。

    k_η = (2S/ν²)^{1/4}/(2π)，S = Σ(2πk)²E  →  ν = sqrt(2S)/(2π·k_η)²
    """
    k, E = compute_energy_spectrum(u_d, v_d)
    S = float(np.sum((2 * np.pi * k) ** 2 * E))
    return float(np.sqrt(2 * S) / (2 * np.pi * target) ** 2)


def _mode_field(coeffs):
    """由 {(kx, ky): amplitude} 疊出 (u, v)；u 帶模態、v 置零，便於精確控制頻譜。"""
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((N, N))
    for (kx, ky), a in coeffs.items():
        u += a * np.cos(2 * np.pi * (kx * X + ky * Y))
    return u, np.zeros((N, N))


def test_uv_rel_err_matches_definition():
    """vector-velocity rel-L2 = sqrt(‖Δu‖²+‖Δv‖²)/sqrt(‖u‖²+‖v‖²)，非 u/v 各自誤差的平均。"""
    rng = np.random.default_rng(0)
    u_d, v_d = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    u_p, v_p = u_d + 0.1, v_d - 0.2
    m = compute_metrics(u_p, v_p, u_d, v_d)
    expect = (np.sqrt(np.sum((u_p - u_d) ** 2) + np.sum((v_p - v_d) ** 2))
              / np.sqrt(np.sum(u_d ** 2) + np.sum(v_d ** 2)))
    assert m["uv_rel_err"] == pytest.approx(expect, rel=1e-9)


def test_nu_helper_pins_keta():
    """helper 自身的正確性：反推的 ν 必須讓 k_η 落在指定值。"""
    u, v = _mode_field({(3, 0): 1.0, (9, 0): 0.5})
    k, E = compute_energy_spectrum(u, v)
    assert dissipation_wavenumber(k, E, _nu_for_keta(u, v)) == pytest.approx(KETA, rel=1e-9)


def test_band_fields_absent_without_nu():
    """未提供 ν 時不得產生 band/γ 欄位——不退回固定邊界，避免同名不同義。"""
    u, v = _mode_field({(3, 0): 1.0})
    m = compute_metrics(u, v, u, v)
    for key in ("band_rel_err_low", "gamma_low", "k_eta"):
        assert key not in m, f"無 ν 時不應產生 {key}"
    assert "k_cut" in m, "k_cut 不依賴 ν，應仍存在"


def test_perfect_prediction_is_clean():
    u, v = _mode_field({(3, 0): 1.0, (9, 2): 0.5, (20, 5): 0.2})
    m = compute_metrics(u, v, u, v, nu=_nu_for_keta(u, v))
    assert m["uv_rel_err"] == pytest.approx(0.0, abs=1e-12)
    for b in ("low", "mid", "high"):
        assert m[f"band_rel_err_{b}"] == pytest.approx(0.0, abs=1e-9)
    # 有能量的 shell 上 γ 應為 1
    assert m["gamma_low"] == pytest.approx(1.0, abs=1e-9)
    assert m["gamma_mid"] == pytest.approx(1.0, abs=1e-9)


def test_band_shellwise_catches_what_integrated_misses():
    """核心：band 內把能量從 k=6 搬到 k=15，總能量守恆。

    integrated（Σ 後再比）讀 ~0；shell-wise 必須讀到大誤差。
    """
    a = 1.0
    u_d, v_d = _mode_field({(6, 0): a})
    u_p, v_p = _mode_field({(15, 0): a})   # 同振幅、同 band(5<k<=16)、不同 shell

    nu = _nu_for_keta(u_d, v_d)
    md = compute_metrics(u_d, v_d, u_d, v_d, nu=nu)  # sanity: 自比為 0
    assert md["band_rel_err_mid"] == pytest.approx(0.0, abs=1e-9)

    m = compute_metrics(u_p, v_p, u_d, v_d, nu=nu)
    # integrated 版本（舊語意）確認確實被抵消；範圍取 k_η-based mid band
    E_p, E_d = np.array(m["E_pred_k"]), np.array(m["E_dns_k"])
    k = np.arange(len(E_d))
    mid = (k > 0.1 * m["k_eta"]) & (k <= 0.4 * m["k_eta"])
    integrated = abs(E_p[mid].sum() - E_d[mid].sum()) / E_d[mid].sum()
    assert integrated < 1e-6, "構造失敗：integrated 應該被抵消才能證明論點"
    # shell-wise 版本抓得到
    assert m["band_rel_err_mid"] > 1.0, "shell-wise band 誤差未抓到跨 shell 搬移"


def test_gamma_blind_to_pure_amplitude_scaling():
    """γ 對純幅值縮放不敏感 → 必須與 band 譜誤差配對解讀。"""
    u_d, v_d = _mode_field({(3, 0): 1.0, (9, 1): 0.6})
    u_p, v_p = 0.5 * u_d, 0.5 * v_d          # 能量剩 1/4，結構完全正確

    m = compute_metrics(u_p, v_p, u_d, v_d, nu=_nu_for_keta(u_d, v_d))
    assert m["gamma_low"] == pytest.approx(1.0, abs=1e-9), "γ 不應懲罰純幅值縮放"
    assert m["gamma_mid"] == pytest.approx(1.0, abs=1e-9)
    # 但譜誤差必須抓到能量不足（E ∝ amp² → 少 75%）
    assert m["band_rel_err_low"] == pytest.approx(0.75, rel=1e-6)


def test_gamma_detects_decorrelated_realization():
    """同一 shell、同能量、不同相位 → 譜幾乎一致但 γ 應明顯掉下來。"""
    u_d, v_d = _mode_field({(4, 0): 1.0})
    u_p, v_p = _mode_field({(0, 4): 1.0})   # 同 |k|=4 shell、正交方向

    m = compute_metrics(u_p, v_p, u_d, v_d, nu=_nu_for_keta(u_d, v_d))
    E_p, E_d = np.array(m["E_pred_k"]), np.array(m["E_dns_k"])
    assert abs(E_p[4] - E_d[4]) < 1e-9, "構造失敗：兩者 shell 能量應相同"
    assert m["gamma_low"] < 0.5, "γ 未偵測到同能量但去相關的 realization"


def test_lowpass_prediction_signature():
    """顯式低通（截掉 k>5）的指標簽名。"""
    u_d, v_d = _mode_field({(3, 0): 1.0, (9, 0): 0.8, (25, 0): 0.5})
    u_p, v_p = _mode_field({(3, 0): 1.0})    # 只留低頻

    m = compute_metrics(u_p, v_p, u_d, v_d, nu=_nu_for_keta(u_d, v_d))
    assert m["band_rel_err_low"] == pytest.approx(0.0, abs=1e-9), "低頻應保留"
    assert m["band_rel_err_mid"] == pytest.approx(1.0, rel=1e-6), "mid band 應全失"
    assert m["band_rel_err_high"] == pytest.approx(1.0, rel=1e-6), "high band 應全失"


def test_spectral_coherence_shape_and_range():
    rng = np.random.default_rng(1)
    u_d, v_d = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    k_arr, g = spectral_coherence(u_d, v_d, u_d, v_d)
    assert k_arr.shape == g.shape
    # 自比：有能量的 shell γ=1
    assert np.allclose(g[np.isfinite(g)], 1.0, atol=1e-9)
    # 反相場 → γ = -1
    _, g_neg = spectral_coherence(-u_d, -v_d, u_d, v_d)
    assert np.allclose(g_neg[np.isfinite(g_neg)], -1.0, atol=1e-9)


def test_kcut_over_keta_diagnoses_overextended_band():
    """釘住 k_η-based band 的已知限制，並確認 kcut_over_keta 能診斷出它。

    k_η 假設慣性區存在。當譜實際延伸遠短於 k_η（未發展的初始場；真實 DNS 於 t=0
    實測 k_cut=9 而 k_η=65），high band 會落在無訊號區，其相對誤差不可信。
    kcut_over_keta << 1 就是這個狀態的訊號——high band 因此不作為 headline。
    """
    u_d, v_d = _mode_field({(3, 0): 1.0})            # 譜只到 k=3
    rng = np.random.default_rng(0)
    u_d = u_d + 1e-10 * rng.standard_normal((N, N))  # 極弱噪音底，模擬 FFT corner
    u_p, v_p = _mode_field({(3, 0): 1.0})
    u_p = u_p + 1e-6 * rng.standard_normal((N, N))

    m = compute_metrics(u_p, v_p, u_d, v_d, nu=_nu_for_keta(u_d, v_d))
    assert m["k_cut"] < 0.4 * m["k_eta"], "構造失敗：本測試要的是譜遠短於 k_η"
    assert m["kcut_over_keta"] < 1.0, "kcut_over_keta 未反映譜延伸不足"


def test_kcut_uses_dns_not_pred():
    """k_cut 必須由 DNS 定義——若用 pred，不同方法會有不同 band 邊界而不可比。"""
    u_d, v_d = _mode_field({(3, 0): 1.0})
    # pred 在高頻加入大量能量；k_cut 不應因此變大
    u_p, v_p = _mode_field({(3, 0): 1.0, (28, 0): 5.0})
    m = compute_metrics(u_p, v_p, u_d, v_d)
    m_self = compute_metrics(u_d, v_d, u_d, v_d)
    assert m["k_cut"] == m_self["k_cut"], "k_cut 受 pred 影響，破壞跨方法可比性"
