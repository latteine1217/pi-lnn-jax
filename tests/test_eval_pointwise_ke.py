"""pointwise KE 誤差指標的單元測試。

背景：舊的 headline KE MAPE 先把整場塌縮成純量 E(t) 再比，凡是「空間分布錯了但
總量對」的失效（渦位置偏移、能量在空間中搬家）都會被空間平均抵消掉。本檔釘住
改用 pointwise-first 定義後的行為，其中 test_spatial_cancellation_is_caught 就是
這次語意變更的理由本身——舊定義在該案例讀 0，新定義必須讀到真實誤差。

兩個新指標（皆為 pointwise 先取絕對值 → 空間平均；時間平均由 agg_keys 機制完成）：
  ke_pw_mape : mean_x |KE_p - KE_d| / (KE_d + eps)      純 MAPE 語意，低能量區權重被放大
  ke_pw_nmae : mean_x |KE_p - KE_d| / mean_x(KE_d)      normalized MAE，無 eps 依賴
"""
import numpy as np
import pytest

from pi_lnn_jax.evaluate import compute_metrics, energy_timeseries_errors


def _uv_from_ke(ke):
    """由目標 pointwise KE 反推一組 (u, v)：取 v=0, u=sqrt(2*KE)。"""
    return np.sqrt(2.0 * ke), np.zeros_like(ke)


def test_identical_fields_are_zero():
    ke = np.array([[1.0, 2.0], [3.0, 4.0]])
    u, v = _uv_from_ke(ke)
    m = compute_metrics(u, v, u, v)
    assert m["ke_pw_mape"] == pytest.approx(0.0, abs=1e-9)
    assert m["ke_pw_nmae"] == pytest.approx(0.0, abs=1e-9)


def test_spatial_cancellation_is_caught():
    """核心迴歸：正負誤差在空間上完全相消時，舊定義讀 0、新定義必須讀到真實誤差。

    DNS KE = 1 均勻場；預測在一半格點高估 0.5、另一半低估 0.5。
    空間平均後 mean(KE_p) == mean(KE_d) → 舊的 spatial-mean MAPE = 0（盲點）。
    pointwise 則每點都有 |Δ|=0.5、KE_d=1 → 相對誤差 0.5。
    """
    ke_dns = np.ones((2, 2))
    ke_pred = np.array([[1.5, 1.5], [0.5, 0.5]])
    u_d, v_d = _uv_from_ke(ke_dns)
    u_p, v_p = _uv_from_ke(ke_pred)

    # 舊定義（純量 E(t) 序列）在此完全看不到誤差
    old = energy_timeseries_errors([ke_pred.mean()], [ke_dns.mean()])
    assert old["ke_t_mape_spatialmean"] == pytest.approx(0.0, abs=1e-12)

    # 新定義抓得到
    m = compute_metrics(u_p, v_p, u_d, v_d)
    assert m["ke_pw_mape"] == pytest.approx(0.5, rel=1e-9)
    assert m["ke_pw_nmae"] == pytest.approx(0.5, rel=1e-9)


def test_hand_computed_pointwise_values():
    """分母不同 → 兩指標讀數不同，各自手算釘住。"""
    ke_dns = np.array([[1.0, 2.0], [4.0, 8.0]])
    ke_pred = np.array([[1.5, 2.0], [3.0, 8.0]])   # |Δ| = [0.5, 0, 1.0, 0]
    u_d, v_d = _uv_from_ke(ke_dns)
    u_p, v_p = _uv_from_ke(ke_pred)
    m = compute_metrics(u_p, v_p, u_d, v_d)

    # MAPE: mean([0.5/1, 0/2, 1.0/4, 0/8]) = mean([0.5, 0, 0.25, 0]) = 0.1875
    assert m["ke_pw_mape"] == pytest.approx(0.1875, rel=1e-6)
    # nMAE: mean(|Δ|) / mean(KE_d) = (1.5/4) / (15/4) = 0.1
    assert m["ke_pw_nmae"] == pytest.approx(0.1, rel=1e-6)


def test_mask_semantics_excludes_body():
    """cylinder 語意：遮罩外（body 內）格點不得進入 pointwise 平均。

    body 格點給一個極端錯誤值；只要遮罩生效，兩個指標都應仍為 0。
    """
    ke_dns = np.ones((2, 2))
    ke_pred = np.ones((2, 2))
    ke_pred[0, 0] = 1e6                      # body 內的垃圾值
    mask = np.array([[False, True], [True, True]])
    u_d, v_d = _uv_from_ke(ke_dns)
    u_p, v_p = _uv_from_ke(ke_pred)

    axis = np.array([0.0, 1.0])
    m = compute_metrics(u_p, v_p, u_d, v_d, mask=mask, periodic=False,
                        dns_x=axis, dns_y=axis)
    assert m["ke_pw_mape"] == pytest.approx(0.0, abs=1e-9)
    assert m["ke_pw_nmae"] == pytest.approx(0.0, abs=1e-9)


def test_old_spatialmean_key_renamed_not_deleted():
    """取代 headline 語意，但舊定義的值必須仍可查（對照既有已發表數字）。"""
    out = energy_timeseries_errors([1.1, 2.0], [1.0, 2.0])
    assert "ke_t_mape_spatialmean" in out
    assert out["ke_t_mape_spatialmean"] == pytest.approx(0.05, rel=1e-9)
    # 舊 key 不得再以舊語意存在於此函數，避免同名不同義的靜默誤讀
    assert "ke_t_mape" not in out
