"""scalar E(t) 能量時序誤差指標（relative L2(0,T) / L∞ / final-time）的單元測試。

對應 GPT 建議的 kinetic-energy 時序指標：把每幀空間平均 KE 當一條時間序列來評，
與現行 ke_rel_err（每幀整場 Frobenius L2、再對 t 平均）是不同的量。
"""
import numpy as np
import pytest

from pi_lnn_jax.evaluate import energy_timeseries_errors


def test_identical_series_is_zero():
    et = np.array([1.0, 2.0, 3.0, 4.0])
    out = energy_timeseries_errors(et, et)
    assert out["ke_t_mape_spatialmean"] == pytest.approx(0.0, abs=1e-9)
    assert out["ke_t_rel_l2"] == pytest.approx(0.0, abs=1e-9)
    assert out["ke_t_rel_linf"] == pytest.approx(0.0, abs=1e-9)
    assert out["ke_t_final_rel"] == pytest.approx(0.0, abs=1e-9)


def test_hand_computed_example():
    # diff = [0.1, 0, -0.3, 0]
    et = np.array([1.0, 2.0, 3.0, 4.0])
    ep = np.array([1.1, 2.0, 2.7, 4.0])
    out = energy_timeseries_errors(ep, et)
    # MAPE = <|diff|/et>_t = mean([0.1/1, 0/2, 0.3/3, 0/4]) = mean([0.1,0,0.1,0]) = 0.05
    assert out["ke_t_mape_spatialmean"] == pytest.approx(0.05, rel=1e-9)
    # rel L2 = ||diff||_2 / ||et||_2 = sqrt(0.10)/sqrt(30)
    assert out["ke_t_rel_l2"] == pytest.approx(np.sqrt(0.10) / np.sqrt(30.0), rel=1e-9)
    # rel L∞ = max|diff| / max|et| = 0.3 / 4
    assert out["ke_t_rel_linf"] == pytest.approx(0.3 / 4.0, rel=1e-9)
    # final-time = |ep[-1]-et[-1]| / |et[-1]| = 0
    assert out["ke_t_final_rel"] == pytest.approx(0.0, abs=1e-12)


def test_final_time_drift():
    et = np.array([1.0, 1.0, 2.0])
    ep = np.array([1.0, 1.0, 2.5])
    out = energy_timeseries_errors(ep, et)
    assert out["ke_t_final_rel"] == pytest.approx(0.5 / 2.0, rel=1e-9)


def test_accepts_python_lists():
    # evaluate_time_series 回傳的是 list（.tolist()），helper 應能直接吃
    out = energy_timeseries_errors([1.0, 2.0], [1.0, 2.0])
    assert out["ke_t_rel_l2"] == pytest.approx(0.0, abs=1e-9)
