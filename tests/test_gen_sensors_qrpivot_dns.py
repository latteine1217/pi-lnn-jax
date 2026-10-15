"""`gen_sensors_qrpivot_dns` 的軸向、特徵定義與輸出契約。

舊生成器的缺陷正是軸向：把 [T, x, y] 當 [T, y, x] 讀，"omega" 變成 ∂v/∂y − ∂u/∂x、
座標與取值差一個轉置，而且不會有任何錯誤訊息。這裡用解析的無散度場釘住三件事：
軸向閘門擋下轉置的陣列（擋不下「轉置且 u、v 標籤互換」，見函式 docstring）、渦度特徵是真的旋度、
寫出的索引、座標與取值三者一致。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pi_lnn_jax.data import load_sensors_from_path
from scripts._common.periodic_fields import reference_pressure
from scripts.gen_sensors_qrpivot_dns import (
    FEATURES,
    check_axis_convention,
    physical_features,
    rank_grid_points,
    write_budgets,
)

L = 1.0


def _streamfunction_field(N: int, T: int = 3):
    """ψ = sin(2πx) sin(4πy) + 0.5 cos(2π·3x)，u = ∂ψ/∂y、v = −∂ψ/∂x，陣列 [T, x, y]。

    刻意對 x、y 不對稱，否則對調軸向後仍無散度，閘門分不出來。
    """
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    tw = 2 * np.pi
    u = np.sin(tw * X) * 2 * tw * np.cos(2 * tw * Y)
    v = -(tw * np.cos(tw * X) * np.sin(2 * tw * Y) - 0.5 * 3 * tw * np.sin(3 * tw * X))
    omega = -(-(tw ** 2) * np.sin(tw * X) * np.sin(2 * tw * Y)
              - (2 * tw) ** 2 * np.sin(tw * X) * np.sin(2 * tw * Y)
              - 0.5 * (3 * tw) ** 2 * np.cos(3 * tw * X))  # ω = −∇²ψ
    scale = 1.0 + 0.1 * np.arange(T)[:, None, None]
    return u[None] * scale, v[None] * scale, omega[None] * scale


def test_axis_gate_accepts_x_first_and_rejects_swapped():
    u, v, _ = _streamfunction_field(64)
    check_axis_convention(u, v, L)
    with pytest.raises(ValueError, match="軸向"):
        check_axis_convention(u.transpose(0, 2, 1), v.transpose(0, 2, 1), L)


def test_omega_feature_is_the_curl():
    u, v, omega = _streamfunction_field(128)
    feat = physical_features(u, v, L)["omega"]
    rel = np.linalg.norm(feat - omega) / np.linalg.norm(omega)
    assert rel < 2e-2  # 二階中央差分在最高模態 k=3 上的截斷誤差約 1%
    # 舊生成器的讀法（對 axis 2 取 ∂/∂x）得到的不是旋度
    gx_u = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * L / 128)
    gy_v = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * L / 128)
    legacy = gy_v - gx_u
    assert np.linalg.norm(legacy - omega) / np.linalg.norm(omega) > 0.5


def test_budgets_are_nested_and_values_match_coordinates(tmp_path):
    N, T = 32, 4
    u, v, _ = _streamfunction_field(N, T)
    rng = np.random.default_rng(0)
    u = u + 0.05 * rng.standard_normal(u.shape)  # 打破對稱造成的排名平手
    v = v + 0.05 * rng.standard_normal(v.shape)
    time = np.linspace(0.0, 0.3, T)
    feats = physical_features(u, v, L)
    ranking = rank_grid_points([feats[n] for n in FEATURES], 10)
    write_budgets(u, v, time, L, ranking, [5, 10], tmp_path, "unit", {"source_file": "synthetic"})

    small = json.loads((tmp_path / "sensors_qrpivot_K5_unit.json").read_text())
    large = json.loads((tmp_path / "sensors_qrpivot_K10_unit.json").read_text())
    assert small["selected_coordinates"] == large["selected_coordinates"][:5]
    assert small["features"] == list(FEATURES)

    loaded = load_sensors_from_path(tmp_path / "sensors_qrpivot_K10_unit.json", time_stride=1)
    coords = np.asarray(large["selected_coordinates"])
    ix = np.rint(coords[:, 0] * N).astype(int)
    iy = np.rint(coords[:, 1] * N).astype(int)
    npz = np.load(tmp_path / "sensors_qrpivot_K10_unit_dns_values.npz")
    np.testing.assert_array_equal(npz["u"], u[:, ix, iy].T.astype(np.float32))
    np.testing.assert_array_equal(npz["v"], v[:, ix, iy].T.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(large["indices"]), ix * N + iy)  # 扁平索引依 [x, y] 展開
    np.testing.assert_allclose(np.asarray(loaded["sensor_pos"]), coords.astype(np.float32), rtol=0, atol=1e-7)

    with pytest.raises(FileExistsError):
        write_budgets(u, v, time, L, ranking, [5], tmp_path, "unit", {})


def test_pressure_feature_is_poisson_solution():
    u, v, _ = _streamfunction_field(32)
    np.testing.assert_array_equal(physical_features(u, v, L)["p_ref"], reference_pressure(u, v, L))


def test_existing_output_refuses_before_writing_anything(tmp_path):
    """任一預算的輸出已存在時，其他預算也不得寫出（不留半套產出）。"""
    N, T = 16, 3
    u, v, _ = _streamfunction_field(N, T)
    time = np.linspace(0.0, 0.2, T)
    ranking = rank_grid_points([physical_features(u, v, L)[n] for n in FEATURES], 10)
    write_budgets(u, v, time, L, ranking, [10], tmp_path, "unit", {})
    with pytest.raises(FileExistsError):
        write_budgets(u, v, time, L, ranking, [5, 10], tmp_path, "unit", {})
    assert not (tmp_path / "sensors_qrpivot_K5_unit.json").exists()
    assert not (tmp_path / "sensors_qrpivot_K5_unit_dns_values.npz").exists()
