"""Tests for scripts/_common/cylinder_sensors.py — cylinder placement 腳本的共用件。

What: 驗證 body/fluid 幾何推導與 sensor set 輸出 schema。用合成 shard（不需 Arrow /
      pyarrow），因此本測試可在本機 macOS 跑。

Why: gen_sensors_{uniform,boundary,sdf_isocontour} 三支各自複製了同一段「載 shard →
     body_mask → fluid_indices → coords_grid → body center/radius」前導與同一段 JSON+NPZ
     輸出（原始碼註解自承「與 boundary 版一致」「schema 同 boundary/sdf」）。三份 schema
     若漂移，產出的 sensor set 會在下游 CylinderDataset 靜默錯配。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from _paths import REPO_ROOT

for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common.cylinder_sensors import (  # noqa: E402
    cylinder_geometry,
    load_qrpivot_module,
    write_sensor_set,
)


H, W, T = 8, 10, 5


def _shard(re_value: float = 1781.0) -> dict:
    """合成 shard：中央 2×2 為 body（速度 0），其餘為流體。"""
    u = np.ones((T, H, W), np.float32)
    v = np.ones((T, H, W), np.float32)
    u[:, 3:5, 4:6] = 0.0
    v[:, 3:5, 4:6] = 0.0
    y2d, x2d = np.meshgrid(
        np.linspace(0.0, 1.0, W), np.linspace(0.0, 2.0, H),
    )
    return {
        "u": u, "v": v, "x": x2d, "y": y2d,
        "t": np.arange(T, dtype=np.float64), "Re": re_value,
    }


def _detect_mask(u, v, threshold):
    return (np.abs(u) + np.abs(v)).mean(axis=0) < threshold


def test_geometry_separates_body_from_fluid():
    g = cylinder_geometry([_shard()], body_threshold=1e-4, detect_body_mask=_detect_mask)
    assert g.body_mask.sum() == 4          # 中央 2×2
    assert g.fluid_mask.sum() == H * W - 4
    assert g.fluid_indices.shape[0] == H * W - 4
    # coords_grid 是 (row, col)，供 farthest-point 使用
    assert g.coords_grid.shape == (H * W - 4, 2)
    np.testing.assert_array_equal(g.coords_grid[:, 0], g.fluid_indices // W)


def test_geometry_body_center_and_radius():
    g = cylinder_geometry([_shard()], body_threshold=1e-4, detect_body_mask=_detect_mask)
    # body 是 row 3-4 / col 4-5 的 2×2；中心應落在其質心
    assert g.body_center[0] == pytest.approx(np.linspace(0, 2, H)[3:5].mean())
    assert g.body_center[1] == pytest.approx(np.linspace(0, 1, W)[4:6].mean())
    assert g.body_radius > 0.0


def test_geometry_intersects_body_across_shards():
    """多 shard 時 body 取交集——任一 shard 判為流體的格點就不是 body。"""
    s1, s2 = _shard(), _shard()
    s2["u"][:, 3, 4] = 1.0  # 在 s2 這格是流體
    s2["v"][:, 3, 4] = 1.0
    g = cylinder_geometry([s1, s2], body_threshold=1e-4, detect_body_mask=_detect_mask)
    assert g.body_mask.sum() == 3
    assert not g.body_mask[3, 4]


def test_write_sensor_set_schema():
    shard = _shard()
    g = cylinder_geometry([shard], body_threshold=1e-4, detect_body_mask=_detect_mask)
    picks = np.array([0, 5, 11])
    with tempfile.TemporaryDirectory() as td:
        jp, npzp = write_sensor_set(
            out_dir=Path(td), base="sensors_test_K3_cylinder_Re1781",
            geom=g, sensor_fluid_idx=picks, shard=shard,
            K=3, body_threshold=1e-4, method="unit_test", time_stride=20,
        )
        payload = json.loads(jp.read_text())
        arr = np.load(npzp)

    assert payload["K"] == 3
    assert payload["method"] == "unit_test"
    assert payload["grid"] == [H, W]
    assert payload["n_body_cells"] == 4
    assert payload["Re_list"] == [1781.0]
    assert len(payload["selected_coordinates"]) == 3
    assert len(payload["sensor_i"]) == len(payload["sensor_j"]) == 3
    assert payload["values_npz"] == str(npzp)
    # npz: u/v 為 [K, T]（CylinderDataset 契約），t 為 [T]
    assert arr["u"].shape == (3, T)
    assert arr["v"].shape == (3, T)
    assert arr["t"].shape == (T,)


def test_write_sensor_set_extra_fields_merge():
    """各 placement 法的專屬欄位（n_levels / n_boundary 等）要能併進 payload。"""
    shard = _shard()
    g = cylinder_geometry([shard], body_threshold=1e-4, detect_body_mask=_detect_mask)
    with tempfile.TemporaryDirectory() as td:
        jp, _ = write_sensor_set(
            out_dir=Path(td), base="b", geom=g, sensor_fluid_idx=np.array([0, 1]),
            shard=shard, K=2, body_threshold=1e-4, method="m", time_stride=20,
            extra={"n_levels": 8, "band": 0.25},
        )
        payload = json.loads(jp.read_text())
    assert payload["n_levels"] == 8
    assert payload["band"] == 0.25


def test_sensor_values_match_picked_grid_cells():
    """釘住取值路徑：npz 的 u[k] 必須是該 sensor 格點的時序，不可錯軸。"""
    shard = _shard()
    shard["u"][:, 0, 0] = 7.0  # 標記第一個 fluid 格點
    g = cylinder_geometry([shard], body_threshold=1e-4, detect_body_mask=_detect_mask)
    assert g.fluid_indices[0] == 0  # (0,0) 是流體
    with tempfile.TemporaryDirectory() as td:
        _, npzp = write_sensor_set(
            out_dir=Path(td), base="b", geom=g, sensor_fluid_idx=np.array([0]),
            shard=shard, K=1, body_threshold=1e-4, method="m", time_stride=20,
        )
        arr = np.load(npzp)
    np.testing.assert_allclose(arr["u"][0], 7.0)


def test_load_qrpivot_module_defaults_to_bundled():
    """未給 gen_dir（或該目錄無外部模組）時，用本專案內建的 qrpivot 模組。

    2026-08-21 從 pi-lnn 移植進 `_common.qrpivot_cylinder` 後，這條路不再需要
    外部 repo；既有 sbatch 傳的 `--gen-dir scripts` 也必須安靜落回內建。
    """
    for gen_dir in (None, tempfile.mkdtemp()):
        mod = load_qrpivot_module(gen_dir)
        assert mod.__name__.endswith("qrpivot_cylinder")
        for fn in ("load_shard", "detect_cylinder_mask", "farthest_point_sampling",
                   "build_snapshot_matrix", "qr_pivot_sensors"):
            assert callable(getattr(mod, fn)), f"內建 qrpivot 模組缺 {fn}"


def test_load_qrpivot_module_honours_external_override(tmp_path):
    """gen_dir 內有外部同名模組時優先用它（測試 stub 與舊 pi-lnn 佈局靠這條）。"""
    (tmp_path / "generate_sensors_qrpivot_cylinder.py").write_text(
        "SENTINEL = 'external'\n")
    mod = load_qrpivot_module(str(tmp_path))
    assert getattr(mod, "SENTINEL", None) == "external"
