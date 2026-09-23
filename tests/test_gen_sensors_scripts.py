"""End-to-end tests for scripts/gen_sensors_{uniform,boundary,sdf_isocontour}.py。

What: 用 stub 版的 `generate_sensors_qrpivot_cylinder` 實際跑三支腳本的 main()，
      檢查產出的 JSON/NPZ schema 三者一致。

Why: Arrow I/O 已於 2026-08-21 隨 qrpivot 模組移植進 `_common.qrpivot_cylinder`，
     但真實 Arrow shard 是 GB 級且不在版控內，本機與 CI 仍拿不到。stub 掉 shard 讀取
     這一層後，placement 邏輯與輸出契約就能在本機驗證——三份複製品各自漂移正是
     長期缺少這層驗證的後果。
"""
from __future__ import annotations

import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

STUB = '''
import numpy as np

def load_shard(path):
    H, W, T = 12, 16, 6
    u = np.ones((T, H, W), np.float32)
    v = np.ones((T, H, W), np.float32)
    u[:, 5:7, 7:9] = 0.0
    v[:, 5:7, 7:9] = 0.0
    y2d, x2d = np.meshgrid(np.linspace(0.0, 1.0, W), np.linspace(0.0, 2.0, H))
    return {"u": u, "v": v, "x": x2d, "y": y2d,
            "t": np.arange(T, dtype=np.float64), "Re": 1781.0}

def detect_cylinder_mask(u, v, threshold=1e-4):
    return (np.abs(u) + np.abs(v)).mean(axis=0) < threshold

def farthest_point_sampling(coords, n, seed=0):
    # 確定性的等間隔取樣即可——本測試驗的是 schema 與接線，不是取樣品質
    n = min(n, len(coords))
    return np.linspace(0, len(coords) - 1, n).astype(int)

def build_snapshot_matrix(shards, time_stride, fluid_mask):
    n_fluid = int(fluid_mask.sum())
    rng = np.random.RandomState(0)
    return rng.randn(8, n_fluid)

def qr_pivot_sensors(A, K):
    return np.arange(min(K, A.shape[1]))
'''

# (腳本檔名, 額外 CLI 參數, 預期 method)
CASES = [
    ("gen_sensors_uniform.py", ["--K", "20"], "farthest_point_uniform"),
    (
        "gen_sensors_boundary.py",
        ["--K", "20", "--n-uniform", "6", "--n-boundary", "6", "--bl-thresh", "0.3"],
        "uniform_boundary_qr",
    ),
    (
        "gen_sensors_sdf_isocontour.py",
        ["--K", "20", "--n-levels", "4"],
        "sdf_isocontour_linear",
    ),
]


def _run(script: str, extra: list[str], gen_dir: Path, out_dir: Path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        [script, "--shards", "fake.arrow", "--gen-dir", str(gen_dir),
         "--out", str(out_dir), *extra],
    )
    runpy.run_path(str(_SCRIPTS / script), run_name="__main__")


@pytest.mark.parametrize("script, extra, method", CASES)
def test_script_writes_expected_schema(script, extra, method, monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        gen_dir = Path(td) / "gen"
        gen_dir.mkdir()
        (gen_dir / "generate_sensors_qrpivot_cylinder.py").write_text(STUB)
        out_dir = Path(td) / "out"

        _run(script, extra, gen_dir, out_dir, monkeypatch)

        jsons = sorted(out_dir.glob("*.json"))
        npzs = sorted(out_dir.glob("*_values.npz"))
        assert len(jsons) == 1 and len(npzs) == 1
        payload = json.loads(jsons[0].read_text())
        arr = np.load(npzs[0])

    assert payload["method"] == method
    assert payload["K"] == 20
    assert payload["grid"] == [12, 16]
    assert payload["n_body_cells"] == 4
    assert payload["Re_list"] == [1781.0]
    # 三支共用的 schema 欄位
    for key in ("domain", "n_fluid_cells", "body_threshold", "time_stride_qr",
                "selected_coordinates", "sensor_i", "sensor_j", "sensor_flat",
                "values_npz"):
        assert key in payload, f"{script} 產出缺欄位 {key}"
    n = len(payload["sensor_i"])
    assert len(payload["sensor_j"]) == len(payload["selected_coordinates"]) == n
    assert arr["u"].shape == (n, 6)
    assert arr["v"].shape == (n, 6)
    assert arr["t"].shape == (6,)


def test_all_three_agree_on_shared_schema_keys(monkeypatch):
    """三支的共用欄位集合必須完全相同——這是它們曾經各自複製的那份 schema。"""
    shared = None
    for script, extra, _ in CASES:
        with tempfile.TemporaryDirectory() as td:
            gen_dir = Path(td) / "gen"
            gen_dir.mkdir()
            (gen_dir / "generate_sensors_qrpivot_cylinder.py").write_text(STUB)
            out_dir = Path(td) / "out"
            _run(script, extra, gen_dir, out_dir, monkeypatch)
            payload = json.loads(next(out_dir.glob("*.json")).read_text())

        # 扣掉各法專屬欄位後，剩下的必須一致
        method_specific = {"n_levels", "band", "n_uniform", "n_boundary", "n_qr"}
        keys = set(payload) - method_specific
        if shared is None:
            shared = keys
        else:
            assert keys == shared, f"{script} 的共用 schema 與其他兩支不一致"


def test_sensor_coordinates_avoid_body(monkeypatch):
    """placement 不得選到 body 格點——這是三支共同的正確性前提。"""
    with tempfile.TemporaryDirectory() as td:
        gen_dir = Path(td) / "gen"
        gen_dir.mkdir()
        (gen_dir / "generate_sensors_qrpivot_cylinder.py").write_text(STUB)
        out_dir = Path(td) / "out"
        _run("gen_sensors_uniform.py", ["--K", "20"], gen_dir, out_dir, monkeypatch)
        payload = json.loads(next(out_dir.glob("*.json")).read_text())

    for i, j in zip(payload["sensor_i"], payload["sensor_j"]):
        assert not (5 <= i < 7 and 7 <= j < 9), f"sensor ({i},{j}) 落在 body 內"
