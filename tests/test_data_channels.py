"""load_sensors_from_path 多 channel 行為 + 向後相容回歸。"""
from __future__ import annotations

import json
import numpy as np
import pytest

from pi_lnn_jax.data import load_sensors_from_path


def _make_fixture(tmp_path, channels):
    K, T = 4, 6
    rng = np.random.RandomState(0)
    arrays = {"time": np.linspace(0.0, 1.0, T)}
    for c in channels:
        arrays[c] = rng.randn(K, T).astype(np.float32)
    npz_name = "fx_values.npz"
    np.savez(tmp_path / npz_name, **arrays)
    meta = {
        "K": K,
        "resolution": 16,
        "selected_coordinates": [[float(i), float(i)] for i in range(K)],
        "dns_values_npz": npz_name,
        "channels": channels,
    }
    json_path = tmp_path / "fx.json"
    json.dump(meta, open(json_path, "w"))
    return json_path


def test_uvp_three_channels(tmp_path):
    jp = _make_fixture(tmp_path, ["u", "v", "p"])
    d = load_sensors_from_path(jp, time_stride=1)
    assert d["sensor_vals"].shape[-1] == 3
    for key in ("u_mean", "u_std", "v_mean", "v_std", "p_mean", "p_std"):
        assert key in d["norm_stats"], key


def test_default_uv_backward_compat(tmp_path):
    # 無 channels key → 預設 [u, v]，2-channel，無 p_mean
    jp = _make_fixture(tmp_path, ["u", "v"])
    meta = json.load(open(jp))
    del meta["channels"]
    json.dump(meta, open(jp, "w"))
    d = load_sensors_from_path(jp, time_stride=1)
    assert d["sensor_vals"].shape[-1] == 2
    assert "p_mean" not in d["norm_stats"]


def test_bad_channel_order_raises(tmp_path):
    jp = _make_fixture(tmp_path, ["v", "u"])  # 順序錯，違反 decoder (u,v,p) 輸出順序
    with pytest.raises(ValueError, match="開頭"):
        load_sensors_from_path(jp, time_stride=1)


def test_missing_channel_raises(tmp_path):
    # meta 宣告含 p，但 npz 實際沒寫 p
    jp = _make_fixture(tmp_path, ["u", "v", "p"])
    meta = json.load(open(jp))
    import numpy as _np
    # 重存一份缺 p 的 npz（沿用 meta 指向的檔名）
    npz_path = tmp_path / meta["dns_values_npz"]
    K, T = 4, 6
    _rng = _np.random.RandomState(2)
    _np.savez(npz_path, u=_rng.randn(K, T).astype(_np.float32),
              v=_rng.randn(K, T).astype(_np.float32),
              time=_np.linspace(0.0, 1.0, T))
    with pytest.raises(FileNotFoundError, match="缺 channel"):
        load_sensors_from_path(jp, time_stride=1)


def test_constant_channel_std_zero_raises(tmp_path):
    # 某 channel 為常數場 → std≈0 → 應 Fail Fast
    import numpy as _np
    K, T = 4, 6
    npz_name = "fx_values.npz"
    _np.savez(tmp_path / npz_name,
              u=_np.ones((K, T), dtype=_np.float32),   # 常數
              v=_np.random.RandomState(1).randn(K, T).astype(_np.float32),
              time=_np.linspace(0.0, 1.0, T))
    meta = {
        "K": K, "resolution": 16,
        "selected_coordinates": [[float(i), float(i)] for i in range(K)],
        "dns_values_npz": npz_name, "channels": ["u", "v"],
    }
    jp = tmp_path / "fx.json"
    json.dump(meta, open(jp, "w"))
    with pytest.raises(ValueError, match="std"):
        load_sensors_from_path(jp, time_stride=1)
