"""oracle 監督（`supervision_sensor_json`）：encoder 吃保留幀，監督目標換成完整序列。

Why 這組測試存在：
    間歇重建的診斷臂要回答「gap 內沒有資料監督」是不是模型用不上下一幀的原因
    （knowledge/experiments/kolmogorov-cfc-isolation-2026-10.md 的 oracle 2×2）。對照要成立需要：

      * 未設定時 dataset 與 ReBatch 與舊版相同（不產生覆寫欄位）。
      * 設定時 encoder 輸入、norm_stats 與不開 oracle 完全相同；監督目標是完整序列，
        且以 encoder 檔的 norm_stats 正規化。
      * 兩檔對不齊（位置、時刻、值、缺 pre_stride）一律報錯，不默默拿錯的目標去監督。
      * 與會改動時間軸的其他設定同時開啟時拒收。

    預設路徑逐位元不變的最終背書是 lab-server 的 CPU A/B 對拍（CLAUDE.md §7.1）。
"""
from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from pi_lnn_jax.data import load_sensors_from_path
from pi_lnn_jax.pipeline.kolmogorov import assembly
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs

K, T_FULL = 6, 21
KEEP = [2, 3, 7, 12, 13, 20]          # 間歇保留幀（完整序列的子集）


def _write_set(tmp_path, name, time, u, v, pos, pre_stride=None):
    npz = tmp_path / f"{name}_dns_values.npz"
    np.savez(npz, time=time.astype(np.float32), u=u.astype(np.float32), v=v.astype(np.float32))
    meta = {"K": K, "selected_coordinates": pos.tolist(), "dns_values_npz": npz.name}
    if pre_stride is not None:
        meta["pre_stride"] = pre_stride
    js = tmp_path / f"{name}.json"
    js.write_text(json.dumps(meta))
    return js


def _files(tmp_path, *, pos_shift=0.0, time_shift=0.0, val_shift=0.0, pre_stride=1):
    rng = np.random.default_rng(0)
    pos = rng.uniform(size=(K, 2))
    t = np.linspace(0.0, 1.0, T_FULL)
    u, v = rng.normal(size=(K, T_FULL)), rng.normal(size=(K, T_FULL)) + 0.5
    full = _write_set(tmp_path, "full", t, u, v, pos)
    gu, gv = u[:, KEEP].copy(), v[:, KEEP].copy()
    gu[0, 0] += val_shift
    gap = _write_set(tmp_path, "gap", t[KEEP] + time_shift, gu, gv, pos + pos_shift,
                     pre_stride=pre_stride)
    return gap, full


def _dataset(gap):
    d = load_sensors_from_path(gap, time_stride=1)
    d["re_value"], d["re_norm"] = 1.0e4, 1.0
    return d


def test_unset_dataset_and_rebatch_are_unchanged(tmp_path):
    gap, _ = _files(tmp_path)
    d = _dataset(gap)
    assert "sup_vals" not in d
    rb = assembly._make_re_batch(d)
    assert rb.enc_vals is None and rb.enc_time is None


def test_oracle_routes_encoder_and_target(tmp_path):
    gap, full = _files(tmp_path)
    d = _dataset(gap)
    da = assembly._attach_supervision(d, str(full))
    rb = assembly._make_re_batch(da)
    enc_vals, enc_time = assembly.encoder_context(rb)
    # encoder 輸入與 norm_stats 和不開 oracle 時相同
    np.testing.assert_array_equal(np.asarray(enc_vals), d["sensor_vals"])
    np.testing.assert_array_equal(np.asarray(enc_time), d["sensor_time"])
    assert da["norm_stats"] == d["norm_stats"]
    # 監督目標是完整序列，保留幀上的值與 encoder 輸入一致（以 encoder 檔的 norm_stats 正規化）
    assert rb.sensor_vals.shape == (T_FULL, K, 2)
    np.testing.assert_allclose(np.asarray(rb.sensor_vals)[KEEP], d["sensor_vals"], atol=1e-5)
    # 非保留幀用的是同一組常數：還原成物理量要等於完整檔的原值
    ns = d["norm_stats"]
    raw = np.load(tmp_path / "full_dns_values.npz")
    u_phys = np.asarray(rb.sensor_vals)[..., 0] * ns["u_std"] + ns["u_mean"]
    np.testing.assert_allclose(u_phys.T, raw["u"], atol=1e-5)


@pytest.mark.parametrize("kw, msg", [
    ({"pos_shift": 0.01}, "位置"),
    ({"time_shift": 0.013}, "子集"),
    ({"val_shift": 0.5}, "不一致"),
    ({"pre_stride": None}, "pre_stride"),
])
def test_misaligned_files_fail_fast(tmp_path, kw, msg):
    gap, full = _files(tmp_path, **kw)
    with pytest.raises(ValueError, match=msg):
        assembly._attach_supervision(_dataset(gap), str(full))


def test_oracle_rejects_stride_override(tmp_path):
    gap, full = _files(tmp_path)
    da = assembly._attach_supervision(_dataset(gap), str(full))
    with pytest.raises(ValueError, match="supervision_sensor_json"):
        assembly._make_re_batch(da, encoder_time_stride=2)


_GUARD = dict(supervision_sensor_json="x.json", encoder_time_stride=1, target_time_stride=1,
              arch="liquid", n_datasets=1, sensor_cut_min_frac=0.0, autoreg_pseudo_dt=0.0,
              sensor_dropout_rate=0.0, use_continuous_re_physics=False, refine_optimizer="none")


def test_guard_passes_when_unset_or_clean():
    assembly.check_supervision_oracle(**{**_GUARD, "supervision_sensor_json": "",
                                         "sensor_dropout_rate": 0.5})
    assembly.check_supervision_oracle(**_GUARD)


@pytest.mark.parametrize("override", [
    {"encoder_time_stride": 2}, {"arch": "vanilla"}, {"n_datasets": 2},
    {"sensor_cut_min_frac": 0.5}, {"autoreg_pseudo_dt": 0.05}, {"sensor_dropout_rate": 0.3},
    {"use_continuous_re_physics": True}, {"refine_optimizer": "lbfgs"},
])
def test_guard_rejects_unverified_combos(override):
    with pytest.raises(ValueError, match="supervision_sensor_json"):
        assembly.check_supervision_oracle(**{**_GUARD, **override})


def test_config_default_and_toml(tmp_path):
    base = "[train]\niterations = 10\nseed = 1\n"
    p0 = tmp_path / "a.toml"
    p0.write_text(base)
    assert resolve_inputs(["--config", str(p0)]).config.data.supervision_sensor_json == ""
    p1 = tmp_path / "b.toml"
    p1.write_text(base + textwrap.dedent('supervision_sensor_json = "data/x.json"\n'))
    r = resolve_inputs(["--config", str(p1)])
    assert r.config.data.supervision_sensor_json == "data/x.json"
    assert r.provenance.for_field("data.supervision_sensor_json").source == "toml"
