# tests/test_continuous_re_physics.py
"""連續-Re 物理正則的 config 旗標：對齊既有 causal_eps 合約
（load_config 不灌 train 預設；schema 註冊只保證「不被當 unknown 丟棄」）。"""
from pi_lnn_jax.config import load_config


def test_crp_flags_absent_not_in_train_kwargs(tmp_path):
    # 未提供時不灌入 train_kwargs（由 train script merge 補預設，對齊 causal_eps 慣例）
    p = tmp_path / "c.toml"
    p.write_text('[train]\niterations=10\n')
    r = load_config(p)
    assert "use_continuous_re_physics" not in r["train_kwargs"]
    assert "continuous_re_physics_weight" not in r["train_kwargs"]


def test_crp_flags_parse_and_recognized(tmp_path):
    # 提供時：被 schema 認可、型別解析正確、不進 unknown_keys（防 silent-drop）
    p = tmp_path / "c.toml"
    p.write_text('[train]\niterations=10\n'
                 'use_continuous_re_physics=true\ncontinuous_re_physics_weight=0.5\n')
    r = load_config(p)
    tk = r["train_kwargs"]
    assert tk["use_continuous_re_physics"] is True
    assert tk["continuous_re_physics_weight"] == 0.5
    assert "use_continuous_re_physics" not in r["unknown_keys"]
    assert "continuous_re_physics_weight" not in r["unknown_keys"]


# ─── Task 2: _build_crp_interp + _make_re_batch_crp ────────────────────────

import numpy as np
from pi_lnn_jax.pipeline.kolmogorov.assembly import (
    _build_crp_interp,
    _make_re_batch,
)
# per-step 取樣屬執行期 → _make_re_batch_crp 住在 kolmogorov/run.py（Task 6）
from pi_lnn_jax.pipeline.kolmogorov.run import _make_re_batch_crp


def _ds(re, u_std):
    K, T = 4, 3
    return {
        "sensor_vals": np.zeros((T, K, 2), np.float32),
        "sensor_pos": np.zeros((K, 2), np.float32),
        "sensor_time": np.linspace(0, 1, T).astype(np.float32),
        "re_value": float(re),
        "re_norm": float(np.log(re) / np.log(1e6)),
        "norm_stats": {"u_mean": 0.0, "u_std": u_std, "v_mean": 0.0,
                       "v_std": u_std, "p_mean": 0.0, "p_std": 1.0},
    }


def test_crp_interp_node_consistency():
    ds = [_ds(1000, 0.3), _ds(10000, 0.45), _ds(1000000, 0.46)]
    interp = _build_crp_interp(ds)
    active = _make_re_batch(ds[0])
    rn = float(np.log(10000) / np.log(1e6))
    rb = _make_re_batch_crp(active, rn, interp, re_norm_scale=1e6)
    assert abs(float(rb.u_std) - 0.45) < 1e-5
    assert abs(float(rb.re_norm) - rn) < 1e-6
    assert abs(float(rb.nu) - 1.0 / 10000.0) < 1e-3
    assert rb.sensor_vals.shape == active.sensor_vals.shape


def test_crp_interp_midpoint():
    ds = [_ds(1000, 0.3), _ds(1000000, 0.5)]
    interp = _build_crp_interp(ds)
    active = _make_re_batch(ds[0])
    rn_mid = 0.5 * (interp["re_norm_min"] + interp["re_norm_max"])
    rb = _make_re_batch_crp(active, rn_mid, interp, re_norm_scale=1e6)
    assert 0.3 < float(rb.u_std) < 0.5
