"""驗 config loader 可吃 pi-lnn 既有 EXP-030 TOML 而不爆。

判定 PASS：
  1. 可成功 load
  2. model_kwargs 含必要 keys + 對齊 EXP-030 值
  3. train_kwargs 含 iterations 等
  4. POC_NOT_YET_KEYS 觸發 warning 但不 raise
  5. unknown_keys 為空（pi-lnn 既有 TOML 應全 covered）
"""
from __future__ import annotations

from pi_lnn_jax.config import load_config


def test_model_enum_validators_actually_raise():
    """relpos_bias_mode / attention_kind 非法值必須 raise（原本是 silent no-op）。"""
    import pytest
    from pi_lnn_jax.config import _validate_one, MODEL_SCHEMA

    # 合法值通過
    assert _validate_one("relpos_bias_mode", "radial", MODEL_SCHEMA) == "radial"
    assert _validate_one("relpos_bias_mode", "vector", MODEL_SCHEMA) == "vector"
    assert _validate_one("attention_kind", "scalar", MODEL_SCHEMA) == "scalar"
    assert _validate_one("attention_kind", "vector", MODEL_SCHEMA) == "vector"
    # 非法值 raise
    with pytest.raises(ValueError):
        _validate_one("relpos_bias_mode", "bogus", MODEL_SCHEMA)
    with pytest.raises(ValueError):
        _validate_one("attention_kind", "bogus", MODEL_SCHEMA)
    print("✓ model_enum_validators_actually_raise")


def test_train_range_validators_actually_raise():
    """num_sensor_query_points / gradnorm_ema_momentum / t_early_threshold 邊界驗證生效。"""
    import pytest
    from pi_lnn_jax.config import _validate_one, TRAIN_SCHEMA

    # 合法值（含現有 config 實際使用值）通過
    assert _validate_one("num_sensor_query_points", 0, TRAIN_SCHEMA) == 0
    assert _validate_one("num_sensor_query_points", 2000, TRAIN_SCHEMA) == 2000
    assert _validate_one("gradnorm_ema_momentum", 0.9, TRAIN_SCHEMA) == 0.9
    assert _validate_one("t_early_threshold", 0.05, TRAIN_SCHEMA) == 0.05

    # 非法值 raise
    with pytest.raises(ValueError):
        _validate_one("num_sensor_query_points", -1, TRAIN_SCHEMA)
    with pytest.raises(ValueError):
        _validate_one("gradnorm_ema_momentum", 1.0, TRAIN_SCHEMA)   # [0,1) 排除 1.0
    with pytest.raises(ValueError):
        _validate_one("t_early_threshold", 0.0, TRAIN_SCHEMA)       # (0,1) 排除 0.0
    with pytest.raises(ValueError):
        _validate_one("t_early_threshold", 1.0, TRAIN_SCHEMA)       # (0,1) 排除 1.0
    print("✓ train_range_validators_actually_raise")


def test_al_constraint_mode_validator():
    """AL constraint mode 必須明確枚舉，避免 typo 靜默回退成舊行為。"""
    import pytest
    from pi_lnn_jax.config import _validate_one, TRAIN_SCHEMA

    assert _validate_one("al_constraint_mode", "mse", TRAIN_SCHEMA) == "mse"
    assert _validate_one("al_constraint_mode", "signed_mean", TRAIN_SCHEMA) == "signed_mean"
    with pytest.raises(ValueError):
        _validate_one("al_constraint_mode", "signed", TRAIN_SCHEMA)
    print("✓ al_constraint_mode_validator")


def test_validate_one_rejects_tuple_return_validator():
    """防呆 guard：validator 回傳非 None（如 tuple-return lambda）必須被擋下。"""
    import pytest
    from pi_lnn_jax.config import _validate_one

    fake_schema = {
        "bad_key": (str, "x", lambda v: (v == "x", "must be x")),  # 經典 no-op 寫法
    }
    with pytest.raises(RuntimeError):
        _validate_one("bad_key", "x", fake_schema)
    print("✓ validate_one_rejects_tuple_return_validator")


def test_uvp_channels_derive_sensor_value_dim_3(tmp_path):
    # load_config 已在檔頂 import；回傳 dict 含 model_kwargs/train_kwargs/data_kwargs
    toml = tmp_path / "c.toml"
    toml.write_text(
        '[train]\n'
        'sensor_jsons = ["x.json"]\n'
        're_values = [10000.0]\n'
        'observed_sensor_channels = ["u", "v", "p"]\n'
    )
    cfg = load_config(str(toml))
    assert cfg["model_kwargs"]["sensor_value_dim"] == 3


def test_existing_configs_still_load():
    """Never Break Userspace：所有現有 configs/*.toml 啟用真實驗證後仍可 load。"""
    import glob
    import warnings as _warnings
    from pi_lnn_jax.config import load_config

    tomls = sorted(glob.glob("configs/*.toml"))
    assert tomls, "找不到 configs/*.toml"
    for p in tomls:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # 忽略 unknown/not_yet warning，只驗不 raise
            load_config(p)
    print(f"✓ existing_configs_still_load: {len(tomls)} configs OK")
