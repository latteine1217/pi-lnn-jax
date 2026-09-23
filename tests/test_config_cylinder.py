"""Cylinder case discriminator + keys route through load_config (spec 2026-06-10)."""
from pi_lnn_jax.config import load_config, TRAIN_SCHEMA


def test_case_default_is_kolmogorov():
    assert TRAIN_SCHEMA["case"][1] == "kolmogorov"


def test_cylinder_keys_recognized(tmp_path):
    toml = tmp_path / "cyl.toml"
    toml.write_text(
        'case = "cylinder"\n'
        'cylinder_data_npz = "data/cylinder_v1.npz"\n'
        'bc_weight = 0.1\n'
        'bc_body_weight = 2.0\n'
        'bc_n = 128\n'
        'use_physics_denormalization = true\n'
    )
    cfg = load_config(str(toml))
    assert cfg["train_kwargs"]["case"] == "cylinder"
    assert cfg["train_kwargs"]["bc_weight"] == 0.1
    assert cfg["train_kwargs"]["bc_n"] == 128
    assert cfg["train_kwargs"]["use_physics_denormalization"] is True
    assert cfg["data_kwargs"]["cylinder_data_npz"] == "data/cylinder_v1.npz"
    assert cfg["unknown_keys"] == []   # 所有 cylinder key 都被 schema 認得
