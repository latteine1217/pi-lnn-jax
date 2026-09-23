"""Characterization tests for typed effective configuration resolution."""
from __future__ import annotations

import textwrap

import pytest

from pi_lnn_jax.config import (
    ConfigurationLayer,
    UnknownOverrideError,
    merge_config_layers,
)
from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs as resolve_cylinder
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs as resolve_kolmogorov


def _write(tmp_path, body: str):
    path = tmp_path / "c.toml"
    path.write_text(textwrap.dedent(body))
    return path


def _resolve(tmp_path, body: str, *cli: str):
    path = _write(tmp_path, body)
    return resolve_kolmogorov(["--config", str(path), *cli])


_BASE = """
    [train]
    iterations = 100
    seed = 7
"""


def test_cli_none_falls_through_to_toml(tmp_path):
    result = _resolve(tmp_path, _BASE)
    assert result.config.run.steps == 100
    assert result.config.run.seed == 7
    assert result.provenance.for_field("run.steps").source == "toml"


def test_cli_explicit_beats_toml_and_records_shadow_chain(tmp_path):
    result = _resolve(tmp_path, _BASE, "--steps", "3", "--seed", "99")
    assert result.config.run.steps == 3
    field = result.provenance.for_field("run.steps")
    assert field.source == "cli"
    assert [(entry.layer, entry.value) for entry in field.chain] == [
        ("schema_default", 5000), ("toml", 100), ("cli", 3),
    ]


def test_missing_keys_keep_historical_second_defaults(tmp_path):
    config = _resolve(tmp_path, _BASE).config
    assert config.loss.gradnorm_freq == 1000
    assert config.loss.al_update_freq == 100
    assert config.loss.al_warmup_steps == 0
    assert config.loss.t_early_threshold == 0.05


def test_store_true_flag_ors_with_toml_bool(tmp_path):
    assert _resolve(tmp_path, _BASE).config.loss.use_gradnorm is False
    assert _resolve(tmp_path, _BASE, "--use_gradnorm").config.loss.use_gradnorm is True
    assert _resolve(
        tmp_path, _BASE + "    use_gradnorm = true\n"
    ).config.loss.use_gradnorm is True


def test_flatten_order_and_raw_container_types_are_preserved(tmp_path):
    result = _resolve(tmp_path, """
        [a]
        t_early_weight = 1.5
        [b]
        t_early_weight = 9.5
        sensor_channel_weights = [1, 2, 3]
    """)
    assert result.config.loss.t_early_weight == 9.5
    assert type(result.config.loss.sensor_channel_weights) is list
    assert result.config.loss.sensor_channel_weights == [1, 2, 3]


def test_schema_float_coercion_remains_visible_in_effective_config(tmp_path):
    config = _resolve(tmp_path, _BASE + "    data_loss_weight = 1\n").config
    assert type(config.loss.data_weight) is float
    assert config.loss.data_weight == 1.0


def test_layer_engine_defensively_copies_lists():
    supplied = [1, 2]
    values, _ = merge_config_layers(
        [ConfigurationLayer("schema_default", {"x": supplied})],
        known_fields=frozenset({"x"}),
    )
    supplied.append(3)
    assert values["x"] == [1, 2]
    assert type(values["x"]) is list


def test_provenance_values_do_not_alias_effective_values():
    values, provenance = merge_config_layers(
        [ConfigurationLayer("schema_default", {"x": [1, 2]})],
        known_fields=frozenset({"x"}),
    )
    values["x"].append(3)
    assert provenance["x"].chain[-1].value == [1, 2]


def test_future_strict_override_rejects_unknown_field():
    with pytest.raises(UnknownOverrideError, match="unknown fields"):
        merge_config_layers(
            [
                ConfigurationLayer("schema_default", {"x": 1}),
                ConfigurationLayer("explicit_override", {"typo": 2}, reject_unknown=True),
            ],
            known_fields=frozenset({"x"}),
        )


def test_cylinder_guards_run_before_loading_toml(capsys):
    with pytest.raises(SystemExit) as error:
        resolve_cylinder(["--config", "/does/not/exist.toml", "--resume", "x"])
    assert error.value.code == 4
    assert "禁止 resume" in capsys.readouterr().err


def test_cylinder_cli_override_and_derived_warmup(tmp_path):
    path = _write(tmp_path, """
        [train]
        case = "cylinder"
        iterations = 100
        time_marching_warmup = 0.25
        [data]
        cylinder_data_npz = "missing.npz"
    """)
    result = resolve_cylinder([
        "--config", str(path), "--allow-cpu", "--steps", "20", "--seed", "9",
    ])
    assert result.config.run.steps == 20
    assert result.config.run.seed == 9
    assert result.config.time_marching_warmup_steps == 5
    assert result.provenance.for_field("run.steps").source == "cli"


# ── physics loss schedule：曾被錯貼「POC 尚未實作」標籤 ────────────────────
#
# 這兩個鍵的接線一直是完整的（typed 欄位、policy 預設、TOML 映射都在），
# 卻同時掛在 POC_NOT_YET_KEYS 裡，於是每次使用都收到一句
# 「POC 尚未實作 keys（GPU 階段補）」——警告主動勸阻了一個能用的功能。
# 2026-08-04 移入 TRAIN_SCHEMA，整個 POC_NOT_YET 機制隨之移除（該集合已空）。


@pytest.mark.parametrize("key,path,value", [
    ("physics_loss_warmup_steps", "loss.physics_warmup_steps", 5000),
    ("physics_loss_ramp_steps", "loss.physics_ramp_steps", 1200),
])
def test_physics_loss_schedule_is_settable_from_toml(tmp_path, key, path, value):
    """設得了、值對、來源歸因正確——三者缺一，這個鍵就還是名存實亡。"""
    result = _resolve(tmp_path, _BASE + f"{key} = {value}\n")

    group, field = path.split(".")
    assert getattr(getattr(result.config, group), field) == value
    assert result.provenance.for_field(path).source == "toml"


@pytest.mark.parametrize("key", ["physics_loss_warmup_steps", "physics_loss_ramp_steps"])
def test_physics_loss_schedule_does_not_warn(tmp_path, key, recwarn):
    """假警告不得回來。

    先前實測：設 physics_loss_warmup_steps=5000 會警告「POC 尚未實作」，
    而 config.loss.physics_warmup_steps 確實是 5000——警告與行為互相矛盾。
    """
    _resolve(tmp_path, _BASE + f"{key} = 3000\n")

    bad = [str(w.message) for w in recwarn
           if "尚未實作" in str(w.message) or key in str(w.message)]
    assert not bad, f"{key} 又被當成未實作／未認識的鍵：{bad}"


def test_the_not_yet_mechanism_is_gone(tmp_path):
    """自證：整個 POC_NOT_YET 機制已移除，不是只把兩個鍵搬走。

    留著一個空集合與它的警告分支，下一個人還是會往裡面加鍵——
    而那個分支的語意（「已知但未實作」）對本 repo 的每一個歷史案例都是錯的：
    兩個是接線完整的、一個是 CLI 實作的、五個是真死鍵。
    """
    import pi_lnn_jax.config as C

    assert not hasattr(C, "POC_NOT_YET_KEYS"), "POC_NOT_YET_KEYS 又回來了"
    assert "not_yet_toml_keys" not in {
        f.name for f in __import__("dataclasses").fields(C.ConfigProvenance)
    }, "provenance 仍帶 not_yet_toml_keys——機制沒清乾淨"
