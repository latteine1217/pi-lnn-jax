"""baseline 家族的評估協定：模式必填、取樣不得有預設值。

本檔的前身釘的是**分歧**：五支腳本各自寫 `--sensor-time-stride` / `--sensor-T`
的預設值，`classical_baselines_fair.py` 是 `1 / 200`、另外四支是 `4 / 50`，
而它們的數字並排進同一張論文對照表。那份紀錄的用途是「讓分歧有名字，從此無法
在 code review 中隱形，也無法被靜靜修掉」。

分歧已於 2026-08-06 解決，修法不是把數值統一——那不可實作：82 份 config 的
`time_strides` 有 7 種取值，正確的 stride 取決於該 sensor set 的原生 cadence。
改成統一**來源**：`pi_lnn_jax.evaluation_protocol` 由訓練 config 決定取樣，
呼叫端必須明示協定模式。

本檔因此改釘新契約：

1. 每支都宣告 `--protocol` 且 `required=True`、**無預設**。有預設就會有人吃到
   錯的那一個，而錯的不會 crash，只會在別的時間解析度上算數字（TD-4 的失效形狀，
   實測已發生於 12 份既有產物）。
2. `--sensor-time-stride` 的預設必須是 `None`——「不給」表示由協定決定，而不是
   悄悄套一個數字。
"""
from __future__ import annotations

import ast
import pathlib

import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

#: 數字並排進同一張 baseline 對照表的五支。
FAMILY = (
    "evaluate_baselines.py",
    "eval_gappy_cross_re.py",
    "cost_accuracy.py",
    "train_baseline_shred.py",
    "classical_baselines_fair.py",
)

#: 三種模式；與 `evaluation_protocol.ProtocolMode` 一致。硬編碼字面值——
#: 從被檢查之物推導出來的期望值恆真。
EXPECTED_MODES = ("follow_training", "fixed_grid", "sensor_time_independent")


def _add_argument_calls(path: pathlib.Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            flag = next((a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("--")), None)
            if flag is not None:
                yield flag, {kw.arg: kw.value for kw in node.keywords}


def _spec(path: pathlib.Path) -> dict[str, dict]:
    return {flag: kws for flag, kws in _add_argument_calls(path)}


SPECS = {f: _spec(_SCRIPTS / f) for f in FAMILY}


def test_every_family_member_is_scanned():
    """自證：掃不到 add_argument 就代表下面每一條都空過。"""
    for name, spec in SPECS.items():
        assert len(spec) > 3, f"{name} 只掃到 {len(spec)} 個旗標——掃描器壞了"


@pytest.mark.parametrize("name", FAMILY)
def test_protocol_mode_is_required_and_has_no_default(name):
    spec = SPECS[name]
    assert "--protocol" in spec, (
        f"{name} 未宣告 --protocol。評估協定必須由呼叫端明示：\n"
        "  follow_training（跟隨訓練 time_strides）／fixed_grid（固定格點，須說明理由）／\n"
        "  sensor_time_independent（不對齊，thesis §7.2 間歇評估）")
    kws = spec["--protocol"]
    assert "default" not in kws, (
        f"{name} 的 --protocol 有預設值。這正是 TD-4 的失效形狀——"
        "預設會讓人吃到錯的那一個，而錯的不會 crash。")
    assert "required" in kws and ast.literal_eval(kws["required"]) is True, (
        f"{name} 的 --protocol 不是 required")


@pytest.mark.parametrize("name", FAMILY)
def test_protocol_choices_are_the_three_modes(name):
    kws = SPECS[name]["--protocol"]
    assert "choices" in kws, f"{name} 的 --protocol 未限制 choices——打錯字會變成新模式"
    src = ast.unparse(kws["choices"])
    assert "ProtocolMode" in src, (
        f"{name} 的 choices 未由 ProtocolMode 導出（實際：{src}）——"
        "手抄一份字串清單會與 enum 漂移")


@pytest.mark.parametrize("name", FAMILY)
def test_sampling_flags_have_no_silent_default(name):
    """`--sensor-time-stride` 不給 ⇒ 由協定決定，而不是悄悄套一個數字。"""
    kws = SPECS[name].get("--sensor-time-stride")
    assert kws is not None, f"{name} 未宣告 --sensor-time-stride"
    assert "default" in kws, f"{name} 的 --sensor-time-stride 未寫 default"
    assert ast.literal_eval(kws["default"]) is None, (
        f"{name} 的 --sensor-time-stride 預設不是 None"
        f"（實際 {ast.unparse(kws['default'])}）。有數值預設就會有人在沒發覺的"
        "情況下用它評估——那正是本檔前身記錄的分歧。")


def test_the_modes_match_the_enum():
    """反向斷言：enum 加了新模式而本檔沒跟上時要紅。"""
    from pi_lnn_jax.evaluation_protocol import ProtocolMode

    assert tuple(m.value for m in ProtocolMode) == EXPECTED_MODES, (
        "ProtocolMode 的成員已改變。新模式代表一種新的評估協定——"
        "更新本檔，並確認 scripts/CLAUDE.md 與 ADR 有記下它為何存在。")


def test_the_old_divergence_is_gone():
    """反向斷言：舊分歧若復活（有人把數值預設加回來）要紅。

    沒有這條，本檔就只是描述現況，而不是守住它。
    """
    for name in FAMILY:
        stride = SPECS[name].get("--sensor-time-stride", {}).get("default")
        assert stride is not None and ast.literal_eval(stride) is None, name
