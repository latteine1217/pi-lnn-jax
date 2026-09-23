"""Every typed effective-config field used as a divisor has a zero guard."""
from __future__ import annotations

import ast

import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.config import TRAIN_SCHEMA

_PIPELINE = REPO_ROOT / "pi_lnn_jax" / "pipeline"
_CONFIG_LOCALS = {"run", "loss", "curriculum", "refinement"}


def _config_field(node: ast.AST) -> str | None:
    """Return the leaf name for direct typed config access such as `loss.gradnorm_freq`."""
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in _CONFIG_LOCALS):
        return node.attr
    return None


def _divisor_uses() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(_PIPELINE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, (ast.Mod, ast.Div, ast.FloorDiv))):
                continue
            field = _config_field(node.right)
            if field:
                out.setdefault(field, []).append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                )
    return out


def _short_circuited(field: str) -> bool:
    for path in sorted(_PIPELINE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)):
                continue
            guards = any(
                isinstance(value, ast.Compare)
                and _config_field(value.left) == field
                and any(isinstance(op, (ast.Gt, ast.GtE, ast.NotEq)) for op in value.ops)
                for value in node.values
            )
            divides = any(
                isinstance(child, ast.BinOp)
                and isinstance(child.op, (ast.Mod, ast.Div, ast.FloorDiv))
                and _config_field(child.right) == field
                for value in node.values for child in ast.walk(value)
            )
            if guards and divides:
                return True
    return False


_SCHEMA_KEYS = {
    "gradnorm_freq": "gradnorm_freq",
    "al_update_freq": "al_update_freq",
    "save_every": "checkpoint_period",
}


def _schema_rejects_zero(field: str) -> bool:
    key = _SCHEMA_KEYS.get(field)
    if key is None:
        return False
    validator = TRAIN_SCHEMA[key][2]
    if validator is None:
        return False
    try:
        validator(0)
    except Exception:
        return True
    return False


def _clamped_at_resolution(field: str) -> bool:
    if field != "log_every":
        return False
    source = (_PIPELINE / "kolmogorov" / "config.py").read_text()
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "max" and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == 1
        for node in ast.walk(tree)
    )


DIVISOR_USES = _divisor_uses()


def test_scan_finds_the_known_divisor_fields():
    assert DIVISOR_USES, "掃不到任何 typed config divisor——掃描器壞了"
    for expected in ("gradnorm_freq", "al_update_freq", "save_every", "eval_every", "log_every"):
        assert expected in DIVISOR_USES, f"已知除數欄位 {expected} 沒被掃到"


@pytest.mark.parametrize("field", sorted(DIVISOR_USES), ids=sorted(DIVISOR_USES))
def test_divisor_field_has_a_protection(field):
    protections = {
        "呼叫端短路": _short_circuited(field),
        "schema 拒絕 0": _schema_rejects_zero(field),
        "resolution clamp": _clamped_at_resolution(field),
    }
    assert any(protections.values()), (
        f"typed field {field!r} 被當除數用於 {DIVISOR_USES[field]}，但沒有 zero guard:\n"
        + "\n".join(f"    {name}: {ok}" for name, ok in protections.items())
    )
