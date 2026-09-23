"""評估腳本不得自己把 metric 落盤 —— 從 AST 推導，不看拼法。

問題（架構審查候選 2）：`pi_lnn_jax/metric_artifact.py` 是評估結果的正典記錄，
而 `evaluate_field_series` 這道 deep seam 一度**零 production 呼叫端**——五支
baseline 評估器各自跑 per-t 迴圈、各自聚合、各自 `json.dump`，metric 的語意標記
（`ke_mape_def`）由人手寫進字串。`compare_baselines.py` 再把兩種 schema 合成
論文的四方對照表。

後果不是抽象的：`ke_t_mape` 這個鍵在不同產出端指不同的量（見 scripts/CLAUDE.md §6），
而兩支消費端曾因此壞掉。語意若是可選附註，遲早會有一份產出把它寫錯或漏寫。

本檔掃 AST：`scripts/` 下任何檔案把 `json.dump` / `json.dumps` / `write_text` 的
結果寫進**評估產物**，就是繞過 seam。判準是「有沒有呼叫 json.dump(s)」加上該檔
是否屬於評估家族——後者用硬編碼清單，理由見下。

**豁免必須是硬編碼的字面清單。** 若豁免條件由被檢查的東西自己構成（例如
「檔名含 diag 就跳過」），守衛就會隨新增檔案自動失效。
"""
from __future__ import annotations

import ast
import pathlib

import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

#: 產出進論文表格 / 圖的評估腳本。這些的結果必須經 metric artifact seam。
#: 硬編碼：新增評估腳本時要有人決定它屬不屬於這一族，不靠檔名猜。
METRIC_PRODUCERS = frozenset({
    "evaluate_exp245.py",
    "evaluate_multi_re.py",
    "evaluate_baselines.py",
    "classical_baselines_fair.py",
    "eval_gappy_cross_re.py",
    "train_baseline_shred.py",
    "cost_accuracy.py",
})

#: 落盤只能經這兩個入口——兩者都在 metric_artifact.py，且都 atomic + allow_nan=False。
SANCTIONED_WRITERS = frozenset({
    "write_metric_artifact", "write_compatibility_projection",
})

EXEMPT: dict[str, str] = {}


def _json_dump_calls(tree: ast.AST) -> list[int]:
    """`json.dump(...)` / `json.dumps(...)` 的行號。"""
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr in {"dump", "dumps"}
                and isinstance(func.value, ast.Name) and func.value.id == "json"):
            out.append(node.lineno)
    return out


def _called_names(tree: ast.AST) -> set[str]:
    return {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _producers() -> list[pathlib.Path]:
    return [p for p in sorted(_SCRIPTS.rglob("*.py"))
            if "legacy" not in p.parts and p.name in METRIC_PRODUCERS]


PRODUCERS = _producers()


def test_every_named_producer_exists():
    """自證：清單裡的檔案若被改名或刪除，這條會紅而不是靜默少驗一支。"""
    found = {p.name for p in PRODUCERS}
    assert found == set(METRIC_PRODUCERS), (
        f"清單與磁碟不符——只找到 {sorted(found)}；"
        f"缺 {sorted(set(METRIC_PRODUCERS) - found)}")


def test_the_guard_can_actually_fire():
    """鑑別力自檢：對一段確實手寫落盤的程式碼，掃描器必須抓到。

    少了這條，`_json_dump_calls` 哪天因 AST 結構變動而永遠回空，
    下面的參數化測試會全綠而什麼都沒驗。
    """
    tree = ast.parse("import json\njson.dump(metrics, open(p, 'w'))\n")
    assert _json_dump_calls(tree) == [2]
    assert _json_dump_calls(ast.parse("json.dumps(d)\n")) == [1]


@pytest.mark.parametrize("path", PRODUCERS, ids=lambda p: p.name)
def test_producer_writes_through_the_artifact_seam(path):
    tree = ast.parse(path.read_text())
    hits = _json_dump_calls(tree)

    if path.name in EXEMPT:
        assert hits, (
            f"{path.name} 列在 EXEMPT 卻沒有手寫落盤——豁免已過期，請移除。"
            f"（理由當初是：{EXEMPT[path.name]}）")
        return

    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} 在第 {hits} 行手寫 json.dump——"
        "評估產物必須經 metric artifact seam。\n"
        "  改用 metric_artifact.write_metric_artifact（正典記錄）與 "
        "write_compatibility_projection（既有 schema 的投影）。\n"
        "  Why：手寫落盤的那一刻，metric 的語意就變成可選附註——"
        "ke_mape_def 寫錯或漏寫都不會有人發現（scripts/CLAUDE.md §6）。")


@pytest.mark.parametrize("path", PRODUCERS, ids=lambda p: p.name)
def test_producer_actually_calls_a_sanctioned_writer(path):
    """反向斷言：不寫 json.dump 也可能是「根本沒落盤」。

    只驗「沒有手寫」會讓一支忘了寫結果的腳本靜默通過。
    """
    called = _called_names(tree=ast.parse(path.read_text()))
    assert called & SANCTIONED_WRITERS, (
        f"{path.name} 沒有呼叫任何合規的落盤入口"
        f"（{sorted(SANCTIONED_WRITERS)}）——它真的有把結果寫出去嗎？")


def test_exemptions_are_a_literal_list_not_derived():
    """豁免若由被檢查的東西自己構成，守衛就恆真。"""
    assert isinstance(EXEMPT, dict)
    assert all(isinstance(k, str) and k.endswith(".py") for k in EXEMPT)
    assert all(isinstance(v, str) and v for v in EXEMPT.values()), "每個豁免都要有理由"
    assert len(EXEMPT) <= 2, (
        f"豁免長到 {len(EXEMPT)} 個——那不再是例外，是這條規則沒人在遵守")
