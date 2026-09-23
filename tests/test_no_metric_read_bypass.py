"""已遷移的讀側 consumer 不得自己手挑 metric 值 —— 從 AST 推導，不看拼法。

問題（架構審查候選 2，讀側）：producer 寫出的 compatibility projection 一度由每支
consumer 各自 `json.load` + 硬挑鍵讀。`compare_baselines.py` 手刻 `_rows()`／method
taxonomy／`d["metrics_mean"].get(k)`；`aggregate_pv_campaign.py` 也各自 `metrics_mean`
索引。投影鍵一旦漂移，consumer 只會安靜給錯數字。

`pi_lnn_jax/result_table.py` 這道 deep 讀側 seam 把讀取收斂成一處：(method×Re×split)
表、method taxonomy 單源、metric 存取一律經語意層（漂移就大聲失敗）。已遷移的 consumer
必須**走這道 seam**，不得繞回自己 `payload["metrics_mean"][...]` 手挑值。

本檔掃 AST：清單裡的 consumer（1）必須 import `pi_lnn_jax.result_table`，（2）不得出現
`…["metrics_mean"]` 這種手挑 metric 值的下標。判準窄縮到常量鍵 `"metrics_mean"`——
`["metrics_per_t"]`（frame 數，aggregate_pv 合法讀）、`["rows"]`（在 result_table 內）
都不在範圍。

**豁免必須是硬編碼的字面清單。** 若豁免條件由被檢查的東西自己構成，守衛就會隨新增
檔案自動失效。清單只列**已遷移**的 consumer——plot 腳本尚未遷移，不列進來（誠實）。
"""
from __future__ import annotations

import ast
import pathlib

import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

#: 已遷移到 deep 讀側 seam 的 consumer：對 producer 投影建視圖（(method×Re×split) 表 /
#: 跨-run 聚合）的那些。硬編碼：新增遷移 consumer 時要有人把它列進來，不靠檔名猜。
#: 尚未遷移的 plot 腳本刻意不列——把清單保持在「真的已遷移」的範圍才誠實。
MIGRATED_READERS = frozenset({
    "compare_baselines.py",
    "aggregate_pv_campaign.py",
    # EXP-510 的 K-sweep band 聚合：metrics_mean 一律經語意層，per-t 直接讀（不在本守衛範圍）。
    "aggregate_ksweep_bands.py",
})

_RESULT_TABLE_MODULE = "pi_lnn_jax.result_table"

EXEMPT: dict[str, str] = {}


def _imports_result_table(tree: ast.AST) -> bool:
    """該檔是否 `from pi_lnn_jax.result_table import …`。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _RESULT_TABLE_MODULE:
            return True
    return False


def _metrics_mean_subscripts(tree: ast.AST) -> list[int]:
    """`…["metrics_mean"]` 下標的行號（手挑 metric 值的 ad-hoc 讀）。

    只認常量鍵 `"metrics_mean"`——`["metrics_per_t"]`（frame 數）／`["rows"]` 不算。
    """
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and key.value == "metrics_mean":
            out.append(node.lineno)
    return out


def _readers() -> list[pathlib.Path]:
    return [p for p in sorted(_SCRIPTS.rglob("*.py"))
            if "legacy" not in p.parts and p.name in MIGRATED_READERS]


READERS = _readers()


def test_every_named_reader_exists():
    """自證：清單裡的檔案若被改名或刪除，這條會紅而不是靜默少驗一支。"""
    found = {p.name for p in READERS}
    assert found == set(MIGRATED_READERS), (
        f"清單與磁碟不符——只找到 {sorted(found)}；"
        f"缺 {sorted(set(MIGRATED_READERS) - found)}")


def test_the_guard_can_actually_fire():
    """鑑別力自檢：對一段確實手挑 metric 值的程式碼，掃描器必須抓到。

    少了這條，`_metrics_mean_subscripts` 哪天因 AST 結構變動而永遠回空，
    下面的參數化測試會全綠而什麼都沒驗。
    """
    tree = ast.parse('v = payload["metrics_mean"]["u_rel_err"]\n')
    assert _metrics_mean_subscripts(tree) == [1]
    # 反面：只讀 frame 數 / rows 不該誤觸。
    assert _metrics_mean_subscripts(ast.parse('n = d["metrics_per_t"]\n')) == []
    assert _metrics_mean_subscripts(ast.parse('rs = d["rows"]\n')) == []


@pytest.mark.parametrize("path", READERS, ids=lambda p: p.name)
def test_reader_goes_through_the_deep_seam(path):
    tree = ast.parse(path.read_text())
    assert _imports_result_table(tree), (
        f"{path.relative_to(REPO_ROOT)} 沒有 import {_RESULT_TABLE_MODULE}——"
        "已遷移的 consumer 必須經 deep 讀側 seam（ResultTable / read_metric_value）。")


@pytest.mark.parametrize("path", READERS, ids=lambda p: p.name)
def test_reader_does_not_hand_pick_metric_values(path):
    tree = ast.parse(path.read_text())
    hits = _metrics_mean_subscripts(tree)

    if path.name in EXEMPT:
        assert hits, (
            f"{path.name} 列在 EXEMPT 卻沒有手挑 metric 值——豁免已過期，請移除。"
            f"（理由當初是：{EXEMPT[path.name]}）")
        return

    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} 在第 {hits} 行下標 [\"metrics_mean\"] 手挑 "
        "metric 值——讀 metric 必須經 result_table 的語意層。\n"
        "  改用 result_table.read_metric_value / ResultTable.value：投影鍵漂移時"
        "大聲失敗，而不是靜默給錯數字（scripts/CLAUDE.md §2）。")


def test_exemptions_are_a_literal_list_not_derived():
    """豁免若由被檢查的東西自己構成，守衛就恆真。"""
    assert isinstance(EXEMPT, dict)
    assert all(isinstance(k, str) and k.endswith(".py") for k in EXEMPT)
    assert all(isinstance(v, str) and v for v in EXEMPT.values()), "每個豁免都要有理由"
    assert len(EXEMPT) <= 2, (
        f"豁免長到 {len(EXEMPT)} 個——那不再是例外，是這條規則沒人在遵守")
