"""腳本索引的 metric 鍵，必須是庫真的會回傳的鍵。

背景：2026-08-01 把 `energy_timeseries_errors` 的 `ke_t_mape` 改名為
`ke_t_mape_spatialmean`（純鍵名，公式未動）。改名沒有跟到消費端，結果是
`scripts/ke_timeseries_errors.py:36` 從那天起**每次執行都 KeyError**——
而沒有任何測試會紅，因為沒有測試碰過那支腳本。

這類斷裂的共同形狀是：**函式回傳的是字串 dict，呼叫端用字面量索引，
兩者之間沒有任何東西把它們綁在一起**。加一支測試釘住某個鍵不夠——下一次
改名會出現在別的鍵上。所以這裡改為**從程式碼推導**：掃出所有
`x = energy_timeseries_errors(...)` 之後對 `x` 的字面量索引，逐一要求該鍵
真的在回傳裡。新增消費端自動納入，不必記得回來改測試。

覆蓋邊界（誠實話）：只掃「直接把回傳綁成區域變數再索引」這個形式。
先傳給別的函式、或存進 dict 再取出的，需要資料流分析，這裡抓不到。
`backfill_ksweep_metrics.py` 刻意有自己的 stdlib 抄本（見 scripts/CLAUDE.md），
判準因此是「從 pi_lnn_jax import 該函式」而非「出現這個名字」——只比對名字會
把那支的自有鍵集誤判為壞（初版就是這樣，被本檔自己的參數化測試抓出來）。
那份的鍵對齊由它自己的 gate 負責。
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.evaluate import energy_timeseries_errors

_SCRIPTS = REPO_ROOT / "scripts"
_FUNC = "energy_timeseries_errors"


def _actual_keys() -> frozenset[str]:
    """實跑一次取得真實回傳鍵——不手列，手列會跟著腐爛。"""
    t = np.linspace(1.0, 2.0, 8)
    return frozenset(energy_timeseries_errors(t * 1.01, t))


def _indexed_keys(tree: ast.AST) -> dict[str, list[int]]:
    """`x = energy_timeseries_errors(...)` 之後所有 `x["k"]` 的 k → 行號。"""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == _FUNC):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))

    out: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id in bound and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            out.setdefault(node.slice.value, []).append(node.lineno)
    return out


def _imports_library_func(tree: ast.AST) -> bool:
    """是否**從庫 import** 這個函式——而不是自己定義一個同名的。

    判準必須是來源而非名字：`backfill_ksweep_metrics.py` 刻意自帶一份純 stdlib
    抄本（設計為 `ssh lab-server "python3 -"` 無 repo 執行，見 scripts/CLAUDE.md），
    它的 dict 有自己的鍵集，拿庫的回傳去比對它是無意義的。
    只比對名字就會把它誤判——初版正是這樣，被本檔自己的參數化測試抓出來。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pi_lnn_jax"):
            if any(a.name == _FUNC for a in node.names):
                return True
    return False


def _consumers() -> dict[pathlib.Path, dict[str, list[int]]]:
    found: dict[pathlib.Path, dict[str, list[int]]] = {}
    for path in sorted(_SCRIPTS.rglob("*.py")):
        if "legacy" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        if not _imports_library_func(tree):
            continue
        keys = _indexed_keys(tree)
        if keys:
            found[path] = keys
    return found


CONSUMERS = _consumers()


def test_scan_finds_the_known_consumer():
    """自證：掃描器真的在看。抓不到消費端＝下面每一條都空過。"""
    assert CONSUMERS, f"掃不到任何 {_FUNC} 的字面量索引消費端——掃描器壞了"
    names = {p.name for p in CONSUMERS}
    assert "ke_timeseries_errors.py" in names, (
        f"已知的消費端沒被掃到，實得 {sorted(names)}")


@pytest.mark.parametrize("path", sorted(CONSUMERS), ids=lambda p: p.name)
def test_indexed_keys_exist_in_the_actual_return(path):
    actual = _actual_keys()
    bad = {k: lines for k, lines in CONSUMERS[path].items() if k not in actual}
    assert not bad, (
        f"{path.relative_to(REPO_ROOT)} 索引了 {_FUNC} 不會回傳的鍵——執行時 KeyError：\n"
        + "\n".join(f"    {k!r} @ 行 {lines}" for k, lines in sorted(bad.items()))
        + f"\n  實際回傳：{sorted(actual)}"
        + "\n  改名時鍵名要跟到消費端；改公式定義則要回查 paper 並提 diff 給人審"
        "（scripts/CLAUDE.md §3.4）。")


def test_the_check_would_catch_a_rename():
    """鑑別力自檢：把一個不存在的鍵餵進比對，必須被判為壞。

    少了這條，`_actual_keys()` 哪天回傳空集合（例如函式簽章變了、
    實跑拋例外被吞掉），上面的參數化測試會全綠而什麼都沒驗。
    """
    actual = _actual_keys()
    assert actual, "實跑取不到任何回傳鍵——比對的另一邊是空的"
    assert "ke_t_mape" not in actual, (
        "`ke_t_mape` 又出現在 energy_timeseries_errors 的回傳裡了。"
        "若是有意回復，更新本測試；若非，那是 2026-08-01 改名被回退了")
    assert "ke_t_mape_spatialmean" in actual
