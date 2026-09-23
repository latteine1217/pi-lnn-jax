"""eval 腳本不得直接呼叫 operator constructor —— 從 AST 推導，不看拼法。

問題（架構審查候選 K）：`model_factory` 存在的理由寫在它自己的 docstring：
訓練端與 eval 端必須是同一個物件，否則 eval 會用「另一個架構」重建 checkpoint 的
模型，而「新增一個影響 forward 但不改 param shape 的 flag」會安靜通過。

守這件事的哨兵在 `tests/test_model_factory.py`，但它斷言的是
`not hasattr(mod, "_build_model")` ——**一個字串拼法**，而且只檢查
`evaluate_exp245` 一個模組。繞過 factory 的方式不只一種：漂移實際上以
`LiquidOperator(**model_kwargs)` 的形式在 `evaluate_multi_re.py` 與
`dump_cp_fields.py` 復發過（2026-08-03 已收），而那個拼法完全在偵測面之外。

本檔改為掃 AST：`scripts/` 下任何檔案直接呼叫三個 operator constructor 之一，
就是繞過 factory。新腳本自動納入，不必記得回來加測試。

**豁免必須是硬編碼的字面清單。** 若豁免條件由被檢查的東西自己構成
（例如「檔名含 bench 就跳過」），守衛就會隨新增檔案自動失效——本 session
已經在別處踩過兩次這個坑（`vc_cleanup` 的白名單、`EXPECT_BASE` 的來源）。
"""
from __future__ import annotations

import ast
import pathlib

import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

#: 這三個是 `model_factory.build_model` 負責建的。直接呼叫＝繞過。
OPERATOR_CTORS = frozenset({
    "LiquidOperator", "VanillaDeepONetOperator", "StandardPINNOperator",
})

#: 豁免：逐檔列出並附理由。硬編碼的字面清單，不從檔案屬性推導。
EXEMPT = {
    # 效能量測腳本，不讀 ckpt、不產數字；建一個模型跑 forward 計時而已。
    "bench_folx.py": "microbenchmark：不 restore ckpt、不產出任何評估數字",
}


def _direct_ctor_calls(tree: ast.AST) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in OPERATOR_CTORS):
            out.setdefault(node.func.id, []).append(node.lineno)
    return out


def _scripts() -> list[pathlib.Path]:
    return [p for p in sorted(_SCRIPTS.rglob("*.py")) if "legacy" not in p.parts]


SCRIPTS = _scripts()


def test_scan_sees_something():
    """自證：掃不到任何 .py 就代表下面每一條都空過。"""
    assert len(SCRIPTS) > 20, f"只掃到 {len(SCRIPTS)} 支腳本——路徑或過濾壞了"


def test_the_guard_can_actually_fire():
    """鑑別力自檢：對一段確實繞過 factory 的程式碼，掃描器必須抓到。

    少了這條，`_direct_ctor_calls` 哪天因為 AST 結構變動而永遠回空，
    下面的參數化測試會全綠而什麼都沒驗。
    """
    tree = ast.parse("model = LiquidOperator(**kwargs)\n")
    assert _direct_ctor_calls(tree) == {"LiquidOperator": [1]}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_does_not_bypass_the_factory(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        pytest.skip(f"{path.name} 無法解析（跨 repo 腳本？）")

    hits = _direct_ctor_calls(tree)
    if path.name in EXEMPT:
        assert hits, (
            f"{path.name} 列在 EXEMPT 卻沒有直接建構——豁免已過期，請移除。"
            f"（理由當初是：{EXEMPT[path.name]}）")
        return

    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} 直接呼叫 operator constructor，繞過 model_factory：\n"
        + "\n".join(f"    {name} @ 行 {lines}" for name, lines in sorted(hits.items()))
        + "\n  改用 `model_factory.build_model(arch, model_kwargs, K_sensors=...)`。"
        "\n  沒有 --arch 的腳本就明寫 build_model(\"liquid\", ...)——把「只支援 B3」"
        "這個假設寫出來，而不是靠「剛好只建 B3」。"
        "\n  Why：訓練端與 eval 端漂移不會 crash，只會用另一個架構重建 ckpt 的模型"
        "（見 pi_lnn_jax/model_factory.py 的 module docstring）。")


def test_exemptions_are_a_literal_list_not_derived():
    """豁免若由被檢查的東西自己構成，守衛就恆真。

    本 session 已在 `vc_cleanup` 的白名單與 `EXPECT_BASE` 的來源踩過兩次。
    這裡把「EXEMPT 是有限的硬編碼字面清單」釘住。
    """
    assert isinstance(EXEMPT, dict) and EXEMPT
    assert all(isinstance(k, str) and k.endswith(".py") for k in EXEMPT)
    assert all(isinstance(v, str) and v for v in EXEMPT.values()), "每個豁免都要有理由"
    assert len(EXEMPT) <= 3, (
        f"豁免長到 {len(EXEMPT)} 個——那不再是例外，是這條規則沒人在遵守")
