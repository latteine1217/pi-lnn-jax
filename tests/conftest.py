"""測試套件的 process-global state 隔離守門。

What:
    釘住 `jax_enable_x64` 這個 process-global 旗標在整個 session 維持基準值，
    只允許測試在自己的 fixture 範圍內暫時開啟並還原。

Why:
    x64 開著會靜默改變 dtype 而非報錯：optimistix 的 scan carry 前後不一致、
    RWFDense 與 nn.Dense 的 param dtype 分歧、replay 測試（bit-identical 契約的
    黃金測試，見 CLAUDE.md §7.1）對不上 golden。這類污染只在「全套件一起跑」時
    出現，單檔跑全綠，因此不設守門就會反覆重演。

    污染有兩個入口，兩個都要擋：
      1. import 期：pytest 於 collection 階段 import 所有測試模組，模組層的
         `jax.config.update("jax_enable_x64", True)`（或其 import 到的腳本裡有這行）
         會在任何測試開跑前就把旗標打開，且與測試執行順序無關。
      2. 測試期：fixture 忘了還原。

    守門一律 fail-fast 並點名，不靜默還原——靜默還原會讓下一個洩漏源同樣隱形。
"""
from __future__ import annotations

import pytest
from jax import config as _jax_config

_FLAG = "jax_enable_x64"

# conftest 在 collection 之前 import，此時尚未有任何測試模組被 import，
# 故這裡讀到的是乾淨的 process 預設值。
_BASELINE = _jax_config.read(_FLAG)


def pytest_collection_finish(session: pytest.Session) -> None:
    """collection 結束（＝所有測試模組已 import）時檢查旗標未被 import 期污染。"""
    now = _jax_config.read(_FLAG)
    if now != _BASELINE:
        raise pytest.UsageError(
            f"{_FLAG} 在 collection 期被改成 {now!r}（基準 {_BASELINE!r}）："
            "某個測試模組（或它 import 的腳本）在模組層呼叫了 "
            f'jax.config.update("{_FLAG}", ...)。請改為 autouse fixture 或 '
            "__main__ 內設定；模組層設定會污染整個測試套件。"
        )


@pytest.fixture(autouse=True)
def _assert_x64_restored():
    """每個測試結束後檢查旗標已回到基準值（fixture 忘了還原就在此爆）。"""
    yield
    now = _jax_config.read(_FLAG)
    assert now == _BASELINE, (
        f"{_FLAG} 測試結束後為 {now!r}，未還原到基準 {_BASELINE!r}；"
        "開啟 x64 請用會還原的 autouse fixture。"
    )
