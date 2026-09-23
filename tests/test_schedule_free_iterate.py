"""TD-1：`--iterate x` 從 opt_state 重建 ScheduleFree 的評估 iterate。

背景：訓練端存的與 final eval 用的都是 **y**（訓練 iterate），而 optax 文件建議
在 **x**（Polyak 平均）上評估，PyTorch 對照用的也是 x。差多少從未量化，
`--iterate x` 是補上量測端的旗標。

本檔釘住兩件事：
  1. `_find_schedule_free_state` 的判別邏輯。orbax lenient restore 後 opt_state
     是巢狀 dict/tuple、**型別資訊已丟失**，`isinstance(ScheduleFreeState)` 認不出來，
     所以改用「同時具備 b1 與 z」這組結構特徵。判別特徵一旦寫錯，可能靜默抓到
     別的節點而算出一組沒有意義的參數——那正是要擋的。
  2. x 的重建結果與 optax 正典實作一致（不自己抄 `(y−(1−b1)z)/b1`）。
"""
from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from evaluate_exp245 import _find_schedule_free_state  # noqa: E402


def _sf(b1=0.9, z=None):
    return {"b1": b1, "z": z if z is not None else {"w": jnp.ones((2,))}}


def test_finds_state_nested_in_dicts_and_tuples():
    """restore 後的 opt_state 是巢狀容器，不是具名 tuple。"""
    tree = {"inner": ({"noise": 1}, [ {"deep": _sf(b1=0.75)} ])}
    found = _find_schedule_free_state(tree)
    assert found is not None and float(found.b1) == 0.75


def test_returns_none_when_no_schedule_free_state():
    """非 schedule_free 的 ckpt 必須回 None，讓呼叫端 fail-fast 而不是猜。"""
    assert _find_schedule_free_state({"mu": 1, "nu": 2, "count": 3}) is None
    assert _find_schedule_free_state(None) is None


def test_partial_match_is_not_accepted():
    """只有 b1 或只有 z 都不算——鬆掉這條就可能抓到 adam 的節點。"""
    assert _find_schedule_free_state({"b1": 0.9}) is None
    assert _find_schedule_free_state({"z": {"w": jnp.ones((2,))}}) is None


def test_reconstruction_matches_optax_reference():
    """x 的值必須等於 optax 正典實作，不是我們自己抄的公式。"""
    from optax.contrib import schedule_free_eval_params

    b1 = 0.9
    y = {"w": jnp.array([1.0, 2.0, 3.0])}
    z = {"w": jnp.array([0.5, 0.5, 0.5])}
    sf = _find_schedule_free_state({"opt": {"b1": b1, "z": z}})
    assert sf is not None

    x = schedule_free_eval_params(sf, y)
    expected = (y["w"] - (1.0 - b1) * z["w"]) / b1
    assert jnp.allclose(x["w"], expected)
    # 自證：x 必須真的不等於 y，否則上面的斷言對任何實作都成立
    assert not jnp.allclose(x["w"], y["w"])
