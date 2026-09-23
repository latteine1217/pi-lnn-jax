"""Tests for the diagnostic RNG/schedule ledger.

Ledger 是重構 train_kolmogorov.py 的 first-divergence oracle：逐步記錄
host-side 決策，讓新舊實作可以比對出「哪一步」開始分岔。

紀律紅線（spec §6）：預設關閉、關閉時 get_ledger() 回 None、
大陣列只存 digest。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pi_lnn_jax.pipeline._ledger import (
    Ledger, digest, get_ledger, ledger_enabled, params_digest, reset_ledger,
)


@pytest.fixture(autouse=True)
def _clean_ledger(monkeypatch):
    monkeypatch.delenv("PILNN_RNG_LEDGER", raising=False)
    reset_ledger()
    yield
    reset_ledger()


def test_gate_is_off_by_default(monkeypatch):
    assert ledger_enabled() is False
    assert get_ledger() is None
    print("✓ gate_is_off_by_default")


@pytest.mark.parametrize("val", ["0", ""])
def test_gate_stays_off_for_falsey_values(monkeypatch, val):
    monkeypatch.setenv("PILNN_RNG_LEDGER", val)
    assert ledger_enabled() is False
    assert get_ledger() is None
    print(f"✓ gate_stays_off_for_falsey_values[{val!r}]")


def test_gate_turns_on_and_returns_singleton(monkeypatch):
    monkeypatch.setenv("PILNN_RNG_LEDGER", "1")
    assert ledger_enabled() is True
    a = get_ledger()
    b = get_ledger()
    assert isinstance(a, Ledger)
    assert a is b, "同一 process 內必須是同一個 Ledger，否則 row 會分散"
    print("✓ gate_turns_on_and_returns_singleton")


def test_digest_is_deterministic_and_bit_sensitive():
    x = np.arange(16, dtype=np.float32)
    assert digest(x) == digest(x.copy())
    y = x.copy()
    y[7] = np.nextafter(y[7], np.float32(1e9))   # 動最小的一個 bit
    assert digest(x) != digest(y), "digest 必須對單一 ulp 差異敏感"
    print("✓ digest_is_deterministic_and_bit_sensitive")


def test_digest_separates_shape_and_dtype():
    a = np.zeros(4, dtype=np.float32)
    assert digest(a) != digest(a.reshape(2, 2)), "shape 必須進 digest"
    assert digest(a) != digest(np.zeros(4, dtype=np.float64)), "dtype 必須進 digest"
    print("✓ digest_separates_shape_and_dtype")


def test_digest_handles_non_contiguous_input():
    # ledger 會收到 jnp array 的切片；非連續記憶體不得讓 tobytes 取到錯誤內容
    base = np.arange(20, dtype=np.float32).reshape(4, 5)
    view = base[:, ::2]
    assert digest(view) == digest(np.ascontiguousarray(view))
    print("✓ digest_handles_non_contiguous_input")


def test_params_digest_is_order_independent_of_dict_construction():
    import jax.numpy as jnp
    p1 = {"params": {"a": jnp.ones((2, 2)), "b": jnp.zeros((3,))}}
    p2 = {"params": {"b": jnp.zeros((3,)), "a": jnp.ones((2, 2))}}
    assert params_digest(p1) == params_digest(p2), (
        "digest 必須依 tree path 排序，不能受 dict 插入順序影響"
    )
    print("✓ params_digest_is_order_independent_of_dict_construction")


def test_records_and_dumps_sorted_json(tmp_path, monkeypatch):
    monkeypatch.setenv("PILNN_RNG_LEDGER", "1")
    led = get_ledger()
    led.record(1, re_idx=0, n_collo=32, cx="deadbeef")
    led.record(2, re_idx=1, n_collo=64, cx="cafebabe")
    out = tmp_path / "sub" / "rng_ledger.json"
    led.dump(out)
    rows = json.loads(out.read_text())
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["re_idx"] == 1
    print("✓ records_and_dumps_sorted_json")


def test_params_digest_handles_real_optax_state():
    # ledger 的最終 dump 呼叫 params_digest(opt_state)，但既有測試只蓋過純 dict of jnp array。
    # 該呼叫在 run 的最後才執行，若拋錯會讓整個 lab-server job 的 ledger 白費。
    import jax.numpy as jnp
    from pi_lnn_jax.optimizers import build_optimizer

    # 建構真實 optax optimizer（簡單 adam 路徑，避免 SOAP/schedule_free 依賴）
    optimizer, opt_info = build_optimizer(
        name="adam",
        learning_rate=0.001,
    )
    assert opt_info["name"] == "adam"

    # 簡單 params pytree
    params = {
        "weights": jnp.ones((3, 4)),
        "bias": jnp.zeros((4,)),
    }

    # 初始化 optimizer 狀態
    opt_state = optimizer.init(params)

    # 驗證 params_digest 不拋錯且回傳 16 字 hex
    digest_val = params_digest(opt_state)
    assert isinstance(digest_val, str), "digest 必須回傳 str"
    assert len(digest_val) == 16, f"digest 長度必須為 16，但得到 {len(digest_val)}"
    # 16 個字都應是 hex 字元
    assert all(c in "0123456789abcdef" for c in digest_val), (
        f"digest 必須是 16 字 hex，但得到 {digest_val!r}"
    )

    # 驗證確定性：同一 opt_state 多次呼叫應得到相同值
    digest_val2 = params_digest(opt_state)
    assert digest_val == digest_val2, "params_digest 必須確定性"

    print("✓ params_digest_handles_real_optax_state")


# ─────────────────────────────────────────────────────────────────────────────
# 尾列的執行環境（spec wave2 §8.3 第 7 項）
#
# digest 只在特定平台上有意義：同一份程式碼在 arm64 與 x86_64 上給出不同的
# params_digest（job 4740 對照本機實測），GPU 甚至同機兩跑就不同（job 4729）。
# 尾列若不記環境，「digest 對不上」就無從區分是程式碼變了還是機器換了。
# ─────────────────────────────────────────────────────────────────────────────

def test_environment_reports_the_fields_that_determine_digest_validity():
    from pi_lnn_jax.pipeline._ledger import environment

    env = environment()

    for key in ("machine", "system", "jax", "jaxlib", "backend"):
        assert key in env, f"環境缺欄位 {key}"
        assert env[key], f"環境欄位 {key} 是空值——記了等於沒記"
    print(f"✓ environment_reports_the_fields ({env})")


def test_tail_row_carries_the_environment():
    """尾列（step == -1）自動帶環境。

    Why 放在 Ledger 而不是各 case 的 run.py：尾列是 ledger 自己定義的概念，
    環境屬於「這份錄製」而非任一案的邏輯。放在共用處，兩案不會漂移，
    將來第三個 case 也自動有。
    """
    led = Ledger()
    led.record(0, cx="a")
    led.record(-1, params_digest="deadbeef")

    step_row, tail_row = led.rows
    assert "environment" not in step_row, "逐步列不該帶環境（每步一份是純浪費）"
    assert tail_row["environment"]["machine"], "尾列必須帶環境"
    assert tail_row["params_digest"] == "deadbeef", "既有欄位不得被覆蓋"


def test_explicit_environment_is_not_overwritten():
    """replay 端重建尾列做測試時要能自己指定環境；自動附加不得蓋掉呼叫端給的值。"""
    led = Ledger()
    led.record(-1, params_digest="x", environment={"machine": "made-up"})

    assert led.rows[0]["environment"] == {"machine": "made-up"}


def test_environment_mismatch_is_reported_with_the_offending_field():
    """可比性判斷：不符時要說出是哪一個欄位不符，否則使用者只知道「不能比」。"""
    from pi_lnn_jax.pipeline._ledger import environment, environment_mismatch

    here = environment()

    assert environment_mismatch(here) is None, "與自己比應相容"

    other = dict(here, machine="some-other-arch")
    reason = environment_mismatch(other)
    assert reason and "machine" in reason, reason
    assert here["machine"] in reason and "some-other-arch" in reason, reason


def test_missing_environment_is_reported_as_unknown_not_as_compatible():
    """舊 fixture 沒有這個欄位。必須回報「無從判斷」，不得當成相容——
    那會讓跨平台的 digest 比對靜靜地在錯誤前提下進行。"""
    from pi_lnn_jax.pipeline._ledger import environment_mismatch

    reason = environment_mismatch(None)

    assert reason and "未記錄" in reason, reason


def test_note_fields_land_in_the_tail_row_only():
    """`note()` 是「值在早期階段算出、尾列到最後才寫」的通道。

    `init_params_digest` 只有 initialize 手上有，尾列卻由 finalize 寫——
    沒有這個通道就得把純觀測量塞進 TrainingState（破壞「state 決定數值」的讀法）
    或各案的 journal（兩案不對稱，cylinder 沒有 journal）。
    """
    led = Ledger()
    led.note(init_params_digest="init-abc")
    led.record(0, cx="a")
    led.record(-1, params_digest="final-xyz")

    step_row, tail_row = led.rows
    assert "init_params_digest" not in step_row, "note 不該汙染逐步列"
    assert tail_row["init_params_digest"] == "init-abc"
    assert tail_row["params_digest"] == "final-xyz", "既有欄位不得被 note 蓋掉"


def test_explicit_tail_field_wins_over_note():
    """呼叫端明示的值優先——replay 端重建尾列做測試時需要自己指定。"""
    led = Ledger()
    led.note(init_params_digest="from-note")
    led.record(-1, init_params_digest="explicit")

    assert led.rows[0]["init_params_digest"] == "explicit"
