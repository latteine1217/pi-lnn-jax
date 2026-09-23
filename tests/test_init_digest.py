"""把 fixture 尾列的 `params_digest` 與**重跑一次初始化**的結果比對。

What: 對八份 golden fixture，用尾列記的命令列重走生產路徑
      （`resolve_inputs` → `build_context` → `initialize`），
      比對算出的 digest 與尾列的 **`init_params_digest`**。

比的**不是** `params_digest`——那一欄記在 `finalize`，是**訓練後**的參數。
本檔初版拿它跟 `initialize` 的輸出比，是一條從一開始就不可能通過的測試；
因為它在開發機一律 skip，直到弱路徑門檻（§8.3 第 5 項）強迫它在 lab-server
上執行才暴露（job 4781/4782）。門檻在加入的同一輪就抓出它所守護的測試是錯的。

Why: 缺口在 wave2 spec §8.3——本機測試對 params 的覆蓋止於**結構**
     （建構參數、常數表、參數總數，見 `test_cylinder_assembly.py`），
     從來沒有任何測試碰過 fixture 記的那個**值**。翻掉一個承重的初始化旗標，
     結構層抓得到；但初始化路徑本身接錯（餵錯 key、順序顛倒、少消耗一次
     RNG），結構層抓不到。

Why 走生產路徑而不手工重建模型: 手工重建會變成第二份實作，和 assembly／
     initialize 各自演化後就不再測到真正在跑的東西。這裡直接呼叫它們。

執行前提（兩個，缺一就 skip 並說明理由，不靜靜跳過）:
  1. **執行環境必須與錄製時相符。** digest 跨平台不可比——同碼同輸入，
     arm64 與 x86_64 給出不同的值（job 4740 對照本機實測）。
     `environment_mismatch` 負責判定；尾列沒記環境的舊 fixture 一律拒絕。
  2. **錄製當時的資料檔要在本機。** `build_context` 會真的載入資料。

因此本檔在開發機上全部 skip，在錄製 fixture 的那台機器上才實際比對——
這正是 §8.3 第 6 項要的「在錄製架構上補完值這一層」。

fixture 出處（出處是 fixture 的一部分，改值必須同時改這裡）:
  ledger/f1–f4                Kolmogorov，錄製 job 見 `test_fingerprint_recording`
  cylinder_ledger/c1–c4       lab-server job **5497** 的 HEAD 側
                              （`artifacts/ab/c{1..4}/rng_ledger.json`，
                              main @ 9af6db7，Linux/x86_64/CPU）

⚠️ 這四份於 2026-09-03 重錄。原因是模型**有意改變**：`use_rwf` 與
`cfc_input_dependent_tau` 兩個預設 False→True 後，cylinder 的參數量
3,139,146 → 3,272,010，初始化 digest 隨之由 `bed0b931c5dbee29` 變成
`06a2ed459d1f67a1`（見 knowledge/codebase/technical-debt.md TD-4）。
重錄而非放寬比對——舊值對應的模型已不存在，留著它只會讓這條測試恆紅。
同一輪的 job 5497 中，段 (3) BASE vs HEAD 四組態全部 BIT-IDENTICAL，
即 digest 改變來自那個裁決，不是重構造成的分岔。
"""
from __future__ import annotations

import json
import pathlib

import pytest

import _acceptance_strictness
from _paths import REPO_ROOT

from pi_lnn_jax.pipeline._ledger import environment_mismatch, params_digest

#: (fixture 路徑, case 模組名)。兩案的協定同形，故可參數化。
FIXTURES = [
    ("ledger/f1_single_re.json", "kolmogorov"),
    ("ledger/f2_multi_re_crp.json", "kolmogorov"),
    ("ledger/f3_rar.json", "kolmogorov"),
    ("ledger/f4_prod_scale.json", "kolmogorov"),
    ("cylinder_ledger/c1.json", "cylinder"),
    ("cylinder_ledger/c2.json", "cylinder"),
    ("cylinder_ledger/c3.json", "cylinder"),
    ("cylinder_ledger/c4.json", "cylinder"),
]


def _tail(path: pathlib.Path) -> dict:
    rows = json.loads(path.read_text())
    tails = [r for r in rows if r.get("step") == -1]
    assert len(tails) == 1, f"{path.name}: 尾列應恰有一列，實得 {len(tails)}"
    return tails[0]


def _initialize(case: str, argv: list[str]):
    """走該案的生產路徑到 initialize 為止，回傳 params。"""
    if case == "kolmogorov":
        from pi_lnn_jax.pipeline.kolmogorov.assembly import build_context
        from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs
        from pi_lnn_jax.pipeline.kolmogorov.run import RunJournal, initialize
        ctx = build_context(resolve_inputs(argv).config)
        return initialize(ctx, RunJournal()).params

    from pi_lnn_jax.pipeline.cylinder.assembly import build_context
    from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs
    from pi_lnn_jax.pipeline.cylinder.run import initialize
    ctx = build_context(resolve_inputs(argv).config)
    return initialize(ctx).params


@pytest.mark.parametrize("rel,case", FIXTURES, ids=[f[0].split("/")[-1] for f in FIXTURES])
def test_init_digest_matches_fixture_tail(rel, case, monkeypatch):
    if _acceptance_strictness.out_of_scope(case):
        pytest.skip(f"{rel}: 本次驗收 job 只驗另一案，{case} 的資料未掛載")
    tail = _tail(REPO_ROOT / "tests" / "fixtures" / rel)

    reason = environment_mismatch(tail.get("environment"))
    _acceptance_strictness.require_no_skip(reason, rel)
    if reason is not None:
        pytest.skip(f"{rel}: {reason}——digest 的值只能在錄製架構上比")

    argv = tail.get("argv")
    assert argv, f"{rel}: 尾列缺 argv，無法重走生產路徑"

    # config 路徑在 argv 內是相對的；錄製當時的 cwd 是 repo root。
    monkeypatch.chdir(REPO_ROOT)
    try:
        params = _initialize(case, argv)
    except FileNotFoundError as e:
        _acceptance_strictness.require_no_skip(f"資料檔不在本機（{e}）", rel)
        pytest.skip(f"{rel}: 錄製當時的資料檔不在本機（{e}）——build_context 需要真實資料")

    want = tail.get("init_params_digest")
    if want is None:
        _acceptance_strictness.require_no_skip(
            "尾列沒有 init_params_digest（fixture 早於該欄位，需重錄）", rel)
        pytest.skip(f"{rel}: 尾列沒有 init_params_digest——需以現行程式碼重錄 fixture")

    got = params_digest(params)
    assert got == want, (
        f"{rel}: 初始化算出的 digest 與尾列的 init_params_digest 不符\n"
        f"  尾列 {want}\n  重算 {got}\n"
        "執行環境已確認相符，故這是真的分岔：模型建構或初始化路徑已改變。")


@pytest.mark.parametrize("rel,case", FIXTURES, ids=[f[0].split("/")[-1] for f in FIXTURES])
def test_fixture_was_recorded_on_a_reproducible_backend(rel, case):
    """fixture 必須錄自 CPU backend。

    GPU 上同一份程式碼兩跑就給不同的 params_digest（job 4729），故 GPU 錄的
    fixture 其 digest 不構成任何判準。先前這件事只能從命令列旗標**推斷**
    （wave2 spec §8.4 第一條）；`backend` 現在直接記在尾列，可以斷言。
    """
    tail = _tail(REPO_ROOT / "tests" / "fixtures" / rel)
    env = tail.get("environment")

    assert env, f"{rel}: 尾列未記執行環境，無從判斷 backend"
    assert env.get("backend") == "cpu", (
        f"{rel}: 錄自 backend={env.get('backend')!r}。GPU 上同碼兩跑 digest 就不同"
        "（job 4729），該 fixture 的 digest 不構成判準——必須在 CPU 上重錄。")


def test_the_fixture_list_covers_every_committed_fixture():
    """自證：清單漏一份，那份就永遠不會被比對，而測試依然全綠。"""
    on_disk = {
        f"{d}/{p.name}"
        for d in ("ledger", "cylinder_ledger")
        for p in (REPO_ROOT / "tests" / "fixtures" / d).glob("*.json")
    }

    assert on_disk == {rel for rel, _ in FIXTURES}, (
        f"fixture 清單與磁碟不符：\n  只在磁碟 {on_disk - {r for r, _ in FIXTURES}}"
        f"\n  只在清單 {{r for r, _ in FIXTURES}} - on_disk")
