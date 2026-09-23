"""驗收 job 上「弱路徑即失敗」的門檻（開發機不受影響）。

背景（wave2 spec §8.3 第 5 項）：`replay_schedule` 有兩條建 ctx 的路。
`build_context` 會重新從 npz 算出時間軸、K/T、幾何等衍生量；`data_meta` 回退
則是把錄製當時就算好的同一批數字原樣讀回來——後者不覆蓋那些量本身的正確性。
兩條都是合法且刻意保留的（開發機沒有資料檔，常態就走回退），但強度不同。

問題在於：強路徑「有沒有被執行過」目前是機率事件——取決於跑測試的那台機器
剛好有沒有資料檔。沒人會發現它從此再也沒被走過。

作法：預設不變（兩條路都放行），但驗收 job 匯出本環境變數後，
回退路徑即視為失敗。門檻因此只在「應該有資料」的場合生效，
開發機不受影響。

環境變數名同時出現在本模組與兩支驗收 sbatch，故有一條測試釘住三者一致
（見 `test_replay_strictness_gate.py`）。
"""
from __future__ import annotations

import os

#: 設為非空且非 "0" 時，所有「弱路徑」一律判定失敗：
#:   - replay 走 data_meta 回退而非 build_context
#:   - 初始化 digest 的比對因前提不成立而 skip
#: 兩者是同一類失效——弱路徑「有沒有被走過」取決於環境，沒人會發現它從此沒被走過。
ENV_VAR = "PILNN_ACCEPTANCE_STRICT"


def strict_mode() -> bool:
    val = os.environ.get(ENV_VAR, "")
    return bool(val) and val != "0"


def require_build_context(ctx_source: str, expected: str, name: str) -> None:
    """嚴格模式下，replay 未走 `build_context` 即拋 AssertionError。

    `expected` 由呼叫端傳入該案的 `REPLAY_CTX_SOURCE_BUILD_CONTEXT`——
    不在此處硬寫字串，免得兩案的常數改名後這裡安靜地永遠通過。
    """
    if not strict_mode():
        return
    assert ctx_source == expected, (
        f"{name}: {ENV_VAR} 已設，但 replay 走的是 {ctx_source!r} 而非 {expected!r}。"
        "驗收 job 上資料檔應該在位——走回退代表資料沒掛上，"
        "這次 replay 沒有覆蓋到時間軸／K/T／幾何等衍生量的重算。"
    )


def require_no_skip(reason: str | None, name: str) -> None:
    """嚴格模式下，`reason` 非 None（即該測試將 skip）就拋 AssertionError。

    給初始化 digest 的比對用：它有兩條 skip 路徑（執行環境不符、資料檔不在本機），
    在開發機上兩條都是常態。但在驗收 job 上兩者都**不該**成立——fixture 就錄自
    該機器、資料也在位。不設門檻的話，那條比對可能從寫好之後就再也沒真正執行過，
    而測試永遠是綠的（skip 不算失敗）。
    """
    if not strict_mode():
        return
    assert reason is None, (
        f"{name}: {ENV_VAR} 已設，但這條比對被跳過了——理由：{reason}。"
        "驗收 job 上 fixture 錄自本機、資料也在位，skip 代表前提沒被滿足，"
        "而非「這次不用比」。"
    )


#: 本次驗收 job 所驗的 case（"kolmogorov" / "cylinder"）。未設＝不限定。
CASE_ENV_VAR = "PILNN_ACCEPTANCE_CASE"


def out_of_scope(case: str) -> bool:
    """True 若 `case` 不是本次驗收 job 要驗的那一案。

    Why 需要區分「不在範圍」與「弱路徑」：每支驗收 job 只掛載自己那一案的
    資料檔（Kolmogorov worktree 沒有 cylinder 的 npz，反之亦然）。
    跨案的 fixture 因而必然缺資料——那不是「前提沒被滿足」，是**這支 job
    本來就不負責它**。若一律套嚴格模式，job 會因為別案的資料不在而失敗
    （job 4828/4829 就是這樣紅的），而那個紅什麼也沒說明。
    """
    want = os.environ.get(CASE_ENV_VAR, "").strip()
    return bool(want) and want != case
