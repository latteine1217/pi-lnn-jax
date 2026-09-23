"""`models.py` 的 dataclass 預設與 `config.py` 的 schema 預設，必須一致或在此有名字。

問題（架構審查候選 B）：同一個 model 旗標的預設值宣告在兩個地方，而兩條建構路徑
各自吃其中一個——Kolmogorov 走 `config.py` schema，cylinder 的
`pipeline/cylinder/assembly.py` 直接 `LiquidOperator(**CFG, ...)`，CFG 未指定的鍵
落到 `models.py` 的 dataclass 預設。

2026-09-03 起 `cfc_input_dependent_tau` / `use_rwf` / `cfc_tau_mod_scale` 三鍵
兩邊對齊（True / True / 0.5），只剩 `query_mlp_hidden_dim` 一筆分岔。

那次對齊示範了本檔要防的正是哪種事故：打開 `cfc_input_dependent_tau` 會讓
cylinder 的 `cfc_tau_mod_scale` 落到 `models.py` 的 `2.0`，而 `config.py:114-115`
自己註記那個值「讓 τ 動態範圍暴衝 ~400× 全面崩壞 +27~58%」。tau 關著時 scale
是惰性的，所以這個陷阱在對齊前一直沒有症狀。scale 因此同批改為 `0.5`。

本檔不主動改預設值（那是行為變更，論文投稿中需 A/B 背書），只讓分岔有名字：
新增欄位時若兩邊給了不同預設，測試立刻紅，而不是幾個月後由某個崩壞的實驗發現。
"""
from __future__ import annotations

import dataclasses

import pytest

from pi_lnn_jax.config import MODEL_SCHEMA
from pi_lnn_jax.models import LiquidOperator

#: 已知分岔。硬編碼字面值，附後果。不是豁免，是釘子。
#: 形式：鍵 → (models.py 的值, config.py schema 的值, 為什麼要記著)
KNOWN_DIVERGENCE = {
    # 曾在此、已對齊而移除的分岔（移除紀錄正是這個守衛的用意：分岔被解決時
    # 要有人來改紀錄，而不是靜靜消失）：
    #   use_temporal_anchor           2026-08-03 兩邊對齊 True（架構候選 H）
    #   cfc_input_dependent_tau       2026-09-03 兩邊對齊 True
    #   use_rwf                       2026-09-03 兩邊對齊 True
    #   cfc_tau_mod_scale             2026-09-03 兩邊對齊 0.5（配套，見 docstring）
    # 後兩批的量測證據全在 Kolmogorov，cylinder 未驗——對齊是使用者明示的決定，
    # 不是證據外推。cylinder 下次跑之前該知道它的預設變了。
    "query_mlp_hidden_dim": (
        256, 64,
        "純容量差異，無記錄在案的後果；列在此處是為了讓「兩邊預設不同」這件事完整。"),
}


def _dataclass_defaults() -> dict[str, object]:
    return {
        f.name: f.default
        for f in dataclasses.fields(LiquidOperator)
        if f.default is not dataclasses.MISSING
    }


DEFAULTS = _dataclass_defaults()
#: 兩邊都宣告了預設值的鍵——只有這些比得起來。
COMPARABLE = sorted(set(DEFAULTS) & set(MODEL_SCHEMA))


def test_there_is_something_to_compare():
    """自證：兩邊的交集若是空的，下面的參數化就零案例而全綠。"""
    assert len(COMPARABLE) > 20, (
        f"只有 {len(COMPARABLE)} 個鍵兩邊都宣告預設——掃描或介面已變")


@pytest.mark.parametrize("key", COMPARABLE)
def test_default_matches_or_is_a_recorded_divergence(key):
    mine, theirs = DEFAULTS[key], MODEL_SCHEMA[key][1]
    if key in KNOWN_DIVERGENCE:
        exp_mine, exp_theirs, why = KNOWN_DIVERGENCE[key]
        assert (mine, theirs) == (exp_mine, exp_theirs), (
            f"{key} 的分岔已改變：紀錄 models.py={exp_mine!r} schema={exp_theirs!r}，"
            f"實際 models.py={mine!r} schema={theirs!r}。\n"
            f"  當初記著的理由：{why}\n"
            "  若已刻意對齊，從 KNOWN_DIVERGENCE 移除；若是新的分岔，更新紀錄。")
        return

    assert mine == theirs, (
        f"{key} 的預設值在兩處不同：models.py={mine!r}，config.py schema={theirs!r}。\n"
        "  Kolmogorov 走 schema，cylinder 的 assembly.CFG 未指定時落到 models.py——\n"
        "  兩案會用不同的值建同一個模型，而不會有任何提示。\n"
        "  對齊其中一邊，或（若分岔是刻意的）加進本檔的 KNOWN_DIVERGENCE 並寫下後果。")


def test_recorded_divergences_are_still_real():
    """反向斷言：紀錄裡的分岔若已消失，這裡要紅。

    沒有這條，KNOWN_DIVERGENCE 會變成沒人維護的歷史殘留，
    而「分岔被解決」這件該被看見的事會靜靜發生。
    """
    for key, (exp_mine, exp_theirs, _) in KNOWN_DIVERGENCE.items():
        assert key in COMPARABLE, f"{key} 已不在兩邊的交集——紀錄過期"
        assert exp_mine != exp_theirs, f"{key} 列為分岔卻記著相同的值"


def test_the_documented_catastrophic_value_is_not_the_schema_default():
    """schema 是 Kolmogorov 的來源，它不得指向記錄有案的災難值。

    這條與上面的分岔紀錄互補：分岔本身可以存在，但**受 schema 保護的那條路徑**
    必須落在安全值上。
    """
    assert MODEL_SCHEMA["cfc_tau_mod_scale"][1] == 0.5, (
        "schema 的 cfc_tau_mod_scale 不再是 0.5——config.py:114-115 記載 2.0 會讓"
        "τ 動態範圍暴衝 ~400× 全面崩壞 +27~58%，改動前請回查那組實驗（exp_521-524）")


def test_cylinder_path_does_not_silently_enable_tau_at_the_dangerous_scale():
    """cylinder 的安全性目前靠「tau 是關的」撐著——把這個前提釘住。

    哪天 CFG 開始指定 cfc_input_dependent_tau=True 而沒一併給 scale，
    這條會紅，提醒補上 scale 而不是讓它落到 2.0。
    """
    from pi_lnn_jax.pipeline.cylinder.assembly import CFG

    tau_on = CFG.get("cfc_input_dependent_tau", DEFAULTS["cfc_input_dependent_tau"])
    scale = CFG.get("cfc_tau_mod_scale", DEFAULTS["cfc_tau_mod_scale"])

    assert not (tau_on and scale == 2.0), (
        f"cylinder 會以 tau_mod_scale={scale} 開啟 input-dependent tau——"
        "那是 config.py:114-115 記載的災難值。在 CFG 明設 cfc_tau_mod_scale=0.5。")
