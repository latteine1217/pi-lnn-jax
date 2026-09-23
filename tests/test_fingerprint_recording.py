"""訓練端寫的指紋，必須等於 eval 端從同一份 config 重建出來的指紋。

`test_model_fingerprint.py` 驗判定邏輯，`test_restore_eval_params.py` 驗閘門的
四條路徑——兩者都用合成資料。它們證明不了本檔要證明的那件事：

    真實訓練 run 記錄下來的那串字，與 eval 端重建時算出來的那串字，真的相等。

這是閘門會不會**誤擋每一次正當 eval** 的關鍵。任何一邊的取值方式改變
（欄位排除規則、str() 之外的序列化、model_kwargs 的解析順序）都會讓兩者分岔，
而合成測試兩邊都用同一個函式，永遠看不到分岔。

fixture 是真實產物，不是手寫的：`artifacts/cpuab/f1/summary.json`，由 lab-server
job 5469 的 HEAD 側（main @ 2c07d68）產生。整份原樣收下，不裁剪——裁剪過的
「真實產物」就不再是證據。該 job 的驗收層 A 全綠（四份 fresh run 皆 BIT-IDENTICAL、
ledger-off 亦然），所以這份 fixture 同時是「指紋面改動未破壞 §7.1 契約」的證據。

⚠️ 這個 fixture 會在 model 欄位變動時報紅，那是**設計如此**：新增或移除
`LiquidOperator` 欄位確實改變了指紋面，該重錄一份而不是放寬比對。重錄方式見
本檔末尾。

跨架構可比（與 `test_init_digest.py` 相反）：指紋是 config 值的字串化，不含浮點
運算。fixture 錄於 x86_64，在 arm64 開發機上逐欄位相等——所以本檔不 skip。
"""
from __future__ import annotations

import json

import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.config import load_config
from pi_lnn_jax.model_factory import build_model, fingerprint_diff, model_fingerprint

_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "summary" / "f1_single_re.json"
_CONFIG = REPO_ROOT / "configs" / "_ledger_single_re.toml"


@pytest.fixture(scope="module")
def recorded_and_current() -> tuple[dict, dict]:
    summary = json.loads(_FIXTURE.read_text())
    cfg = load_config(str(_CONFIG))
    model, _ = build_model(summary["arch"], cfg["model_kwargs"], K_sensors=8)
    return summary["model_construction"], model_fingerprint(model)


def test_fixture_is_a_real_training_artifact():
    """自證：fixture 必須是真的訓練產物，不是為了讓測試變綠而手寫的。"""
    summary = json.loads(_FIXTURE.read_text())
    for key in ("arch", "model_name", "n_params", "ckpt_steps", "final_metrics"):
        assert key in summary, (
            f"fixture 缺 {key}——這不像真實 summary.json，"
            "而像為了通過本測試而拼出來的最小 dict")
    assert summary["model_construction"], "fixture 沒有 model_construction"


def test_recorded_fingerprint_matches_eval_side_reconstruction(recorded_and_current):
    """本檔的存在理由。"""
    recorded, current = recorded_and_current
    problems = fingerprint_diff(recorded, current)
    assert not problems, (
        "訓練記錄的指紋 ≠ eval 端重建的指紋——閘門會誤擋每一次正當的 eval：\n"
        + "\n".join(f"    {p}" for p in problems)
        + f"\n  若是因為 {_CONFIG.name} 或 LiquidOperator 欄位有意變更，"
        "重錄 fixture（見本檔 module docstring 末段），不要放寬比對。")


def test_the_comparison_has_discriminating_power(recorded_and_current):
    """竄改一個欄位必須被抓到。少了這條，上面的「相等」不代表任何事
    ——兩個空 dict 也相等。"""
    recorded, current = recorded_and_current
    assert recorded, "指紋是空的，比對無意義"

    flag = "disable_cross_attention"
    assert flag in recorded, f"fixture 沒有 {flag}，換一個欄位做這個自檢"
    flipped = {**recorded, flag: "True" if recorded[flag] == "False" else "False"}

    caught = fingerprint_diff(flipped, current)
    assert any(flag in p for p in caught), (
        f"竄改 {flag} 後仍判定相等 → 比對器沒有鑑別力")


# 重錄 fixture（lab-server，需要一次真實訓練）：
#   scripts/slurm/submit_verify.sh kolmo
#   scp lab-server:~/pi-lnn-jax-ledger/artifacts/cpuab/f1/summary.json \
#       tests/fixtures/summary/f1_single_re.json
# 並在上面的 docstring 更新 job id 與 commit sha——fixture 的出處是它的一部分。


def test_fingerprint_hides_drag_alpha_when_off():
    """阻尼關閉（alpha=0）時指紋不含 drag_alpha → 與加功能前的 ckpt 逐鍵相同。

    Why 這條測試存在：`drag_alpha` 加進 model 時漏了這個省略（c5d89fa），
    當下 **226 個帶指紋的 ckpt 全部**在 `restore_eval_params` 硬 fail——
    訊息是「訓練時無此欄位」，而 α=0 在 `physics.py` 是靜態分支、逐位元等同
    從未有此功能。這是 opt-in 旗標的通則：關閉值必須讓指紋回到「功能不存在」
    的形狀（另兩個實例是 mid_band 與 trainable_fourier）。
    """
    from pi_lnn_jax.models import LiquidOperator

    base = dict(
        sensor_value_dim=2, d_model=16, d_time=4,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=0, num_query_mlp_layers=0,
        query_mlp_hidden_dim=16, operator_rank=8, decoder_attention_heads=1,
        use_temporal_anchor=True, T_total=1.0, temporal_anchor_harmonics=2,
        domain_length=1.0,
    )
    off = LiquidOperator(**base)
    on = LiquidOperator(**{**base, "drag_alpha": 0.3})

    assert "drag_alpha" not in model_fingerprint(off), (
        "關閉時仍留在指紋裡 → 加功能前訓練的 ckpt 全數評估不了")
    assert model_fingerprint(on)["drag_alpha"] == "0.3", (
        "開啟時必須留在指紋裡，否則 flag-on 會被誤判為同一個模型")
    # 鑑別力自檢：省略不能是「兩邊都空」造成的假綠
    assert fingerprint_diff(model_fingerprint(off), model_fingerprint(on)), (
        "on/off 判定相同 → 省略規則寫過頭了")
