"""預設建構下的指紋鍵集是凍結的——新增鍵會讓既有 ckpt 全數評估不了。

## 為什麼需要這一條

`model_fingerprint` 把 model dataclass 的欄位寫進 `summary.json` 的
`model_construction`，`fingerprint_diff` 則對「訓練時無此欄位」硬 raise。
兩者相加的後果是：**往 model 加一個 dataclass 欄位，就會讓所有更早訓練的
ckpt 在 `restore_eval_params` 失敗**——訊息是「訓練時無此欄位」，看起來像
ckpt 壞了，實際上壞的是新加的欄位沒做相容處理。

這不是假想。2026-09-02 `c5d89fa` 加入 `drag_alpha` 後實測：lab-server 上
401 份 `summary.json`、226 份有指紋，**226 份全數**過不了閘門，含論文 2×2
消融的 `abl_b0_s*`。而 `drag_alpha=0` 在 `physics.py` 是靜態分支、逐位元
等同「從未有此功能」，那是純粹的假紅。

相容做法已經有三個實例（見 `model_fingerprint` 尾端）：**opt-in 旗標在關閉值
時要從指紋省略**，讓 flag-off 的指紋與「從未有此功能」的模型逐鍵相同。
問題是沒有東西強迫下一個人想到這件事——本檔就是那個強迫。

## 這張表怎麼運作

凍結的是**預設建構下的鍵集**，不是值。於是：

* 新欄位有做「關閉值省略」→ 鍵集不變 → 靜默通過。**相容的路不需要登錄。**
* 新欄位沒做 → 鍵集變大 → 本檔失敗，並在訊息裡把該做的判斷講清楚。

不對稱是刻意的：只有**不相容**的那條路需要有人明確表態。真的必須讓某個
欄位在預設值下也進指紋（它的預設值本身就改變行為），那就更新這張表——
而更新它等於宣告「所有既有 ckpt 從此不可評估」，那正是該被看見的時刻。

慣例沿用 `test_projection_keys_frozen.py`：硬編碼字面表，不從被檢查的物件推導
（由被檢查之物構成的斷言會恆真）。
"""
from __future__ import annotations

import pytest

from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.models import (
    LiquidOperator,
    StandardPINNOperator,
    VanillaDeepONetOperator,
)

#: 各 arch 在**只給必填欄位**時的指紋鍵集。硬編碼字面值。
FROZEN_FINGERPRINT_KEYS: dict[str, set[str]] = {
    "liquid": {
        "T_total", "attention_kind", "body_center_x", "body_center_y", "body_radius",
        "cfc_input_dependent_tau", "cfc_log_tau_max", "cfc_log_tau_min",
        "cfc_tau_mod_scale", "d_model", "d_time", "decoder_attention_heads",
        "disable_cross_attention", "domain_length", "forcing_A_init", "forcing_k_f_init",
        "forcing_k_f_max", "forcing_k_f_min", "fourier_embed_dim",
        "fusion_temperature_init", "learn_forcing_A", "learn_forcing_k_f",
        "num_query_mlp_layers", "num_spatial_encoder_layers", "num_temporal_cfc_layers",
        "num_token_attention_layers", "operator_rank", "output_head_gain",
        "periodic_domain", "query_mlp_hidden_dim", "relpos_bias_mode", "rwf_mean",
        "rwf_stddev", "sensor_value_dim", "temporal_anchor_harmonics",
        "token_attention_heads", "torch_style_init", "use_decoder_remat",
        "use_locality_decay", "use_modified_mlp", "use_rwf", "use_sdf_features",
        "use_temporal_anchor",
    },
    "vanilla": {
        "K_sensors", "T_total", "d_time", "domain_length", "forcing_A_init",
        "forcing_k_f_init", "forcing_k_f_max", "forcing_k_f_min", "fourier_embed_dim",
        "hidden_dim", "learn_forcing_A", "learn_forcing_k_f", "num_branch_layers",
        "num_trunk_layers", "operator_rank", "output_head_gain", "sensor_value_dim",
        "temporal_anchor_harmonics", "use_temporal_anchor",
    },
    "pinn": {
        "T_total", "d_time", "domain_length", "forcing_A_init", "forcing_k_f_init",
        "forcing_k_f_max", "forcing_k_f_min", "fourier_embed_dim", "hidden_dim",
        "learn_forcing_A", "learn_forcing_k_f", "num_layers", "output_head_gain",
        "temporal_anchor_harmonics", "use_temporal_anchor",
    },
}

#: 只給必填欄位（其餘走 dataclass 預設）——「預設建構」的定義。
_MINIMAL = {
    "liquid": lambda: LiquidOperator(
        sensor_value_dim=2, d_model=16, d_time=4,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1),
    "vanilla": lambda: VanillaDeepONetOperator(K_sensors=8),
    "pinn": lambda: StandardPINNOperator(),
}


@pytest.mark.parametrize("arch", sorted(FROZEN_FINGERPRINT_KEYS))
def test_default_construction_fingerprint_keys_are_frozen(arch):
    model = _MINIMAL[arch]()
    actual = set(model_fingerprint(model))
    expected = FROZEN_FINGERPRINT_KEYS[arch]

    added = sorted(actual - expected)
    removed = sorted(expected - actual)
    assert not (added or removed), (
        f"{arch} 在預設建構下的指紋鍵集變了。\n"
        f"  新增：{added}\n  消失：{removed}\n\n"
        "新增鍵代表：**所有更早訓練、帶指紋的 ckpt 從此在 restore_eval_params 失敗**\n"
        "（訊息會是「訓練時無此欄位」，看起來像 ckpt 壞了）。2026-09-02 實測一次，\n"
        "226 份帶指紋的 summary.json 全中。先回答一個問題再決定怎麼修：\n\n"
        "  這個欄位在它的**預設值**下，行為與「從未有此功能」相同嗎？\n\n"
        "  相同 → 在 model_fingerprint 尾端加一段，預設值時 fp.pop 掉它。\n"
        "         （已有三個實例：mid_band / trainable_fourier / drag_alpha）\n"
        "         這樣鍵集不變，本測試自動通過，既有 ckpt 不受影響。\n\n"
        "  不同 → 那它確實改變了模型身分，既有 ckpt 本來就不該被拿來評估。\n"
        "         更新上面那張表，並在 PR 說明裡寫清楚哪些既有結果因此失效。\n\n"
        "消失的鍵同樣要看：移除欄位會讓帶該欄位的舊指紋報「eval 無此欄位」。")


def test_the_frozen_table_has_discriminating_power():
    """自證：表若是從被檢查之物推導出來的，斷言會恆真而測不到任何東西。

    這裡反向構造一個「新增了欄位」的情境——把 drag_alpha 打開，它就會出現在
    指紋裡——並確認上面那條斷言確實會抓到多出來的鍵。
    """
    on = set(model_fingerprint(LiquidOperator(
        sensor_value_dim=2, d_model=16, d_time=4,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        drag_alpha=0.35)))
    assert on - FROZEN_FINGERPRINT_KEYS["liquid"] == {"drag_alpha"}, (
        "凍結表抓不到多出來的鍵——它沒有鑑別力")
