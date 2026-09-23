"""arch flag → Flax Module 的唯一 factory。

What: `build_model(arch, model_kwargs, K_sensors)` 依 `--arch` 建構 B3 / B0 / B2
      三種 operator，並負責過濾各 constructor 不吃的 kwargs。

Why: 這份 factory 原本在 `train_kolmogorov.py` 與 `scripts/evaluate_exp245.py`
     各存一份逐行複製。兩份漂移的後果不是 crash，而是 eval 用「另一個架構」去
     重建 checkpoint 的模型——orbax 的 lenient restore 只比對 params 樹，
     架構層級的差異（例如新增一個影響 forward 但不改 param shape 的 flag）
     會安靜通過，產出看似合理卻錯誤的 headline 數字。放進套件讓訓練與 eval
     共用同一個物件，是唯一能從源頭消除漂移的做法（見 tests/test_model_factory.py）。
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .models import (
    LiquidOperator,
    StandardPINNOperator,
    VanillaDeepONetOperator,
)

#: Flax 在每個 Module 上掛的框架欄位，與「這個模型是怎麼建的」無關。
_NON_CONSTRUCTION_FIELDS = frozenset({"parent", "name"})

# LiquidOperator 以外的 operator 共同接受的 kwargs。
# 不在此集合的 key（d_model、num_attention_heads 等 B3 專用）會被濾掉，
# 否則 vanilla / pinn 的 constructor 會直接 TypeError。
COMMON_KEYS = frozenset({
    "sensor_value_dim", "d_time", "domain_length",
    "use_temporal_anchor", "T_total", "temporal_anchor_harmonics",
    "output_head_gain", "fourier_embed_dim",
    "learn_forcing_A", "learn_forcing_k_f",
    "forcing_A_init", "forcing_k_f_init",
    "forcing_k_f_min", "forcing_k_f_max",
    "drag_alpha",
})


def build_model(arch: str, model_kwargs: dict, K_sensors: int):
    """根據 arch flag 建構對應的 Flax Module。

    返回 `(model, model_name)` — model_name 供 log 標示是哪個 baseline。
    不修改傳入的 `model_kwargs`（呼叫端常重複使用同一個 dict）。
    """
    if arch == "liquid":
        kwargs = dict(model_kwargs)  # 不改傳入 dict（呼叫端常重用）
        # 方向1: TOML array 解析成 list，但 Flax module field 需 hashable → 轉 tuple
        if isinstance(kwargs.get("mid_band_wavenumbers"), list):
            kwargs["mid_band_wavenumbers"] = tuple(kwargs["mid_band_wavenumbers"])
        return LiquidOperator(**kwargs), "LiquidOperator (B3)"

    if arch == "vanilla":
        # B0: VanillaDeepONet 需要 K_sensors 作編譯期常數
        vanilla_kwargs = {k: v for k, v in model_kwargs.items() if k in COMMON_KEYS}
        # B0 自己的 hidden_dim ≈ query_mlp；branch/trunk 深度共用 query MLP 層數
        vanilla_kwargs["hidden_dim"] = model_kwargs.get("query_mlp_hidden_dim", 64)
        vanilla_kwargs["operator_rank"] = model_kwargs.get("operator_rank", 64)
        vanilla_kwargs["num_branch_layers"] = model_kwargs.get("num_query_mlp_layers", 3) or 3
        vanilla_kwargs["num_trunk_layers"] = model_kwargs.get("num_query_mlp_layers", 3) or 3
        return (
            VanillaDeepONetOperator(K_sensors=K_sensors, **vanilla_kwargs),
            f"VanillaDeepONet (B0, K={K_sensors})",
        )

    if arch == "pinn":
        pinn_kwargs = {k: v for k, v in model_kwargs.items() if k in COMMON_KEYS}
        # pinn 無 sensor branch，不吃 sensor_value_dim
        pinn_kwargs.pop("sensor_value_dim", None)
        pinn_kwargs["hidden_dim"] = model_kwargs.get("query_mlp_hidden_dim", 64)
        pinn_kwargs["num_layers"] = max(1, model_kwargs.get("num_query_mlp_layers", 6) or 6)
        return StandardPINNOperator(**pinn_kwargs), "StandardPINN (B2)"

    raise ValueError(f"unknown arch: {arch}")


def model_fingerprint(model: Any) -> dict[str, Any]:
    """建構指紋：這個 model 實例實際是用什麼建出來的。

    What: 取 dataclass 欄位的實際值（排除 Flax 框架欄位），JSON-safe。

    Why 取自實例而非 config：
      本模組開頭那段 Why 講的是「兩份 factory 程式碼漂移」，共用 factory 已經解決。
      但還有一個它擋不住的漂移——訓練用的 config 與 eval 用的 config 不同。
      `ckpt.verify_params_tree` 只比 path+shape，所以**改 forward 卻不改參數樹**的
      旗標完全隱形（實測 `disable_cross_attention` 兩側都是 14546 個參數、樹全等，
      而同一份 params 的輸出差 4.6e-2）。指紋補的就是這一段。

      取自實例才是「實際建構」：`build_model` 對 vanilla/pinn 會用 `COMMON_KEYS`
      濾掉一半的 kwargs，cylinder 那條路更是完全繞過 config 用模組級 CFG——
      config 的 `model_kwargs` 描述不了這兩者，dataclass 欄位可以。

    值一律轉成字串：指紋只用於相等比較與「差在哪」的回報，不需還原型別，
    而 tuple/enum/np scalar 混在 config 裡會讓 JSON round-trip 後比對出假差異。
    """
    fp = {
        f.name: str(getattr(model, f.name))
        for f in dataclasses.fields(model)
        if f.name not in _NON_CONSTRUCTION_FIELDS
    }
    # 方向1 opt-in：mid-band 分支關閉（wavenumbers=()）時，指紋與「從未有此功能」的
    # 模型逐鍵相同 → 保住既有 ckpt / §7.1 A/B 契約，不因新增旗標而假紅。開啟時兩欄
    # 都保留 → flag-on 正確被辨為不同 arch。
    if getattr(model, "mid_band_wavenumbers", ()) == ():
        fp.pop("mid_band_wavenumbers", None)
        fp.pop("mid_band_embed_dim", None)
        fp.pop("mid_band_init_sigma", None)
    # 同理，方向1 exp_508 的 trainable-frequency 分支關閉時也整組省略。
    if not getattr(model, "use_trainable_fourier", False):
        fp.pop("use_trainable_fourier", None)
        fp.pop("trainable_fourier_dim", None)
        fp.pop("trainable_fourier_init_scale", None)
        fp.pop("trainable_fourier_lowpass_kc", None)
    # 同理，線性阻尼關閉（alpha=0）時省略。`physics.py` 的 `if drag_alpha:` 是靜態
    # 分支，α=0 時整段不進運算圖 → 逐位元等同「從未有此功能」。
    # 少這一條的後果實測過：c5d89fa 加入本欄位後，**226 個帶指紋的 ckpt 全部**
    # 在 restore_eval_params 硬 fail（訊息是「訓練時無此欄位」），含論文 2×2 消融的
    # abl_b0_s*。新增 opt-in 旗標時這是必要配套，不是可選的。
    if not getattr(model, "drag_alpha", 0.0):
        fp.pop("drag_alpha", None)
    # 同理，Wang 2021 跨分支 mMLP 的兩個 opt-in 旗標**同時**關閉時整組省略：兩者皆 off
    # 時 forward 與「從未有此功能」逐位元相同。任一開啟則兩欄都留，讓 flag-on 被辨為
    # 不同 arch。
    if not (getattr(model, "mmlp_branch_u", False) or getattr(model, "mmlp_gate_branch", False)):
        fp.pop("mmlp_branch_u", None)
        fp.pop("mmlp_gate_branch", None)
    # 同理，relpos bias 的兩個 opt-in 旗標在關閉值時各自省略。"layernorm" 是既有行為
    # （那條 bias 在多數 config 下是 no-op），zero_init=False 亦然。
    if getattr(model, "relpos_bias_norm", "layernorm") == "layernorm":
        fp.pop("relpos_bias_norm", None)
    if not getattr(model, "relpos_bias_zero_init", False):
        fp.pop("relpos_bias_zero_init", None)
    return fp


def fingerprint_diff(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """兩份指紋的差異，逐欄位列出；空 list = 相符。

    回報必須點名欄位——使用者深夜撞到這個閘門時，唯一想知道的是「差在哪」，
    只說「不同」等於把診斷工作原封退回去。
    """
    problems: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        if key not in recorded:
            problems.append(f"{key}: 訓練時無此欄位，eval 為 {current[key]!r}")
        elif key not in current:
            problems.append(f"{key}: 訓練時為 {recorded[key]!r}，eval 無此欄位")
        elif recorded[key] != current[key]:
            problems.append(f"{key}: 訓練 {recorded[key]!r} ≠ eval {current[key]!r}")
    return problems
