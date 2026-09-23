"""physics / loss 測試共用的最小 LiquidOperator 配方。

What:
    `minimal_kwargs(sensor_value_dim, **over)` 回傳建一個「小到不能再小、但四條
    路徑（spatial encoder / temporal CfC / token attention / decoder）都still在」
    的 LiquidOperator 所需的 kwargs。

Why:
    這組 10 個必填 kwargs 原本逐字複製在 7 處（4 處 `sensor_value_dim=2`、
    3 處 `=3`，其餘欄位完全相同）。承重的不是「它們必須相等」——沒有測試依賴
    這件事——而是**成本**：`LiquidOperator` 每多一個必填欄位，就要改 7 個地方，
    而漏掉的那個是在跑起來才炸。

    ⚠️ 這裡不省時間。同 config 同 shape 的圖在單一 process 內本來就共用 XLA
    executable，`--dist loadfile` 下各檔又跑在不同 worker。它買的是單一定義。

為什麼不是 `conftest.py` 的 fixture：
    這 7 處全部位於 **module 層的 helper**（`_build()`／`_sensor_loss()`／
    `_build_loss_and_batch()`）之中，不在測試函式裡。fixture 到不了那裡，硬要用
    就得讓每個測試多收一個參數再往下傳。而且被共用的是**常數**，不是需要
    setup/teardown 的資源——用 fixture 包一個 constructor 只是把 import 寫成
    fixture。走本 repo 既有的裸模組慣例（`_boundary_scan` / `_acceptance_strictness`）。
"""
from __future__ import annotations

#: 四條路徑都保留、但每條都壓到最小寬度。改這裡等於同時改 7 個測試檔的模型，
#: 所以只在 `LiquidOperator` 的必填介面變動時動它，不要拿它調某一個測試的數值。
_MINIMAL = {
    "d_model": 16, "d_time": 4,
    "num_spatial_encoder_layers": 1, "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 1, "token_attention_heads": 2,
    "query_mlp_hidden_dim": 16, "operator_rank": 8, "decoder_attention_heads": 1,
}


def minimal_kwargs(sensor_value_dim: int, **over) -> dict:
    """`sensor_value_dim` 必填而非預設——2（uv）與 3（uvp）的差別是承重的，
    讓呼叫端在自己的檔案裡明說它測的是哪一種 target。
    """
    return {"sensor_value_dim": sensor_value_dim, **_MINIMAL, **over}
