"""實驗2 sensor dropout 核心（zero-mask，非移除 sensor）。

Why zero-mask 而非移除：
  B0 VanillaDeepONetOperator 的 branch 把 sensor flatten 成 [N, K*C] 餵 MLP，
  K_sensors 是編譯期常數（models.py），移除 sensor 會改 branch 權重 shape。
  故 B0/B3 統一用「保留 K 維、drop 的 sensor value 設 0」以維持公平對照——
  差異純粹來自架構能否學會忽略死掉的 sensor（CfC 時間記憶 vs vanilla MLP）。

用法（兩階段共用）：
  eval-time：固定 key 生成一組 mask，多 realization 平均。
  train-time：每 step 用新 key 生成 mask，比照 sensor_idx 在 jit 外預生成、
    shape 固定 [K] → jit train_step 不 retrace。mask 只作用於模型輸入，
    data-loss target 用未 mask 的原始 sensor（呼叫端負責分離）。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def make_keep_mask(key: jax.Array, K: int, dropout_rate: float) -> jnp.ndarray:
    """回傳 [K] float mask（1=keep, 0=drop）。

    drop 數 = round(dropout_rate * K)，隨機選哪些 sensor。K 與 dropout_rate 為
    static Python 值（此函式在 jit 外呼叫），故 n_drop 可作 static slice。
    """
    n_drop = int(round(float(dropout_rate) * int(K)))
    mask = jnp.ones((int(K),), dtype=jnp.float32)
    if n_drop <= 0:
        return mask
    drop_idx = jax.random.permutation(key, int(K))[:n_drop]
    return mask.at[drop_idx].set(0.0)


def apply_sensor_dropout(sensor_vals: jnp.ndarray, keep_mask: jnp.ndarray) -> jnp.ndarray:
    """對 sensor_vals [T, K, C] 沿 K 維套 zero-mask。K 維與 shape 不變（B0 相容）。"""
    m = keep_mask.astype(sensor_vals.dtype)
    return sensor_vals * m[None, :, None]
