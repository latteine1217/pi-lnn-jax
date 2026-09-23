"""Branch 自迴歸：用模型自己的預測把 sensor 序列延伸到資料時窗之外。

What:
    `extend_sensor_sequence` 在 sensor 位置上解出 t > 資料末端的 pseudo 觀測，
    接回序列尾端，回傳延伸後的 `(sensor_vals, sensor_time)`。分輪進行：第 r 輪用
    第 r−1 輪延伸後的序列重新 encode，故後段的 pseudo 幀是**基於前段 pseudo 幀**
    解出來的——那才是自迴歸，不是拿同一個凍結狀態解全部。

Why:
    decoder 對 t > sensor_time[-1] 的 query 一律取最後一幀的 `h_states`
    （`models.py` 的 idx clip）。於是外推段所有時刻共用同一份 branch 狀態，
    能隨時間變的只剩 trunk。EXP-530/531/532 的證據把瓶頸指向這裡：
    dt>0 的監督（EXP-532）只補回可改善空間的約 1/4。

    延伸序列讓 branch 重新擁有隨時間演化的狀態。代價是誤差會沿著 rollout 累積，
    這正是要用實驗量的東西。

契約：**訓練端與 eval 端必須呼叫同一個函式**（scripts/CLAUDE.md §1）。
    pseudo 觀測走 `stop_gradient`：它們是輸入不是預測目標，梯度若回穿整條 rollout，
    記憶體與時間都會隨輪數爆炸，而且會把「延伸序列」變成一個被優化的物件。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def pseudo_frame_times(t_last: float, t_end: float, dt: float) -> np.ndarray:
    """(t_last, t_end] 上間距為 dt 的 pseudo 幀時刻。dt 不整除時 fail-fast。"""
    if not (dt > 0.0):
        raise ValueError(f"pseudo 幀間距須為正，收到 {dt}")
    if t_end <= t_last:
        raise ValueError(f"t_end={t_end} 未超過資料末端 {t_last}，無可延伸的區間")
    n = round((t_end - t_last) / dt)
    if abs(n * dt - (t_end - t_last)) > 1e-9:
        raise ValueError(
            f"pseudo 幀間距 {dt} 無法整除延伸區間 {t_end - t_last}（不寬鬆對齊）")
    return t_last + dt * np.arange(1, n + 1, dtype=np.float64)


def check_extension_inputs(pseudo_times, t_last: float, rounds: int) -> np.ndarray:
    """建構期的前提檢查；回傳正規化後的 pseudo_times。

    Why 不放在 `extend_sensor_sequence` 內：那支在 `jax.jit` 之下執行，`sensor_time`
    是 traced array，取 `[-1]` 的**值**會 TracerArrayConversionError。值層面的前提
    在建構期就已知（呼叫端有資料時窗），故在那裡驗；runtime 那支只碰 shape。
    """
    if rounds < 1:
        raise ValueError(f"rounds 須 ≥ 1，收到 {rounds}")
    pseudo_times = np.asarray(pseudo_times, dtype=np.float64)
    if pseudo_times.size == 0:
        raise ValueError("pseudo_times 為空")
    if not np.all(pseudo_times > t_last):
        raise ValueError(f"pseudo_times 必須全部晚於資料末端 {t_last}")
    if rounds > pseudo_times.size:
        raise ValueError(f"rounds={rounds} 多於 pseudo 幀數 {pseudo_times.size}")
    return pseudo_times


def extend_sensor_sequence(
    model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
    pseudo_times, rounds: int, *, encode_method, decode_method,
):
    """回傳 `(vals_ext, time_ext)`：尾端接上模型自己解出的 pseudo 觀測。

    Args:
        pseudo_times: 要補的時刻（遞增，皆 > sensor_time[-1]；前提由
            `check_extension_inputs` 在建構期驗過）。
        rounds: 自迴歸輪數。分成 `rounds` 段依序解；每段都先用「含前面所有 pseudo
            幀」的序列重新 encode。1 = 全部用凍結狀態一次解完（最弱的版本）。

    **本函式在 jit 之下執行**：只碰 shape，不取任何 traced array 的值。
    各輪的序列長度不同 → Python 迴圈展開、shape 靜態，jit 對每輪各編一次。
    """
    pseudo_times = np.asarray(pseudo_times, dtype=np.float64)
    if rounds < 1 or pseudo_times.size == 0 or rounds > pseudo_times.size:
        raise ValueError(
            f"rounds={rounds} 與 pseudo 幀數 {pseudo_times.size} 不相容；"
            "值層面的前提請在建構期用 check_extension_inputs 驗")

    K = int(sensor_pos.shape[0])          # shape 在 tracer 上可取，值不行
    C = int(sensor_vals.shape[-1])
    vals, times = sensor_vals, sensor_time

    for chunk in np.array_split(pseudo_times, rounds):
        h_states = model.apply(
            params, vals, sensor_pos, re_norm, times, method=encode_method)
        n = int(chunk.size)
        xy_q = jnp.tile(sensor_pos, (n, 1))                       # [n*K, 2]
        t_q = jnp.repeat(jnp.asarray(chunk, times.dtype), K)      # [n*K]
        pred = model.apply(
            params, xy_q, t_q, h_states, times, sensor_pos, method=decode_method)
        new_vals = jax.lax.stop_gradient(pred[:, :C]).reshape(n, K, C)
        vals = jnp.concatenate([vals, new_vals.astype(vals.dtype)], axis=0)
        times = jnp.concatenate([times, jnp.asarray(chunk, times.dtype)], axis=0)

    return vals, times
