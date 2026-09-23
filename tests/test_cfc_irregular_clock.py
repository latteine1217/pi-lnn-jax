"""CfC 的不等距時鐘：真實間距要進到狀態，且第一步的慣例不得靜默改變。

Why this file exists:
    「tolerate the irregular, unsynchronised timestamps of field sensors」是選用
    CfC 的理由（`chapter02.tex`），但那條路**零測試覆蓋**：把
        dts = concat([sensor_time[:1], diff(sensor_time)])
    換成「用平均間隔取代真實 gap」——語意上徹底廢掉不等距支援，而在等間隔
    時鐘上逐位元相同——全庫沒有任何測試偵測得到。根因是每個測試的
    `sensor_time` 都是 `linspace` / `arange`，全部等距。

    本檔用**同 t0、同平均間隔、不同間距**的兩條時間軸當判別器：真的吃間距
    才會有差，用平均值就會相同。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator

K, T = 5, 6


@pytest.fixture(scope="module")
def encode():
    """回 encode(sensor_time) -> h_states，其餘輸入固定。"""
    model = LiquidOperator(**minimal_kwargs(3))
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(
        jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1, jnp.linspace(0.0, 1.0, T),
        jax.random.uniform(jax.random.PRNGKey(9), (4, 2)), jnp.zeros((4,)))

    def _encode(sensor_time):
        return model.apply(params, sensor_vals, sensor_pos, 0.1,
                           jnp.asarray(sensor_time, dtype=jnp.float32),
                           method=LiquidOperator.encode)
    return _encode


# 同 t0=0、同 t_end=1、同平均間隔 0.2，但間距不等。用平均值取代真實 gap
# 會讓這條與 UNIFORM 完全等價——那正是要抓的失效。
UNIFORM = np.linspace(0.0, 1.0, T)
IRREGULAR = np.array([0.0, 0.05, 0.10, 0.55, 0.60, 1.00])


def test_the_two_axes_are_a_valid_discriminator():
    """先證明判別器本身成立：兩條軸的平均間隔相同，只有間距不同。"""
    assert np.allclose(np.diff(UNIFORM).mean(), np.diff(IRREGULAR).mean())
    assert not np.allclose(np.diff(UNIFORM), np.diff(IRREGULAR))
    assert UNIFORM[0] == IRREGULAR[0], "t0 必須相同，否則測到的是 dts[0] 而非間距"


def test_irregular_spacing_reaches_the_recurrent_state(encode):
    """真實 gap 要進到 CfC——換成平均間隔就會讓這個斷言變綠而語意已死。"""
    delta = float(jnp.max(jnp.abs(encode(UNIFORM) - encode(IRREGULAR))))
    assert delta > 1e-3, (
        f"不等距與等距（同平均間隔）的隱狀態只差 {delta:.3e}——"
        "CfC 沒有真的吃到 per-step 間距")


def test_first_step_uses_the_absolute_time_not_an_interval(encode):
    """`dts[0] = sensor_time[0]`：第一步餵的是絕對時刻，不是間隔。

    這與 `chapter02.tex:265` 的 Δt_n = t_n − t_{n−1} 不符，是**已知的既有行為**，
    在此釘住而非修正：主資料集 t0=0 故無害，但 §7.2 的間歇集 t0=0.30/0.40
    會在第 0 步注入一個數倍於名目步長的假 gap。改成 `concat([zeros(1), diff])`
    會動到 cylinder 與 intermittent 的數字，屬 CLAUDE.md §7.1 的 bit-identical
    契約，必須走 A/B 對拍並由人裁決。

    判別器：兩條**間距完全相同**、只差 t0 的軸。現行行為下輸出不同；
    改成正確形式後兩者會相同——所以這個斷言會在修好的那天變紅，
    那正是它該做的事（提醒一起更新契約與稿子）。
    """
    shifted = UNIFORM + 0.30                      # 間距不變，只平移原點
    assert np.allclose(np.diff(UNIFORM), np.diff(shifted))
    delta = float(jnp.max(jnp.abs(encode(UNIFORM) - encode(shifted))))
    assert delta > 1e-3, (
        f"時間原點平移後隱狀態只差 {delta:.3e}——若 dts[0] 已改為 0，"
        "請同步更新本測試、chapter02.tex 的 Δt 定義，並跑 §7.1 的 A/B 對拍")
