"""effective_sensor_query_points 的 isolated 單元測試（純 Python）。

sensor mini-batch 走 replace=False 抽樣，設定值一旦大於 population（訓練實際
餵入的 frames × K）就是 ValueError。這個組合在每個「時間軸變短」的 campaign
都會重現——cross-Re 的 K=10、T=11 快照密度、90% 間歇——而各自的 config 生成
器要記得自己算一次 population，漏掉就是 45 秒後才爆的 job。

判斷因此收在訓練端：設定的語意是「每步最多取這麼多」，population 不足時
全取是唯一合理行為，不是使用者設錯。降級會明確印出來，不是靜默改設定。
"""
from __future__ import annotations

import pytest

from pi_lnn_jax.pipeline.kolmogorov.assembly import effective_sensor_query_points


def test_below_population_is_unchanged():
    """一般情況：設定值小於 population，原樣沿用（既有 config 行為不變）。"""
    assert effective_sensor_query_points(2000, population=10100) == 2000


def test_above_population_degrades_to_full():
    """population 不足 → 0（全取）。這是 T=11 / gap90 / K=10 那批的情形。"""
    assert effective_sensor_query_points(2000, population=1100) == 0


def test_equal_to_population_is_kept():
    """恰好等於 population 仍可抽樣（replace=False 取全部是合法的）。"""
    assert effective_sensor_query_points(1000, population=1000) == 1000


def test_zero_stays_zero():
    """0 本來就是「全取」，不受影響。"""
    assert effective_sensor_query_points(0, population=500) == 0


def test_negative_population_raises():
    """population 應由 T×K 推得，非正值代表上游資料已經壞了。"""
    with pytest.raises(ValueError):
        effective_sensor_query_points(100, population=0)


@pytest.mark.parametrize("requested,population,expected", [
    (2000, 1000, 0),      # gap90: 10 frames × K=100
    (2000, 1100, 0),      # T=11:  11 frames × K=100
    (2000, 1010, 0),      # cross-Re K=10 @ Re=500
    (2000, 3000, 2000),   # gap70: 30 frames × K=100 —— 這格本來就過得去
    (2000, 20100, 2000),  # 主線 201 × 100
])
def test_real_campaign_cases(requested, population, expected):
    assert effective_sensor_query_points(requested, population) == expected
