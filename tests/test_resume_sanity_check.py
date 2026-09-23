"""釘住 resume 那道守衛——它先前在生產規模上跑不起來，而且會發假警報。

兩個獨立的缺陷疊在同一個函式上（model-audit §3）：

1. **記憶體**：`sensor_idx` 寫死 `None` → 拿**全量** T*K 網格過一次 decoder，
   而訓練走的是 mini-batch。K=200 是 20200 對 2000，cross-attention 的中介張量
   隨之放大，在 r740 上直接 `RESOURCE_EXHAUSTED`。lab-server 的 955 份 job log 裡
   沒有任何 production run 走過 resume，所以從沒被觸發——「訓練中斷可以續跑」
   一直是**未驗證的能力**。

2. **語意**：它拿全量的 sensor loss 去比 `last_logged_loss`，而後者是訓練時
   **隨機 mini-batch** 算的。兩者是同一期望值的不同估計量，實測 n=2000 的
   相對標準誤約 3.3%，而容忍度訂 1e-3——**噪聲是容忍度的 33 倍**，所以就算
   resume 完美無缺也保證會警告。假警報與靜默通過一樣沒有守衛效果。

修法是讓兩側算同一個量：`finalize` 用 `deterministic_sensor_idx` 落下參考值，
resume 端用同一組索引重算。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.pipeline.kolmogorov.run import deterministic_sensor_idx


def test_subsample_is_smaller_than_the_full_grid():
    """這就是記憶體那一半：拿到的點數必須是 mini-batch 尺度，不是全量。"""
    T, K, n_sq = 101, 200, 2000
    idx = deterministic_sensor_idx(T * K, n_sq)
    assert idx is not None and idx.shape == (n_sq,), (
        f"K={K} 下應取 {n_sq} 點，實得 {None if idx is None else idx.shape}——"
        "回到全量就會在 r740 上 RESOURCE_EXHAUSTED")
    assert int(idx.max()) < T * K and int(idx.min()) >= 0


def test_subsample_spans_the_whole_time_axis():
    """攤平是 time-major，取前綴只會看到最早的 n/K 個時刻。

    這與 GradNorm probe 的那條是同一個失效形狀，所以用同樣的判準釘住。
    """
    T, K, n_sq = 101, 200, 2000
    idx = np.asarray(deterministic_sensor_idx(T * K, n_sq))
    rows = idx // K                      # 每個點落在哪一個時刻
    assert rows.min() == 0, "沒有涵蓋最早的時刻"
    assert rows.max() >= T - 2, f"最晚只到第 {rows.max()} 列（共 {T}）——像是取了前綴"
    assert len(np.unique(rows)) >= T // 2, (
        f"只碰到 {len(np.unique(rows))} 個相異時刻——時間覆蓋不足")


def test_subsample_does_not_lock_onto_the_first_sensors():
    """`index mod K` 必須循環，否則會鎖定 QR-pivot 的前幾個高重要度感測器。"""
    T, K, n_sq = 101, 200, 2000
    idx = np.asarray(deterministic_sensor_idx(T * K, n_sq))
    cols = idx % K
    assert len(np.unique(cols)) > K // 4, (
        f"只碰到 {len(np.unique(cols))} 個相異感測器（共 {K}）——取樣鎖在同一批上")


def test_full_grid_is_returned_as_none_when_it_already_fits():
    """小 K 下全量本來就跑得動，回 None 讓 loss_fn 走既有的全量路徑。"""
    assert deterministic_sensor_idx(500, 2000) is None      # n_take >= n_total
    assert deterministic_sensor_idx(500, 500) is None
    # 0 與 None 都代表「不做 mini-batch」，與 loss_fn 對 sensor_idx=None 的語意一致
    assert deterministic_sensor_idx(500, 0) is None
    assert deterministic_sensor_idx(500, None) is None


def test_it_is_deterministic():
    """兩側必須算同一個量——不確定的取樣會讓比較失去意義。"""
    a = deterministic_sensor_idx(101 * 200, 2000)
    b = deterministic_sensor_idx(101 * 200, 2000)
    assert jnp.array_equal(a, b)
