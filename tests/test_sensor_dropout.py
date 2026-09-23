"""實驗2 sensor dropout 核心純函式測試。

語意（zero-mask，非移除）：
  - B0 VanillaDeepONet 的 K_sensors 是編譯期常數 → 不能移除 sensor，只能保留 K 維把
    drop 的 sensor value 設 0。B3/B0 共用同一機制以維持公平對照。
  - keep_mask [K]：1=keep, 0=drop；drop 數 = round(rate*K)，隨機選哪些。
  - 套用只作用於「模型輸入」；data-loss target 用未 mask 的原始 sensor（呼叫端負責分離）。
"""
import jax
import jax.numpy as jnp

from pi_lnn_jax.sensor_dropout import make_keep_mask, apply_sensor_dropout


def test_rate_zero_keeps_all():
    mask = make_keep_mask(jax.random.PRNGKey(0), K=100, dropout_rate=0.0)
    assert mask.shape == (100,)
    assert int(jnp.sum(mask)) == 100


def test_drop_count_is_round_rate_times_K():
    # rate=0.3, K=100 → drop 30 → keep 70
    mask = make_keep_mask(jax.random.PRNGKey(1), K=100, dropout_rate=0.3)
    assert int(jnp.sum(mask)) == 70
    # mask 只含 0/1
    assert set(jnp.unique(mask).tolist()) <= {0.0, 1.0}


def test_drop_count_rounds_half():
    # rate=0.335, K=100 → round(33.5)=34 drop → 66 keep（Python round=banker's：round(33.5)=34? 用 int(x+0.5)）
    mask = make_keep_mask(jax.random.PRNGKey(2), K=100, dropout_rate=0.5)
    assert int(jnp.sum(mask)) == 50


def test_same_key_reproducible():
    m1 = make_keep_mask(jax.random.PRNGKey(7), K=100, dropout_rate=0.3)
    m2 = make_keep_mask(jax.random.PRNGKey(7), K=100, dropout_rate=0.3)
    assert bool(jnp.all(m1 == m2))


def test_different_key_different_mask():
    m1 = make_keep_mask(jax.random.PRNGKey(7), K=100, dropout_rate=0.3)
    m2 = make_keep_mask(jax.random.PRNGKey(8), K=100, dropout_rate=0.3)
    # 極不可能兩組隨機選點完全相同
    assert not bool(jnp.all(m1 == m2))


def test_apply_zeros_dropped_keeps_rest():
    T, K, C = 5, 100, 2
    vals = jax.random.normal(jax.random.PRNGKey(3), (T, K, C))
    mask = make_keep_mask(jax.random.PRNGKey(4), K=K, dropout_rate=0.3)
    out = apply_sensor_dropout(vals, mask)
    assert out.shape == vals.shape
    # dropped 位置全 0（跨 T,C）
    dropped = mask == 0.0
    assert bool(jnp.all(out[:, dropped, :] == 0.0))
    # kept 位置不變
    kept = mask == 1.0
    assert bool(jnp.allclose(out[:, kept, :], vals[:, kept, :]))


def test_apply_shape_preserved_for_b0_compat():
    # 關鍵：K 維不變（B0 K_sensors 編譯期常數要求）
    T, K, C = 3, 100, 2
    vals = jnp.ones((T, K, C))
    mask = make_keep_mask(jax.random.PRNGKey(5), K=K, dropout_rate=0.5)
    out = apply_sensor_dropout(vals, mask)
    assert out.shape == (T, K, C)
