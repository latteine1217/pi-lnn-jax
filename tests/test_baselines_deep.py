"""Phase B 深度 baseline（SHRED）的 isolated unit 測試（TDD: 先寫）。

只做 isolated unit 驗證（合成張量、CPU、毫秒級）：forward 形狀、可微分/可訓練、window 切分。
※ 不在本機跑完整訓練（遵守 no-local-training）；完整訓練走 lab-server。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pi_lnn_jax.baselines_deep import SHRED, build_windows


def test_shred_forward_shape():
    model = SHRED(out_dim=8 * 8 * 2, hidden=16, decoder_hidden=(32,))
    x = jnp.zeros((4, 5, 10 * 2))  # B=4, L=5, K=10, C=2 → K*C=20
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)
    assert y.shape == (4, 8 * 8 * 2)


def test_shred_one_training_run_reduces_loss():
    """isolated unit：合成資料上跑數步 Adam，MSE 應下降（驗證 loss/grad wiring 可訓練）。"""
    model = SHRED(out_dim=8 * 8 * 2, hidden=16, decoder_hidden=(32,))
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (8, 5, 20))
    y_true = jax.random.normal(jax.random.PRNGKey(1), (8, 8 * 8 * 2))
    params = model.init(rng, x)

    def loss_fn(p):
        return jnp.mean((model.apply(p, x) - y_true) ** 2)

    l0 = float(loss_fn(params))
    opt = optax.adam(1e-2)
    state = opt.init(params)
    for _ in range(30):
        g = jax.grad(loss_fn)(params)
        upd, state = opt.update(g, state)
        params = optax.apply_updates(params, upd)
    assert float(loss_fn(params)) < l0


def test_build_windows_shapes_and_alignment():
    T, K, nf, L = 10, 4, 6, 3
    sensor_vals = np.arange(T * K * 2, dtype=np.float32).reshape(T, K, 2)
    fields = np.arange(T * nf, dtype=np.float32).reshape(T, nf)
    X, Y = build_windows(sensor_vals, fields, L)
    assert X.shape == (T - L + 1, L, K * 2)
    assert Y.shape == (T - L + 1, nf)
    # 對齊：第 i 個 window 的 target = window 末端時間 (i+L-1) 的 field
    assert np.allclose(Y[0], fields[L - 1])
    assert np.allclose(X[0], sensor_vals[0:L].reshape(L, K * 2))


def test_build_windows_fail_when_too_short():
    import pytest
    with pytest.raises(ValueError):
        build_windows(np.zeros((2, 4, 2)), np.zeros((2, 6)), L=5)
