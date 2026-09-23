"""GPT-5.5 審查驗證後 confirmed bug 修復的回歸守門。

涵蓋：
- H2：causal_weights 改 Wang time-slab → 同時刻點同權重 + 採樣數不變。
- H9：compute_energy_spectrum 涵蓋 corner modes → Parseval 完整（checkerboard）。
- H8：compute_metrics 的 u/v rel-err 套用 mask（與 KE/div/vorticity 一致）。
"""
import numpy as np
import jax.numpy as jnp

from pi_lnn_jax.causal import causal_weights
from pi_lnn_jax.evaluate import compute_energy_spectrum, compute_metrics


def test_causal_weights_slab_permutation_invariant():
    """同一時刻的點必須得到相同 causal weight（slab 聚合，非逐點 prefix sum）。"""
    ct = jnp.array([0.0, 1.0, 1.0, 2.0])
    r = jnp.array([1.0, 4.0, 0.0, 1.0])
    w = causal_weights(ct, r, 0.5)
    assert abs(float(w[1]) - float(w[2])) < 1e-6, "同時刻點權重不同 → slab 不變性破壞"


def test_causal_weights_sampling_count_invariant():
    """某時刻的 slab 權重不應隨「同殘差的其他時刻採樣點數」改變。"""
    def w_last(n0):
        ct = jnp.concatenate([jnp.zeros(n0), jnp.array([1.0])])
        r = jnp.ones(n0 + 1)
        return float(causal_weights(ct, r, 0.5)[-1])
    assert abs(w_last(1) - w_last(4)) < 1e-6, "t=1 權重隨 t=0 採樣數變 → 採樣數不變性破壞"


def test_causal_weights_eps_zero_uniform():
    """eps=0 → 全 1（關閉）。"""
    ct = jnp.array([0.0, 1.0, 2.0, 1.0])
    r = jnp.array([1.0, 2.0, 3.0, 0.5])
    w = causal_weights(ct, r, 0.0)
    assert np.allclose(np.asarray(w), 1.0)


def test_energy_spectrum_parseval_checkerboard():
    """checkerboard 場能量全在對角 corner mode；spectrum sum 須 == 真 KE（0.5）。"""
    N = 8
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    u = ((-1.0) ** (xx + yy)).astype(float)
    v = np.zeros((N, N))
    _k, E = compute_energy_spectrum(u, v)
    ke_true = 0.5 * np.mean(u ** 2 + v ** 2)  # 0.5
    assert abs(E.sum() - ke_true) < 1e-9, f"spectrum sum {E.sum()} != KE {ke_true}（corner mode 漏）"


def test_energy_spectrum_parseval_random():
    """一般場：sum E(k) 須等於 0.5*mean(u²+v²)（Parseval 完整）。"""
    rng = np.random.RandomState(0)
    u = rng.randn(16, 16); v = rng.randn(16, 16)
    _k, E = compute_energy_spectrum(u, v)
    ke_true = 0.5 * np.mean(u ** 2 + v ** 2)
    assert abs(E.sum() - ke_true) / ke_true < 1e-9, "spectrum 非 Parseval 完整"


def test_compute_metrics_velocity_err_respects_mask():
    """u/v rel-err 須套用 mask（body 內大錯被遮罩 → rel-err ≈ 0），與 KE 等量一致。"""
    u_pred = np.ones((4, 4)); u_dns = np.ones((4, 4))
    v_pred = np.zeros((4, 4)); v_dns = np.zeros((4, 4))
    mask = np.ones((4, 4), bool); mask[0, 0] = False  # 排除 body 格點
    u_pred[0, 0] = 100.0  # body 內大錯
    # periodic=True 走 np.roll div/vorticity（不需 dns_x/dns_y）；u_err mask 邏輯與 periodic 無關
    m = compute_metrics(u_pred, v_pred, u_dns, v_dns, periodic=True, mask=mask)
    assert m["u_rel_err"] < 1e-6, "u_rel_err 未套 mask（body 內錯誤洩漏進速度誤差）"
