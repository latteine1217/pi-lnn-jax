"""兩條「開了才會發現壞掉」的路：Poisson 接線與 data 權重的靜默 no-op。

兩者都因為生產 config 從不啟用而長期無人踩到，也因此沒有任何測試覆蓋。
稽核把它們挖出來後在此釘住——修好之後要有東西擋住它再壞一次。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.kolmogorov import assembly

K, T, N = 5, 4, 8


def _loss_fn_with_poisson(w_poisson: float):
    model = LiquidOperator(**minimal_kwargs(3))
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1, sensor_time,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=w_poisson > 0.0, use_al=False,
        al_rho=0.0, w_poisson=w_poisson, T_total=1.0)
    re_batch = assembly.ReBatch(
        sensor_vals=sensor_vals, sensor_pos=sensor_pos, sensor_time=sensor_time,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0))
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    tw = jnp.array([1.0, 0.057, 0.057])
    return lambda: loss_fn(params, cx, cy, ct, tw, jnp.asarray(0.0),
                           jnp.asarray(1.0), re_batch, 0.0)


def test_poisson_path_runs_when_enabled():
    """`poisson_loss_weight > 0` 這條路要真的跑得起來。

    先前呼叫端多傳了 `drag_alpha=`，而 `poisson_residual` 沒有這個參數 →
    一開就 TypeError。全庫 config 的 poisson_loss_weight 都是 0，所以那個
    錯誤直到 chapter05.tex:72 那條未來工作被實作才會浮現。
    """
    total, aux = _loss_fn_with_poisson(0.5)()
    assert jnp.isfinite(total), "Poisson 開啟後 total 不是有限值"
    assert jnp.isfinite(aux[4]), "poisson 殘差不是有限值"


def test_poisson_weight_actually_changes_the_total():
    """權重 > 0 必須改變 total——否則「跑得起來」只是跑了個 0。"""
    off, _ = _loss_fn_with_poisson(0.0)()
    on, aux = _loss_fn_with_poisson(0.5)()
    assert float(aux[4]) > 0.0, "poisson 殘差恆為 0，這個測試量不到東西"
    assert not jnp.allclose(on, off), "開啟 Poisson 後 total 沒變"


# ── data 權重的靜默 no-op ─────────────────────────────────────────────────

def _cfg(data_weight: float, refine: str = "none"):
    class _Loss:
        pass
    class _Refine:
        pass
    class _Cfg:
        loss = _Loss()
        refinement = _Refine()
    _Cfg.loss.data_weight = data_weight
    _Cfg.loss.cont_gradnorm = False
    _Cfg.loss.use_gradnorm = True
    _Cfg.refinement.optimizer = refine
    return _Cfg


def _guard(cfg):
    from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy
    KolmogorovPolicy().guard(None, cfg)


def test_nonunit_data_weight_is_rejected_when_refinement_is_off():
    """主 loss 不讀 `data_loss_weight`，設它是逐位元無效的介入。

    危險在於 schema 收下它、provenance 還會記錄，於是「調了 data 權重」的
    ablation 會拿到與對照組**完全相同**的結果卻被當成有做。
    """
    with pytest.raises(ValueError, match="gradnorm_init_weights"):
        _guard(_cfg(2.0))


def test_unit_data_weight_and_refinement_paths_stay_open():
    """1.0 是全庫 388 份 config 的值；refinement 開啟時該鍵有真實作用。"""
    _guard(_cfg(1.0))                     # 全庫 388 份 config 的值
    _guard(_cfg(2.0, refine="lbfgs"))     # refinement 路徑會讀它，不該被擋
