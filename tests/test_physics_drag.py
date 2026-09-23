"""線性阻尼項 `-alpha*u` 進入 NS residual 的行為。

要證明兩件事：
  1. `drag_alpha=0`（預設）與不傳該參數逐位元相同 —— 既有 config 零影響。
     實作上靠 residual 內的 `if drag_alpha:` 是靜態 Python 分支，alpha=0 時
     整段不進運算圖；本測試把這個實作細節釘成行為契約。
  2. `drag_alpha>0` 時，殘差恰好多出 `alpha*u`（動量式），不多不少。
     殘差寫作 `mom = LHS - f`，真方程為 `LHS + alpha*u - f = 0`，故是 **加** alpha*u。

第 2 條用「同一組場、兩個 alpha」相減檢驗，不重算場，避免把模型誤差混進來。
"""
import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs
from jax import config as _jax_config

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn

ALPHA = 0.35   # 定案組態的值


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)


def _build():
    model = LiquidOperator(**minimal_kwargs(2))
    K, T, N = 5, 4, 12
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(2), (T, K, 2))
    params = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    h = model.apply(params, sv, sp, 0.1, st, method=LiquidOperator.encode)
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,), maxval=1.0)
    return model, params, h, sp, st, cx, cy, ct


_STATS = (1e-4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)   # nu, u_mean, u_std, v_mean, v_std, p_mean, p_std


def _call(ns, params, h, cx, cy, ct, sp, st, **kw):
    return ns(params, h, cx, cy, ct, 0.1, 2.0, sp, st, *_STATS,
              return_per_point=True, **kw)


def test_alpha_zero_is_bit_identical_to_omitting_it():
    model, params, h, sp, st, cx, cy, ct = _build()
    ns, _ = make_ns_residual_fn(model)
    base = _call(ns, params, h, cx, cy, ct, sp, st)
    zero = _call(ns, params, h, cx, cy, ct, sp, st, drag_alpha=0.0)
    for a, b, name in zip(base, zero, ("mu", "mv", "cont", "mom_u", "mom_v", "cont_pp")):
        assert jnp.array_equal(a, b), f"{name}: drag_alpha=0 與不傳不逐位元相同"


def test_drag_adds_exactly_alpha_times_u():
    model, params, h, sp, st, cx, cy, ct = _build()
    ns, _ = make_ns_residual_fn(model)
    _, _, _, mu0, mv0, c0 = _call(ns, params, h, cx, cy, ct, sp, st, drag_alpha=0.0)
    _, _, _, mu1, mv1, c1 = _call(ns, params, h, cx, cy, ct, sp, st, drag_alpha=ALPHA)

    # 同一組場下取出 (u, v)：用 decode_query 直接求，與 residual 內部同一條路徑
    out = model.apply(params, jnp.stack([cx, cy], axis=-1), ct, h, st, sp,
                      method=LiquidOperator.decode_query)
    u, v = out[:, 0], out[:, 1]     # norm_stats 為 (0,1) → 已是物理值

    assert jnp.allclose(mu1 - mu0, ALPHA * u, atol=1e-9), "mom_u 的增量 ≠ alpha*u"
    assert jnp.allclose(mv1 - mv0, ALPHA * v, atol=1e-9), "mom_v 的增量 ≠ alpha*v"
    assert jnp.array_equal(c0, c1), "連續方程不該被阻尼影響"


def test_drag_alpha_is_a_construction_field_so_the_fingerprint_sees_it():
    """把 alpha 放進 model（而非 config data 段）的理由：data 段的 kolmogorov_A/k_f
    是 inert 死鍵，且只有 model dataclass 欄位會進 model_fingerprint——那道閘門
    才擋得住「阻尼資料訓的 ckpt 被無阻尼 residual 拿去 eval」。

    ⚠️ 但 alpha=0 時本欄位**刻意不進指紋**（見 model_fingerprint 尾端）。
    α=0 在 physics.py 是靜態分支、逐位元等同「從未有此功能」，留著會讓加入本
    功能之前訓練的 ckpt 全數評估不了——實測 226 份帶指紋的 summary.json 全中
    （commit 2c07d68）。這與 mid_band / trainable_fourier 是同一個慣例。

    本測試原本要的保護**兩個方向都仍然成立**，因為「缺鍵」本身就是差異：
    缺鍵 vs 有值會被 fingerprint_diff 報成「訓練時無此欄位」或「eval 無此欄位」。
    因此下面改成直接驗兩個方向，比原本的單一字面斷言更貼近意圖。
    """
    from pi_lnn_jax.model_factory import fingerprint_diff, model_fingerprint
    kw = dict(sensor_value_dim=2, d_model=16, d_time=4,
              num_spatial_encoder_layers=1, num_temporal_cfc_layers=1)
    off = model_fingerprint(LiquidOperator(**kw, drag_alpha=0.0))
    on = model_fingerprint(LiquidOperator(**kw, drag_alpha=ALPHA))

    assert "drag_alpha" in on, "阻尼開啟時未進指紋 → 閘門看不到它"
    assert "drag_alpha" not in off, (
        "阻尼關閉時仍留在指紋 → 加入本功能之前訓練的 ckpt 全數評估不了")
    assert any("drag_alpha" in p for p in fingerprint_diff(on, off)), \
        "訓練有阻尼、eval 無阻尼——沒抓到"
    assert any("drag_alpha" in p for p in fingerprint_diff(off, on)), \
        "訓練無阻尼、eval 有阻尼——沒抓到"
