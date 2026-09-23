"""回歸鎖：physics.py 的 NS residual 在物理空間（denormalized）計算。

背景：`make_ns_residual_fn` 的 field function 先把 model 的 normalized 輸出還原成
物理量（`u = u_n·u_std + u_mean`，v/p 同理）再組 NS residual，故 residual 在
**物理空間**，並非 normalized-space identity。physics.py 檔頭 docstring 先前沿用
舊敘述（宣稱 identity / 不乘 std/mean）已於 2026-07 更正為與實作一致。

此測試鎖定該行為，防止 docstring 與實作再次漂移：改變 velocity std 必改變 momentum
residual——若 residual 落在 normalized space（identity 不用 std），改 std 不會有影響。
對照 knowledge/theory/navier-stokes-nondim.md。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs
from jax import config as _jax_config

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn


@pytest.fixture(autouse=True)
def _enable_x64():
    # residual 對 std 的敏感度比較需要 float64 精度；x64 是 process-global 旗標，
    # 用 autouse fixture 只在本檔範圍開啟、結束還原，避免污染其他測試。
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)


def _build():
    model = LiquidOperator(**minimal_kwargs(2))
    K, T, N = 5, 4, 12
    sensor_pos = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    sensor_time = jnp.linspace(0.0, 1.0, T)
    sensor_vals = jax.random.normal(jax.random.PRNGKey(2), (T, K, 2))
    params = model.init(jax.random.PRNGKey(3), sensor_vals, sensor_pos, 0.1,
                        sensor_time, jax.random.uniform(jax.random.PRNGKey(9), (N, 2)),
                        jnp.zeros((N,)))
    h_states = model.apply(params, sensor_vals, sensor_pos, 0.1, sensor_time,
                           method=LiquidOperator.encode)
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,), maxval=1.0)
    return model, params, h_states, sensor_pos, sensor_time, cx, cy, ct


def test_ns_residual_in_physical_space_uses_velocity_denorm():
    """改變 velocity std → momentum residual 必改變，證明 field function 先 denorm
    (u = u_n·u_std + u_mean) 再組 residual = 物理空間，而非 normalized identity。"""
    model, params, h, sp, st, cx, cy, ct = _build()
    ns_fn, _ = make_ns_residual_fn(model)
    A, k_f, nu = 0.1, 2.0, 1e-4
    um, vm, pm, ps = 0.0, 0.0, 0.0, 1.0
    # 簽名: ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs, pm, ps)
    r_std1 = ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, 1.0, vm, 1.0, pm, ps)
    r_std2 = ns_fn(params, h, cx, cy, ct, A, k_f, sp, st, nu, um, 2.0, vm, 2.0, pm, ps)
    mom_u_1 = float(jnp.asarray(r_std1[0]))
    mom_u_2 = float(jnp.asarray(r_std2[0]))
    assert not jnp.isclose(mom_u_1, mom_u_2, rtol=1e-6), (
        f"velocity std 未改變 momentum residual (std1={mom_u_1:.3e}, std2={mom_u_2:.3e}) "
        f"→ residual 疑落在 normalized space，與 physics.py docstring 與實作(denorm) 矛盾"
    )
    print(f"✓ residual 隨 velocity std 變化（物理空間 denorm）: "
          f"mom_u(std=1)={mom_u_1:.3e}  mom_u(std=2)={mom_u_2:.3e}")
