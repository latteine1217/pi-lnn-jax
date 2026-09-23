"""NS 殘差對解析解為零 —— 補上「對不對」那一半。

Why this file exists:
    `tests/test_ns_residual_parity.py` 釘住的是 fused ≡ baseline（兩份實作互比），
    `CLAUDE.md` §7.1 的 A/B 對拍釘住的是重構前 ≡ 重構後。**兩者都不檢查「對」。**
    實測：把壓力梯度符號在 `physics.py` 的兩份實作裡**同時**反號，
    `test_ns_residual_parity` / `test_physics_residual_space` /
    `test_physics_anisotropic` / `test_physics_drag` 合計 18 項全綠。

    本檔提供缺的那一半：把已知的精確解餵進殘差，斷言它為零。
    與 parity 相乘才等於「兩份實作都正確」——
        parity:   fused ≡ baseline
        本檔:     baseline ≡ 真值
        ⇒         fused ≡ 真值

Why two solutions and not one:
    平行 Kolmogorov 解的 **p ≡ 0**，壓力梯度的符號在它底下完全隱形——
    正好就是上面那個測試套件抓不到的錯。Taylor–Green 有非平凡的 p，
    是唯一壓得住壓力項的那個。兩個都要，缺一個就留下同一個洞。

Solutions used:
    平行 Kolmogorov（含 forcing，v = p = 0）：
        u(y,t) = A/(ν κ²) · (1 − e^{−ν κ² t}) · sin(κ y),  κ = 2π k_f
    Taylor–Green 衰減渦（無 forcing）：
        u = −cos(κx) sin(κy) e^{−2νκ²t}
        v =  sin(κx) cos(κy) e^{−2νκ²t}
        p = −¼(cos 2κx + cos 2κy) e^{−4νκ²t}

Scope:
    只驗 `make_ns_residual_fn_baseline`（它吃 `model.apply(...)` 的整體 forward，
    可以餵任意場）。fused 路徑由 parity 測試接上，見上方推理。
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jax import config as _jax_config

from pi_lnn_jax.physics import make_ns_residual_fn_baseline

N_POINTS = 64
K_F = 2.0
KAPPA = 2.0 * jnp.pi * K_F


# 解析解的殘差在 fp32 下被捨入淹沒（~1e-7），分不出「零」與「小錯」。
# x64 是 process-global，故用會還原的 autouse fixture（同 test_autodiff.py）。
@pytest.fixture(autouse=True)
def _enable_x64():
    prev = _jax_config.read("jax_enable_x64")
    _jax_config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        _jax_config.update("jax_enable_x64", prev)


class _AnalyticField(nn.Module):
    """把解析解包成 baseline 殘差期望的 forward 介面。

    `scale` / `p_sign` 是給突變用的旋鈕：解被破壞後殘差必須不再為零，
    否則這個測試自己就是裝飾品。
    """
    kind: str
    nu: float
    scale: float = 1.0
    p_sign: float = 1.0

    @nn.compact
    def __call__(self, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t):
        # baseline 的 field_uvp 會 model.apply(params, ...)，故需至少一個參數
        _ = self.param("unused", nn.initializers.zeros, (1,))
        x, y = xy[:, 0], xy[:, 1]
        if self.kind == "kolmogorov":
            amp = A_FORCE / (self.nu * KAPPA ** 2)
            u = amp * (1.0 - jnp.exp(-self.nu * KAPPA ** 2 * t)) * jnp.sin(KAPPA * y)
            u = u * self.scale
            z = jnp.zeros_like(u)
            return jnp.stack([u, z, z], axis=-1)
        decay = jnp.exp(-2.0 * self.nu * KAPPA ** 2 * t)
        u = -jnp.cos(KAPPA * x) * jnp.sin(KAPPA * y) * decay * self.scale
        v = jnp.sin(KAPPA * x) * jnp.cos(KAPPA * y) * decay * self.scale
        p = -0.25 * (jnp.cos(2 * KAPPA * x) + jnp.cos(2 * KAPPA * y)) * decay ** 2
        return jnp.stack([u, v, p * self.p_sign], axis=-1)


A_FORCE = 0.1


def _residuals(kind: str, *, nu: float, scale: float = 1.0, p_sign: float = 1.0,
               nu_used: float | None = None, a_used: float | None = None):
    """回 (mom_u, mom_v, cont)。`nu_used`/`a_used` 讓突變只動殘差端的常數。"""
    model = _AnalyticField(kind=kind, nu=nu, scale=scale, p_sign=p_sign)
    ns = make_ns_residual_fn_baseline(model)
    xs = jax.random.uniform(jax.random.PRNGKey(1), (N_POINTS,), dtype=jnp.float64)
    ys = jax.random.uniform(jax.random.PRNGKey(2), (N_POINTS,), dtype=jnp.float64)
    t_hi = 5.0 if kind == "kolmogorov" else 1.0
    ts = jax.random.uniform(jax.random.PRNGKey(3), (N_POINTS,), dtype=jnp.float64) * t_hi
    params = model.init(jax.random.PRNGKey(0), None, None, 0.1, None,
                        jnp.zeros((1, 2)), jnp.zeros((1,)))
    forcing = A_FORCE if kind == "kolmogorov" else 0.0
    out = ns(params, None, None, 0.1, None, xs, ys, ts,
             forcing if a_used is None else a_used, K_F,
             nu if nu_used is None else nu_used,
             0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    return tuple(float(v) for v in out)


# 精確解在 fp64 下的殘差量級；四個突變最小的一個給 8.7e-6，中間隔了 20+ 個數量級。
ZERO = 1e-20


def test_parallel_kolmogorov_solution_has_zero_residual():
    """含 forcing 的平行流精確解：三個殘差都該是機器零。"""
    mom_u, mom_v, cont = _residuals("kolmogorov", nu=1e-4)
    assert mom_u < ZERO, f"mom_u={mom_u:.3e}"
    assert mom_v < ZERO and cont < ZERO


@pytest.mark.parametrize("label,kw", [
    ("解被縮放 1.5×", {"scale": 1.5}),
    ("殘差端的 ν 改 2×", {"nu_used": 2e-4}),
    ("殘差端的 forcing A 改 2×", {"a_used": 2 * A_FORCE}),
])
def test_kolmogorov_probe_is_not_vacuous(label, kw):
    """探針必須會紅：破壞解或殘差常數之後 mom_u 不可再是零。"""
    mom_u, _, _ = _residuals("kolmogorov", nu=1e-4, **kw)
    assert mom_u > 1e-8, f"{label} 之後 mom_u 仍為 {mom_u:.3e}——這個探針抓不到東西"


def test_taylor_green_solution_has_zero_residual():
    """Taylor–Green（無 forcing、p 非平凡）：三個殘差都該是機器零。"""
    mom_u, mom_v, cont = _residuals("taylor_green", nu=1e-2)
    assert mom_u < ZERO and mom_v < ZERO and cont < ZERO


def test_pressure_gradient_sign_is_pinned():
    """p 反號必須讓動量殘差爆掉。

    這是本檔存在的核心理由：同一個突變在既有的 18 項物理測試下**全綠**，
    因為 parity 比的是兩份實作彼此，而 p ≡ 0 的平行流看不見壓力項。
    """
    mom_u, mom_v, cont = _residuals("taylor_green", nu=1e-2, p_sign=-1.0)
    assert mom_u > 1.0, f"p 反號後 mom_u 只有 {mom_u:.3e}——壓力項沒有被釘住"
    assert mom_v > 1.0
    assert cont < ZERO, "連續性不含 p，反號不該影響它"
