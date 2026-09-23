"""2D Kolmogorov NS residual computed via JAX autograd.

對齊 pi_con/physics.py:unsteady_ns_residuals（Lx=Ly=1.0 簡化版，週期域）。

設計重點（Wave 5 convergence-efficiency）：
  1) 融合逐點導數：透過 autodiff.make_fused_field_derivatives 以
     forward-over-forward（jacfwd 取一階 + 巢狀 jvp 取二階對角）一次取齊
     value / 一階 (x,y,t) / 二階 (xx,yy)，取代舊版逐量 jax.grad grad-of-grad
     (reverse-over-reverse)。GPU 實測（docs/archive/POC_RESULTS.md §4.5）：對本案小輸入/輸出維度，
     fof 為三種 AD 模式中峰值記憶體與 wall 雙優（for ~1.8-2.1×、ror ~2.5-5.9× 更耗）。
  2) **encode-once 模式**：sensor encode 由 caller 預算一次 (model.apply method=encode)，
     h_states 作 dynamic arg 傳入。PDE grad 內部只走 decoder（model.apply method=decode_query），
     XLA 看到的 graph 不含重複 lax.scan over T。
  3) **Physics-space residual（denormalized；always-on 唯一路徑）**：
     model 出 normalized 量級 (u_n, v_n, p_n)，field_uvp_phys 先還原成物理量
     u = u_n·u_std + u_mean（v, p 同理）再組 NS residual，故 residual 在
     **物理空間**計算，並非 normalized space。這是刻意設計：velocity 必 denorm；
     config `use_physics_denormalization` 目前是 dead flag，不 gate 任何行為
     （見 train_cylinder.py 的 NOTE），實際行為由 caller 傳入的真實 std/mean 決定
     （train_kolmogorov 一律傳 dataset norm_stats，非 identity）。
     Why denorm: NS 方程對有量綱物理場成立；直接拿 z-score 量 u_n 當 u 代入，
     因 advection (u·u_x) 與黏性項 scaling 未補償，得到的並非真正的 NS residual。
     量級 trade-off: 物理空間 residual 量級較大（u_std<1、Re 大時可達 O(10)）。
     GradNorm 不因此壓垮 physics weight：真正 load-bearing 的是 reference-layer
     選擇。**實際生效的是 ("query_decoder", "trunk_out")**（liquid 路徑）／
     ("trunk_out",)（其餘）——見 pipeline/kolmogorov/assembly.py 的 gn_ref_path 覆蓋值，
     所有歷史 Kolmogorov production run 皆以此訓練。_build_grad_norm_fn 簽章上的
     ("temporal_encoder",) 是**從未被使用的函式預設值**，勿據它推論實際行為。
     min_weight floor（gradnorm/lra 預設 0.05）僅兜底。機制與實證缺口見
     knowledge/theory/navier-stokes-nondim.md。
     歷史脈絡: pi-lnn EXP-245 曾用 use_physics_denormalization=False 在 normalized
     space 算 residual 以迴避早期 GradNorm 崩潰；此 jax 版改走物理正確的 denorm
     路徑。本註解先前沿用該舊敘述（宣稱 identity / 不乘 std/mean）已過時，
     2026-07 更正為與實作一致。
  4) Pressure-Poisson residual 共用同一個 decode-only field function。

forcing: f_x = A * sin(2π · k_f · y)，f_y = 0
  EXP-030: A=0.1, k_f=2.0

API (Wave 5 multi-Re；norm/Re 改 runtime args，jit 不因 Re retrace)：
  ns_residuals(params, h_states, xs, ys, ts, A, k_f,
               sensor_pos, sensor_time, nu, u_mean, u_std, v_mean, v_std) → (mom_u², mom_v², cont²)
  poisson_residual(params, h_states, xs, ys, ts,
                   sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std) → poisson²

Caller pattern (in loss_fn):
  h_states = model.apply(params, sv, sp, re_norm, st, method=LiquidOperator.encode)
  mom_u, mom_v, cont = ns_fn(params, h_states, cx, cy, ct, A, k_f,
                             sensor_pos, sensor_time, nu, u_mean, u_std, v_mean, v_std)
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def make_ns_residual_fn(
    model,
    sensor_pos: jnp.ndarray | None = None,
    sensor_time: jnp.ndarray | None = None,
    norm_stats: dict | None = None,
    re_value: float | None = None,
) -> Callable:
    """Return (ns_residuals, poisson_residual)。

    Wave 5 signature change (multi-Re support):
      `sensor_pos`、`sensor_time`、`norm_stats`、`re_value` 不再 closure capture，
      改為 ns_residuals / poisson_residual 的 runtime args：
        ns_residuals(params, h_states, xs, ys, ts, A, k_f,
                     sensor_pos, sensor_time, nu, u_mean, u_std, v_mean, v_std)
      這四個過去 closure 的舊參數保留只為 backward-compat（傳了會 warn-ignore），
      不再影響行為。

    Why: 舊版把 (nu, u_mean, ...) bake 進 closure，multi-Re 訓練時 dataset 切換
         會 silent 失效（外層改 Python 變數，jit graph 內仍是 d0 的常數）。
         改 runtime arg 後 jit 只 trace 一次，Re 切換零成本。
    """
    if any(x is not None for x in (sensor_pos, sensor_time, norm_stats, re_value)):
        import warnings as _w
        _w.warn(
            "make_ns_residual_fn() 舊參數 (sensor_pos/sensor_time/norm_stats/re_value) "
            "已改為 ns_residuals 的 runtime args；傳入值會被忽略。",
            DeprecationWarning,
            stacklevel=2,
        )

    # 必須 lazy import 避免循環依賴
    from pi_lnn_jax.models import LiquidOperator

    # ── single-point scalar functions（denormalized 物理單位，decode-only）──
    # 簽名：(params, h_states, x, y, t, sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std)
    # grad 只對 (x=2, y=3, t=4) 求；其餘為 dynamic constants（jax 不會對它們建立 grad graph，
    # 但會把它們納入 trace metadata，跨 Re 切換不需 retrace）。
    def field_uvp_phys(params, h_states, x, y, t,
                       sensor_pos, sensor_time,
                       u_mean, u_std, v_mean, v_std, p_mean, p_std):
        out = model.apply(
            params,
            jnp.array([[x, y]]), jnp.array([t]),
            h_states, sensor_time, sensor_pos,
            method=LiquidOperator.decode_query,
        )[0]   # [3] normalized
        u = out[0] * u_std + u_mean
        v = out[1] * v_std + v_mean
        p = out[2] * p_std + p_mean   # 對稱化（殘差只用梯度，p_mean 微分後消失）
        # concatenate 取代 stack：stack 不在 folx registry（output fallback full hessian）
        return jnp.concatenate([u[None], v[None], p[None]])

    # ── 融合逐點導數（取代逐量 jax.grad + vmap，消除 decoder 重複前向）──
    # field_uvp_phys signature: (params, h, x, y, t, sensor_pos, sensor_time, um, us, vm, vs)
    # extra = (sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std) 不參與微分。
    from pi_lnn_jax.autodiff import make_fused_field_derivatives
    _fused = make_fused_field_derivatives(field_uvp_phys)

    def _eval(params, h_states, xs, ys, ts, common):
        # value [N,3], jac [N,3,3] (d/dx,d/dy,d/dt), d2x [N,3], d2y [N,3]
        return _fused(params, h_states, xs, ys, ts, *common)

    def ns_residuals(params, h_states, xs, ys, ts, A, k_f,
                     sensor_pos, sensor_time,
                     nu, u_mean, u_std, v_mean, v_std, p_mean, p_std,
                     Lx: float = 1.0, Ly: float = 1.0,
                     drag_alpha: float = 0.0,
                     return_per_point=False):
        """Return (mom_u², mom_v², cont²) MSE means。

        return_per_point=True 額外回傳 (mom_u, mom_v, cont) 逐點 signed 值，
        供 causal weighting / RAR 使用（mean(per_point²) == 純量回傳）。

        Wave 5 multi-Re args（runtime，jit 不因 Re 變化 retrace）：
            sensor_pos, sensor_time, nu=1/Re, u_mean/u_std/v_mean/v_std。
        """
        common = (sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std, p_mean, p_std)
        value, jac, d2x, d2y = _eval(params, h_states, xs, ys, ts, common)
        u = value[:, 0]; v = value[:, 1]
        # anisotropic domain scaling: x_phys=x_norm·Lx → ∂/∂x_phys=(1/Lx)·∂/∂x_norm
        # 預設 Lx=Ly=1.0 → 與週期單位域 (Kolmogorov) 完全一致。
        u_xv = jac[:, 0, 0] / Lx; u_yv = jac[:, 0, 1] / Ly; u_tv = jac[:, 0, 2]
        v_xv = jac[:, 1, 0] / Lx; v_yv = jac[:, 1, 1] / Ly; v_tv = jac[:, 1, 2]
        p_xv = jac[:, 2, 0] / Lx; p_yv = jac[:, 2, 1] / Ly
        u_xxv = d2x[:, 0] / Lx ** 2; u_yyv = d2y[:, 0] / Ly ** 2
        v_xxv = d2x[:, 1] / Lx ** 2; v_yyv = d2y[:, 1] / Ly ** 2

        f_x = A * jnp.sin(2.0 * jnp.pi * k_f * ys)
        f_y = 0.0

        mom_u = u_tv + u * u_xv + v * u_yv + p_xv - nu * (u_xxv + u_yyv) - f_x
        mom_v = v_tv + u * v_xv + v * v_yv + p_yv - nu * (v_xxv + v_yyv) - f_y
        # 線性阻尼 -alpha*u（渦量式 -alpha*omega 的動量式對應，curl 為線性）。
        # drag_alpha 是靜態 float，alpha=0 時整段不進運算圖 → 逐位元等同舊行為。
        if drag_alpha:
            mom_u = mom_u + drag_alpha * u
            mom_v = mom_v + drag_alpha * v
        cont = u_xv + v_yv

        mu = jnp.mean(mom_u ** 2)
        mv = jnp.mean(mom_v ** 2)
        c = jnp.mean(cont ** 2)
        if return_per_point:
            return mu, mv, c, mom_u, mom_v, cont
        return mu, mv, c

    def poisson_residual(params, h_states, xs, ys, ts,
                         sensor_pos, sensor_time,
                         u_mean, u_std, v_mean, v_std, p_mean, p_std,
                         Lx=1.0, Ly=1.0):
        """Pressure-Poisson residual MSE: ∇²p = -(∂u/∂x)² - (∂v/∂y)² - 2(∂u/∂y)(∂v/∂x)。

        anisotropic domain scaling 與 ns_residuals 一致（x_phys=x_norm·Lx →
        ∂/∂x_phys=(1/Lx)∂/∂x_norm）。預設 Lx=Ly=1.0 → 與週期單位域 (Kolmogorov) 相同；
        Lx≠Ly（cylinder）若開 Poisson 必須傳實際尺度，否則 ∇²p 會差 1/Lx²、1/Ly² 倍。
        """
        common = (sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std, p_mean, p_std)
        value, jac, d2x, d2y = _eval(params, h_states, xs, ys, ts, common)
        u_xv = jac[:, 0, 0] / Lx; u_yv = jac[:, 0, 1] / Ly
        v_xv = jac[:, 1, 0] / Lx; v_yv = jac[:, 1, 1] / Ly
        p_xxv = d2x[:, 2] / Lx ** 2; p_yyv = d2y[:, 2] / Ly ** 2
        laplacian_p = p_xxv + p_yyv
        rhs = -(u_xv ** 2 + v_yv ** 2 + 2.0 * u_yv * v_xv)
        return jnp.mean((laplacian_p - rhs) ** 2)

    return ns_residuals, poisson_residual


def make_ns_residual_fn_baseline(model) -> Callable:
    """NS residual for non-LiquidOperator 架構（B0 vanilla / Standard PINN）。

    這些 baseline 無 encode/decode_query 拆分（h_states），改走全 ``__call__``：
    每個 collocation 點重新前向（vanilla 含 branch 編碼；PINN 忽略 sensor），
    用標準 ``jax.jacfwd`` 取一/二階導數。perf 比 liquid 的 fused/folx 路徑差
    （per-point 全前向），但 B0/PINN 是單 seed 10k baseline，可接受。

    NS 物理與 ``make_ns_residual_fn`` 的 ``ns_residuals`` 一致，**包含 anisotropic
    domain scaling**（`Lx`/`Ly`，預設 1.0）。實測（同 model/params/collocation）：
    `Lx=Ly=1` 時兩者逐點相對差 max 2.3e-05、純量 MSE 一致到 7 位有效數字
    ——float32 捨入層級。

    ⚠️ 先前此處寫「完全一致」而簽章根本沒有 `Lx`/`Ly`：那句話只在 `Lx=Ly=1`
    成立，若 B0/PINN 哪天跑在 anisotropic 域（如 cylinder 的 Lx≈0.6, Ly≈0.3），
    兩者的 residual MSE 實測差 5.7–7.9 倍。`Lx`/`Ly` 於 2026-08-03 補上，
    `tests/test_ns_residual_parity.py` 釘住兩者的一致性。

    Returns ns_residuals(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                         xs, ys, ts, A, k_f, nu, u_mean, u_std, v_mean, v_std,
                         p_mean, p_std, Lx=1.0, Ly=1.0, return_per_point=False)。
    """
    def field_uvp(params, x, y, t,
                  sensor_vals, sensor_pos, re_norm, sensor_time,
                  u_mean, u_std, v_mean, v_std, p_mean, p_std):
        out = model.apply(
            params, sensor_vals, sensor_pos, re_norm, sensor_time,
            jnp.array([[x, y]]), jnp.array([t]),
        )[0]   # [3] normalized (u, v, p)
        u = out[0] * u_std + u_mean
        v = out[1] * v_std + v_mean
        p = out[2] * p_std + p_mean   # 殘差只用梯度，p_mean 微分後消失
        return jnp.concatenate([u[None], v[None], p[None]])

    def _derivs(params, x, y, t, common):
        f = lambda xx, yy, tt: field_uvp(params, xx, yy, tt, *common)
        val = f(x, y, t)                                              # [3]
        jx = jax.jacfwd(f, argnums=0)(x, y, t)                        # [3] ∂/∂x
        jy = jax.jacfwd(f, argnums=1)(x, y, t)                        # [3] ∂/∂y
        jt = jax.jacfwd(f, argnums=2)(x, y, t)                        # [3] ∂/∂t
        d2x = jax.jacfwd(jax.jacfwd(f, argnums=0), argnums=0)(x, y, t)  # [3] ∂²/∂x²
        d2y = jax.jacfwd(jax.jacfwd(f, argnums=1), argnums=1)(x, y, t)  # [3] ∂²/∂y²
        return val, jx, jy, jt, d2x, d2y

    def ns_residuals(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                     xs, ys, ts, A, k_f, nu,
                     u_mean, u_std, v_mean, v_std, p_mean, p_std,
                     Lx: float = 1.0, Ly: float = 1.0,
                     drag_alpha: float = 0.0,
                     return_per_point=False):
        common = (sensor_vals, sensor_pos, re_norm, sensor_time,
                  u_mean, u_std, v_mean, v_std, p_mean, p_std)
        val, jx, jy, jt, d2x, d2y = jax.vmap(
            lambda x, y, t: _derivs(params, x, y, t, common)
        )(xs, ys, ts)
        u = val[:, 0]; v = val[:, 1]
        # anisotropic domain scaling，與 ns_residuals（fused）逐字對應：
        # x_phys=x_norm·Lx → ∂/∂x_phys=(1/Lx)·∂/∂x_norm。預設 1.0 ⇒ 除以 1.0，
        # IEEE754 下為精確運算，既有 B0/PINN 路徑逐位元不變。
        u_xv = jx[:, 0] / Lx; u_yv = jy[:, 0] / Ly; u_tv = jt[:, 0]
        v_xv = jx[:, 1] / Lx; v_yv = jy[:, 1] / Ly; v_tv = jt[:, 1]
        p_xv = jx[:, 2] / Lx; p_yv = jy[:, 2] / Ly
        u_xxv = d2x[:, 0] / Lx ** 2; u_yyv = d2y[:, 0] / Ly ** 2
        v_xxv = d2x[:, 1] / Lx ** 2; v_yyv = d2y[:, 1] / Ly ** 2
        f_x = A * jnp.sin(2.0 * jnp.pi * k_f * ys)
        mom_u = u_tv + u * u_xv + v * u_yv + p_xv - nu * (u_xxv + u_yyv) - f_x
        mom_v = v_tv + u * v_xv + v * v_yv + p_yv - nu * (v_xxv + v_yyv)
        if drag_alpha:   # 見 fused 版本的說明
            mom_u = mom_u + drag_alpha * u
            mom_v = mom_v + drag_alpha * v
        cont = u_xv + v_yv
        mu = jnp.mean(mom_u ** 2); mv = jnp.mean(mom_v ** 2); c = jnp.mean(cont ** 2)
        if return_per_point:
            return mu, mv, c, mom_u, mom_v, cont
        return mu, mv, c

    return ns_residuals
