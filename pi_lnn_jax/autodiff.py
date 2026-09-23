"""融合逐點空間/時間導數（給 PINN physics 殘差用）。

What:
  make_fused_field_derivatives(field_fn) -> fused(params, h, xs, ys, ts, *extra)
    一次回傳 (value, jacobian, d2/dx2, d2/dy2)，四者共用同一份 decoder 前向，
    取代「每個導數量各自 jax.grad + vmap」造成的重複前向。

Why:
  PINN 殘差需要 u,v,p 的值、一階 (x,y,t) 導、Laplacian 對角 (xx, yy)。
  樸素做法對同一場函式重複前向十餘次；本工廠用 collapsed forward-Laplacian
  （folx）一次前向同時帶 jacobian 與累加二階（= Laplacian）。詳見下方常數註解。

Generality:
  與 PDE / case 無關；field_fn 只需 (params, h, x, y, t, *extra) -> [C] scalar-input。
"""
from __future__ import annotations

import os
from typing import Callable

import jax
import jax.numpy as jnp

# 導數後端固定為 folx（collapsed forward-Laplacian，NeurIPS-2025 Collapsing Taylor
# 的成熟 JAX 實作）。EXP-245 scale 實測 vs jet：Laplacian grad 1.53× 快（−35%）、
# forward 1.34× 快，數值等價 3e-13。
#   原理：一次前向同時帶 D 個一階 tangent + 1 個「累加二階」（= Laplacian），
#         取代「每方向獨立 jet 再相加」。
#   實作：folx.forward_laplacian 對 (x,y) 取 value/jacobian/laplacian；t 方向一階另用 jvp。
#
# 2026-08-03：移除 PILNN_AD_MODE 與其三條分支（ror / taylor / fof）。它們是選出 folx
# 的過程留下的量測工具，不是研究開關——production sbatch 零使用，只有 bench/ 與
# scripts/legacy/ 在設。留著一個沒人跑的 fallback，在真的需要它的那天最不可靠；
# 而更糟的是舊 sbatch 若還設著那個 env，會拿到與宣稱不符的後端。開關直接移除，
# 就不存在被誤設的可能。

_FOF_CHUNK = int(os.environ.get("PILNN_FOF_CHUNK", "0"))




def make_fused_field_derivatives(field_fn: Callable) -> Callable:
    """Return fused(params, h, xs, ys, ts, *extra).

    field_fn signature: (params, h, x, y, t, *extra) -> jnp.ndarray [C]
      其中 x, y, t 為 scalar；*extra 為不參與微分的 dynamic constants。

    fused returns:
      value [N, C], jac [N, C, 3], d2x [N, C], d2y [N, C]
      jac[..., 0/1/2] = d/dx, d/dy, d/dt
    """
    def _per_point(params, h, x, y, t, *extra):
        inp = jnp.stack([x, y, t])

        def f(coords):
            return field_fn(params, h, coords[0], coords[1], coords[2], *extra)

        value = f(inp)                       # [C]

        # collapsed forward-Laplacian：一次前向同時取 value/jacobian/laplacian。
        # 只對 (x,y)——t 方向的一階另用 jvp（見下方），二階不需要。
        # 注意：d2x/d2y **必須分開回**，不能把 collapsed laplacian 塞進 d2x
        # 而讓 d2y=0。消費端是分開用的：physics.py:140-141 是 d2x/Lx² + d2y/Ly²，
        # refiners.py:260-261 同樣分開取。理由與作法見下方 anisotropic 那段。
        # （此處原有一句「消費端都只用 d2x+d2y 的和，已確認」——那在
        #   anisotropic 修正之後就不成立了，且與下方 15 行的實作直接矛盾。）
        import folx
        xy = jnp.stack([x, y])

        def f_xy(c2):
            return field_fn(params, h, c2[0], c2[1], t, *extra)

        out = folx.forward_laplacian(f_xy)(xy)
        value = out.x                              # [C]
        jac_xy = out.jacobian.dense_array.T        # [C, 2] = (∂/∂x, ∂/∂y)
        lap = out.laplacian                        # [C] = ∂²/∂x² + ∂²/∂y²

        def f_t(tt):
            return field_fn(params, h, x, y, tt, *extra)
        _, d1t = jax.jvp(f_t, (t,), (jnp.ones_like(t),))

        jac = jnp.stack([jac_xy[:, 0], jac_xy[:, 1], d1t], axis=-1)   # [C, 3]
        # 必須拆 per-direction 二階導：anisotropic 域（Lx≠Ly，如 cylinder）的 physics
        # 用 d2x/Lx² + d2y/Ly²，不能把 collapsed laplacian 全塞 d2x（否則退化成
        # (uxx+uyy)/Lx²，Lx≠Ly 時黏性項錯）。folx 只給 collapsed laplacian，額外一個
        # directional forward-over-forward pass 取 ∂²/∂y²，再 d2x = lap - d2y。
        ey2 = jnp.array([0.0, 1.0], dtype=xy.dtype)
        _dfy = lambda c2_: jax.jvp(f_xy, (c2_,), (ey2,))[1]
        d2y = jax.jvp(_dfy, (xy,), (ey2,))[1]      # [C] = ∂²/∂y²
        d2x = lap - d2y                            # [C] = ∂²/∂x²

        return value, jac, d2x, d2y

    def fused(params, h, xs, ys, ts, *extra):
        n_extra = len(extra)
        axes = (None, None, 0, 0, 0) + (None,) * n_extra
        N = xs.shape[0]
        if _FOF_CHUNK <= 0 or N <= _FOF_CHUNK:
            return jax.vmap(_per_point, in_axes=axes)(params, h, xs, ys, ts, *extra)
        # ── collocation 分塊（省峰值記憶體）──
        # OOM 主因：vmap over N 把每點的 attention 中間張量 [3,K,hidden] 放大成
        # [N,3,K,hidden]（vector attention 大 K 時爆）。lax.map 序列處理每塊 CHUNK 點，
        # 峰值降到 ∝ CHUNK×K×hidden，且 lax.map 的 backward 也序列（不存全 N activation）。
        # 數值與單批 vmap 完全等價（每點獨立）。pad 到 CHUNK 倍數後切回 [:N]。
        C = _FOF_CHUNK
        pad = (-N) % C
        if pad:
            xs = jnp.concatenate([xs, jnp.zeros(pad, xs.dtype)])
            ys = jnp.concatenate([ys, jnp.zeros(pad, ys.dtype)])
            ts = jnp.concatenate([ts, jnp.zeros(pad, ts.dtype)])
        nc = (N + pad) // C
        xr = xs.reshape(nc, C); yr = ys.reshape(nc, C); tr = ts.reshape(nc, C)

        def per_chunk(sl):
            xc, yc, tc = sl
            return jax.vmap(_per_point, in_axes=axes)(params, h, xc, yc, tc, *extra)

        # remat：lax.map(=scan) 的 backward 預設存所有 chunk 的 forward residual（反而更耗），
        # jax.checkpoint 讓 backward 重算每 chunk forward → 峰值真正降到單塊。
        outs = jax.lax.map(jax.checkpoint(per_chunk), (xr, yr, tr))   # 各 [nc, C, ...]
        return jax.tree_util.tree_map(
            lambda a: a.reshape((nc * C,) + a.shape[2:])[:N], outs)

    return fused
