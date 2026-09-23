"""Causal weighting for time-dependent PINN（Wang, Sankaran, Perdikaris 2022,
"Respecting Causality is all you need for training PINNs"）。

What:
  causal_weights(ct, residual_per_point, eps) -> w [N]
    Wang time-slab：同一時刻的點先聚合成一個 slab loss（slab 內 mean），
    w_slab = exp(-eps * Σ_{t_j < t_i} L_slab_j)，再 broadcast 回 slab 內各點。
    迫使網路先學前段時間。eps=0 → w≡1（等同關閉/均勻）。
    用 slab 聚合（非逐點 prefix sum）以保證對「同時刻點排列」與「各時刻採樣數」不變。

Why:
  時間相依 PDE 的 PINN 若同時硬學所有時間，後段（chaotic）梯度會污染前段，
  導致收斂停滯。causal weighting 是通用、與 PDE 無關的軟性因果排序。

時間分箱（TD-5 的修法，2026-09-13）：
  `ct` 先**等寬分箱到 `n_slabs` 個 slab**（預設 `DEFAULT_N_SLABS = 32`），再做上述聚合。
  這是 Wang 原文的前提——它的 eps 是對 **slab 數**校準的。

  不分箱會出事，而且是無聲的：`run.py` 的 `ct` 走 `jax.random.uniform` 連續取樣，
  1024 個點各不相同 → `jnp.unique` 給 1024 個 slab，退化成本檔明說要避免的逐點
  prefix sum，權重沿 1024 個點連乘衰減。實測（每點殘差能量取 1）：

      eps=0.0   sum(w) = 1024.000 / 1024     （關閉）
      eps=0.1   sum(w) =   10.508 / 1024
      eps=1.0   sum(w) =    1.582 / 1024     ← schema 預設
      分箱成 21 個 slab 後，eps=1.0 給 56.475

  即未分箱時照 schema 預設打開 `use_causal` 會關掉 99.85% 的 physics 殘差。
  分箱邊界取自**該批 ct 的 min/max** 而非固定 [0, T]，因為 time-marching curriculum
  會讓取樣時窗隨訓練前進——固定邊界會讓早期所有點擠進最前面幾個 slab。

Note:
  w 以 stop_gradient 凍結（僅作為權重，不對其反傳）；與 GradNorm 跨-task 加權正交。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


DEFAULT_N_SLABS = 32
"""ct 的分箱數。Wang 原文的 eps 對此數校準；改它等同改 eps 的尺度。"""


def causal_weights(ct: jnp.ndarray, residual_per_point: jnp.ndarray,
                   eps: float | jnp.ndarray,
                   n_slabs: int = DEFAULT_N_SLABS) -> jnp.ndarray:
    """Return causal weights w [N] in (0, 1]。

    Args:
      ct:                  [N] collocation 時間（任意順序）
      residual_per_point:  [N] 非負殘差能量（如 mom_u²+mom_v²+cont²）
      eps:                 >= 0 因果強度；0 → 全 1
      n_slabs:             ct 等寬分箱數；邊界取自該批 ct 的 min/max
    """
    # Wang time-slab：ct 等寬分箱成 n_slabs 個 slab，slab 內取 mean 當該 slab 的 loss，
    # 對 slab（時間遞增）做 exclusive prefix sum，再 broadcast 回各點。
    # 分箱是必要的，不是效能考量——見 module docstring 的實測數字。
    m = max(int(n_slabs), 1)
    lo = jnp.min(ct)
    span = jnp.maximum(jnp.max(ct) - lo, 1e-12)
    # 落在右端點的點 clip 回最後一個 slab（否則 idx == m 會越界）
    inv = jnp.clip(((ct - lo) / span * m).astype(jnp.int32), 0, m - 1)
    dt = residual_per_point.dtype
    slab_sum = jnp.zeros(m, dt).at[inv].add(residual_per_point)
    slab_cnt = jnp.zeros(m, dt).at[inv].add(1.0)
    # 空 slab 的 loss 記 0：該時段沒有樣本，就不對它之後的 slab 施加任何衰減。
    slab_loss = slab_sum / jnp.maximum(slab_cnt, 1.0)
    cum_excl = jnp.cumsum(slab_loss) - slab_loss       # 每 slab 只看嚴格在前的 slab 累積
    w_slab = jnp.exp(-eps * cum_excl)
    return jax.lax.stop_gradient(w_slab[inv])
