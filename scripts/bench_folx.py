"""Benchmark folx 修復（tile→_repeat3, stack→concat）對 Kolmogorov-size fof physics step 的影響。

量 EXP-245 規模 LiquidOperator 的 fused physics step（fused field derivatives 的 forward+backward；後端固定 folx）
ms/step + peak-mem。修復前/後由 PYTHONPATH 指向不同 worktree 的 pi_lnn_jax 決定（程式碼本身固定）。

用法：PYTHONPATH=<worktree> python bench_folx.py
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.autodiff import make_fused_field_derivatives

# ── EXP-245 (B3 full) 規模 ──
m = LiquidOperator(
    fourier_embed_dim=128, sensor_value_dim=2,
    d_model=256, d_time=16,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=2, token_attention_heads=4,
    num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
    decoder_attention_heads=4, use_temporal_anchor=True, T_total=5.0,
    temporal_anchor_harmonics=2, domain_length=1.0,
)
T, K, NC = 51, 100, 1024
rng = np.random.RandomState(0)
sv = jnp.asarray(rng.standard_normal((T, K, 2)), jnp.float32)
sp = jnp.asarray(rng.uniform(0, 1, (K, 2)), jnp.float32)
st = jnp.asarray(np.linspace(0, 5, T), jnp.float32)
RE = float(np.log(10000.0) / np.log(10000.0))
ix = jnp.asarray(rng.uniform(0, 1, (8, 2)), jnp.float32)
it = jnp.asarray(rng.uniform(0, 5, (8,)), jnp.float32)
params = m.init(jax.random.PRNGKey(0), sv, sp, RE, st, ix, it)
h = m.apply(params, sv, sp, RE, st, method=LiquidOperator.encode)  # encode-once（physics grad 只走 decoder）

n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))


def field_fn(p_, h_, x, y, t, sp_, st_):
    out = m.apply(p_, jnp.array([[x, y]]), jnp.array([t]), h_, st_, sp_,
                  method=LiquidOperator.decode_query)[0]
    return jnp.concatenate([out[0][None], out[1][None], out[2][None]])


fused = make_fused_field_derivatives(field_fn)
cx = jnp.asarray(rng.uniform(0, 1, (NC,)), jnp.float32)
cy = jnp.asarray(rng.uniform(0, 1, (NC,)), jnp.float32)
ct = jnp.asarray(rng.uniform(0, 5, (NC,)), jnp.float32)


@jax.jit
def step(p):
    def loss(pp):
        value, jac, d2x, d2y = fused(pp, h, cx, cy, ct, sp, st)
        return jnp.sum(value ** 2) + jnp.sum(jac ** 2) + jnp.sum(d2x ** 2) + jnp.sum(d2y ** 2)
    return jax.value_and_grad(loss)(p)


print(f"backend={jax.default_backend()} params={n_params:,} K={K} NC={NC} d_model=256")
# warmup（含 jit 編譯）
l, g = step(params)
jax.block_until_ready(g)
# 穩態量測
N = 50
t0 = time.perf_counter()
for _ in range(N):
    l, g = step(params)
    jax.block_until_ready(g)
ms = (time.perf_counter() - t0) / N * 1000.0

peak = 0.0
try:
    stats = jax.local_devices()[0].memory_stats()
    peak = stats.get("peak_bytes_in_use", 0) / 1e9
except Exception:
    pass

print(f"RESULT ms/step={ms:.2f} peak_mem={peak:.3f}GB loss={float(l):.4e}")
