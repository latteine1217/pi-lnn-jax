"""Collapsing Taylor concept proof（零依賴）：量 D=2 Laplacian 的 collapse 空間。

不實作 folx/forward-laplacian。改測「1 方向 vs 2 方向」的 jet-Laplacian grad 成本：
  collapse 的省幅來自跨方向共享 value+1階前向。
  - 2-dir ≈ 2×1-dir → 沒共享 → collapse 有頭空間（folx 值得做）
  - 2-dir ≈ 1-dir   → XLA 已 CSE 共享 → collapse 幫不上（省下實作）

EXP-245 scale (d_model=256, T=101, K=100, N_collo=1024)。量 forward + grad(params) + peak-mem。
"""
from __future__ import annotations
import statistics
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet as _jet
from bench.autodiff_modes import LiquidOperator, make_field_fn


def eprint(*a): print(*a, file=sys.stderr, flush=True)


def median_ms(fn, *a, n=20):
    o = fn(*a); jax.block_until_ready(o)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); o = fn(*a); jax.block_until_ready(o)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dir", type=int, choices=[1, 2], required=True,
                    help="Laplacian 用幾個空間方向（1=只 ∂²/∂x²；2=∂²/∂x²+∂²/∂y² 真 Laplacian）")
    ap.add_argument("--n-collo", type=int, default=1024)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    backend = jax.default_backend(); dev = jax.devices()[0]
    if backend != "gpu" and not args.allow_cpu:
        eprint(f"[FATAL] backend={backend}. add --allow-cpu"); sys.exit(5)

    MK = dict(sensor_value_dim=2, d_model=256, d_time=16,
              num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
              num_token_attention_layers=2, token_attention_heads=4,
              num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
              use_temporal_anchor=True, T_total=5.0, temporal_anchor_harmonics=2,
              domain_length=1.0)
    RE = 10000.0; RE_NORM = float(np.log(RE) / np.log(10000.0))
    T, K, N = 101, 100, args.n_collo
    um, us, vm, vs = (jnp.float32(c) for c in (0.0, 0.4167, 0.0, 0.4167))
    rng = np.random.RandomState(0)
    sv = jnp.asarray(rng.standard_normal((T, K, 2)), jnp.float32)
    sp = jnp.asarray(rng.uniform(0, 1, (K, 2)), jnp.float32)
    st = jnp.asarray(np.linspace(0, 5, T), jnp.float32)
    model = LiquidOperator(**MK)
    params = model.init(jax.random.PRNGKey(0), sv, sp, RE_NORM, st,
                        jnp.asarray(rng.uniform(0,1,(8,2)),jnp.float32),
                        jnp.asarray(rng.uniform(0,5,(8,)),jnp.float32))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    h = model.apply(params, sv, sp, RE_NORM, st, method=LiquidOperator.encode)
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    cx = jax.random.uniform(k[0], (N,), jnp.float32, 0.0, 1.0)
    cy = jax.random.uniform(k[1], (N,), jnp.float32, 0.0, 1.0)
    ct = jax.random.uniform(k[2], (N,), jnp.float32, 0.0, 5.0)
    field_fn = make_field_fn(model)
    common = (sp, st, um, us, vm, vs)
    dirs = [jnp.array([1., 0.], jnp.float32), jnp.array([0., 1.], jnp.float32)][:args.n_dir]
    z2 = jnp.zeros(2, jnp.float32)

    def per_point(p, x, y, t):
        xy = jnp.stack([x, y])
        def f(c2):
            return field_fn(p, h, c2[0], c2[1], t, *common)   # [3]=(u,v,p)
        lap = jnp.zeros(3, jnp.float32)
        for e in dirs:
            _, (_, d2) = _jet.jet(f, (xy,), ((e, z2),))
            lap = lap + d2
        return lap

    def residual_sum(p):
        lap = jax.vmap(lambda x, y, t: per_point(p, x, y, t),
                       in_axes=(0, 0, 0))(cx, cy, ct)          # [N,3]
        return jnp.mean(lap ** 2)

    f_fwd = jax.jit(residual_sum)
    f_grad = jax.jit(jax.grad(residual_sum))
    t0 = time.perf_counter(); jax.block_until_ready(f_grad(params))
    compile_ms = (time.perf_counter() - t0) * 1e3
    fwd_ms = median_ms(f_fwd, params)
    grad_ms = median_ms(f_grad, params)
    peak = (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6

    print(f"[n_dir={args.n_dir}] params={n_params:,} N={N} backend={backend}")
    print(f"  laplacian fwd = {fwd_ms:.2f} ms")
    print(f"  laplacian grad= {grad_ms:.2f} ms")
    print(f"  compile       = {compile_ms:.0f} ms")
    print(f"  peak_mem      = {peak:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
