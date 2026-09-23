"""量測單一配置 (variant×mode×dtype×N) 的 GPU peak-memory + step wall。

三變體 AD 模式（見 bench/autodiff_modes.py）：
  fof  forward-over-forward  (現行 production)
  for  forward-over-reverse  (使用者原始命題)
  ror  reverse-over-reverse  (舊 baseline 代表)

Why 獨立 process：JAX peak_bytes_in_use 為 process 累計峰值、無 reset，
  公平比較必須各跑獨立 process（由 bench_fwd_mem.sbatch 驅動）。

  --mode residual : 殘差 forward（含二階座標導數）
  --mode grad     : grad-w.r.t-params（真實訓練 AD 結構，記憶體峰值最大）

輸出：stdout 一行 JSON（給 summarize_mem 聚合），stderr 人類可讀摘要。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import jax


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def peak_mb(dev):
    s = dev.memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 1e6


def limit_mb(dev):
    s = dev.memory_stats() or {}
    return s.get("bytes_limit", 0) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fof", "for", "ror"], required=True)
    ap.add_argument("--mode", choices=["residual", "grad"], required=True)
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--n-collo", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="預設要求 GPU；加此 flag 才允許 CPU（數據會標 device=cpu）")
    args = ap.parse_args()

    if args.dtype == "f64":
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np
    from bench.autodiff_modes import LiquidOperator, make_residual_sum

    backend = jax.default_backend()
    dev = jax.devices()[0]
    if backend != "gpu" and not args.allow_cpu:
        eprint(f"[FATAL] backend={backend}（非 gpu）。此 benchmark 目的是量 GPU peak-mem。"
               f"\n        確認 JAX_PLATFORMS=cuda 且 .venv 裝了 jax[cuda12]；"
               f"或加 --allow-cpu 明示要在 CPU 跑。")
        sys.exit(5)

    fdtype = jnp.float64 if args.dtype == "f64" else jnp.float32

    CFG_MINI = dict(
        sensor_value_dim=2, d_model=32, d_time=8,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=1, token_attention_heads=4,
        num_query_mlp_layers=1, query_mlp_hidden_dim=32, operator_rank=32,
        decoder_attention_heads=1, use_temporal_anchor=True, T_total=5.0,
        temporal_anchor_harmonics=2, domain_length=1.0, fourier_embed_dim=0,
    )
    RE_VALUE = 1000.0
    RE_NORM = float(np.log(RE_VALUE) / np.log(10000.0))
    T, K, N = 11, 25, args.n_collo
    nu = jnp.asarray(1.0 / RE_VALUE, fdtype)
    u_mean, u_std, v_mean, v_std = (jnp.asarray(c, fdtype) for c in (0.1, 0.8, -0.05, 0.7))

    rng = np.random.RandomState(0)
    sensor_vals = jnp.asarray(rng.standard_normal((T, K, 2)), dtype=fdtype)
    sensor_pos = jnp.asarray(rng.uniform(0, 1, (K, 2)), dtype=fdtype)
    sensor_time = jnp.asarray(np.linspace(0, 5, T), dtype=fdtype)

    model = LiquidOperator(**CFG_MINI)
    init_xy = jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=fdtype)
    init_t = jnp.asarray(rng.uniform(0, 5, (8,)), dtype=fdtype)
    params = model.init(jax.random.PRNGKey(0), sensor_vals, sensor_pos,
                        RE_NORM, sensor_time, init_xy, init_t)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    h_states = model.apply(params, sensor_vals, sensor_pos, RE_NORM, sensor_time,
                           method=LiquidOperator.encode)

    k = jax.random.split(jax.random.PRNGKey(1), 3)
    cx = jax.random.uniform(k[0], (N,), dtype=fdtype, minval=0.0, maxval=1.0)
    cy = jax.random.uniform(k[1], (N,), dtype=fdtype, minval=0.0, maxval=1.0)
    ct = jax.random.uniform(k[2], (N,), dtype=fdtype, minval=0.0, maxval=5.0)
    A, k_f = jnp.asarray(0.1, fdtype), jnp.asarray(2.0, fdtype)
    common = (sensor_pos, sensor_time, u_mean, u_std, v_mean, v_std)

    res_sum = make_residual_sum(model, args.variant, h_states, cx, cy, ct, A, k_f, common, nu)
    fn = jax.jit(res_sum if args.mode == "residual" else jax.grad(res_sum))

    out = fn(params)                         # warmup / compile
    jax.block_until_ready(out)

    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = fn(params)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    step_ms = statistics.median(ts) * 1e3

    rec = dict(
        variant=args.variant, mode=args.mode, dtype=args.dtype, n_collo=N,
        device=backend, n_params=n_params,
        peak_mb=round(peak_mb(dev), 2), limit_mb=round(limit_mb(dev), 1),
        step_ms=round(step_ms, 4),
    )
    eprint(f"  [{args.variant}|{args.mode:>8s}|{args.dtype}|N={N:>5d}] "
           f"peak={rec['peak_mb']:>9.2f}MB  step={rec['step_ms']:>8.3f}ms")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
