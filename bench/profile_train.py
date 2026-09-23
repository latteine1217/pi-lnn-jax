"""Performance benchmark for train_full mini smoke pipeline.

What: 跑 mini config 並量化 (jit_wall, per-step wall, peak RSS)，與 jit-cached steps
      single-loss-component timing breakdown。可重複跑（baseline + post-opt 比較）。

Why: 「優化前後比較」必須 deterministic, fixed seed + fixed config + fixed warmup。

Usage:
  PYTHONPATH=. uv run python bench/profile_train.py --label baseline --n_steps 30
  PYTHONPATH=. uv run python bench/profile_train.py --label post_opt --n_steps 30
  # 比較：
  PYTHONPATH=. uv run python bench/profile_train.py --compare baseline post_opt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil

from pi_lnn_jax.data import SENSOR_JSON, load_sensors_from_path
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn


# ──────────────────────────────────────────────────────────────────────────────
# Fixed mini config (與 train_full mini_smoke 等價但 inline；確保 benchmark 可重現)
# ──────────────────────────────────────────────────────────────────────────────
CFG_MINI = dict(
    sensor_value_dim=2, d_model=32, d_time=8,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=1, token_attention_heads=4,
    num_query_mlp_layers=1, query_mlp_hidden_dim=32, operator_rank=32,
    decoder_attention_heads=1, use_temporal_anchor=True,
    T_total=5.0, temporal_anchor_harmonics=2, domain_length=1.0,
    fourier_embed_dim=0,
)
# Wave 4 perf opt-in (env var REMAT=1 for benchmark)
import os as _os
CFG_MINI_REMAT = {**CFG_MINI, "use_decoder_remat": bool(_os.environ.get("REMAT", "0") == "1")}
RE_VALUE = 1000.0
RE_NORM = float(np.log(RE_VALUE) / np.log(10000.0))
SEED = 42
N_COLLO = 16
W_PHYS = 0.01


def subsample(d, T=11, K=25):
    Tf = d["sensor_vals"].shape[0]
    stride = max(1, (Tf - 1) // (T - 1))
    tidx = np.arange(0, Tf, stride)[:T]
    return {
        "sensor_vals": d["sensor_vals"][tidx][:, :K, :],
        "sensor_pos": d["sensor_pos"][:K],
        "sensor_time": d["sensor_time"][tidx],
        "norm_stats": d["norm_stats"],
    }


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def run_benchmark(n_steps: int = 30, n_warmup: int = 2) -> dict:
    """跑 n_steps + n_warmup 步 mini training；measure jit wall + per-step wall + peak RSS."""
    print(f"=== Benchmark: n_steps={n_steps}, n_warmup={n_warmup} ===")

    # ── 資料 + 模型 init ──
    # load_sensors 已刪（用 assert 而非 raise，python -O 下守衛全消失）。
    # load_sensors_from_path 是同一份邏輯 + 三道 fail-fast（coords shape /
    # std≈0 / channel 順序）。SENSOR_JSON 是該檔既有的常數。
    d = subsample(load_sensors_from_path(SENSOR_JSON, time_stride=2))
    sv = jnp.asarray(d["sensor_vals"])
    sp = jnp.asarray(d["sensor_pos"])
    st = jnp.asarray(d["sensor_time"])
    norm_stats = d["norm_stats"]

    model = LiquidOperator(**CFG_MINI_REMAT)
    print(f"  use_decoder_remat: {CFG_MINI_REMAT.get('use_decoder_remat', False)}")
    rng = jax.random.PRNGKey(SEED)
    rng, rng_init = jax.random.split(rng)
    init_xy = jnp.asarray(np.random.RandomState(SEED).uniform(0, 1, (8, 2)).astype(np.float32))
    init_t = jnp.asarray(np.random.RandomState(SEED).uniform(0, 5, (8,)).astype(np.float32))
    rss_before_init = get_rss_mb()
    params = model.init(rng_init, sv, sp, RE_NORM, st, init_xy, init_t)
    rss_after_init = get_rss_mb()
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  params: {n_params:,} | RSS init: {rss_before_init:.0f} → {rss_after_init:.0f} MB")

    # ── Sensor query (固定) ──
    T, K = sv.shape[0], sv.shape[1]
    xy_q = jnp.broadcast_to(sp[None], (T, K, 2)).reshape(T * K, 2)
    t_q = jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)
    target = sv.reshape(T * K, 2)

    # ── NS residual (Wave 4 perf API: encode-once) ──
    ns_fn, _ = make_ns_residual_fn(
        model, sp, st, norm_stats=norm_stats, re_value=RE_VALUE,
    )

    def loss_fn(p, cx, cy, ct):
        # encode once → 共用給 sensor pred + PDE residual (XLA trace 只看一次 lax.scan)
        h_states = model.apply(p, sv, sp, RE_NORM, st, method=LiquidOperator.encode)
        pred = model.apply(p, xy_q, t_q, h_states, st, sp,
                            method=LiquidOperator.decode_query)
        sensor_loss = jnp.mean((pred[:, :2] - target) ** 2)
        A_f, k_f = model.apply(p, method=LiquidOperator.get_forcing)
        mom_u, mom_v, cont = ns_fn(p, h_states, cx, cy, ct, A_f, k_f)
        total = sensor_loss + W_PHYS * (mom_u + mom_v + cont)
        return total

    tx = optax.adam(3e-3)
    opt_state = tx.init(params)

    @jax.jit
    def step(p, ost, cx, cy, ct):
        l, g = jax.value_and_grad(loss_fn)(p, cx, cy, ct)
        u, ost = tx.update(g, ost)
        p = optax.apply_updates(p, u)
        return p, ost, l

    # ── jit compile (n_warmup steps) ──
    rng_collo = jax.random.PRNGKey(SEED + 1)
    cx0 = jax.random.uniform(rng_collo, (N_COLLO,))
    cy0 = jax.random.uniform(rng_collo, (N_COLLO,))
    ct0 = jax.random.uniform(rng_collo, (N_COLLO,), minval=0.0, maxval=5.0)

    jit_start = time.perf_counter()
    params, opt_state, l = step(params, opt_state, cx0, cy0, ct0)
    jax.block_until_ready(l)
    jit_wall = time.perf_counter() - jit_start
    print(f"  jit compile wall (step 1): {jit_wall:.2f}s")
    rss_after_jit = get_rss_mb()
    print(f"  RSS after jit: {rss_after_jit:.0f} MB (+{rss_after_jit - rss_after_init:.0f})")

    # Warmup extra step(s) to stabilize (XLA HLO 可能 first-few-steps 還在 graph cache)
    for _ in range(n_warmup):
        rng_collo, sub = jax.random.split(rng_collo)
        keys = jax.random.split(sub, 3)
        cx = jax.random.uniform(keys[0], (N_COLLO,))
        cy = jax.random.uniform(keys[1], (N_COLLO,))
        ct = jax.random.uniform(keys[2], (N_COLLO,), minval=0.0, maxval=5.0)
        params, opt_state, l = step(params, opt_state, cx, cy, ct)
        jax.block_until_ready(l)

    # ── 量測：n_steps cached steps ──
    per_step_walls = []
    rss_peaks = []
    for i in range(n_steps):
        rng_collo, sub = jax.random.split(rng_collo)
        keys = jax.random.split(sub, 3)
        cx = jax.random.uniform(keys[0], (N_COLLO,))
        cy = jax.random.uniform(keys[1], (N_COLLO,))
        ct = jax.random.uniform(keys[2], (N_COLLO,), minval=0.0, maxval=5.0)
        t0 = time.perf_counter()
        params, opt_state, l = step(params, opt_state, cx, cy, ct)
        jax.block_until_ready(l)
        per_step_walls.append(time.perf_counter() - t0)
        rss_peaks.append(get_rss_mb())

    step_arr = np.array(per_step_walls) * 1000  # ms
    rss_arr = np.array(rss_peaks)
    summary = {
        "n_params": int(n_params),
        "jit_wall_s": float(jit_wall),
        "step_ms_median": float(np.median(step_arr)),
        "step_ms_mean": float(step_arr.mean()),
        "step_ms_std": float(step_arr.std()),
        "step_ms_min": float(step_arr.min()),
        "step_ms_max": float(step_arr.max()),
        "rss_mb_init": float(rss_after_init),
        "rss_mb_after_jit": float(rss_after_jit),
        "rss_mb_peak_step": float(rss_arr.max()),
        "rss_mb_final": float(rss_arr[-1]),
        "n_steps_measured": n_steps,
        "n_warmup": n_warmup,
        "n_collo": N_COLLO,
        "final_loss": float(l),
    }
    print(f"  step wall (median/mean/min/max): "
          f"{summary['step_ms_median']:.1f} / {summary['step_ms_mean']:.1f} / "
          f"{summary['step_ms_min']:.1f} / {summary['step_ms_max']:.1f} ms")
    print(f"  RSS peak / final: {summary['rss_mb_peak_step']:.0f} / {summary['rss_mb_final']:.0f} MB")
    print(f"  final_loss: {summary['final_loss']:.4e}")
    return summary


def compare(labels: list[str], bench_dir: Path):
    """讀 labels 對應 JSON 並印 side-by-side 對比表。"""
    data = {}
    for lab in labels:
        path = bench_dir / f"{lab}.json"
        if not path.exists():
            print(f"[WARN] {path} 不存在，跳過 {lab}")
            continue
        with open(path) as f:
            data[lab] = json.load(f)

    if len(data) < 2:
        print("需要至少 2 個 label 比較。")
        return

    keys = ["jit_wall_s", "step_ms_median", "step_ms_mean",
            "rss_mb_after_jit", "rss_mb_peak_step", "n_params"]
    labs = list(data.keys())
    print(f"\n=== Compare: {' vs '.join(labs)} ===")
    print(f"{'metric':25s}  " + "  ".join(f"{l:>14s}" for l in labs) + "  " +
          (f"{'speedup vs ' + labs[0]:>14s}" if len(labs) >= 2 else ""))
    print("-" * (25 + 16 * len(labs) + 18))
    for k in keys:
        vals = [data[l][k] for l in labs]
        row = f"{k:25s}  " + "  ".join(f"{v:>14.3f}" for v in vals)
        if k in ("jit_wall_s", "step_ms_median", "step_ms_mean") and len(labs) >= 2:
            speedup = vals[0] / vals[-1]
            row += f"  {speedup:>14.2f}x"
        elif k in ("rss_mb_after_jit", "rss_mb_peak_step") and len(labs) >= 2:
            ratio = vals[-1] / vals[0]
            row += f"  {ratio:>14.2f}x (mem)"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None, help="存檔 label (baseline / post_opt / ...)")
    parser.add_argument("--n_steps", type=int, default=30, help="cached-step 量測數")
    parser.add_argument("--n_warmup", type=int, default=2, help="jit 後 warmup 步數")
    parser.add_argument("--compare", nargs="+", default=None, help="比較這些 labels")
    parser.add_argument("--bench_dir", type=str, default="bench/results")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare(args.compare, bench_dir)
        return

    summary = run_benchmark(n_steps=args.n_steps, n_warmup=args.n_warmup)
    label = args.label or f"run_{int(time.time())}"
    path = bench_dir / f"{label}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
