"""EXP-245 訓練 step 時間剖析：拆 encode / decode / physics / backward / SOAP，
並比 matmul precision (fp32 / tf32 / bf16)。

回答「step 時間花在哪、forward vs backward 誰最久、哪個算法該優化」。
用 production LiquidOperator + make_ns_residual_fn(fof)，EXP-245 scale（d_model=256, T=101,
K=100, N_collo=1024, rank=256）。fp32 arrays，僅變 matmul precision（RTX 3090 tensor core）。

分段（jit 各別、block 計時）：
  enc   : encode（TemporalCfC lax.scan over T）
  encdec: encode + decode(sensor) → sensor loss
  fwd   : + physics 殘差(fof) → total loss            [forward]
  grad  : jax.grad(fwd)                                [forward+backward]
  step  : grad + SOAP update                           [完整 step]
推得：encode / decode / physics / backward / optimizer 佔比。
另印 forward loss 值（tf32/bf16 是否數值漂移）。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax

_PREC = {"fp32": "highest", "tf32": "tensorfloat32", "bf16": "bfloat16"}


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def median_ms(fn, *args, n=20):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    ap.add_argument("--n-collo", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", _PREC[args.precision])

    import jax.numpy as jnp
    import numpy as np
    import optax
    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.physics import make_ns_residual_fn
    from pi_lnn_jax.optimizers import build_optimizer

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        eprint(f"[FATAL] backend={backend}（非 gpu）。加 --allow-cpu 才在 CPU 跑。")
        sys.exit(5)
    dev = jax.devices()[0]

    # ── EXP-245 scale model（對齊 configs/exp_245_b3_les_T50.toml [model]）──
    MK = dict(
        sensor_value_dim=2, d_model=256, d_time=16,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=2, token_attention_heads=4,
        num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
        use_temporal_anchor=True, T_total=5.0, temporal_anchor_harmonics=2,
        domain_length=1.0,
    )
    RE_VALUE = 10000.0
    RE_NORM = float(np.log(RE_VALUE) / np.log(10000.0))
    T, K, N = 101, 100, args.n_collo
    nu = jnp.asarray(1.0 / RE_VALUE, jnp.float32)
    um, us, vm, vs = (jnp.asarray(c, jnp.float32) for c in (0.0, 0.4167, 0.0, 0.4167))

    rng = np.random.RandomState(0)
    sensor_vals = jnp.asarray(rng.standard_normal((T, K, 2)), dtype=jnp.float32)
    sensor_pos = jnp.asarray(rng.uniform(0, 1, (K, 2)), dtype=jnp.float32)
    sensor_time = jnp.asarray(np.linspace(0, 5, T), dtype=jnp.float32)

    model = LiquidOperator(**MK)
    init_xy = jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=jnp.float32)
    init_t = jnp.asarray(rng.uniform(0, 5, (8,)), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), sensor_vals, sensor_pos,
                        RE_NORM, sensor_time, init_xy, init_t)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    xy_q = jnp.broadcast_to(sensor_pos[None], (T, K, 2)).reshape(T * K, 2)
    t_q = jnp.broadcast_to(sensor_time[:, None], (T, K)).reshape(T * K)
    target = sensor_vals.reshape(T * K, 2)
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    cx = jax.random.uniform(k[0], (N,), dtype=jnp.float32, minval=0.0, maxval=1.0)
    cy = jax.random.uniform(k[1], (N,), dtype=jnp.float32, minval=0.0, maxval=1.0)
    ct = jax.random.uniform(k[2], (N,), dtype=jnp.float32, minval=0.0, maxval=5.0)
    A, k_f = jnp.float32(0.1), jnp.float32(2.0)

    ns_fn, _ = make_ns_residual_fn(model)
    LO = LiquidOperator

    def encode_only(p):
        return model.apply(p, sensor_vals, sensor_pos, RE_NORM, sensor_time, method=LO.encode)

    def encdec(p):
        h = encode_only(p)
        pred = model.apply(p, xy_q, t_q, h, sensor_time, sensor_pos, method=LO.decode_query)
        return jnp.mean((pred[:, :2] - target) ** 2)

    def fwd(p):
        h = model.apply(p, sensor_vals, sensor_pos, RE_NORM, sensor_time, method=LO.encode)
        pred = model.apply(p, xy_q, t_q, h, sensor_time, sensor_pos, method=LO.decode_query)
        sensor_loss = jnp.mean((pred[:, :2] - target) ** 2)
        mu, mv, c = ns_fn(p, h, cx, cy, ct, A, k_f, sensor_pos, sensor_time, nu, um, us, vm, vs)
        return sensor_loss + 1.0 * (mu + mv + c)

    grad_fn = jax.grad(fwd)
    tx, _info = build_optimizer(name="schedule_free", learning_rate=1e-3,
                                base_optimizer="soap", soap_precondition_frequency=2)
    opt_state = tx.init(params)

    def step(p, ost):
        g = jax.grad(fwd)(p)
        upd, ost = tx.update(g, ost, p)
        return optax.apply_updates(p, upd), ost

    j_enc = jax.jit(encode_only)
    j_encdec = jax.jit(encdec)
    j_fwd = jax.jit(fwd)
    j_grad = jax.jit(grad_fn)
    j_step = jax.jit(step)

    loss_val = float(j_fwd(params))   # 數值漂移檢查

    t_enc = median_ms(j_enc, params, n=args.iters)
    t_encdec = median_ms(j_encdec, params, n=args.iters)
    t_fwd = median_ms(j_fwd, params, n=args.iters)
    t_grad = median_ms(j_grad, params, n=args.iters)
    t_step = median_ms(lambda p: j_step(p, opt_state)[0], params, n=args.iters)
    peak = (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6

    print(f"\n=== EXP-245 step profile | precision={args.precision} | "
          f"backend={backend} | params={n_params:,} | N_collo={N} ===")
    print(f"  forward loss value = {loss_val:.6e}  (tf32/bf16 漂移檢查)")
    print(f"  peak_mem = {peak:.0f} MB\n")
    print(f"  {'segment':<28s}{'cumulative ms':>14s}{'delta ms':>12s}{'share%':>9s}")
    enc = t_enc; dec = t_encdec - t_enc; phys = t_fwd - t_encdec
    bwd = t_grad - t_fwd; opt = t_step - t_grad
    total = t_step
    rows = [("encode (CfC scan over T)", t_enc, enc),
            ("+ decode(sensor)", t_encdec, dec),
            ("+ physics residual (fof)", t_fwd, phys),
            ("+ backward (grad/params)", t_grad, bwd),
            ("+ SOAP update", t_step, opt)]
    for name, cum, dlt in rows:
        print(f"  {name:<28s}{cum:>14.2f}{dlt:>12.2f}{100*dlt/total:>8.1f}%")
    print(f"\n  FORWARD total  = {t_fwd:.2f} ms ({100*t_fwd/total:.0f}%)")
    print(f"  BACKWARD       = {bwd:.2f} ms ({100*bwd/total:.0f}%)")
    print(f"  OPTIMIZER(SOAP)= {opt:.2f} ms ({100*opt/total:.0f}%)")
    print(f"  STEP total     = {t_step:.2f} ms")


if __name__ == "__main__":
    main()
