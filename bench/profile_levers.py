"""優化槓桿天花板測量：SOAP freq sweep + sensor mini-batch，量穩態 ms/step。

修正 profile_exp245.py 的 SOAP 測量誤差：那邊每次重跑 init_step（完整 eigh，因 opt_state
count 沒推進）→ 高估 SOAP。此處**正確推進 (params, opt_state)** 跑 warmup + 穩態 N 步取 mean，
讓 freq-gated QR refresh 以正確頻率攤提。

configs（由 sbatch 掃）：
  adam, full            → 無 SOAP 基準
  soap freq=2,  full    → 現況
  soap freq=10, full    → SOAP 論文 default
  soap freq=1e9,full    → 純每步 SOAP（無 refresh）→ 推得 refresh 佔比
  soap freq=10, sb=2000 → backward(sensor mini-batch) 疊加

推得：SOAP 穩態開銷 = soap(freq=10) − adam；refresh 佔比 = soap(2) − soap(∞)；
       sensor mini-batch backward 省幅 = soap(10,full) − soap(10,sb=2000)。
不設 tf32（用 jax 預設），對齊真實訓練。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.optimizers import build_optimizer


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["soap", "adam"], default="soap")
    ap.add_argument("--soap-freq", type=int, default=2)
    ap.add_argument("--sensor-batch", type=int, default=0, help="0=full T*K；>0 取子集 query 數")
    ap.add_argument("--n-collo", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        eprint(f"[FATAL] backend={backend}（非 gpu）。加 --allow-cpu 才在 CPU 跑。")
        sys.exit(5)

    MK = dict(
        sensor_value_dim=2, d_model=256, d_time=16,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=2, token_attention_heads=4,
        num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
        use_temporal_anchor=True, T_total=5.0, temporal_anchor_harmonics=2,
        domain_length=1.0,
    )
    RE = 10000.0
    RE_NORM = float(np.log(RE) / np.log(10000.0))
    T, K, N = 101, 100, args.n_collo
    nu = jnp.asarray(1.0 / RE, jnp.float32)
    um, us, vm, vs = (jnp.asarray(c, jnp.float32) for c in (0.0, 0.4167, 0.0, 0.4167))

    rng = np.random.RandomState(0)
    sv = jnp.asarray(rng.standard_normal((T, K, 2)), dtype=jnp.float32)
    sp = jnp.asarray(rng.uniform(0, 1, (K, 2)), dtype=jnp.float32)
    st = jnp.asarray(np.linspace(0, 5, T), dtype=jnp.float32)

    model = LiquidOperator(**MK)
    ix = jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=jnp.float32)
    it = jnp.asarray(rng.uniform(0, 5, (8,)), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), sv, sp, RE_NORM, st, ix, it)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    xy_q = jnp.broadcast_to(sp[None], (T, K, 2)).reshape(T * K, 2)
    t_q = jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)
    target = sv.reshape(T * K, 2)
    # sensor mini-batch：取固定子集（timing 不依賴選哪些點）
    if args.sensor_batch and args.sensor_batch < T * K:
        idx = jnp.asarray(rng.choice(T * K, args.sensor_batch, replace=False))
        xy_q, t_q, target = xy_q[idx], t_q[idx], target[idx]

    k = jax.random.split(jax.random.PRNGKey(1), 3)
    cx = jax.random.uniform(k[0], (N,), dtype=jnp.float32, minval=0.0, maxval=1.0)
    cy = jax.random.uniform(k[1], (N,), dtype=jnp.float32, minval=0.0, maxval=1.0)
    ct = jax.random.uniform(k[2], (N,), dtype=jnp.float32, minval=0.0, maxval=5.0)
    A, k_f = jnp.float32(0.1), jnp.float32(2.0)
    ns_fn, _ = make_ns_residual_fn(model)
    LO = LiquidOperator

    def loss_fn(p):
        h = model.apply(p, sv, sp, RE_NORM, st, method=LO.encode)
        pred = model.apply(p, xy_q, t_q, h, st, sp, method=LO.decode_query)
        sl = jnp.mean((pred[:, :2] - target) ** 2)
        mu, mv, c = ns_fn(p, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs)
        return sl + (mu + mv + c)

    if args.optimizer == "adam":
        tx, info = build_optimizer(name="adam", learning_rate=1e-3)
    else:
        tx, info = build_optimizer(name="soap", learning_rate=1e-3,
                                   soap_precondition_frequency=args.soap_freq)
    opt_state = tx.init(params)

    @jax.jit
    def step(p, ost):
        g = jax.grad(loss_fn)(p)
        upd, ost = tx.update(g, ost, p)
        return optax.apply_updates(p, upd), ost

    # warmup（推進 state，過 init + 填 preconditioner）
    for _ in range(args.warmup):
        params, opt_state = step(params, opt_state)
    jax.block_until_ready(params)

    # 穩態計時（持續推進 state，mean 攤提 freq pattern）
    ts = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        params, opt_state = step(params, opt_state)
        jax.block_until_ready(params)
        ts.append(time.perf_counter() - t0)
    mean_ms = statistics.mean(ts) * 1e3
    med_ms = statistics.median(ts) * 1e3

    tag = (f"{args.optimizer}"
           + (f"_f{args.soap_freq}" if args.optimizer == "soap" else "")
           + (f"_sb{args.sensor_batch}" if args.sensor_batch else "_sbFull"))
    print(f"[{tag}] mean={mean_ms:.2f}ms median={med_ms:.2f}ms "
          f"(N_q={xy_q.shape[0]}, params={n_params:,}, fallback={info.get('fallback_to')})",
          flush=True)


if __name__ == "__main__":
    main()
