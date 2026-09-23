"""量 backward 中「physics 路徑」vs「data 路徑」各佔多少。

做法：比 grad(full_loss) vs grad(data_only_loss)，差值 = physics backward 成本。
EXP-245 scale (d_model=256, T=101, K=100, N_collo=1024)，sensor mini-batch=2000。
這個數字決定：Taylor-mode 若能用於 physics backward 能省多少。
"""
from __future__ import annotations
import statistics
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn


def eprint(*a): print(*a, file=sys.stderr, flush=True)


def median_ms(fn, *args, n=20):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        eprint(f"[FATAL] backend={backend}. Add --allow-cpu."); sys.exit(5)

    MK = dict(
        sensor_value_dim=2, d_model=256, d_time=16,
        num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
        num_token_attention_layers=2, token_attention_heads=4,
        num_query_mlp_layers=1, query_mlp_hidden_dim=256, operator_rank=256,
        use_temporal_anchor=True, T_total=5.0, temporal_anchor_harmonics=2,
        domain_length=1.0,
    )
    RE = 10000.0; RE_NORM = float(np.log(RE) / np.log(10000.0))
    T, K, N, N_SQ = 101, 100, 1024, 2000
    nu = jnp.float32(1.0 / RE)
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
    # sensor mini-batch (fixed subset)
    idx = jax.random.choice(jax.random.PRNGKey(9), T*K, (N_SQ,), replace=False)
    xy_q = jnp.broadcast_to(sp[None],(T,K,2)).reshape(T*K,2)[idx]
    t_q  = jnp.broadcast_to(st[:,None],(T,K)).reshape(T*K)[idx]
    tgt  = sv.reshape(T*K,2)[idx]
    # collocation
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    cx = jax.random.uniform(k[0], (N,), jnp.float32, 0.0, 1.0)
    cy = jax.random.uniform(k[1], (N,), jnp.float32, 0.0, 1.0)
    ct = jax.random.uniform(k[2], (N,), jnp.float32, 0.0, 5.0)
    A, k_f = jnp.float32(0.1), jnp.float32(2.0)
    ns_fn, _ = make_ns_residual_fn(model)
    LO = LiquidOperator

    def data_only(p):
        h = model.apply(p, sv, sp, RE_NORM, st, method=LO.encode)
        pred = model.apply(p, xy_q, t_q, h, st, sp, method=LO.decode_query)
        return jnp.mean((pred[:, :2] - tgt) ** 2)

    def full_loss(p):
        h = model.apply(p, sv, sp, RE_NORM, st, method=LO.encode)
        pred = model.apply(p, xy_q, t_q, h, st, sp, method=LO.decode_query)
        sl = jnp.mean((pred[:, :2] - tgt) ** 2)
        mu, mv, c = ns_fn(p, h, cx, cy, ct, A, k_f, sp, st, nu, um, us, vm, vs)
        return sl + (mu + mv + c)

    g_data = jax.jit(jax.grad(data_only))
    g_full = jax.jit(jax.grad(full_loss))

    t_data = median_ms(g_data, params)
    t_full = median_ms(g_full, params)
    t_phys_bwd = t_full - t_data

    print(f"\n=== backward split (EXP-245 scale, sensor mini-batch={N_SQ}) ===")
    print(f"  params={n_params:,}  N_collo={N}  backend={backend}")
    print(f"  grad(data only)  = {t_data:.1f} ms  ({100*t_data/t_full:.0f}% of full grad)")
    print(f"  grad(data+phys)  = {t_full:.1f} ms")
    print(f"  physics backward = {t_phys_bwd:.1f} ms  ({100*t_phys_bwd/t_full:.0f}% of full grad)")
    print("\n  → Taylor-mode 若能完全消除 physics backward overhead，")
    print(f"    理論最大省幅 = {t_phys_bwd:.1f}ms/{t_full:.1f}ms = {100*t_phys_bwd/t_full:.0f}%")


if __name__ == "__main__":
    main()
