"""JAX Natural-Gradient (Gauss-Newton) feasibility spike。

三方案建 J / 解 GN step（全做、比效能）：
  A1  materialized jacrev : J = jacrev(r_flat)(θ) [M,P] → kernel trick + Jacobi (fp64)
  A2  matrix-free CG      : 不建 J；(JJᵀ+λ)z=r 用 jvp∘vjp matvec + CG；Δθ = Jᵀz
  A3  optimistix LM       : LevenbergMarquardt 1-step（cross-check baseline）

Gauss-Newton 不需真 Hessian：只用殘差對參數的一階 Jacobian J=∂r/∂θ。
殘差 r 內的二階空間導（Laplacian）由 fof（autodiff.py）解決，jacrev 對其再求 param-Jacobian。

隔離於 bench/：沿用 autodiff_modes 的 isolated LiquidOperator + fof（避 torch CUDA 衝突）。
全程 x64（GN solve 需 fp64）。

Modes:
  --mode correctness : 解析 GN 1-step 收斂 + 真實 mini model loss 下降 + A1/A2/LM 方向 cosine
  --mode perf --variant {A1,A2} --n-collo N : 單配置 compile/peak-mem/wall（獨立 process 量峰值）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from bench.autodiff_modes import LiquidOperator, make_field_fn, _make_fof


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# GN solvers
# ─────────────────────────────────────────────────────────────────────────────

def ng_step_kernel(J, r, damping=1e-6, jacobi=True):
    """A1: kernel trick 解 (JJᵀ+λ) → Δθ = Jᵀz（fp64）。直譯 pi-lnn solve_ng_step。"""
    J64 = J.astype(jnp.float64)
    r64 = r.astype(jnp.float64).reshape(-1)
    N = J64.shape[0]
    K = J64 @ J64.T                       # [N,N]
    eye = jnp.eye(N, dtype=jnp.float64)
    if jacobi:
        d = jnp.clip(jnp.diag(K), 1e-12, None)
        dis = 1.0 / jnp.sqrt(d)
        Kt = K * dis[:, None] * dis[None, :]
        y = jnp.linalg.solve(Kt + damping * eye, dis * r64)
        z = dis * y
    else:
        z = jnp.linalg.solve(K + damping * eye, r64)
    return J64.T @ z                       # Δθ flat [P]


def ng_step_cg(r_flat_fn, flat, damping=1e-6, tol=1e-6, maxiter=200):
    """A2: matrix-free。(JJᵀ+λ)z=r 用 jvp∘vjp matvec + CG；Δθ = Jᵀz。

    在 model dtype 運作（jvp/vjp 須對齊 params dtype）；CG 容差/迭代上限可調。
    """
    r0, vjp_fn = jax.vjp(r_flat_fn, flat)   # r0 = r(flat)，vjp_fn(v) = Jᵀv
    def Jt(v):
        return vjp_fn(v)[0]
    def Jv(w):
        return jax.jvp(r_flat_fn, (flat,), (w,))[1]
    def K_matvec(v):
        return Jv(Jt(v)) + damping * v
    z, _ = jax.scipy.sparse.linalg.cg(K_matvec, r0, tol=tol, maxiter=maxiter)
    return Jt(z), r0                        # Δθ flat [P], r0


def build_J_dense(r_flat_fn, flat):
    """A1 用：J = ∂r/∂θ [M,P] via jacrev（M 個 VJP，vmap+jit）。"""
    return jax.jacrev(r_flat_fn)(flat)


# ─────────────────────────────────────────────────────────────────────────────
# 真實 mini-model residual vector r(θ) ∈ R^M（對齊 refiners.make_pinn_residual_vector_fn）
# ─────────────────────────────────────────────────────────────────────────────

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


def build_mini_problem(n_collo, dtype=jnp.float64, seed=0):
    """回傳 (params, residual_vec_fn)。residual_vec_fn(params)->r [M]。"""
    rng = np.random.RandomState(seed)
    T, K = 11, 25
    sensor_vals = jnp.asarray(rng.standard_normal((T, K, 2)), dtype=dtype)
    sensor_pos = jnp.asarray(rng.uniform(0, 1, (K, 2)), dtype=dtype)
    sensor_time = jnp.asarray(np.linspace(0, 5, T), dtype=dtype)
    um, us, vm, vs = (jnp.asarray(c, dtype) for c in (0.1, 0.8, -0.05, 0.7))
    nu = jnp.asarray(1.0 / RE_VALUE, dtype)

    model = LiquidOperator(**CFG_MINI)
    init_xy = jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=dtype)
    init_t = jnp.asarray(rng.uniform(0, 5, (8,)), dtype=dtype)
    params = model.init(jax.random.PRNGKey(seed), sensor_vals, sensor_pos,
                        RE_NORM, sensor_time, init_xy, init_t)

    xy_sensor_q = jnp.broadcast_to(sensor_pos[None], (T, K, 2)).reshape(T * K, 2)
    t_sensor_q = jnp.broadcast_to(sensor_time[:, None], (T, K)).reshape(T * K)
    sensor_target = sensor_vals.reshape(T * K, 2)

    k = jax.random.split(jax.random.PRNGKey(seed + 1), 3)
    cx = jax.random.uniform(k[0], (n_collo,), dtype=dtype, minval=0.0, maxval=1.0)
    cy = jax.random.uniform(k[1], (n_collo,), dtype=dtype, minval=0.0, maxval=1.0)
    ct = jax.random.uniform(k[2], (n_collo,), dtype=dtype, minval=0.0, maxval=5.0)

    field_fn = make_field_fn(model)
    fused = _make_fof(field_fn)
    common = (sensor_pos, sensor_time, um, us, vm, vs)
    sqrt_wd, sqrt_wp = jnp.sqrt(jnp.asarray(1.0, dtype)), jnp.sqrt(jnp.asarray(0.1, dtype))

    def residual_vec_fn(p):
        h = model.apply(p, sensor_vals, sensor_pos, RE_NORM, sensor_time,
                        method=LiquidOperator.encode)
        pred = model.apply(p, xy_sensor_q, t_sensor_q, h, sensor_time, sensor_pos,
                           method=LiquidOperator.decode_query)
        data_r = (pred[:, :2] - sensor_target).reshape(-1)
        A_f, k_f = model.apply(p, method=LiquidOperator.get_forcing)
        value, jac, d2x, d2y = fused(p, h, cx, cy, ct, *common)
        u = value[:, 0]; v = value[:, 1]
        u_x = jac[:, 0, 0]; u_y = jac[:, 0, 1]; u_t = jac[:, 0, 2]
        v_x = jac[:, 1, 0]; v_y = jac[:, 1, 1]; v_t = jac[:, 1, 2]
        p_x = jac[:, 2, 0]; p_y = jac[:, 2, 1]
        u_xx = d2x[:, 0]; u_yy = d2y[:, 0]; v_xx = d2x[:, 1]; v_yy = d2y[:, 1]
        f_x = A_f * jnp.sin(2.0 * jnp.pi * k_f * ct)
        mom_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - f_x
        mom_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont = u_x + v_y
        return jnp.concatenate([sqrt_wd * data_r, sqrt_wp * mom_u,
                                sqrt_wp * mom_v, sqrt_wp * cont])

    return params, residual_vec_fn


def loss_of(r):
    return 0.5 * float(jnp.sum(r ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_analytic():
    """線性殘差 r(θ)=Aθ-b：GN 應 1 步收斂到 LS 解。驗 A1/A2 solve 數學。"""
    rng = np.random.RandomState(0)
    M, P = 80, 200
    A = jnp.asarray(rng.standard_normal((M, P)) / np.sqrt(P), dtype=jnp.float64)
    b = jnp.asarray(rng.standard_normal(M), dtype=jnp.float64)
    theta0 = jnp.zeros(P, dtype=jnp.float64)
    r_fn = lambda th: A @ th - b
    theta_ls = jnp.linalg.lstsq(A, b, rcond=None)[0]   # min ||Aθ-b||

    # A1
    J = build_J_dense(r_fn, theta0)
    d1 = ng_step_kernel(J, r_fn(theta0), damping=1e-10, jacobi=False)
    err1 = float(jnp.linalg.norm((theta0 - d1) - theta_ls))
    # A2
    d2, _ = ng_step_cg(r_fn, theta0, damping=1e-10, tol=1e-12, maxiter=P)
    err2 = float(jnp.linalg.norm((theta0 - d2) - theta_ls))
    print(f"  [analytic] A1 ||θ_GN - θ_LS||={err1:.2e}  A2={err2:.2e}  "
          f"[{'OK' if max(err1, err2) < 1e-4 else 'CHECK'}]")


def cosine(a, b):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def test_real(n_collo=128, steps=5, damping=1e-4):
    params, r_fn = build_mini_problem(n_collo, dtype=jnp.float64)
    flat0, unravel = ravel_pytree(params)
    r_flat = lambda f: r_fn(unravel(f))

    # 方向交叉比對（A1 vs A2 vs LM）@ θ0
    J = build_J_dense(r_flat, flat0)
    r0 = r_flat(flat0)
    dA1 = ng_step_kernel(J, r0, damping=damping, jacobi=True)
    dA2, _ = ng_step_cg(r_flat, flat0, damping=damping, tol=1e-8, maxiter=300)
    print(f"  [real N={n_collo}] M={J.shape[0]} P={J.shape[1]}  loss0={loss_of(r0):.4e}")
    print(f"  [direction] cosine(A1,A2)={cosine(dA1, dA2):.4f}")
    try:
        import optimistix as optx
        sol = optx.least_squares(lambda y, args: r_flat(y),
                                 optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
                                 flat0, max_steps=1, throw=False)
        dLM = flat0 - sol.value
        print(f"  [direction] cosine(A1,LM_1step)={cosine(dA1, dLM):.4f}")
    except Exception as e:
        print(f"  [LM] skipped: {type(e).__name__}: {e}")

    # A1 多步 loss 下降（lr=1.0 GN）
    flat = flat0
    losses = [loss_of(r0)]
    for s in range(steps):
        J = build_J_dense(r_flat, flat)
        d = ng_step_kernel(J, r_flat(flat), damping=damping, jacobi=True)
        flat = flat - d
        losses.append(loss_of(r_flat(flat)))
    mono = all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1))
    print(f"  [A1 {steps}-step loss] " + " → ".join(f"{l:.3e}" for l in losses) +
          f"  [{'monotone↓ OK' if mono else 'NOT monotone'}]")


# ─────────────────────────────────────────────────────────────────────────────
# Perf（單配置；獨立 process 量 peak-mem）
# ─────────────────────────────────────────────────────────────────────────────

def peak_mb(dev):
    return (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6


def median_ms(fn, *args, n=10):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def run_perf(variant, n_collo, damping=1e-4):
    dev = jax.devices()[0]
    params, r_fn = build_mini_problem(n_collo, dtype=jnp.float64)
    flat0, unravel = ravel_pytree(params)
    r_flat = lambda f: r_fn(unravel(f))
    P = flat0.size
    M = int(r_flat(flat0).shape[0])

    if variant == "A1":
        def ng_full(flat):
            J = build_J_dense(r_flat, flat)
            return ng_step_kernel(J, r_flat(flat), damping=damping, jacobi=True)
    elif variant == "A2":
        def ng_full(flat):
            d, _ = ng_step_cg(r_flat, flat, damping=damping, tol=1e-6, maxiter=200)
            return d
    else:
        raise ValueError(variant)

    fn = jax.jit(ng_full)
    t0 = time.perf_counter()
    out = fn(flat0); jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1e3
    step_ms = median_ms(fn, flat0)

    rec = dict(variant=variant, n_collo=n_collo, M=M, P=int(P),
               device=jax.default_backend(),
               peak_mb=round(peak_mb(dev), 2),
               compile_ms=round(compile_ms, 1), step_ms=round(step_ms, 3))
    eprint(f"  [{variant}|N={n_collo}] M={M} P={P} peak={rec['peak_mb']:.1f}MB "
           f"compile={rec['compile_ms']:.0f}ms step={rec['step_ms']:.2f}ms")
    print(json.dumps(rec), flush=True)


def run_convergence(n_collo=128, ng_steps=25, adam_steps=2000, adam_lr=1e-3):
    """A2-NG（matrix-free CG + LM damping 自適應）vs Adam：steps/wall-to-accuracy。

    回答「GN 值不值得」：同一 residual loss 0.5||r||²，比兩者
      - loss vs step、loss vs wall
      - 達固定門檻所需步數/wall、最終可達 loss
    """
    import optax

    params, r_fn = build_mini_problem(n_collo, dtype=jnp.float64)
    flat0, unravel = ravel_pytree(params)
    r_flat = lambda f: r_fn(unravel(f))
    loss_flat = jax.jit(lambda f: 0.5 * jnp.sum(r_flat(f) ** 2))
    ng_cg = jax.jit(lambda f, lam: ng_step_cg(r_flat, f, damping=lam,
                                              tol=1e-6, maxiter=200))

    def thresholds_report(traj):
        """traj: list of (step, loss, wall_s)。回傳達門檻的 (step, wall)。"""
        out = {}
        for thr in (1e-2, 1e-3, 1e-4, 1e-5):
            hit = next((t for t in traj if t[1] <= thr), None)
            out[thr] = (hit[0], hit[2]) if hit else None
        return out

    # ── A2-NG ──
    print(f"\n  --- A2-NG (CG + LM damping) N={n_collo}, {ng_steps} steps ---", flush=True)
    flat = flat0
    lam = 1e-3
    l_cur = float(loss_flat(flat)); jax.block_until_ready(l_cur)
    ng_traj = [(0, l_cur, 0.0)]
    t0 = time.perf_counter()
    for step in range(1, ng_steps + 1):
        d, _ = ng_cg(flat, lam); jax.block_until_ready(d)
        flat_try = flat - d
        l_try = float(loss_flat(flat_try))
        if l_try < l_cur:                  # accept，降 damping
            flat, l_cur, lam = flat_try, l_try, max(lam * 0.7, 1e-8)
        else:                              # reject，升 damping（下一步更保守）
            lam = min(lam * 3.0, 1e2)
        ng_traj.append((step, l_cur, time.perf_counter() - t0))
        if step <= 5 or step % 5 == 0:
            print(f"    ng step {step:>3d}  loss={l_cur:.4e}  λ={lam:.1e}  "
                  f"wall={ng_traj[-1][2]:.1f}s", flush=True)

    # ── Adam ──
    print(f"\n  --- Adam (lr={adam_lr}) N={n_collo}, {adam_steps} steps ---", flush=True)
    opt = optax.adam(adam_lr)
    st = opt.init(params)
    def loss_p(p): return 0.5 * jnp.sum(r_fn(p) ** 2)
    @jax.jit
    def adam_step(p, s):
        l, g = jax.value_and_grad(loss_p)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l
    p = params
    l0 = float(loss_p(p))
    ad_traj = [(0, l0, 0.0)]
    t0 = time.perf_counter()
    for step in range(1, adam_steps + 1):
        p, st, l = adam_step(p, st)
        if step % 50 == 0 or step == adam_steps:
            lf = float(l); ad_traj.append((step, lf, time.perf_counter() - t0))
            if step % 500 == 0 or step == adam_steps:
                print(f"    adam step {step:>5d}  loss={lf:.4e}  wall={ad_traj[-1][2]:.1f}s",
                      flush=True)

    ng_thr = thresholds_report(ng_traj)
    ad_thr = thresholds_report(ad_traj)
    print("\n  === steps/wall to reach loss threshold ===", flush=True)
    print(f"  {'thr':>6s} | {'NG step':>8s} {'NG wall':>9s} | {'Adam step':>10s} {'Adam wall':>10s}")
    for thr in (1e-2, 1e-3, 1e-4, 1e-5):
        n = ng_thr[thr]; a = ad_thr[thr]
        ns = f"{n[0]}" if n else "—"; nw = f"{n[1]:.1f}s" if n else "—"
        as_ = f"{a[0]}" if a else "—"; aw = f"{a[1]:.1f}s" if a else "—"
        print(f"  {thr:>6.0e} | {ns:>8s} {nw:>9s} | {as_:>10s} {aw:>10s}")
    print(f"\n  final loss: NG={ng_traj[-1][1]:.4e} ({ng_traj[-1][2]:.1f}s) | "
          f"Adam={ad_traj[-1][1]:.4e} ({ad_traj[-1][2]:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "perf", "convergence"], required=True)
    ap.add_argument("--variant", choices=["A1", "A2"], default="A1")
    ap.add_argument("--n-collo", type=int, default=128)
    ap.add_argument("--ng-steps", type=int, default=25)
    ap.add_argument("--adam-steps", type=int, default=2000)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu and args.mode == "perf":
        eprint(f"[FATAL] backend={backend}（非 gpu）。perf 量測需 GPU；或加 --allow-cpu。")
        sys.exit(5)

    if args.mode == "correctness":
        print(f"=== correctness (backend={backend}, x64) ===", flush=True)
        test_analytic()
        for n in (128, 256):
            try:
                test_real(n_collo=n, steps=3)
            except Exception as e:   # OOM/其他：保住已印結果，標記失敗
                print(f"  [real N={n}] FAILED: {type(e).__name__}: {e}", flush=True)
    elif args.mode == "convergence":
        print(f"=== convergence: A2-NG vs Adam (backend={backend}, x64) ===", flush=True)
        run_convergence(args.n_collo, args.ng_steps, args.adam_steps)
    else:
        run_perf(args.variant, args.n_collo)


if __name__ == "__main__":
    main()
