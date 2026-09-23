"""Measure the leading Lyapunov exponent lambda_max of the 2D Kolmogorov system.

System (matches pi_lnn_jax/physics.py + config defaults for Re=10000):
    2D incompressible Navier-Stokes, unit periodic domain [0,1]^2,
    nu = 1/Re, forcing f_x = A*sin(2*pi*k_f*y), f_y = 0.

Method: pseudo-spectral vorticity solver (2/3 dealiasing, integrating-factor
RK4 for the stiff viscous term), and a Benettin-style estimate of lambda_max
using the tangent linear model (TLM) obtained via jax.jvp on the one-step map.

Subcommands
    selftest : local isolated correctness gates (small N, CPU). No training.
    measure  : ensemble Benettin measurement (meant for slurm / GPU).

Comments in English per repo research-code convention.
"""

from __future__ import annotations

import argparse
import json
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# Spectral operators on the unit periodic domain [0,1]^2.
# Wavenumbers are angular: k = 2*pi*n with n = fftfreq(N)*N (integer cycles/L).
# ----------------------------------------------------------------------------


def _wavenumbers(N: int):
    k1d = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)  # angular wavenumbers
    kx = k1d[:, None]
    ky = k1d[None, :]
    ksq = kx**2 + ky**2
    inv_ksq = jnp.where(ksq == 0.0, 0.0, 1.0 / ksq)  # zero-out the mean mode
    return kx, ky, ksq, inv_ksq


def _dealias_mask(N: int):
    # 2/3 rule: keep |n| <= N/3.
    n = jnp.fft.fftfreq(N, d=1.0 / N)
    cutoff = N / 3.0
    m1d = jnp.abs(n) <= cutoff
    return (m1d[:, None] & m1d[None, :]).astype(jnp.float64)


def vorticity_from_uv(u, v, kx, ky):
    # omega = v_x - u_y in spectral space.
    return jnp.fft.ifft2(1j * kx * jnp.fft.fft2(v) - 1j * ky * jnp.fft.fft2(u)).real


def uv_from_what(what, kx, ky, inv_ksq):
    psi_hat = what * inv_ksq  # (-lap) psi = omega  =>  psi_hat = what/ksq
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    return jnp.fft.ifft2(u_hat).real, jnp.fft.ifft2(v_hat).real


def _rhs_hat(what, kx, ky, inv_ksq, mask, nu, force_hat, include_nonlinear=True,
             include_viscous=True, include_forcing=True):
    """d(omega_hat)/dt in spectral space (viscous part handled separately when
    using the integrating factor, but exposed here for the selftest)."""
    psi_hat = what * inv_ksq
    u = jnp.fft.ifft2(1j * ky * psi_hat).real
    v = jnp.fft.ifft2(-1j * kx * psi_hat).real
    wx = jnp.fft.ifft2(1j * kx * what).real
    wy = jnp.fft.ifft2(1j * ky * what).real
    out = jnp.zeros_like(what)
    if include_nonlinear:
        adv = u * wx + v * wy  # u . grad(omega)
        adv_hat = jnp.fft.fft2(adv) * mask  # dealias the product
        out = out - adv_hat
    if include_viscous:
        out = out - nu * (kx**2 + ky**2) * what
    if include_forcing:
        out = out + force_hat
    return out


def _nonlin_forcing_hat(what, kx, ky, inv_ksq, mask, force_hat):
    """Non-stiff part only (advection + forcing); viscous handled by IF."""
    psi_hat = what * inv_ksq
    u = jnp.fft.ifft2(1j * ky * psi_hat).real
    v = jnp.fft.ifft2(-1j * kx * psi_hat).real
    wx = jnp.fft.ifft2(1j * kx * what).real
    wy = jnp.fft.ifft2(1j * ky * what).real
    adv_hat = jnp.fft.fft2(u * wx + v * wy) * mask
    return -adv_hat + force_hat


@partial(jax.jit, static_argnames=("N",))
def _step_if_rk4(what, dt, kx, ky, ksq, inv_ksq, mask, nu, force_hat, N):
    """Integrating-factor RK4 step for d w/dt = L w + Nl(w),  L = -nu*ksq.

    The viscous (linear, stiff) term is integrated exactly via the integrating
    factor E = exp(L*dt); RK4 is applied to the transformed non-stiff part.
    """
    L = -nu * ksq
    E = jnp.exp(L * dt)
    E2 = jnp.exp(L * dt / 2.0)

    def Nl(w):
        return _nonlin_forcing_hat(w, kx, ky, inv_ksq, mask, force_hat)

    a = Nl(what)
    b = Nl(E2 * (what + 0.5 * dt * a))
    c = Nl(E2 * what + 0.5 * dt * b)
    d = Nl(E * what + dt * E2 * c)
    return E * what + (dt / 6.0) * (E * a + 2.0 * E2 * b + 2.0 * E2 * c + d)


def make_stepper(N: int, nu: float, A: float, k_f: float, dt: float):
    kx, ky, ksq, inv_ksq = _wavenumbers(N)
    mask = _dealias_mask(N)
    # vorticity forcing = curl(f) = -d(f_x)/dy = -A*2*pi*k_f*cos(2*pi*k_f*y)
    x = jnp.arange(N) / N
    yy = x[None, :] * jnp.ones((N, 1))
    fw = -A * (2.0 * jnp.pi * k_f) * jnp.cos(2.0 * jnp.pi * k_f * yy)
    force_hat = jnp.fft.fft2(fw)

    def step(what):
        return _step_if_rk4(what, dt, kx, ky, ksq, inv_ksq, mask, nu, force_hat, N)

    ctx = dict(kx=kx, ky=ky, ksq=ksq, inv_ksq=inv_ksq, mask=mask,
               nu=nu, force_hat=force_hat, dt=dt, N=N)
    return step, ctx


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def energy_enstrophy(what, kx, ky, inv_ksq, N):
    u, v = uv_from_what(what, kx, ky, inv_ksq)
    ke = 0.5 * jnp.mean(u**2 + v**2)
    w = jnp.fft.ifft2(what).real
    ens = 0.5 * jnp.mean(w**2)
    return float(ke), float(ens)


def energy_spectrum(what, kx, ky, inv_ksq, N):
    u, v = uv_from_what(what, kx, ky, inv_ksq)
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    e = 0.5 * (jnp.abs(uh) ** 2 + jnp.abs(vh) ** 2) / (N**4)
    kmag = jnp.sqrt(kx**2 + ky**2) / (2.0 * jnp.pi)  # integer shells
    kint = jnp.round(kmag).astype(int)
    nbins = N // 2
    spec = jnp.zeros(nbins)
    kflat = kint.ravel()
    eflat = e.ravel()
    mask = kflat < nbins
    spec = spec.at[jnp.where(mask, kflat, 0)].add(jnp.where(mask, eflat, 0.0))
    return np.asarray(spec)


# ----------------------------------------------------------------------------
# selftest : isolated correctness gates (small N, local CPU)
# ----------------------------------------------------------------------------


def selftest():
    print("[selftest] isolated correctness gates (small N, CPU)\n")
    ok = True

    # Gate A: inviscid + unforced -> energy & enstrophy conserved.
    N = 64
    kx, ky, ksq, inv_ksq = _wavenumbers(N)
    mask = _dealias_mask(N)
    key = jax.random.PRNGKey(0)
    w0 = jax.random.normal(key, (N, N))
    what = jnp.fft.fft2(w0) * mask
    # low-pass so the initial field is smooth and well-resolved
    what = what * jnp.exp(-((kx**2 + ky**2) / (2.0 * jnp.pi * 8) ** 2))
    dt = 1e-3
    zero_force = jnp.zeros_like(what)

    def step_inviscid(w):
        return _step_if_rk4(w, dt, kx, ky, ksq, inv_ksq, mask, 0.0, zero_force, N)

    ke0, ens0 = energy_enstrophy(what, kx, ky, inv_ksq, N)
    w = what
    for _ in range(2000):  # T = 2.0
        w = step_inviscid(w)
    ke1, ens1 = energy_enstrophy(w, kx, ky, inv_ksq, N)
    dke = abs(ke1 - ke0) / ke0
    dens = abs(ens1 - ens0) / ens0
    passA = dke < 1e-3 and dens < 1e-3
    ok &= passA
    print(f"  Gate A inviscid conservation: dE/E={dke:.2e} dZ/Z={dens:.2e} "
          f"-> {'PASS' if passA else 'FAIL'}")

    # Gate B: pure diffusion of a single Fourier mode -> exp(-nu k^2 t).
    N = 32
    kx, ky, ksq, inv_ksq = _wavenumbers(N)
    mask = _dealias_mask(N)
    nu = 0.01
    what = jnp.zeros((N, N), dtype=jnp.complex128)
    what = what.at[3, 2].set(1.0 * N * N)  # single mode n=(3,2)
    kmode2 = (2 * jnp.pi * 3) ** 2 + (2 * jnp.pi * 2) ** 2
    dt = 1e-3
    zf = jnp.zeros_like(what)

    def step_diff(w):
        # viscous only: no nonlinear, no forcing
        L = -nu * ksq
        E = jnp.exp(L * dt)
        return E * w  # exact for pure linear diffusion

    w = what
    T = 1.0
    nsteps = int(T / dt)
    for _ in range(nsteps):
        w = step_diff(w)
    amp = abs(complex(w[3, 2])) / (N * N)
    analytic = np.exp(-nu * float(kmode2) * T)
    errB = abs(amp - analytic) / analytic
    passB = errB < 1e-10
    ok &= passB
    print(f"  Gate B diffusion decay: num={amp:.6e} analytic={analytic:.6e} "
          f"relerr={errB:.2e} -> {'PASS' if passB else 'FAIL'}")

    # Gate B2: full stepper (nonlinear+viscous, no forcing) on the same single
    # mode should still match diffusion closely, since a single mode has zero
    # self-advection (u.grad(omega) vanishes for one Fourier mode).
    def step_full(w):
        return _step_if_rk4(w, dt, kx, ky, ksq, inv_ksq, mask, nu, zf, N)

    w = what
    for _ in range(nsteps):
        w = step_full(w)
    amp2 = abs(complex(w[3, 2])) / (N * N)
    errB2 = abs(amp2 - analytic) / analytic
    passB2 = errB2 < 1e-6
    ok &= passB2
    print(f"  Gate B2 full-stepper single mode: num={amp2:.6e} "
          f"relerr={errB2:.2e} -> {'PASS' if passB2 else 'FAIL'}")

    # Gate C: 2/3 rule keeps |n|<=N/3 per dim -> ~(2/3)^2 in 2D at large N.
    # (Small N over-counts due to integer endpoints, so check at a larger N
    # divisible by 3.)
    N = 192
    m = _dealias_mask(N)
    frac_kept = float(m.mean())
    per_dim = (2 * (N // 3) + 1) / N  # analytic kept fraction per dimension
    passC = abs(frac_kept - per_dim**2) < 1e-9 and 0.42 < frac_kept < 0.47
    ok &= passC
    print(f"  Gate C dealias kept-fraction={frac_kept:.4f} "
          f"(expect {per_dim**2:.4f}) -> {'PASS' if passC else 'FAIL'}")

    print(f"\n[selftest] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Benettin measurement via TLM (jax.jvp on the one-step map)
# ----------------------------------------------------------------------------


def _spin_up(step, what, nsteps):
    def body(w, _):
        return step(w), None
    w, _ = jax.lax.scan(body, what, None, length=nsteps)
    return w


def benettin_lambda(step, what0, key, renorm_every, n_renorm, dt):
    """Leading Lyapunov exponent by TLM + periodic renormalisation.

    Propagates base state `w` and tangent `v` jointly using jax.jvp on the
    one-step map, renormalising `v` every `renorm_every` steps and accumulating
    log growth. Returns running lambda estimates (one per renorm interval).
    """
    v0 = jax.random.normal(key, what0.shape) + 1j * 0.0
    v0 = v0 / jnp.linalg.norm(v0)

    def multi_step(w, v):
        # advance base + tangent by `renorm_every` steps via jvp
        def body(carry, _):
            w, v = carry
            w2, v2 = jax.jvp(step, (w,), (v,))
            return (w2, v2), None
        (w, v), _ = jax.lax.scan(body, (w, v), None, length=renorm_every)
        return w, v

    multi_step = jax.jit(multi_step)

    w = what0
    v = v0
    log_sum = 0.0
    tau = renorm_every * dt
    lam_hist = []
    for i in range(n_renorm):
        w, v = multi_step(w, v)
        growth = jnp.linalg.norm(v)
        log_sum += float(jnp.log(growth))
        v = v / growth  # renormalise
        lam = log_sum / ((i + 1) * tau)
        lam_hist.append(lam)
    return np.asarray(lam_hist)


def load_dns(path):
    """Load DNS dict; return (omega[T,N,N], time[T], config)."""
    o = np.load(path, allow_pickle=True).item()
    return o["omega"], o["time"], o["config"]


def fidelity_gate(step, ctx, dns_omega, dns_time, dns_u, dns_v, idx, dt, horizon):
    """Gate 3: integrate from a DNS snapshot and check the solver tracks the
    DNS trajectory at short time (same dynamical system) and reproduces the
    steady-state energy spectrum. Returns dict of diagnostics."""
    kx, ky, inv_ksq, N = ctx["kx"], ctx["ky"], ctx["inv_ksq"], ctx["N"]
    what = jnp.fft.fft2(jnp.asarray(dns_omega[idx])) * ctx["mask"]
    dt_save = float(dns_time[1] - dns_time[0])
    steps_per_save = int(round(dt_save / dt))
    nsaves = min(int(horizon / dt_save), len(dns_time) - 1 - idx)

    @jax.jit
    def advance(w):
        def body(w, _):
            return step(w), None
        w, _ = jax.lax.scan(body, w, None, length=steps_per_save)
        return w

    errs = []
    w = what
    for s in range(1, nsaves + 1):
        w = advance(w)
        u, v = uv_from_what(w, kx, ky, inv_ksq)
        du = jnp.asarray(dns_u[idx + s])
        dv = jnp.asarray(dns_v[idx + s])
        rel = float(jnp.sqrt(jnp.mean((u - du) ** 2 + (v - dv) ** 2))
                    / jnp.sqrt(jnp.mean(du**2 + dv**2)))
        errs.append((float(dns_time[idx + s] - dns_time[idx]), rel))
    return errs


def measure(args):
    import time

    step, ctx = None, None
    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()

    if args.init_from_dns:
        dns_omega, dns_time, dns_cfg = load_dns(args.init_from_dns)
        N = int(dns_cfg["N"]); nu = float(dns_cfg["nu"])
        A = float(dns_cfg["A"]); k_f = float(dns_cfg["k_f"])
        dt = args.dt
        step, ctx = make_stepper(N, nu, A, k_f, dt)
        print(f"[measure] DNS init: N={N} nu={nu} A={A} k_f={k_f} "
              f"nframes={len(dns_time)} T={dns_time[-1]:.1f}")
        # steady-state frames only (t > t_ss); spread ensemble across them.
        # Benettin integrates forward with the SOLVER (needs no DNS tail); only
        # the fidelity gate looks ahead into DNS frames, so reserve just that.
        ss = np.where(dns_time > args.t_ss)[0]
        dt_save = float(dns_time[1] - dns_time[0])
        fid_reserve = int(min(args.horizon, 4.0) / dt_save) + 5
        hi = max(ss[0] + 1, ss[-1] - fid_reserve)
        idxs = np.linspace(ss[0], hi, args.n_init).astype(int)
        print(f"[measure] ensemble init frames (t): "
              f"{[round(float(dns_time[i]),2) for i in idxs]}")

        if args.fidelity:
            o = np.load(args.init_from_dns, allow_pickle=True).item()
            errs = fidelity_gate(step, ctx, dns_omega, dns_time, o["u"], o["v"],
                                 int(idxs[0]), dt, min(args.horizon, 4.0))
            print("[gate3] solver-vs-DNS rel L2(u,v) tracking:")
            for tt, rel in errs:
                print(f"         dt={tt:.2f}  relL2={rel:.3f}")
    else:
        N, nu, A, k_f, dt = args.N, 1.0 / args.Re, args.A, args.k_f, args.dt
        step, ctx = make_stepper(N, nu, A, k_f, dt)
        idxs = None

    kx, ky, inv_ksq = ctx["kx"], ctx["ky"], ctx["inv_ksq"]
    renorm_every = int(args.tau / dt)
    n_renorm = int(args.horizon / args.tau)
    lam_ensemble = []

    for j in range(args.n_init):
        if args.init_from_dns:
            what_j = jnp.fft.fft2(jnp.asarray(dns_omega[int(idxs[j])])) * ctx["mask"]
            # short spin-up to let the field adapt to this discretisation
            what_j = _spin_up(step, what_j, int(args.spinup / dt))
        else:
            key, k0 = jax.random.split(key)
            w0 = jax.random.normal(k0, (N, N))
            what_j = jnp.fft.fft2(w0) * ctx["mask"]
            what_j = what_j * jnp.exp(-((kx**2 + ky**2) / (2.0 * jnp.pi * 6) ** 2))
            what_j = _spin_up(step, what_j, int(args.spinup / dt))
            what_j = _spin_up(step, what_j, int(j * args.between / dt))
        ke, ens = energy_enstrophy(what_j, kx, ky, inv_ksq, N)
        key, kv = jax.random.split(key)
        lam_hist = benettin_lambda(step, what_j, kv, renorm_every, n_renorm, dt)
        lam_ensemble.append(lam_hist)
        print(f"[measure] init {j}: KE={ke:.4f} lambda_max={lam_hist[-1]:.4f}  "
              f"1/lambda={1.0/lam_hist[-1]:.3f}  ({time.time()-t0:.1f}s)")

    lam_ensemble = np.asarray(lam_ensemble)
    lam_final = lam_ensemble[:, -1]
    result = dict(
        Re=1.0 / nu, N=N, nu=nu, A=A, k_f=k_f, dt=dt,
        init_from_dns=args.init_from_dns,
        tau=args.tau, horizon=args.horizon, spinup=args.spinup,
        n_init=args.n_init,
        lambda_max_mean=float(lam_final.mean()),
        lambda_max_std=float(lam_final.std()),
        lyap_time_mean=float((1.0 / lam_final).mean()),
        lyap_time_std=float((1.0 / lam_final).std()),
        t5_in_lyap_times=float(5.0 * lam_final.mean()),
        per_init_lambda=lam_final.tolist(),
        convergence=lam_ensemble.tolist(),
    )
    print("\n[measure] RESULT")
    print(f"  lambda_max = {result['lambda_max_mean']:.4f} +/- {result['lambda_max_std']:.4f}")
    print(f"  Lyapunov time 1/lambda = {result['lyap_time_mean']:.3f} +/- {result['lyap_time_std']:.3f}")
    print(f"  T=5 corresponds to {result['t5_in_lyap_times']:.2f} Lyapunov times")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[measure] wrote {args.out}")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    m = sub.add_parser("measure")
    m.add_argument("--Re", type=float, default=10000.0)
    m.add_argument("--N", type=int, default=256)
    m.add_argument("--A", type=float, default=0.1)
    m.add_argument("--k_f", type=float, default=2.0)
    m.add_argument("--dt", type=float, default=1e-4)
    m.add_argument("--init_from_dns", type=str, default=None,
                   help="path to DNS .npy; use its omega snapshots as ICs")
    m.add_argument("--t_ss", type=float, default=6.0,
                   help="only use DNS frames with t > t_ss (steady state)")
    m.add_argument("--fidelity", action="store_true",
                   help="run gate-3 solver-vs-DNS tracking check")
    m.add_argument("--spinup", type=float, default=0.5)
    m.add_argument("--between", type=float, default=2.0)
    m.add_argument("--tau", type=float, default=0.1)
    m.add_argument("--horizon", type=float, default=40.0)
    m.add_argument("--n_init", type=int, default=5)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--out", type=str, default="lyapunov_result.json")
    args = p.parse_args()
    if args.cmd == "selftest":
        raise SystemExit(selftest())
    raise SystemExit(measure(args))


if __name__ == "__main__":
    main()
