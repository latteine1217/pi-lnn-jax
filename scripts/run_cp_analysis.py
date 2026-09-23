"""Offline conformal-prediction analysis on cached reconstruction npz.

Computes, per channel (u, v, ω) and per CP method (vanilla / locally-adaptive with
sensor-distance σ̂ / locally-adaptive with physics σ̂ / Mondrian by distance / Mondrian
by physics residual), the marginal coverage, conditional coverage, worst-slab coverage
(both stratified by sensor distance), and mean interval width — over a temporal
calibration/test split. The physics σ̂ is the pressure-free vorticity-transport residual
(see pi_lnn_jax.physics_field), testing whether a physics-aware difficulty improves
conditional coverage over a purely geometric one.

Axis A (temporal split) is the reported-with-guarantee path; axis-B group-conditional
numbers are diagnostics. Aggregates across all seed npz passed in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.conformal import (  # noqa: E402
    split_conformal_quantile, normalized_residual, local_difficulty,
    mondrian_quantiles, empirical_coverage, interval_width,
    conditional_coverage, worst_slab_coverage, slab_coverage, temporal_split,
    average_coverage_deviation,
)
from pi_lnn_jax.metric_artifact import compute_vorticity  # noqa: E402
from pi_lnn_jax.physics_field import vorticity_transport_residual  # noqa: E402
from pi_lnn_jax.local_cp import fit_local_quantile, predict_scale, local_cp_intervals  # noqa: E402


def _vorticity_stack(u, v):
    """ω per time slice → [T, Nx, Ny]."""
    return np.stack([compute_vorticity(u[t], v[t]) for t in range(u.shape[0])])


def _sensor_distance_field(sensor_pos, x_grid, y_grid, domain_length=1.0):
    """Per-pixel periodic distance-to-nearest-sensor field [Nx, Ny]."""
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], -1)
    return local_difficulty(sensor_pos, xy, domain_length=domain_length).reshape(
        len(x_grid), len(y_grid))


def _groups_from_field(field, n_groups=4):
    """Quantile-bin a difficulty field into n_groups integer group ids (same shape)."""
    edges = np.quantile(field, np.linspace(0, 1, n_groups + 1)[1:-1])
    return np.digitize(field, edges)


def _la_interval(pc, dc, sc, pt, st, alpha):
    """Locally-adaptive split-conformal interval (lo, hi) for the test slices.

    sc/st are the calibration/test difficulty σ̂ (any shape matching pc/pt). The +1e-12
    floor matches normalized_residual's calibration so σ=0 pixels do not collapse to
    zero-width intervals.
    """
    q = split_conformal_quantile(normalized_residual(pc, dc, sc), alpha)
    st_floored = st + 1e-12
    return pt - q * st_floored, pt + q * st_floored


def _analyze_channel(pred, dns, dist_sigma, dist_groups, phys_sigma, phys_groups, alpha):
    """Compute all CP variants for one channel.

    pred/dns/phys_sigma/phys_groups: [T, Nx, Ny]; dist_sigma/dist_groups: [Nx, Ny].
    Worst-slab coverage AND conditional coverage are stratified by SENSOR DISTANCE for
    every method, so the methods are directly comparable on the same conditional axis
    (the question: does a physics-aware σ̂ improve the worst sensor-distance slab?).
    """
    T = pred.shape[0]
    cal_t, test_t = temporal_split(T, scheme="interleaved")
    dsig = np.broadcast_to(dist_sigma, pred.shape)
    dgrp = np.broadcast_to(dist_groups, pred.shape)

    pc, dc = pred[cal_t], dns[cal_t]
    pt, dt = pred[test_t], dns[test_t]
    dsc, dst = dsig[cal_t], dsig[test_t]          # distance σ̂ cal/test
    dgt = dgrp[test_t]                             # distance groups (WSC/cond axis = test)
    psc, pst = phys_sigma[cal_t], phys_sigma[test_t]   # physics σ̂ cal/test
    pgc, pgt = phys_groups[cal_t], phys_groups[test_t]  # physics groups cal/test
    res = {}

    # vanilla split conformal (σ = 1)
    q = split_conformal_quantile(normalized_residual(pc, dc), alpha)
    res["vanilla"] = _metrics(pt - q, pt + q, dt, dgt, dst)

    # locally-adaptive: sensor-distance σ̂
    lo, hi = _la_interval(pc, dc, dsc, pt, dst, alpha)
    res["locally_adaptive"] = _metrics(lo, hi, dt, dgt, dst)

    # locally-adaptive: physics (vorticity-transport residual) σ̂
    lo, hi = _la_interval(pc, dc, psc, pt, pst, alpha)
    res["la_physics"] = _metrics(lo, hi, dt, dgt, dst)

    # mondrian binned by sensor distance
    qg = mondrian_quantiles(normalized_residual(pc, dc), dgrp[cal_t], alpha)
    qmap = np.vectorize(lambda g: qg.get(int(g), np.inf))(dgt)
    res["mondrian"] = _metrics(pt - qmap, pt + qmap, dt, dgt, dst)

    # mondrian binned by physics residual
    qg = mondrian_quantiles(normalized_residual(pc, dc), pgc, alpha)
    qmap = np.vectorize(lambda g: qg.get(int(g), np.inf))(pgt)
    res["mondrian_physics"] = _metrics(pt - qmap, pt + qmap, dt, dgt, dst)
    return res


def _metrics(lo, hi, truth, groups, difficulty):
    # difficulty here is always the sensor-distance field → WSC and the per-slab
    # conditional-coverage curve are stratified by sensor distance for every method.
    return {
        "marginal_coverage": empirical_coverage(lo, hi, truth),
        "mean_width": interval_width(lo, hi),
        "conditional_coverage": {str(k): v for k, v in
                                 conditional_coverage(lo, hi, truth, groups).items()},
        "worst_slab_coverage": worst_slab_coverage(
            lo.ravel(), hi.ravel(), truth.ravel(), difficulty.ravel(), n_slabs=10),
        "slab_coverage": slab_coverage(
            lo.ravel(), hi.ravel(), truth.ravel(), difficulty.ravel(), n_slabs=8),
    }


def _analyze_all_seeds(npz_paths, alpha, A, k_f):
    """Run the per-seed CP analysis at a single α → list of per-seed result dicts.

    Each entry: {"seed": int, "u": {...}, "v": {...}, "omega": {...}}. Shared by the
    single-α path and the α-sweep path (DRY). A/k_f are the Kolmogorov forcing params for
    the physics (vorticity-transport) σ̂; ν is taken from each npz's re_value (ν=1/Re).
    """
    per_seed = []
    for path in npz_paths:
        d = np.load(path)
        dist_sigma = _sensor_distance_field(d["sensor_pos"], d["x_grid"], d["y_grid"])
        dist_groups = _groups_from_field(dist_sigma)
        nu = 1.0 / float(d["re_value"])
        phys_sigma = vorticity_transport_residual(
            d["pred_u"], d["pred_v"], d["dns_t"], nu=nu, A=A, k_f=k_f)  # [T,Nx,Ny]
        phys_groups = _groups_from_field(phys_sigma)
        omega_pred = _vorticity_stack(d["pred_u"], d["pred_v"])
        omega_dns = _vorticity_stack(d["dns_u"], d["dns_v"])
        args_common = (dist_sigma, dist_groups, phys_sigma, phys_groups, alpha)
        seed_res = {
            "u": _analyze_channel(d["pred_u"], d["dns_u"], *args_common),
            "v": _analyze_channel(d["pred_v"], d["dns_v"], *args_common),
            "omega": _analyze_channel(omega_pred, omega_dns, *args_common),
        }
        per_seed.append({"seed": int(d["seed"]), **seed_res})
    return per_seed


# ── Pointwise i.i.d. multi-draw conformal + Path A/B (aligns with pi-lnn) ──────
def _sensor_grid_cells(sensor_pos, x_grid, y_grid) -> set:
    """Flat pixel ids (i*Ny+j) of the grid cells nearest each training sensor."""
    Nx, Ny = len(x_grid), len(y_grid)
    dx, dy = float(x_grid[1] - x_grid[0]), float(y_grid[1] - y_grid[0])
    xi = (np.round(sensor_pos[:, 0] / dx).astype(int)) % Nx
    yj = (np.round(sensor_pos[:, 1] / dy).astype(int)) % Ny
    return set((xi * Ny + yj).tolist())


def _pointwise_conformal(pred, dns, dist_field, phys_sigma, alpha,
                         n_cal, n_test, n_draws, rng, exclude_cells=None):
    """i.i.d. (time, pixel) multi-draw split conformal → coverage mean ± std.

    Repeats n_draws random calibration/test partitions of i.i.d.-sampled (t, pixel)
    points and reports mean ± std coverage, mean half-width, and distance-stratified
    conditional coverage. exclude_cells drops grid cells from the pool (Path A excludes
    the training-sensor cells so calibration uses only held-out locations).

    pred/dns/phys_sigma: [T, Nx, Ny]; dist_field: [Nx, Ny].
    """
    T, Nx, Ny = pred.shape
    npix = Nx * Ny
    pix_ok = np.ones(npix, dtype=bool)
    if exclude_cells:
        pix_ok[list(exclude_cells)] = False
    ok_pix = np.flatnonzero(pix_ok)

    err_flat = np.abs(pred - dns).reshape(T, npix)
    phys_flat = phys_sigma.reshape(T, npix)
    dist_flat = dist_field.reshape(npix)
    methods = ("fixed", "adaptive_distance", "adaptive_physics")
    acc = {m: {"cov": [], "hw": [], "strat": []} for m in methods}
    n_pool = n_cal + n_test

    for _ in range(n_draws):
        t_idx = rng.integers(0, T, size=n_pool)
        p_idx = ok_pix[rng.integers(0, ok_pix.size, size=n_pool)]
        e = err_flat[t_idx, p_idx]
        d = dist_flat[p_idx]
        ph = phys_flat[t_idx, p_idx]
        ec, et = e[:n_cal], e[n_cal:]
        dt = d[n_cal:]
        edges = np.quantile(dt, [0, 0.25, 0.5, 0.75, 1.0])
        masks = [(dt >= lo) & (dt <= hi) for lo, hi in zip(edges[:-1], edges[1:])]

        def _rec(m, cov_pt):
            acc[m]["cov"].append(float(cov_pt.mean()))
            acc[m]["strat"].append(
                [float(cov_pt[mk].mean()) if mk.any() else np.nan for mk in masks])

        qf = split_conformal_quantile(ec, alpha)
        acc["fixed"]["hw"].append(float(qf))
        _rec("fixed", et <= qf)
        for m, sig in (("adaptive_distance", d), ("adaptive_physics", ph)):
            sc, st_ = sig[:n_cal] + 1e-12, sig[n_cal:] + 1e-12
            q = split_conformal_quantile(ec / sc, alpha)
            acc[m]["hw"].append(float(np.mean(q * st_)))
            _rec(m, et <= q * st_)

    out = {}
    for m in methods:
        strat = np.nanmean(np.array(acc[m]["strat"]), axis=0)
        out[m] = {
            "coverage_mean": float(np.mean(acc[m]["cov"])),
            "coverage_std": float(np.std(acc[m]["cov"])),
            "mean_halfwidth": float(np.mean(acc[m]["hw"])),
            "stratified_coverage": strat.tolist(),
            "coverage_spread": float(np.nanmax(strat) - np.nanmin(strat)),
        }
    return out


def _pixel_xy(x_grid, y_grid):
    """Per-pixel (x, y) coordinates, flat over i*Ny+j → [npix, 2]."""
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def _local_cp_pointwise(pred, dns, dist_field, phys_sigma, pix_xy, dns_t, alpha,
                        n_cal, n_test, n_draws, rng, allowed_pix, backend="gbr"):
    """Learned local-CP (spatial CQR) with a leakage-safe three-way split.

    Pixels in `allowed_pix` are split disjointly into a g_phi-training half and an
    evaluation half (so the learned quantile never sees a calibration/test location).
    g_phi is fit once on i.i.d. (time, g-pixel) points; the multi-draw loop then samples
    calibration/test points from the eval pixels and conformalizes ell* per draw.
    """
    T = pred.shape[0]
    npix = pred.shape[1] * pred.shape[2]
    err_flat = np.abs(pred - dns).reshape(T, npix)
    pred_flat = pred.reshape(T, npix)
    dns_flat = dns.reshape(T, npix)
    phys_flat = phys_sigma.reshape(T, npix)
    dist_flat = dist_field.reshape(npix)

    perm = rng.permutation(allowed_pix)
    g_pix, eval_pix = np.array_split(perm, 2)

    def _features(t_idx, p_idx):
        return np.column_stack([pix_xy[p_idx, 0], pix_xy[p_idx, 1],
                                dns_t[t_idx], dist_flat[p_idx]])

    # fit g_phi once on the g-pixel pool (normalized residual = err / physics σ̂)
    n_g = n_cal + n_test
    tg = rng.integers(0, T, size=n_g)
    pg = g_pix[rng.integers(0, g_pix.size, size=n_g)]
    scores_g = err_flat[tg, pg] / (phys_flat[tg, pg] + 1e-12)
    model = fit_local_quantile(_features(tg, pg), scores_g, alpha, backend=backend)

    acc = {"cov": [], "hw": [], "strat": []}
    for _ in range(n_draws):
        t_idx = rng.integers(0, T, size=n_g)
        p_idx = eval_pix[rng.integers(0, eval_pix.size, size=n_g)]
        g = predict_scale(model, _features(t_idx, p_idx))
        sig = phys_flat[t_idx, p_idx]
        ci, ti_ = slice(0, n_cal), slice(n_cal, n_g)
        lo, hi, _ = local_cp_intervals(
            pred_test=pred_flat[t_idx, p_idx][ti_], g_test=g[ti_], sigma_test=sig[ti_],
            resid_cal=err_flat[t_idx, p_idx][ci], g_cal=g[ci], sigma_cal=sig[ci], alpha=alpha)
        truth = dns_flat[t_idx, p_idx][ti_]
        dt = dist_flat[p_idx][ti_]
        cov_pt = (truth >= lo) & (truth <= hi)
        acc["cov"].append(float(cov_pt.mean()))
        acc["hw"].append(float(np.mean(hi - lo) / 2.0))
        edges = np.quantile(dt, [0, 0.25, 0.5, 0.75, 1.0])
        acc["strat"].append([float(cov_pt[(dt >= lo_) & (dt <= hi_)].mean())
                             if ((dt >= lo_) & (dt <= hi_)).any() else np.nan
                             for lo_, hi_ in zip(edges[:-1], edges[1:])])
    strat = np.nanmean(np.array(acc["strat"]), axis=0)
    return {
        "coverage_mean": float(np.mean(acc["cov"])),
        "coverage_std": float(np.std(acc["cov"])),
        "mean_halfwidth": float(np.mean(acc["hw"])),
        "stratified_coverage": strat.tolist(),
        "coverage_spread": float(np.nanmax(strat) - np.nanmin(strat)),
    }


def run_pointwise_paths(npz_path, alpha, A, k_f, n_cal, n_test, n_draws, seed,
                        local_cp=True, backend="gbr"):
    """Path A (transferable: held-out non-sensor locations) and Path B (oracle: full
    field), each with i.i.d. multi-draw conformal, for u/v/ω. Adds learned local-CP
    when local_cp=True (requires scikit-learn)."""
    d = np.load(npz_path)
    rng = np.random.default_rng(seed)
    dist_field = _sensor_distance_field(d["sensor_pos"], d["x_grid"], d["y_grid"])
    pix_xy = _pixel_xy(d["x_grid"], d["y_grid"])
    nu = 1.0 / float(d["re_value"])
    phys = vorticity_transport_residual(
        d["pred_u"], d["pred_v"], d["dns_t"], nu=nu, A=A, k_f=k_f)
    omega_pred = _vorticity_stack(d["pred_u"], d["pred_v"])
    omega_dns = _vorticity_stack(d["dns_u"], d["dns_v"])
    fields = {"u": (d["pred_u"], d["dns_u"]), "v": (d["pred_v"], d["dns_v"]),
              "omega": (omega_pred, omega_dns)}
    sensor_cells = _sensor_grid_cells(d["sensor_pos"], d["x_grid"], d["y_grid"])
    npix = len(d["x_grid"]) * len(d["y_grid"])

    def _one(exclude):
        allowed = np.setdiff1d(np.arange(npix), np.array(sorted(exclude), dtype=int)) \
            if exclude else np.arange(npix)
        out = {}
        for ch, (p, t) in fields.items():
            res = _pointwise_conformal(p, t, dist_field, phys, alpha,
                                       n_cal, n_test, n_draws, rng, exclude_cells=exclude)
            if local_cp:
                res["local_cp"] = _local_cp_pointwise(
                    p, t, dist_field, phys, pix_xy, d["dns_t"], alpha,
                    n_cal, n_test, n_draws, rng, allowed, backend=backend)
            out[ch] = res
        return out

    return {
        "_meta": {"alpha": alpha, "n_cal": n_cal, "n_test": n_test, "n_draws": n_draws,
                  "local_cp_backend": backend if local_cp else None,
                  "path_A": "held-out non-sensor locations (engineering-transferable)",
                  "path_B": "full-field DNS calibration (research oracle)"},
        "path_A": _one(sensor_cells),
        "path_B": _one(None),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", required=True, help="one or more seed npz")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--pointwise-draws", type=int, default=50,
                   help="multi-draw count for the pointwise Path A/B analysis (0 to skip)")
    p.add_argument("--n-cal", type=int, default=200)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pointwise-out", default=None,
                   help="pointwise Path A/B JSON (defaults alongside --out)")
    p.add_argument("--alpha-sweep", default=None,
                   help="comma-separated alphas, e.g. '0.05,0.1,0.15,0.2'; enables sweep output")
    p.add_argument("--sweep-out", default=None,
                   help="sweep JSON output path (defaults alongside --out)")
    p.add_argument("--forcing-A", type=float, default=0.1,
                   help="Kolmogorov forcing amplitude A (physics σ̂); EXP-245 default 0.1")
    p.add_argument("--forcing-kf", type=float, default=2.0,
                   help="Kolmogorov forcing wavenumber k_f (physics σ̂); EXP-245 default 2.0")
    p.add_argument("--no-local-cp", action="store_true",
                   help="skip the learned local-CP method (avoids the scikit-learn dependency)")
    p.add_argument("--local-cp-backend", default="gbr", choices=("gbr", "linear"),
                   help="conditional-quantile regressor for local CP")
    args = p.parse_args()

    methods = ("vanilla", "locally_adaptive", "la_physics", "mondrian", "mondrian_physics")
    channels = ("u", "v", "omega")

    def _agg(per_seed, channel, method, key):
        vals = [s[channel][method][key] for s in per_seed]
        return float(np.mean(vals))

    # single-α path (unchanged output shape/behavior)
    per_seed = _analyze_all_seeds(args.npz, args.alpha, args.forcing_A, args.forcing_kf)
    summary = {"alpha": args.alpha, "n_seeds": len(per_seed), "per_seed": per_seed}
    for ch in channels:
        for me in methods:
            summary.setdefault(me, {})[ch] = {
                "marginal_coverage": _agg(per_seed, ch, me, "marginal_coverage"),
                "mean_width": _agg(per_seed, ch, me, "mean_width"),
                "worst_slab_coverage": _agg(per_seed, ch, me, "worst_slab_coverage"),
            }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[out] {args.out}")
    for me in methods:
        for ch in channels:
            s = summary[me][ch]
            print(f"  {me:>16s} {ch:>5s}: cov={s['marginal_coverage']:.3f} "
                  f"width={s['mean_width']:.4f}")

    # α-sweep path: nested-by-α structure for plot_cp.py::reliability + ACD.
    if args.alpha_sweep:
        alphas = [float(a) for a in args.alpha_sweep.split(",") if a.strip()]
        sweep: dict = {}
        for a in alphas:
            ps = _analyze_all_seeds(args.npz, a, args.forcing_A, args.forcing_kf)
            sweep[str(a)] = {
                me: {ch: {"marginal_coverage": _agg(ps, ch, me, "marginal_coverage"),
                          "worst_slab_coverage": _agg(ps, ch, me, "worst_slab_coverage")}
                     for ch in channels}
                for me in methods
            }
        # ACD: average coverage deviation over the α grid (global + worst-slab).
        acd: dict = {}
        for me in methods:
            acd[me] = {}
            for ch in channels:
                cov_m = [sweep[str(a)][me][ch]["marginal_coverage"] for a in alphas]
                cov_w = [sweep[str(a)][me][ch]["worst_slab_coverage"] for a in alphas]
                acd[me][ch] = {
                    "acd_marginal": average_coverage_deviation(cov_m, alphas),
                    "acd_worst_slab": average_coverage_deviation(cov_w, alphas),
                }
        sweep["_acd"] = acd
        sweep_out = args.sweep_out or str(Path(args.out).with_name("cp_metrics_sweep.json"))
        Path(sweep_out).write_text(json.dumps(sweep, indent=2))
        print(f"[sweep-out] {sweep_out}")
        print("  ACD (marginal / worst-slab), u channel:")
        for me in methods:
            print(f"    {me:>16s}: {acd[me]['u']['acd_marginal']:.4f} / "
                  f"{acd[me]['u']['acd_worst_slab']:.4f}")

    # Pointwise i.i.d. multi-draw conformal + Path A/B (uses the first npz only).
    if args.pointwise_draws > 0:
        pw = run_pointwise_paths(args.npz[0], args.alpha, args.forcing_A, args.forcing_kf,
                                 args.n_cal, args.n_test, args.pointwise_draws, args.seed,
                                 local_cp=not args.no_local_cp, backend=args.local_cp_backend)
        pw_out = args.pointwise_out or str(Path(args.out).with_name("cp_pointwise.json"))
        Path(pw_out).write_text(json.dumps(pw, indent=2))
        print(f"[pointwise-out] {pw_out}")
        pw_methods = ["fixed", "adaptive_distance", "adaptive_physics"]
        if not args.no_local_cp:
            pw_methods.append("local_cp")
        for path in ("path_A", "path_B"):
            print(f"  --- {path} (α={args.alpha}, {args.pointwise_draws} draws) ---")
            for me in pw_methods:
                s = pw[path]["u"][me]
                print(f"    {me:>18s} u: cov={s['coverage_mean']:.3f}±{s['coverage_std']:.3f} "
                      f"hw={s['mean_halfwidth']:.4f} spread={s['coverage_spread']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
