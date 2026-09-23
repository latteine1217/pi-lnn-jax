"""TMLR-style figures for the conformal-prediction UQ section.

Two full-width (6.5 in) multi-panel figures + the table (in the paper):

  cp_mechanism.pdf   (a) geometric σ̂ half-width map  (b) physics σ̂ (vorticity-transport
                     residual) map  (c) conditional coverage vs sensor distance (ω)
  cp_calibration.pdf (a) reliability across α-sweep  (b) coverage vs time (Axis A)
                     (c) worst-slab-coverage vs interval-width Pareto (u)

Style via journal_style.setup_style("tmlr"): serif/CM, 8 pt min, vector PDF,
colorblind-safe palette + marker/linestyle per series.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # journal_style next to this file

from journal_style import setup_style, figwidth, STYLE_CYCLE, save_figure  # noqa: E402
from pi_lnn_jax.conformal import (  # noqa: E402
    split_conformal_quantile, normalized_residual, local_difficulty, temporal_split,
    empirical_coverage,
)
from pi_lnn_jax.physics_field import vorticity_transport_residual  # noqa: E402

# Display order + label for every CP method (keys match run_cp_analysis output).
METHODS = [
    ("vanilla", "Vanilla"),
    ("locally_adaptive", r"LA, distance $\hat\sigma$"),
    ("la_physics", r"LA, physics $\hat\sigma$"),
    ("mondrian", "Mondrian, dist."),
    ("mondrian_physics", "Mondrian, phys."),
]


def _distance_halfwidth(d, alpha):
    """Locally-adaptive (distance σ̂) interval half-width field [Nx, Ny]."""
    xx, yy = np.meshgrid(d["x_grid"], d["y_grid"], indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], -1)
    sigma = local_difficulty(d["sensor_pos"], xy).reshape(xx.shape)
    T = d["pred_u"].shape[0]
    cal_t, _ = temporal_split(T)
    sig_b = np.broadcast_to(sigma, d["pred_u"].shape)
    q = split_conformal_quantile(
        normalized_residual(d["pred_u"][cal_t], d["dns_u"][cal_t], sig_b[cal_t]), alpha)
    return xx, yy, q * sigma


def figure_mechanism(npz_path, metrics_json, out, alpha, A, k_f):
    d = np.load(npz_path)
    fig, axes = plt.subplots(1, 3, figsize=(figwidth("tmlr", "single"), 2.15))

    # (a) geometric σ̂ half-width
    xx, yy, hw = _distance_halfwidth(d, alpha)
    im = axes[0].pcolormesh(xx, yy, hw, shading="auto", cmap="viridis", rasterized=True)
    axes[0].scatter(d["sensor_pos"][:, 0], d["sensor_pos"][:, 1], s=3, c="red", lw=0)
    axes[0].set_title(r"(a) geometric $\hat\sigma$ half-width ($u$)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # (b) physics σ̂ map
    nu = 1.0 / float(d["re_value"])
    R = vorticity_transport_residual(d["pred_u"], d["pred_v"], d["dns_t"], nu=nu, A=A, k_f=k_f)
    mid = R.shape[0] // 2
    im = axes[1].pcolormesh(xx, yy, np.log10(R[mid] + 1e-8), shading="auto",
                            cmap="magma", rasterized=True)
    axes[1].scatter(d["sensor_pos"][:, 0], d["sensor_pos"][:, 1], s=3, c="cyan", lw=0)
    axes[1].set_title(r"(b) physics $\hat\sigma$: $\log_{10}|R_\omega|$")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes[:2]:
        ax.set_xlabel(r"$x/L$ (–)"); ax.set_aspect("equal")
    axes[0].set_ylabel(r"$y/L$ (–)")

    # (c) conditional coverage vs sensor distance (ω channel)
    m = json.loads(Path(metrics_json).read_text())["per_seed"][0]["omega"]
    for i, (key, lab) in enumerate(METHODS):
        c, mk, ls = STYLE_CYCLE[i]
        sc = m[key]["slab_coverage"]
        axes[2].plot(sc["feature_mid"], sc["coverage"], color=c, marker=mk, linestyle=ls,
                     ms=3, label=lab)
    axes[2].axhline(1 - alpha, color="k", ls=":", lw=0.8)
    axes[2].text(0.02, 1 - alpha + 0.01, r"nominal $1-\alpha$", fontsize=6,
                 transform=axes[2].get_yaxis_transform())
    axes[2].set_xlabel(r"sensor distance $d$ (–)")
    axes[2].set_ylabel(r"conditional coverage (–)")
    axes[2].set_title(r"(c) coverage vs distance ($\omega$)")
    axes[2].legend(fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out); plt.close(fig)
    print(f"[fig] {out}")


def figure_calibration(npz_path, metrics_json, sweep_json, out, alpha):
    d = np.load(npz_path)
    m = json.loads(Path(metrics_json).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(figwidth("tmlr", "single"), 2.15))

    # (a) reliability (u channel) across α-sweep
    s = json.loads(Path(sweep_json).read_text())
    alphas = sorted(float(a) for a in s if not a.startswith("_"))  # skip _acd meta
    nominal = [1 - a for a in alphas]
    axes[0].plot([min(nominal), 1.0], [min(nominal), 1.0], "k:", lw=0.8)
    for i, (key, lab) in enumerate(METHODS):
        c, mk, ls = STYLE_CYCLE[i]
        emp = [s[f"{a}"][key]["u"]["marginal_coverage"] for a in alphas]
        axes[0].plot(nominal, emp, color=c, marker=mk, linestyle=ls, ms=3, label=lab)
    axes[0].set_xlabel(r"nominal $1-\alpha$ (–)")
    axes[0].set_ylabel(r"empirical coverage (–)")
    axes[0].set_title(r"(a) reliability ($u$)")
    axes[0].legend(fontsize=5.5, loc="upper left")

    # (b) coverage vs time (Axis A; locally-adaptive distance σ̂, u)
    xx, yy, hw = _distance_halfwidth(d, alpha)
    pred, dns = d["pred_u"], d["dns_u"]
    cov = [empirical_coverage(pred[t] - hw, pred[t] + hw, dns[t]) for t in range(pred.shape[0])]
    axes[1].plot(d["dns_t"], cov, color=STYLE_CYCLE[1][0], lw=1.0)
    axes[1].axhline(1 - alpha, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel(r"time $t$ (s)")
    axes[1].set_ylabel(r"coverage (–)")
    axes[1].set_title(r"(b) coverage vs time ($u$)")

    # (c) WSC–width Pareto (u channel); ideal = top-left
    for i, (key, lab) in enumerate(METHODS):
        c, mk, _ = STYLE_CYCLE[i]
        w = m[key]["u"]["mean_width"]; wsc = m[key]["u"]["worst_slab_coverage"]
        axes[2].scatter(w, wsc, color=c, marker=mk, s=24, label=lab)
    axes[2].axhline(1 - alpha, color="k", ls=":", lw=0.8)
    axes[2].set_xlabel(r"mean interval width (m/s)")
    axes[2].set_ylabel(r"worst-slab coverage (–)")
    axes[2].set_title(r"(c) coverage–width ($u$)")
    axes[2].legend(fontsize=5.5, loc="lower right")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out); plt.close(fig)
    print(f"[fig] {out}")


def figure_acd(sweep_json, out):
    """ACD (average coverage deviation) per method, global vs worst-slab (u channel).

    A single-scalar calibration summary across the α grid (lower is better); the
    worst-slab ACD exposes conditional miscalibration the global ACD hides.
    """
    acd = json.loads(Path(sweep_json).read_text())["_acd"]
    labels = [lab for _, lab in METHODS]
    glob = [acd[k]["u"]["acd_marginal"] for k, _ in METHODS]
    wsc = [acd[k]["u"]["acd_worst_slab"] for k, _ in METHODS]
    xs = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(figwidth("tmlr", "half") * 1.4, 2.3))
    ax.bar(xs - 0.2, glob, width=0.38, color=STYLE_CYCLE[0][0], label="global")
    ax.bar(xs + 0.2, wsc, width=0.38, color=STYLE_CYCLE[1][0], label="worst-slab")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("ACD (–)")
    ax.set_title(r"Average coverage deviation ($u$)")
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out); plt.close(fig)
    print(f"[fig] {out}")


def figure_local_cp_strat(pointwise_json, alpha, out):
    """Conditional coverage across sensor-distance quartiles for the pointwise methods,
    Path A vs Path B — visualizes that learned local CP stays flattest at nominal."""
    pw = json.loads(Path(pointwise_json).read_text())
    methods = [("fixed", "fixed"), ("adaptive_distance", r"adaptive, distance $\hat\sigma$"),
               ("adaptive_physics", r"adaptive, physics $\hat\sigma$"),
               ("local_cp", "learned local CP")]
    quart = ["Q1\n(near)", "Q2", "Q3", "Q4\n(far)"]
    xs = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(figwidth("tmlr", "single"), 2.4), sharey=True)
    for ax, path, title in [(axes[0], "path_A", "(a) Path A (transferable)"),
                            (axes[1], "path_B", "(b) Path B (oracle)")]:
        ax.axhline(1 - alpha, color="k", ls=":", lw=0.8)
        for i, (key, lab) in enumerate(methods):
            c, mk, ls = STYLE_CYCLE[i]
            ax.plot(xs, pw[path]["u"][key]["stratified_coverage"],
                    color=c, marker=mk, linestyle=ls, ms=4, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(quart, fontsize=6)
        ax.set_xlabel("sensor-distance quartile")
        ax.set_title(title)
    axes[0].set_ylabel(r"conditional coverage ($u$, –)")
    axes[1].legend(fontsize=5.5, loc="lower center")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out); plt.close(fig)
    print(f"[fig] {out}")


def figure_error_vs_time(npz_path, out):
    """Mechanistic diagnostic: reconstruction error vs time vs the stationary true RMS.

    Error is high early and decays as the CfC temporal context accumulates, while the
    true field RMS stays flat — isolating model context build-up (not a flow transient)
    as the cause of the early-time undercoverage seen in coverage-vs-time.
    """
    d = np.load(npz_path)
    t = d["dns_t"]
    eu = np.abs(d["pred_u"] - d["dns_u"]).mean(axis=(1, 2))
    ev = np.abs(d["pred_v"] - d["dns_v"]).mean(axis=(1, 2))
    rms_u = np.sqrt((d["dns_u"] ** 2).mean(axis=(1, 2)))
    fig, ax = plt.subplots(1, 2, figsize=(figwidth("tmlr", "single"), 2.3))
    c0, c1 = STYLE_CYCLE[0][0], STYLE_CYCLE[1][0]
    ax[0].plot(t, eu, color=c0, lw=1.1, label=r"mean $|e_u|$")
    ax[0].plot(t, ev, color=c1, lw=1.1, ls="--", label=r"mean $|e_v|$")
    ax[0].plot(t, rms_u, color="k", lw=0.9, ls=":", label=r"true $u_{\rm rms}$ (stationary)")
    ax[0].set_xlabel(r"time $t$ (s)"); ax[0].set_ylabel("magnitude (m/s)")
    ax[0].set_title("(a) error vs time")
    ax[0].legend(fontsize=6)
    ax[1].plot(t, 100.0 * eu / np.maximum(rms_u, 1e-12), color=STYLE_CYCLE[1][0], lw=1.1)
    ax[1].set_xlabel(r"time $t$ (s)"); ax[1].set_ylabel(r"relative error $u$ (\%)")
    ax[1].set_title("(b) error decays as CfC context accumulates")
    fig.tight_layout(pad=0.4)
    save_figure(fig, out); plt.close(fig)
    print(f"[fig] {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--metrics-json", required=True)
    p.add_argument("--sweep-json", required=True)
    p.add_argument("--pointwise-json", default=None,
                   help="cp_pointwise.json → enables the local-CP stratified-coverage figure")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--forcing-A", type=float, default=0.1)
    p.add_argument("--forcing-kf", type=float, default=2.0)
    p.add_argument("--outdir", default="paper/tmlr-format/figures")
    args = p.parse_args()
    setup_style("tmlr")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figure_mechanism(args.npz, args.metrics_json, str(outdir / "cp_mechanism"),
                     args.alpha, args.forcing_A, args.forcing_kf)
    figure_calibration(args.npz, args.metrics_json, args.sweep_json,
                       str(outdir / "cp_calibration"), args.alpha)
    figure_error_vs_time(args.npz, str(outdir / "cp_error_vs_time"))
    figure_acd(args.sweep_json, str(outdir / "cp_acd"))
    if args.pointwise_json:
        figure_local_cp_strat(args.pointwise_json, args.alpha, str(outdir / "cp_local_cp_strat"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
