#!/usr/bin/env python3
"""plot_grid_independence.py — grid-independence convergence figure for the thesis.

Regenerates `figures/results/grid_indep_convergence.{pdf,png}` of
`paper/thesis-format/back/appendix02.tex` from the raw DNS sweep, using the same
recipe as Table~\\ref{tab:dns_grid_indep} in that appendix:

  * reference is N=1024 (not the next-finer grid);
  * KE and enstrophy differences are the **maximum** over the post-spin-up
    window t in [2, 5] s. That maximum falls on the final frame at N=128 and
    N=256 but not at N=512, where it sits at t~4.7; the script reports the
    discrepancy rather than letting the two readings be conflated;
  * pointwise errors compare the **u component alone** at t = 0.5 s, with the
    reference subsampled onto the coarse grid.

Why this file exists: the original analysis script was lost (TD-30). The table
values were re-derived and independently corroborated against the solver's own
stdout (`gi_test_re10000/run_all.out`), but the figure was never regenerated and
kept plotting a retracted enstrophy series. This script is the recorded recipe.

Data lives on the generating host (home-gpu, `~/gi_test_re10000/`); pass
--data-dir to point elsewhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRIDS = (128, 256, 512)
REF_N = 1024
FNAME = ("kolmogorov_dns_fp64_etdrk4_Re10000_N{n}"
         "_T5_dt2p5e4_si100_seed42_icspectral.npy")

SPIN_UP_START = 2.0
K_SENSOR = 5.64   # sqrt(K/pi) at K=100, the sensor sampling band edge
POINTWISE_TIME = 0.5
TOLERANCE_PCT = 2.0

# Okabe--Ito, matching the sibling grid_indep_main figure.
COLORS = {"ke": "#0072B2", "ens": "#D55E00", "u": "#009E73", "omega": "#000000"}
MARKERS = {"ke": "o", "ens": "s", "u": "^", "omega": "D"}
LABELS = {
    "ke": r"$\Delta$KE",
    "ens": r"$\Delta$Enstrophy",
    "u": r"$\|\Delta u\|_2/\|u\|_2$ at $t=0.5$",
    "omega": r"$\|\Delta \omega\|_2/\|\omega\|_2$ at $t=0.5$",
}


def setup_style():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 5,
    })
    import matplotlib.pyplot as plt

    return plt


def scalar_series(u, v, omega):
    """Spatial-mean kinetic energy and enstrophy per stored frame."""
    ke = 0.5 * (u ** 2 + v ** 2).mean(axis=(1, 2))
    ens = 0.5 * (omega ** 2).mean(axis=(1, 2))
    return ke, ens


def band_energy(u, v, kc):
    """Energy in |k| <= kc, summed over both velocity components, per frame."""
    n = u.shape[-1]
    kx = np.fft.fftfreq(n, d=1.0 / n)
    kk = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
    mask = kk <= kc
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (n ** 4)
    return e[:, mask].sum(axis=1)


def rel_l2(field, ref):
    return float(np.linalg.norm(field - ref) / np.linalg.norm(ref))


def load_case(data_dir: Path, n: int):
    d = np.load(data_dir / FNAME.format(n=n), allow_pickle=True).item()
    t = np.asarray(d["time"])
    ke, ens = scalar_series(d["u"], d["v"], d["omega"])
    eband = band_energy(d["u"], d["v"], K_SENSOR)
    i_pw = int(np.argmin(np.abs(t - POINTWISE_TIME)))
    return {
        "t": t, "ke": ke, "ens": ens, "eband": eband,
        "u_pw": np.asarray(d["u"][i_pw]), "omega_pw": np.asarray(d["omega"][i_pw]),
        "t_pw": float(t[i_pw]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "gi_test_re10000")
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    # The generating host carries the data but not matplotlib; the repo checkout
    # carries matplotlib but not the 27 GB of fields. Hence the two-step split.
    ap.add_argument("--dump-json", type=Path, default=None,
                    help="compute only, write the four series to this file")
    ap.add_argument("--from-json", type=Path, default=None,
                    help="plot only, reading values produced by --dump-json")
    args = ap.parse_args()

    if args.from_json is not None:
        rows = {int(k): v for k, v in json.loads(args.from_json.read_text()).items()}
        _report(rows)
        _plot(rows, args.out_dir)
        return

    ref = load_case(args.data_dir, REF_N)
    post = ref["t"] >= SPIN_UP_START
    assert ref["t_pw"] == POINTWISE_TIME, f"pointwise frame is t={ref['t_pw']}"

    rows = {}
    for n in GRIDS:
        c = load_case(args.data_dir, n)
        stride = REF_N // n
        rows[n] = {
            "ke": _scalar_diff(c["ke"], ref["ke"], post, ref["t"], "KE", n),
            "ens": _scalar_diff(c["ens"], ref["ens"], post, ref["t"], "Enstrophy", n),
            "eband": _scalar_diff(c["eband"], ref["eband"], post, ref["t"], "E(k<=ks)", n),
            "u": rel_l2(c["u_pw"], ref["u_pw"][::stride, ::stride]),
            "omega": rel_l2(c["omega_pw"], ref["omega_pw"][::stride, ::stride]),
        }

    _report(rows)
    if args.dump_json is not None:
        args.dump_json.write_text(json.dumps(rows, indent=2))
        print("wrote", args.dump_json)
        return
    _plot(rows, args.out_dir)


def _report(rows):
    print(f"{'N':>6} {'dKE %':>12} {'dEns %':>12} {'dEband %':>12} {'|du| %':>12} {'|dw| %':>12}")
    for n in GRIDS:
        r = rows[n]
        print(f"{n:>6} {r['ke']*100:>12.4g} {r['ens']*100:>12.4g} "
              f"{r['eband']*100:>12.4g} {r['u']*100:>12.4g} {r['omega']*100:>12.4g}")


def _plot(rows, out_dir):
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for key in ("ke", "ens", "u", "omega"):
        ax.plot(GRIDS, [rows[n][key] * 100 for n in GRIDS],
                marker=MARKERS[key], color=COLORS[key], label=LABELS[key])
    ax.axhline(TOLERANCE_PCT, ls="--", lw=0.7, color="0.5")
    ax.axvline(256, ls="--", lw=0.7, color="0.5")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(GRIDS)
    ax.set_xticklabels([str(n) for n in GRIDS])
    ax.set_xlabel(r"Grid resolution $N$ (per side)")
    ax.set_ylabel(r"Relative difference vs $N=1024$ (\%)".replace("\\%", "%"))
    ax.legend(loc="lower left")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"grid_indep_convergence.{ext}"
        fig.savefig(out)
        print("wrote", out)


def _scalar_diff(series, ref_series, post_mask, t, name, n):
    """Largest relative difference over the post-spin-up window.

    The final frame is *not* an equivalent reading: it coincides with the
    maximum at N=128 and N=256 and diverges at N=512. Conflating the two is
    what put a wrong value in the published table, so report when they differ.
    """
    rel = np.abs(series - ref_series) / np.abs(ref_series)
    final = float(rel[post_mask][-1])
    largest = float(rel[post_mask].max())
    if not np.isclose(final, largest, rtol=1e-9):
        t_max = t[post_mask][int(np.argmax(rel[post_mask]))]
        print(f"  note: {name} at N={n}: window maximum {largest*100:.4g}% "
              f"(t={t_max:.2f}) differs from the final frame {final*100:.4g}%")
    return largest


if __name__ == "__main__":
    main()
