#!/usr/bin/env python3
"""aggregate_published_band_stats.py — recompute the per-seed spectral summaries
quoted in the thesis text but never tabulated.

Three published mean ± sd values had no producer in the repo and were recorded as
unreproducible (`knowledge/codebase/architecture-deepening.md`). The underlying
per-frame series were in `artifacts/` all along; only the seed-level aggregation
was missing. This script is that aggregation.

  * `gamma_high`         spectral coherence in the high band, Eq. (3.x)
  * `low_band_rel_err`   integrated low-band energy error, sensor-anchored bands
                         (NOT `band_rel_err_low`, which is the k_eta-fraction
                         shell error plotted in the per-band figure)
  * `enstrophy_rel_err`  |Z_pred - Z_dns| / Z_dns, spatial-mean enstrophy

Two things the recomputation settled, both of which had drifted in the text:

  * gamma is averaged over t >= 0.5 s, as the text states. The K=200 seed set is
    pinned independently by uv_rel_err (8.521 +/- 0.058 against a published
    8.52 +/- 0.06; sets including ksweep_K200_b3 give 0.054 and round to 0.05),
    and on that set t >= 0.5 gives 0.1086 +/- 0.0093 -- the published
    0.109 +/- 0.009 to the digit. No other window reproduces both.
  * the published enstrophy deficit of 15.7 +/- 0.2 % is reproduced by no window
    (t >= 0.5 gives 15.8, t >= 1 gives 15.3). The text quotes it on t >= 1, where
    the artifacts give 15.31 +/- 0.21 % -- the value already tabulated for the
    same configuration elsewhere in the chapter.
  * low_band_rel_err reproduces its published mean on the full window
    (1.119 against 1.12) at a spread of 0.07 against a published 0.08.

The K=100 seed set is NOT identified: main5_s{1..4,42} gives uv_rel_err
15.032 +/- 0.082 against a published 15.04 +/- 0.06, so the K=100 figures below come
from a neighbouring set and are not evidence about the published K=100 value.

Runs are read from `--artifacts-root`; seeds whose `final_eval/metrics.json` is
absent fall back to `eval_figs/series.npz`, which carries the same series.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

KEYS = ("gamma_high", "low_band_rel_err", "enstrophy_rel_err")

# (label, run globs, key -> t_min). t_min follows the window each quantity is
# quoted on: the scalar enstrophy deficit is post-spin-up, the spectral ones run
# over the whole window.
# Seed sets follow the run naming (s1--s4 plus s42); `ksweep_K200_b3` is the
# n=3 K-sweep series arm, not part of the five-seed headline set, and is excluded.
ARMS = {
    "K=100 (main, n=5)": ["main5_s1", "main5_s2", "main5_s3", "main5_s4", "main5_s42"],
    "K=200 (n=5)": ["ksweep_k200_s1", "ksweep_k200_s2", "ksweep_k200_s3",
                    "ksweep_k200_s4", "ksweep_k200_s42"],
}
T_MIN = {"gamma_high": 0.5, "low_band_rel_err": 0.0, "enstrophy_rel_err": 1.0}
AS_PERCENT = {"low_band_rel_err", "enstrophy_rel_err"}


def load_series(run: Path) -> dict[str, np.ndarray] | None:
    """Per-frame series for one run, from whichever artifact the run carries."""
    js = run / "final_eval" / "metrics.json"
    if js.exists():
        rows = json.loads(js.read_text())["metrics_per_t"]
        return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                for k in ("t", *KEYS)}
    npz = run / "eval_figs" / "series.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=True) as d:
            if not all(k in d for k in ("t", *KEYS)):
                return None
            return {k: np.asarray(d[k], dtype=float) for k in ("t", *KEYS)}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", type=Path,
                    default=Path("artifacts/kolmogorov"))
    args = ap.parse_args()

    for label, globs in ARMS.items():
        runs = sorted({p for g in globs for p in args.artifacts_root.glob(g)})
        series = [(r.name, s) for r in runs if (s := load_series(r)) is not None]
        missing = [r.name for r in runs if load_series(r) is None]
        print(f"\n{label}: {len(series)} runs"
              + (f"  (skipped, no series: {', '.join(missing)})" if missing else ""))
        if len(series) < 2:
            print("  too few runs to aggregate")
            continue
        for key in KEYS:
            per_seed = []
            for _name, s in series:
                m = (s["t"] >= T_MIN[key]) & np.isfinite(s[key])
                per_seed.append(float(np.mean(s[key][m])))
            scale = 100.0 if key in AS_PERCENT else 1.0
            unit = " %" if key in AS_PERCENT else ""
            print(f"  {key:20s} t>={T_MIN[key]:<4} "
                  f"{st.mean(per_seed) * scale:.4g} +/- "
                  f"{st.stdev(per_seed) * scale:.3g}{unit}")


if __name__ == "__main__":
    main()
