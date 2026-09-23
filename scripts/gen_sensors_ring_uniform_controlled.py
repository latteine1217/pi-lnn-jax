#!/usr/bin/env python3
"""Ring + uniform sensor placement for RealPDEBench controlled-cylinder.

Why not reuse the cylinder QR sensors: CylinderDataset reads `sensor_vals`
straight out of `<stem>_values.npz`, so borrowing the cylinder Re=10031 set would
feed a DIFFERENT flow's observations into a controlled reconstruction. Values
must be sampled from the controlled shard itself.

Why not QR here: the body oscillates, so a placement optimised on a static field
is not the point. Physically, a forced-vibration rig mounts sensors on/near the
body and covers the rest of the domain; the body sweeps a disk of radius
(R + amp), and any sensor inside it is periodically swallowed by the body. So:

  - ring     : n_ring sensors on a circle at r = R + amp + ring_gap, i.e. just
               OUTSIDE the swept disk -> never occluded, still boundary-layer /
               near-wake close. Angles that fall outside the domain are dropped
               (the body sits near the inlet, so the upstream arc is clipped).
  - uniform  : the remaining K - n_ring via farthest-point sampling over cells
               outside the swept disk (+ margin) -> global coverage.

Geometry and motion are MEASURED from the fields (pi_lnn_jax.controlled), not
taken from the filename: control_freq is NOT the body oscillation frequency.

Output matches the cylinder sensor contract consumed by CylinderDataset:
    <out>/<base>.json         -> {"selected_coordinates": [[x, y], ...], ...}
    <out>/<base>_values.npz   -> t, u [K,T], v [K,T], x [K], y [K]

2026-08-21 起 Arrow I/O 已移植進本專案（_common.qrpivot_cylinder），不再借 pi-lnn env：

  uv run python scripts/gen_sensors_ring_uniform_controlled.py \
      --shard <controlled.arrow> --out data/controlled_sensors --K 100 --n-ring 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_controlled():
    """Import pi_lnn_jax.controlled by path (its package __init__ pulls jax)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "pi_lnn_jax", "controlled.py")
    spec = importlib.util.spec_from_file_location("_pilnn_controlled", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True, help="controlled Arrow shard")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--K", type=int, default=100, help="total sensors")
    ap.add_argument("--n-ring", type=int, default=20,
                    help="sensors on the ring just outside the swept disk; "
                         "the rest are uniform (user spec: uniform-first)")
    ap.add_argument("--ring-gap", type=float, default=0.004,
                    help="physical gap between swept-disk edge and the ring")
    ap.add_argument("--margin", type=float, default=0.002,
                    help="extra physical margin excluded around the swept disk "
                         "for the uniform sensors")
    ap.add_argument("--osc-axis", default="0,1")
    ap.add_argument("--traj-stride", type=int, default=5)
    ap.add_argument("--min-r2", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common.qrpivot_cylinder import farthest_point_sampling, load_shard
    ctrl = _load_controlled()

    s = load_shard(Path(a.shard))
    u, v, x2d, y2d, t = s["u"], s["v"], s["x"], s["y"], s["t"]
    T, H, W = u.shape
    xa = x2d[0, :].astype(np.float64); ya = y2d[:, 0].astype(np.float64)
    Xp, Yp = np.meshgrid(xa, ya)
    axis = tuple(float(z) for z in a.osc_axis.split(","))

    # ── measured geometry + motion ───────────────────────────────────────────
    idx = np.arange(0, T, max(1, a.traj_stride))
    tr = ctrl.infer_trajectory(u[idx], v[idx], t[idx], xa, ya,
                               axis=axis, min_r2=a.min_r2)
    bx, by = tr["base_center"]; R = tr["radius"]; amp = abs(tr["amp"])
    swept_r = R + amp
    ring_r = swept_r + a.ring_gap
    print(f"sim_id={s['sim_id']}  tracked={tr['n_tracked']}/{len(idx)}")
    print(f"  measured: center=({bx:.5f},{by:.5f}) R={R:.5f} amp={amp:.5f} "
          f"freq={tr['freq']:.4f} r2={tr['r2']:.3f}  A/D={amp/(2*R):.3f}")
    print(f"  swept_r={swept_r:.5f}  ring_r={ring_r:.5f}")

    dist = np.sqrt((Xp - bx) ** 2 + (Yp - by) ** 2)          # [H, W] physical

    # ── ring: angular-uniform, drop points outside the domain, snap to grid ──
    ring_ij: list[tuple[int, int]] = []
    dropped = 0
    for th in np.linspace(0.0, 2 * np.pi, a.n_ring, endpoint=False):
        px, py = bx + ring_r * np.cos(th), by + ring_r * np.sin(th)
        if not (xa.min() <= px <= xa.max() and ya.min() <= py <= ya.max()):
            dropped += 1
            continue
        j = int(np.argmin(np.abs(xa - px))); i = int(np.argmin(np.abs(ya - py)))
        ring_ij.append((i, j))
    ring_ij = sorted(set(ring_ij))
    n_ring = len(ring_ij)
    if dropped:
        print(f"  ring: {dropped}/{a.n_ring} angles fall outside the domain "
              f"(body sits near the inlet) -> dropped; {n_ring} kept after dedupe")
    if n_ring == 0:
        raise ValueError("no ring sensor lands inside the domain; lower --ring-gap")

    # ── uniform: farthest-point over cells outside swept disk (+margin) ──────
    cand = (dist > swept_r + a.margin)
    for (i, j) in ring_ij:
        cand[i, j] = False
    cand_flat = np.argwhere(cand.reshape(-1)).ravel()
    n_uniform = a.K - n_ring
    if n_uniform < 0:
        raise ValueError(f"--n-ring {n_ring} exceeds --K {a.K}")
    if n_uniform > len(cand_flat):
        raise ValueError(f"need {n_uniform} uniform sensors but only "
                         f"{len(cand_flat)} candidate cells outside the swept disk")
    coords = np.stack([cand_flat // W, cand_flat % W], axis=1)
    pick = farthest_point_sampling(coords, n_uniform, seed=a.seed)
    uni_ij = [(int(coords[k, 0]), int(coords[k, 1])) for k in pick]

    all_ij = ring_ij + uni_ij
    if len(set(all_ij)) != len(all_ij):
        raise RuntimeError("ring and uniform picked the same cell (logic bug)")
    sensor_i = np.array([p[0] for p in all_ij], dtype=int)
    sensor_j = np.array([p[1] for p in all_ij], dtype=int)
    K = len(all_ij)

    n_swallowed = int((dist[sensor_i, sensor_j] <= swept_r).sum())
    if n_swallowed:
        raise RuntimeError(f"{n_swallowed} sensors lie inside the swept disk")

    sensor_x = xa[sensor_j]; sensor_y = ya[sensor_i]
    print(f"  placed K={K} ({n_ring} ring + {n_uniform} uniform); "
          f"0 inside swept disk. x[{sensor_x.min():.4f},{sensor_x.max():.4f}] "
          f"y[{sensor_y.min():.4f},{sensor_y.max():.4f}]")

    # ── sample values FROM THE CONTROLLED SHARD ─────────────────────────────
    u_s = u[:, sensor_i, sensor_j].T.astype(np.float32)   # [K, T]
    v_s = v[:, sensor_i, sensor_j].T.astype(np.float32)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(s["sim_id"]).replace(".h5", "").replace(".", "p")
    base = f"sensors_ringuniform_K{K}_controlled_{tag}"
    json_path = out_dir / f"{base}.json"
    npz_path = out_dir / f"{base}_values.npz"

    payload = {
        "K": K, "n_ring": n_ring, "n_uniform": n_uniform,
        "method": "ring_outside_swept_plus_uniform",
        "domain": "controlled_cylinder", "grid": f"{H}x{W}",
        "sim_id": s["sim_id"], "source_shard": str(a.shard),
        # measured, not from the filename
        "measured_base_center": [float(bx), float(by)],
        "measured_radius": float(R), "measured_amp": float(amp),
        "measured_freq": float(tr["freq"]), "measured_phase": float(tr["phase"]),
        "traj_r2": float(tr["r2"]), "traj_n_tracked": int(tr["n_tracked"]),
        "swept_radius": float(swept_r), "ring_radius": float(ring_r),
        "ring_gap": a.ring_gap, "margin": a.margin,
        "selected_coordinates": np.stack([sensor_x, sensor_y], 1).tolist(),
        "sensor_i": sensor_i.tolist(), "sensor_j": sensor_j.tolist(),
        "ring_sensor_flat": [int(i * W + j) for (i, j) in ring_ij],
        "uniform_sensor_flat": [int(i * W + j) for (i, j) in uni_ij],
        "values_npz": str(npz_path),
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved: {json_path}")
    np.savez(npz_path, t=t, u=u_s, v=v_s, x=sensor_x, y=sensor_y)
    print(f"Saved: {npz_path}  (u/v shape {u_s.shape})")
    print(f"stem for dump: --sensor-dir {a.out} --sensor-stem {base}")


if __name__ == "__main__":
    main()
