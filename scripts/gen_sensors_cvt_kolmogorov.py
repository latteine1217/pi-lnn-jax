"""gen_sensors_cvt_kolmogorov.py — Centroidal Voronoi Tessellation (DNS-free) placement.

Periodic (toroidal) Lloyd relaxation on the N x N grid with UNIFORM density: each
generator converges to the centroid of its Voronoi cell, yielding a near-hexagonal,
maximally-regular blue-noise layout. Like FPS this is a coverage/uniformity placement
that needs NO LES / DNS for the LAYOUT (sensor VALUES still come from the DNS).

Why: sec:placement's benefit was shown to be geometric uniformity, not flow-specific
information (FPS ties the DNS oracle, beats LES-QR; see
knowledge/experiments/kolmogorov-placement-uniformity-2026-07-06.md). CVT is a SECOND,
independent uniformity method. If CVT also lands on the ~4.6% floor, it hardens the
"uniformity, not learning" attribution and directly rebuts CVT-based learned-placement
claims (e.g. VSOPINN). Init is random-seeded (NOT FPS-seeded) so CVT and FPS stay
methodologically independent.

Output schema is byte-for-byte compatible with sensors_spacefill_*.{json,npz} (same
cand construction, same u[:, x_idx, y_idx].T value axis) so it drops into
exp_pv_cvt_s*.toml with zero training-code change.

Usage (numpy only, CPU; head node or local, NOT a training job):
  python scripts/gen_sensors_cvt_kolmogorov.py \
      --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
      --K 100 --seed 42 \
      --out data/kolmogorov_sensors/re10000 \
      --tag cvt_K100_N256_t0-5_si100
"""
import argparse
import json
from pathlib import Path
import numpy as np


def _periodic_d2(cand: np.ndarray, g: np.ndarray, L: float) -> np.ndarray:
    """Squared toroidal distance from every candidate row to a single point g."""
    diff = np.abs(cand - g)
    diff = np.minimum(diff, L - diff)
    return (diff ** 2).sum(1)


def cvt_periodic(cand: np.ndarray, K: int, L: float, origin: np.ndarray,
                 seed: int, max_iter: int = 100, tol: float = 1e-5):
    """Lloyd relaxation, toroidal distance, uniform density on the grid.

    cand: [n, 2] all grid points (the uniform-density support).
    Returns (selected row indices into cand, iterations run).
    """
    rng = np.random.default_rng(seed)
    n = cand.shape[0]
    gen = cand[rng.choice(n, size=K, replace=False)].astype(np.float64)  # random init
    two_pi = 2.0 * np.pi
    iters = max_iter
    for it in range(max_iter):
        # E-step: assign each grid cell to its nearest generator (toroidal)
        d2 = np.empty((n, K))
        for k in range(K):
            d2[:, k] = _periodic_d2(cand, gen[k], L)
        assign = d2.argmin(1)
        # M-step: move each generator to the toroidal centroid of its cell set
        new = gen.copy()
        for k in range(K):
            pts = cand[assign == k]
            if len(pts) == 0:
                # dead generator: reseed to the globally worst-covered grid point
                new[k] = cand[d2.min(1).argmax()]
                continue
            ang = (pts - origin) / L * two_pi          # map box -> circle per axis
            cx = np.arctan2(np.sin(ang[:, 0]).mean(), np.cos(ang[:, 0]).mean())
            cy = np.arctan2(np.sin(ang[:, 1]).mean(), np.cos(ang[:, 1]).mean())
            new[k, 0] = origin[0] + (cx % two_pi) / two_pi * L
            new[k, 1] = origin[1] + (cy % two_pi) / two_pi * L
        move = np.abs(new - gen); move = np.minimum(move, L - move)
        max_move = float(np.sqrt((move ** 2).sum(1)).max())
        gen = new
        if max_move < tol:
            iters = it + 1
            break
    # snap each generator to its nearest grid point; dedup any collisions
    sel, used = [], set()
    for k in range(K):
        for idx in np.argsort(_periodic_d2(cand, gen[k], L)):
            i = int(idx)
            if i not in used:
                used.add(i); sel.append(i); break
    return np.asarray(sel, dtype=int), iters


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage-optimal CVT sensor placement (DNS-free)")
    ap.add_argument("--dns", required=True, help="DNS .npy (sensor values + grid)")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="seed for Lloyd random init")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    raw = np.load(Path(args.dns).expanduser(), allow_pickle=True).item()
    x = np.asarray(raw["x"], np.float64); y = np.asarray(raw["y"], np.float64)
    u = np.asarray(raw["u"], np.float32); v = np.asarray(raw["v"], np.float32)
    t = np.asarray(raw["time"], np.float32)
    N = len(x); L = float(x[-1] - x[0] + (x[1] - x[0]))   # periodic box length
    origin = np.array([x[0], y[0]], np.float64)

    # candidate grid: cand[i*N + j] = (x[i], y[j])  (row-major, axis_1=x) — same as FPS
    X, Y = np.meshgrid(x, y, indexing="ij")
    cand = np.stack([X.ravel(), Y.ravel()], 1)
    sel_flat, iters = cvt_periodic(cand, args.K, L, origin, args.seed, args.max_iter)
    assert len(set(sel_flat.tolist())) == args.K, "CVT produced duplicate sensors after dedup"
    x_idx, y_idx = np.unravel_index(sel_flat, (N, N))      # x_idx=i, y_idx=j
    coords = np.stack([x[x_idx], y[y_idx]], 1)             # [K,2]

    # coverage diagnostic (same metric as FPS generator, comparable to its printout)
    dd = np.abs(coords[:, None] - coords[None]); dd = np.minimum(dd, L - dd)
    nn = np.sqrt((dd ** 2).sum(-1)); np.fill_diagonal(nn, 9)
    gg = np.abs(cand[:, None] - coords[None]); gg = np.minimum(gg, L - gg)
    cover_max = np.sqrt((gg ** 2).sum(-1)).min(1).max()
    nn_min = nn.min(1)   # per-sensor nearest neighbour
    print(f"CVT K={args.K} (Lloyd {iters} iters): min inter-sensor dist={nn.min():.4f}, "
          f"max coverage gap={cover_max:.4f}  |  cvNN={nn_min.std()/nn_min.mean():.3f} "
          f"(FPS ref: minPD 0.0884, cvNN 0.151, cover_gap 0.086)")

    # sensor values from DNS: convention u_full[t, x_idx, y_idx] — identical to FPS
    sensor_u = u[:, x_idx, y_idx].T                        # [K, T]
    sensor_v = v[:, x_idx, y_idx].T

    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"sensors_{args.tag}.json"
    npath = out / f"sensors_{args.tag}_dns_values.npz"
    meta = {
        "K": args.K, "resolution": f"{N}x{N}",
        "spatial_downsample_res": f"{N}x{N}", "spatial_downsample_stride": 1,
        "method": "centroidal_voronoi_tessellation_periodic",
        "features": [], "time_stride": 1,
        "time_range": [float(t[0]), float(t[-1])], "time_steps": int(len(t)),
        "selected_coordinates": coords.tolist(),
        "source_file": str(args.dns), "seed": args.seed,
        "sensor_dt": float(t[1] - t[0]), "sensor_time_points": int(len(t)),
        "engineering_pipeline_note":
            "DNS-free coverage-optimal placement (CVT/Lloyd, no LES); sensor values from DNS.",
    }
    json.dump(meta, open(jpath, "w"), indent=2)
    np.savez(npath, time=t, u=sensor_u, v=sensor_v)
    print(f"Saved JSON: {jpath}\nSaved NPZ:  {npath}\nDone.")


if __name__ == "__main__":
    main()
