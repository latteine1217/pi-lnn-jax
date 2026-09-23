"""2D-compatibility gate for RealPDEBench cases (read-only diagnostic).

Question: is the field on each stored 2D slice actually 2D divergence-free?
If yes, our 2D incompressible NS residual (physics.py) is valid for that case.
If the slice comes from a 3D simulation (e.g. Foil = WaterLily 3D, spanwise
slice at z=95), then du/dx + dv/dy = -dw/dz != 0 and the 2D continuity / NS
residual is systematically wrong -> the PDE-residual route does not apply and
that case can only be treated as pure data-driven reconstruction.

Metric: relative divergence r = RMS(du/dx + dv/dy) / RMS(|du/dx| + |dv/dy|),
measured over the fluid region (speed above a threshold, to exclude the
near-zero body interior whose finite-difference gradients are spurious). r is
unit-free and independent of the (unknown) physical grid spacing.

Interpretation (heuristic, calibrated against 2D-generated cylinder shards):
  r < 0.05   -> 2D incompressible holds well; PDE-residual route valid.
  0.05-0.15  -> marginal; spanwise leakage non-negligible, flag it.
  r > 0.15   -> significant 3D; 2D NS residual NOT trustworthy.

2D-generated baseline (cylinder numerical shards, verified locally 2026-07-12):
  cylinder Re1781  r = 0.020   |  cylinder Re10031 r = 0.068 (turbulent -> higher
  finite-difference discretization residual, still << 0.15).

Usage (needs pyarrow):
  uv run --with pyarrow python scripts/check_2d_divergence.py \
      --arrow <shard1.arrow> [<shard2.arrow> ...] --out gate_results.json

Read-only numpy diagnostic. Safe on head node (not training, not GPU).
The script prints sim_params so you can confirm the actual Re / AoA of each shard.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def load_fields(shard_path: str, n_frames: int, frame_stride: int):
    """Load u,v + x,y axes from one RealPDEBench Arrow shard.

    Schema: fields stored as float32 binary blobs; shape in shape_t/h/w ints.
    """
    with pa.memory_map(shard_path, "r") as s:
        tbl = ipc.RecordBatchFileReader(s).read_all()
    r = {n: tbl.column(n)[0] for n in tbl.column_names}

    def gi(k):
        return int(r[k].as_py()) if k in r else None

    def blob(k):
        v = r[k].as_py()
        return (np.frombuffer(v, np.float32) if isinstance(v, (bytes, bytearray))
                else np.asarray(v, np.float32))

    T, H, W = gi("shape_t"), gi("shape_h"), gi("shape_w")
    if None in (T, H, W):
        raise ValueError(f"missing shape_t/h/w in {shard_path}; cols={list(r)}")
    u = blob("u").reshape(T, H, W)
    v = blob("v").reshape(T, H, W)

    idx = np.arange(T // 5, T, frame_stride)[:n_frames]  # skip initial transient
    u, v = u[idx], v[idx]

    x = blob("x"); y = blob("y")
    if x.size == H * W:
        x = x.reshape(H, W)[0, :]
    elif x.size != W:
        x = np.arange(W, dtype=np.float32)
    if y.size == H * W:
        y = y.reshape(H, W)[:, 0]
    elif y.size != H:
        y = np.arange(H, dtype=np.float32)

    params = r["sim_params"].as_py() if "sim_params" in r else None
    return u, v, x, y, (T, H, W), params


def divergence_stats(u, v, x, y, speed_frac=0.1):
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.median(np.diff(y))) if y.size > 1 else 1.0
    dudx = np.gradient(u, dx, axis=-1)  # x along W (last axis)
    dvdy = np.gradient(v, dy, axis=-2)  # y along H
    div = dudx + dvdy
    terms = np.abs(dudx) + np.abs(dvdy)

    speed = np.sqrt(u**2 + v**2)
    thr = speed_frac * np.median(speed[speed > 0])
    fluid = speed > thr

    def rms(a, m):
        return float(np.sqrt(np.mean(a[m]**2)))

    per_frame = []
    for t in range(u.shape[0]):
        ft = fluid[t]
        if ft.sum() >= 10:
            per_frame.append(rms(div[t], ft) / (rms(terms[t], ft) + 1e-30))
    per_frame = np.array(per_frame) if per_frame else np.array([np.nan])
    return dict(
        dx=dx, dy=dy,
        r_rel=rms(div, fluid) / (rms(terms, fluid) + 1e-30),
        rms_div=rms(div, fluid),
        rms_dudx=rms(dudx, fluid),   # cross-check: ~= rms_dvdy iff 2D incompressible
        rms_dvdy=rms(dvdy, fluid),
        pf_median=float(np.median(per_frame)),
        pf_p90=float(np.percentile(per_frame, 90)),
        fluid_frac=float(fluid.mean()),
    )


def verdict(r):
    if r < 0.05:
        return "2D-OK"
    if r < 0.15:
        return "MARGINAL"
    return "3D-SIGNIFICANT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", nargs="+", required=True)
    ap.add_argument("--tag", default=None, help="label; default = basename")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--frame-stride", type=int, default=50)
    ap.add_argument("--speed-frac", type=float, default=0.1)
    a = ap.parse_args()

    results = {}
    for path in a.arrow:
        tag = a.tag or path.split("/")[-1]
        u, v, x, y, shp, params = load_fields(path, a.n_frames, a.frame_stride)
        st = divergence_stats(u, v, x, y, a.speed_frac)
        vd = verdict(st["r_rel"])
        results[tag] = dict(sim_params=params, shape=list(shp),
                            frames=int(u.shape[0]), **st, verdict=vd)
        print(f"### {tag}  sim_params={params}  shape(T,H,W)={shp}")
        print(f"    RMS du/dx={st['rms_dudx']:.5f}  RMS dv/dy={st['rms_dvdy']:.5f}"
              f"  (equal iff 2D-incompressible)  RMS(div)={st['rms_div']:.5f}")
        print(f"    relative divergence r = {st['r_rel']:.4f}"
              f"  (per-frame median={st['pf_median']:.4f} p90={st['pf_p90']:.4f})")
        print(f"    VERDICT: {vd}   [2D-gen cylinder baseline: 0.020-0.068]")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[wrote] {a.out}  ({len(results)} case(s))")


if __name__ == "__main__":
    main()
