"""Dump RealPDEBench controlled-cylinder to npz.
z geometry is static + sim_params=[Re,control_freq] has no amplitude, so the
oscillation amplitude is reconstructed via pi_lnn_jax.controlled.infer_trajectory.
Usage (2026-08-21 起 CylinderDataset / Arrow I/O 已移植進本專案，不再借 pi-lnn env):
  uv run python scripts/dump_controlled_cylinder_v2.py \
      --arrow <controlled shard.arrow> --out ~/pi-lnn-jax/data/controlled_v2.npz \
      --sensor-stem sensors_qrpivot_K100_cylinder_Re10031
"""
import argparse, os, sys
import importlib.util
import numpy as np


def _load_controlled():
    """Load pi_lnn_jax.controlled (pure numpy) by file path.

    Importing it as `pi_lnn_jax.controlled` triggers pi_lnn_jax/__init__, which
    pulls jax/flax. The dump runs in the pi-lnn (PyTorch) env, which has pyarrow
    but no jax — so import the numpy-only module directly, bypassing the package
    __init__. cwd-independent (resolves relative to this file).
    """
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "pi_lnn_jax", "controlled.py")
    spec = importlib.util.spec_from_file_location("_pilnn_controlled", cpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.infer_trajectory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor-stem", required=True, help="sensor placement stem")
    ap.add_argument("--sensor-dir", default="data/cylinder_sensors")
    ap.add_argument("--osc-axis", default="0,1")
    ap.add_argument("--traj-stride", type=int, default=5)
    ap.add_argument("--min-r2", type=float, default=0.5)
    ap.add_argument("--n-eval-times", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow as pa
    from _common.cylinder_dataset import CylinderDataset
    from _common.cylinder_arrow import load_arrow_fields
    infer_trajectory = _load_controlled()

    # Re + control_freq live only in sim_id (e.g. '1781_0.7.h5'); the Arrow shard has
    # no sim_params column and load_arrow_fields does not return them. Fail loudly on
    # an unexpected format rather than silently defaulting.
    with open(a.arrow, "rb") as _fh:
        sim_id = str(pa.ipc.open_stream(_fh).read_next_batch().column("sim_id")[0].as_py())
    stem = sim_id[:-3] if sim_id.endswith(".h5") else sim_id
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"cannot parse Re/control_freq from sim_id={sim_id!r}")
    re_val, control_freq = float(parts[0]), float(parts[1])

    ds = CylinderDataset(
        sensor_json=f"{a.sensor_dir}/{a.sensor_stem}.json",
        sensor_npz=f"{a.sensor_dir}/{a.sensor_stem}_values.npz",
        arrow_shard=a.arrow, re_value=re_val, sensor_subsample=20,
    )
    f = load_arrow_fields(a.arrow); T,H,W = f["T"],f["H"],f["W"]
    xlo,xhi,ylo,yhi = ds.x_lo,ds.x_hi,ds.y_lo,ds.y_hi
    Lx_, Ly_ = float(ds.Lx), float(ds.Ly)
    dns_x = ((f["x"][0,:]-xlo)/(xhi-xlo)).astype(np.float32)
    dns_y = ((f["y"][:,0]-ylo)/(yhi-ylo)).astype(np.float32)
    axis = tuple(float(z) for z in a.osc_axis.split(","))
    t = np.asarray(f["t"], np.float64)

    # Body geometry + motion are MEASURED from the fields on PHYSICAL axes (the body
    # is a true circle only there). freq is measured, NOT taken from control_freq:
    # on 1781_0.7 the body oscillates at 0.150 while control_freq=0.7 holds 0.1% of
    # the power. infer_trajectory raises if the fit is noise.
    idx = np.arange(0, T, max(1, a.traj_stride))
    tr = infer_trajectory(f["u"][idx], f["v"][idx], t[idx],
                          f["x"][0,:].astype(np.float64), f["y"][:,0].astype(np.float64),
                          axis=axis, min_r2=a.min_r2)
    print(f"sim_id={sim_id}  Re={re_val:.0f}  control_freq={control_freq:.4f} (metadata only)")
    print(f"MEASURED body: center=({tr['base_center'][0]:.5f},{tr['base_center'][1]:.5f}) "
          f"R={tr['radius']:.5f}+-{tr['radius_std']:.5f} (D={2*tr['radius']:.4f})")
    print(f"MEASURED motion: freq={tr['freq']:.4f} amp={tr['amp']:.5f} phase={tr['phase']:.4f} "
          f"r2={tr['r2']:.3f} power={tr['power_frac']:.1%} tracked={tr['n_tracked']}/{len(idx)} "
          f"A/D={tr['amp']/(2*tr['radius']):.3f}")
    if tr["touches_border"]:
        print("[WARN] fitted body touches a domain border; circle fit may be biased")

    # physical -> normalized (training space). radius_norm uses the same RMS convention
    # as the cylinder dump: sqrt(2*mean(|p-c|^2_norm)) over a disk = R*sqrt((1/Lx^2+1/Ly^2)/2).
    # NOTE: boundary.py treats the body as a circle in NORMALIZED space, i.e. an ellipse
    # in physical space when Lx != Ly (pre-existing convention, shared with cylinder).
    base = ((tr["base_center"][0]-xlo)/(xhi-xlo), (tr["base_center"][1]-ylo)/(yhi-ylo))
    radius = float(tr["radius"] * np.sqrt((1.0/Lx_**2 + 1.0/Ly_**2)/2.0))
    amp_norm = float(tr["amp"] * np.sqrt((axis[0]/Lx_)**2 + (axis[1]/Ly_)**2))
    print(f"normalized: base_center=({base[0]:.5f},{base[1]:.5f}) radius={radius:.5f} amp={amp_norm:.5f}")
    ti = np.arange(0,T,20); us,vs = f["u"][ti],f["v"][ti]
    vi = np.sort(ds.val_t_idx); et = vi[::max(1,len(vi)//a.n_eval_times)][:a.n_eval_times]
    np.savez_compressed(a.out,
        sensor_vals=ds.sensor_vals.astype(np.float32), sensor_pos=ds.sensor_pos.astype(np.float32),
        sensor_time=ds.sensor_time.astype(np.float32), obs_mean=ds.observed_channel_mean.astype(np.float32),
        obs_std=ds.observed_channel_std.astype(np.float32), train_t_idx=ds.train_t_idx.astype(np.int32),
        val_t_idx=ds.val_t_idx.astype(np.int32),
        base_center=np.asarray(base, np.float32), body_radius=np.float32(radius),
        # MEASURED kinematics (normalized). osc_freq is the measured body frequency,
        # NOT control_freq — see module doc and pi_lnn_jax.controlled.
        osc_amp=np.float32(amp_norm), osc_phase=np.float32(tr["phase"]),
        osc_freq=np.float32(tr["freq"]), osc_axis=np.asarray(axis, np.float32),
        # provenance / quality of the recovered trajectory (for audit, not training)
        traj_r2=np.float32(tr["r2"]), traj_power_frac=np.float32(tr["power_frac"]),
        traj_n_tracked=np.int32(tr["n_tracked"]), body_radius_phys=np.float32(tr["radius"]),
        body_radius_phys_std=np.float32(tr["radius_std"]),
        Lx=np.float32(ds.Lx), Ly=np.float32(ds.Ly), re_value=np.float32(ds.re_value),
        control_freq=np.float32(control_freq),
        # real inflow speed measured from the DNS by CylinderDataset -> train_cylinder
        # picks this up instead of the 0.33 fallback (wrong for this rig: p50 speed ~0.078)
        bc_inflow_u=np.float32(ds.bc_inflow_u),
        dns_u=us[et].astype(np.float32), dns_v=vs[et].astype(np.float32),
        dns_x=dns_x, dns_y=dns_y, dns_t_idx=et.astype(np.int32))
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
