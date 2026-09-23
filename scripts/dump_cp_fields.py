"""Reconstruct full (u,v) fields from PI-CON checkpoints across all DNS time slices,
cache them to .npz for offline conformal-prediction analysis.

Why: CP is post-hoc. We reconstruct once per seed (inference only) and cache
(pred_u, pred_v, dns_u, dns_v, grid, sensor geometry, t). The CP module then runs
locally on these arrays without touching the GPU or retraining.

Usage (lab-server, per seed via its own artifacts_dir):
    PYTHONPATH=. uv run python scripts/dump_cp_fields.py \
        --config configs/exp_245_b3_les_T50.toml \
        --ckpt 20000 --artifacts_dir artifacts/kolmogorov/exp245_seed42 \
        --out artifacts/cp/seed42.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params  # noqa: E402
from pi_lnn_jax.config import load_config  # noqa: E402
from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path  # noqa: E402
from pi_lnn_jax.evaluate import reconstruct_field  # noqa: E402
from pi_lnn_jax.model_factory import build_model  # noqa: E402

RE_NORM_SCALE = float(np.log(10000.0))


def parse_args():
    p = argparse.ArgumentParser(description="Dump CP reconstruction fields")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="20000")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--time-stride", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_kwargs, model_kwargs = cfg["data_kwargs"], cfg["model_kwargs"]
    train_kwargs = cfg["train_kwargs"]

    re_value = float(data_kwargs.get("re_values", [10000.0])[0])
    re_norm = float(np.log(re_value) / RE_NORM_SCALE)
    seed = int(train_kwargs.get("seed", 42))

    d = load_sensors_from_path(data_kwargs["sensor_jsons"][0], time_stride=args.time_stride)
    sensor_vals = jnp.asarray(d["sensor_vals"])
    sensor_pos = jnp.asarray(d["sensor_pos"])
    sensor_time = jnp.asarray(d["sensor_time"])
    norm_stats = d["norm_stats"]
    T = sensor_vals.shape[0]

    dns_u_f, dns_v_f, dns_t_f = load_dns_from_path(
        Path(data_kwargs["dns_paths"][0]), time_stride=args.time_stride,
    )
    stride = max(1, dns_u_f.shape[0] // T)
    dns_u, dns_v, dns_t = dns_u_f[::stride][:T], dns_v_f[::stride][:T], dns_t_f[::stride][:T]
    Nx, Ny = dns_u.shape[1], dns_u.shape[2]
    x_grid = np.linspace(0.0, 1.0, Nx, endpoint=False).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny, endpoint=False).astype(np.float32)

    # 走 factory：訓練端與 eval 端必須是同一個物件（scripts/CLAUDE.md §2）。
    # 本腳本沒有 --arch，只支援 B3——把假設寫出來而非靠「剛好只建 B3」。
    model, _ = build_model("liquid", model_kwargs,
                           K_sensors=int(sensor_pos.shape[0]))
    init_params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)

    ckpt_dir = Path(args.artifacts_dir).resolve() / "checkpoints"
    params, step, ckpt_provenance = restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=init_params, model=model)

    pred_u = np.zeros((T, Nx, Ny), np.float32)
    pred_v = np.zeros((T, Nx, Ny), np.float32)
    t0 = time.time()
    for tidx in range(T):
        u, v = reconstruct_field(
            model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
            norm_stats, x_grid, y_grid, float(dns_t[tidx]))
        pred_u[tidx], pred_v[tidx] = u, v
    print(f"[recon] T={T} fields in {time.time()-t0:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, pred_u=pred_u, pred_v=pred_v, dns_u=dns_u, dns_v=dns_v,
        dns_t=dns_t, x_grid=x_grid, y_grid=y_grid,
        sensor_pos=np.asarray(sensor_pos), seed=seed, re_value=re_value,
        # npz 存不了巢狀 dict，序列化成字串；印記仍跟著這份場資料走。
        ckpt_provenance=json.dumps(ckpt_provenance))
    print(f"[out] {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
