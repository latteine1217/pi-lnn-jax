#!/bin/bash
#SBATCH --job-name=dump_cp_k200
#SBATCH --output=logs/dump_cp_k200_%j.out
#SBATCH --error=logs/dump_cp_k200_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
# Eval-class inference job (no training): half-resource per repo policy.
# JAX repo has a single trained seed (42); the paper's 5-seed main table was
# produced in the PyTorch repo. CP here runs Axis A+B on seed 42 only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 UV_OFFLINE=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# PILNJAX_DATA_ROOT rebases the config's relative sensor/DNS paths to the lab-server data
# tree (default points at the Mac dev path). Same as train_exp.sbatch.tmpl.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] PILNJAX_DATA_ROOT=$PILNJAX_DATA_ROOT has no data/ dir" >&2; exit 4; }

CONFIG=configs/exp_ksweep_k200_s42.toml
ARTIFACTS_DIR=artifacts/kolmogorov/ksweep_k200_s42   # K=200 operating point, seed 42

PYTHONPATH=. uv run python scripts/dump_cp_fields.py \
  --config "$CONFIG" --ckpt 20000 \
  --artifacts_dir "$ARTIFACTS_DIR" --out artifacts/cp/k200_seed42.npz

echo "[dump_cp] K=200 seed 42 done"
