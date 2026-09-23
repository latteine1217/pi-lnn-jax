#!/bin/bash
#SBATCH --job-name=cp_recheck
#SBATCH --output=logs/cp_recheck_%j.out
#SBATCH --error=logs/cp_recheck_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#
# Post-hoc CP re-analysis (NO GPU, NO training).
#
# Why no --gres=gpu: run_cp_analysis.py is an offline analysis over the cached
# reconstruction npz (artifacts/cp/seed42.npz). It never loads a model or a
# checkpoint. Requesting a GPU would idle-block one of acmt20's two cards behind
# the EXP-301 training queue for no benefit, so this job is CPU-only and slots
# into the node's free CPU/mem instead.
#
# Resource sanity vs acmt20 (CPUTot=24, RealMemory=112G, gpu:2) while two
# half-resource training jobs are running (16 CPU / 96G / 2 GPU allocated):
#   4 CPU  <= 8 free    ok
#   12 G   <= ~14 G free ok
#   0 GPU  <= 0 free    ok  (does not queue behind training)
#
# Purpose: determine whether the paper's Table tab:cp matches what the CURRENT
# CP code produces. Evidence so far (git archaeology, local): the paper's table
# reproduces artifacts/cp/cp_metrics.json as of commit 75838c9 (20/20 cells) but
# only 12/20 against 7abfa32, which fixed local_difficulty to use the periodic
# minimum-image distance ("correct for periodic domain"). This job regenerates
# the metrics with today's code so the table can be updated from a verified run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# No GPU is allocated to this job; pin JAX to CPU so it cannot contend for a card
# that slurm did not grant us.
export JAX_PLATFORMS=cpu

export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
[ -x .venv/bin/python ] || { echo "[ERR] .venv missing; run 'uv sync --python 3.12' on the head node" >&2; exit 3; }

NPZ=artifacts/cp/seed42.npz
[ -f "$NPZ" ] || { echo "[ERR] $NPZ not found" >&2; exit 4; }

# Write to a NEW path. Do NOT overwrite artifacts/cp/cp_metrics.json: that file is
# the existing evidence we are comparing against, and clobbering it would destroy
# the very thing under test.
OUT=artifacts/cp/cp_metrics_recheck.json

echo "======================================================================"
echo "[cp_recheck] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  repo:   $(pwd)"
echo "  npz:    $NPZ ($(stat -c%s "$NPZ" 2>/dev/null || echo ?) bytes)"
echo "  out:    $OUT  (existing cp_metrics.json is left untouched)"
echo "  code:   $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "  JAX:    JAX_PLATFORMS=$JAX_PLATFORMS (no GPU requested)"
echo "======================================================================"

# --pointwise-draws 0 / --no-local-cp: the learned local-CP method needs scikit-learn,
# which is not in pyproject and not installed in .venv, and the compute node is
# UV_OFFLINE=1 so it cannot be fetched. It is also irrelevant here: Table tab:cp only
# reports vanilla / locally-adaptive (distance, physics) / Mondrian (distance, physics),
# none of which touch local CP. Skipping it keeps the run to exactly the five methods
# under test.
PYTHONPATH=. uv run python -u scripts/run_cp_analysis.py \
  --npz "$NPZ" \
  --alpha 0.1 \
  --forcing-A 0.1 \
  --forcing-kf 2.0 \
  --pointwise-draws 0 \
  --no-local-cp \
  --out "$OUT"

echo "[cp_recheck] done -> $OUT"
