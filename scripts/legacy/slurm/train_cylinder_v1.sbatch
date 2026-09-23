#!/bin/bash
#SBATCH --job-name=jax_cyl_v1
#SBATCH --output=logs/cyl_v1_%j.out
#SBATCH --error=logs/cyl_v1_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Cylinder v1 A/B：baseline (Kolmogorov-style attention) vs geo (geometry-aware #1/#2/#3)。
# 比 DNS field-L2（held-out val 時刻）。資料：data/cylinder_v1.npz（dump_cylinder_v1.py 產）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在（先跑 dump_cylinder_v1.py）" >&2; exit 4; }

echo "[cyl_v1] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

STEPS=8000
for MODE in baseline geo; do
  echo
  echo "######################## MODE=${MODE} ########################"
  uv run python -u train_cylinder_v1.py --mode "$MODE" --steps $STEPS --w_phys 0.01 \
    || echo "[WARN] mode=$MODE rc=$?"
done

echo
echo "[$(date '+%F %T')] cyl_v1 完成"
