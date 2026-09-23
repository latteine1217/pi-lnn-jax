#!/bin/bash
#SBATCH --job-name=jax_folx
#SBATCH --output=logs/folx_%j.out
#SBATCH --error=logs/folx_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# folx (collapsed forward-laplacian) vs jet (current 2-dir taylor)：EXP-245 scale Laplacian。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "[folx] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax,folx; print(f'JAX {jax.__version__} folx {folx.__version__} backend={jax.default_backend()}')"

for IMPL in jet folx; do
  echo; echo "########## impl=${IMPL} ##########"
  uv run python -m bench.bench_folx --impl "$IMPL" --n-collo 1024 2>&1 | grep -vE "WARNING:\[folx\]" || echo "[WARN] impl=$IMPL rc=$?"
done
echo; echo "[$(date '+%F %T')] folx vs jet 完成"
