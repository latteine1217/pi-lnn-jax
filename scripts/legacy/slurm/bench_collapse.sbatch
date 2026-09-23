#!/bin/bash
#SBATCH --job-name=jax_collapse
#SBATCH --output=logs/collapse_%j.out
#SBATCH --error=logs/collapse_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Collapsing Taylor concept proof：1-dir vs 2-dir Laplacian grad（量 collapse 在 D=2 的頭空間）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "[collapse] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

for ND in 1 2; do
  echo; echo "########## n_dir=${ND} ##########"
  uv run python -m bench.bench_collapse --n-dir "$ND" --n-collo 1024 || echo "[WARN] n_dir=$ND rc=$?"
done
echo; echo "[$(date '+%F %T')] collapse concept proof 完成"
