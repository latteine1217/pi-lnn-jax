#!/bin/bash
#SBATCH --job-name=jax_ng_conv
#SBATCH --output=logs/ng_conv_%j.out
#SBATCH --error=logs/ng_conv_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# NG spike convergence：A2-NG (matrix-free CG + LM damping) vs Adam，steps/wall-to-accuracy。
# 回答「Gauss-Newton 值不值得」。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

mkdir -p logs
echo "[ng_conv] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

# N=128（A2 在 perf 已驗最省最快的規模）；NG 25 步 vs Adam 2000 步
uv run python -m bench.ng_spike --mode convergence --n-collo 128 --ng-steps 25 --adam-steps 2000

echo "[$(date '+%F %T')] ng_conv 完成"
