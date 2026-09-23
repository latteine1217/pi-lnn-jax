#!/bin/bash
#SBATCH --job-name=jax_prof245
#SBATCH --output=logs/prof245_%j.out
#SBATCH --error=logs/prof245_%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# EXP-245 step 時間剖析：encode/decode/physics/backward/SOAP 分段 × matmul precision(fp32/tf32/bf16)。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "[prof245] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

for PREC in fp32 tf32 bf16; do
  echo
  echo "######################## precision=${PREC} ########################"
  uv run python -m bench.profile_exp245 --precision "$PREC" --n-collo 1024 \
    || echo "[WARN] precision=$PREC rc=$?"
done

echo
echo "[$(date '+%F %T')] prof245 完成"
