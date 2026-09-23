#!/bin/bash
#SBATCH --job-name=jax_eval_T20_rand_v2
#SBATCH --output=logs/eval_re10000_T20_random_v2_%j.out
#SBATCH --error=logs/eval_re10000_T20_random_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# eval-only：從 re10000_T20_random_v2 最新 ckpt（step=20000）跑評估
# 訓練已在 job 3984 完成；GN phase 因 OOM 未執行，此 job 只取 step-20000 結果

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PILNJAX_ROOT="${PILNJAX_ROOT:-$HOME/pi-lnn-jax}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH" >&2; exit 2; }
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="configs/exp_re10000_T20_random_v2.toml"

echo "======================================================================"
echo "[eval re10000_T20_random_v2] $(date '+%F %T')  job=${SLURM_JOB_ID}"
echo "  config: ${CONFIG}  ckpt: latest (step=20000)"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || true

uv run python -u scripts/evaluate_exp245.py \
  --protocol follow_training \
  --config "${CONFIG}" \
  --ckpt latest \
  --optimizer schedule_free \
  --base_optimizer soap \
  --soap_precondition_frequency 2 \
  --time-stride 8 \
  --no-plot

echo "[$(date '+%F %T')] eval 完成"
