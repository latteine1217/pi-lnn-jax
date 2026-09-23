#!/bin/bash
#SBATCH --job-name=jax_re10k_T20_rand
#SBATCH --output=logs/re10000_T20_random_v1_%j.out
#SBATCH --error=logs/re10000_T20_random_v1_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Exp A: Re=10000, T=20s, random K=100 sensor, vector attention, no W_O.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PILNJAX_ROOT="${PILNJAX_ROOT:-$HOME/pi-lnn-jax}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH" >&2; exit 2; }
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync'" >&2; exit 3; }
[ -f "data/sensors/re10000/sensors_random_K100_N256_t0-20_si128_seed42.json" ] || \
  { echo "[ERR] sensor JSON 不存在；請先執行 scripts/gen_sensors_re10000_T20.py" >&2; exit 4; }

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="configs/exp_re10000_T20_random_v1.toml"

echo "======================================================================"
echo "[re10000_T20_random_v1] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  config: ${CONFIG}"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || true
uv run python -c "import jax; print(f'JAX {jax.__version__}  devices={jax.devices()}')"

uv run python -u train_kolmogorov.py \
  --config "${CONFIG}" \
  --arch liquid \
  --optimizer schedule_free \
  --base_optimizer soap \
  --soap_precondition_frequency 2 \
  --use_gradnorm \
  --use_al \
  --use_time_marching \
  --tm_start_frac 0.1 \
  --tm_warmup_steps 2000 \
  --tm_ramp_steps 6000 \
  --eval_every 0 \
  --save_every 1000 \
  --log_every 500 \
  --refine_optimizer none

TRAIN_RC=$?
echo "[$(date '+%F %T')] 訓練結束 rc=${TRAIN_RC}"

if [ "${TRAIN_RC}" = "0" ]; then
  uv run python -u scripts/evaluate_exp245.py \
    --config "${CONFIG}" \
    --ckpt latest \
    --optimizer schedule_free \
    --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --time-stride 1 \
    --no-plot \
    || echo "[WARN] eval 失敗，ckpt 仍保留可手動 retry"
fi

exit "${TRAIN_RC}"
