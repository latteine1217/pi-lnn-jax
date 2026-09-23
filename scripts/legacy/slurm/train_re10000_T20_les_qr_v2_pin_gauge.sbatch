#!/bin/bash
#SBATCH --job-name=jax_re10k_T20_lesqr_v2_pin_gauge
#SBATCH --output=logs/re10000_T20_les_qr_v2_pin_gauge_%j.out
#SBATCH --error=logs/re10000_T20_les_qr_v2_pin_gauge_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# v2_pin_gauge: 在 v2_pin 基礎上加 gauge 錨定正則項 (gauge_loss_weight=0.01)。唯一變因 vs v2_pin。
#               AL warmup 5000步 + 主訓練 20000步；不跑 GN refinement。半節點資源。

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

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="configs/exp_re10000_T20_les_qr_v2_pin_gauge.toml"

echo "======================================================================"
echo "[re10000_T20_les_qr_v2_pin_gauge] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  config: ${CONFIG}"
echo "  AL warmup: 5000 steps | Main: 20000 steps | NO GN refine | gauge=0.01"
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
  --log_every 500

TRAIN_RC=$?
echo "[$(date '+%F %T')] 訓練結束 rc=${TRAIN_RC}"

if [ "${TRAIN_RC}" = "0" ]; then
  uv run python -u scripts/evaluate_exp245.py \
    --config "${CONFIG}" \
    --ckpt latest \
    --optimizer schedule_free \
    --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --time-stride 8 \
    --no-plot \
    || echo "[WARN] eval 失敗，ckpt 仍保留可手動 retry"
fi

exit "${TRAIN_RC}"
