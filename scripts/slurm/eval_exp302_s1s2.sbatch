#!/bin/bash
#SBATCH --job-name=jax_eval302_s1s2
#SBATCH --output=logs/eval_exp302_s1s2_%j.out
#SBATCH --error=logs/eval_exp302_s1s2_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: 補 EXP-302 s1/s2 的 eval（campaign 取消時已訓完 step-50000 但未 eval）。
#   chunk=2000（PILNN_EVAL_CHUNK_BUDGET=400000）——seed42 已手動驗證 chunk=2000 於報告精度
#   等同 chunk=1000（job 4553 診斷：headline abs 差 ≤ 3e-4，論文 2 位%精度一致），故不再 gate。
# 對齊金標準：--time-stride 1（T_eval=101）；optimizer flags 對齊訓練。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:." UV_OFFLINE=1
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[eval302-s1s2] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HEAD=$(git rev-parse --short HEAD)"
for s in 1 2; do
  echo ""
  echo "=== s${s} chunk=2000 → final_eval ==="
  t0=$(date +%s)
  PILNN_EVAL_CHUNK_BUDGET=400000 \
  uv run python -u scripts/evaluate_exp245.py \
    --protocol follow_training \
    --config "configs/exp_302_b3_re1e6_K200_les_T50_s${s}.toml" \
    --ckpt latest --optimizer schedule_free --base_optimizer soap \
    --soap_precondition_frequency 2 --time-stride 1 \
    --output-subdir final_eval
  echo "  s${s} eval rc=$?  wall=$(( $(date +%s) - t0 ))s"
done
echo ""
echo "[eval302-s1s2] $(date '+%F %T') done"
