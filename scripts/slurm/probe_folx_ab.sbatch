#!/bin/bash
#SBATCH --job-name=jax_probe_folx
#SBATCH --output=logs/probe_folx_ab_%j.out
#SBATCH --error=logs/probe_folx_ab_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: 只量 folx wrap stop_gradient 的訓練 s/step 效益（A/B）。
#   前一版探針把訓練輸出寫 compute-node /tmp（job 結束即讀不到）→ 崩了看不到原因。
#   本版全部寫進共享 logs/，崩就地 dump。
#
# 量法：同一 config 各跑 400 步（log_every 100），sg=0（舊 full-Hessian）vs sg=1（stop_gradient）。
#   比第 300→400 步的 wall 差（跳過前段 JIT 編譯）。artifacts 寫 /tmp，不碰正式產出。

set -uo pipefail   # 注意：不加 -e，讓崩掉的訓練不會中止整個腳本
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[probe-folx] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HEAD=$(git rev-parse --short HEAD)"
CFG=configs/exp_302_b3_re1e6_K200_les_T50.toml
LOGDIR="logs/probe_folx_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

for sg in 0 1; do
  echo ""
  echo "================ [A] PILNN_WRAP_STOPGRAD=${sg} ================"
  L="$LOGDIR/train_sg${sg}.log"
  PILNN_WRAP_STOPGRAD=$sg \
  uv run python -u train_kolmogorov.py \
    --config "$CFG" \
    --arch liquid --optimizer schedule_free --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --use_gradnorm --use_al --use_time_marching \
    --tm_start_frac 0.1 --tm_warmup_steps 2000 --tm_ramp_steps 6000 \
    --eval_every 0 --save_every 100000 --log_every 100 \
    --refine_optimizer none \
    --steps 400 \
    --artifacts_dir /tmp/probe_folx_sg${sg} > "$L" 2>&1
  rc=$?
  echo "  rc=${rc}"
  echo "  step 計時行（wall 欄）："
  grep -E "^ +[0-9]+ " "$L" | tail -5 | sed 's/^/    /'
  echo "  folx 'round not in registry' 次數: $(grep -c 'round not in registry' "$L")"
  if [ "$rc" != "0" ]; then
    echo "  !! 訓練非零退出，log 尾巴："
    tail -25 "$L" | sed 's/^/    /'
  fi
done

echo ""
echo "[probe-folx] $(date '+%F %T') done — 完整 log 見 $LOGDIR/"
