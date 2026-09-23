#!/bin/bash
#SBATCH --job-name=jax_sb_ab
#SBATCH --output=logs/sb_ab_%j.out
#SBATCH --error=logs/sb_ab_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Sensor mini-batch A/B：full (10100) vs mini-batch (2000) sensor query。
# 同 EXP-245 config + 模板旗標，各跑 300 步。
# 比 ms/step + loss 軌跡，確認 KE 不退步後再整合 20k。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] $PILNJAX_DATA_ROOT/data 不存在" >&2; exit 4; }

CONFIG="configs/exp_245_b3_les_T50.toml"
STEPS=5000
COMMON="--config $CONFIG --arch liquid --optimizer schedule_free --base_optimizer soap \
  --soap_precondition_frequency 2 --use_gradnorm --use_al --use_time_marching \
  --tm_start_frac 0.1 --tm_warmup_steps 2000 --tm_ramp_steps 6000 \
  --steps $STEPS --log_every 200 --eval_every 0 --save_every 0 --refine_optimizer none"

echo "[sb_ab] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "===== A: full sensor query (10100) ====="
uv run python -u train_kolmogorov.py $COMMON \
  2>&1 | tee logs/sb_full_${SLURM_JOB_ID:-local}.log || echo "[WARN] full rc=$?"

echo; echo "===== B: sensor mini-batch (2000/10100) ====="
uv run python -u train_kolmogorov.py $COMMON --n_sensor_query 2000 \
  2>&1 | tee logs/sb_mb_${SLURM_JOB_ID:-local}.log || echo "[WARN] minibatch rc=$?"

echo
echo "===== ms/step 比較 ====="
for tag in full mb; do
  f=logs/sb_${tag}_${SLURM_JOB_ID:-local}.log
  if [ -f "$f" ]; then
    wall=$(grep -E "^\s+[0-9]+\s" "$f" | awk '{print $NF}' | tail -5 | awk '{sum+=$1;n++} END{printf "%.2f", sum/n}')
    total=$(grep "train wall" "$f" | tail -1)
    echo "  [$tag] last-5-window avg wall/200steps=${wall}s => ~$(echo "scale=0; $wall*1000/200" | bc)ms/step | $total"
  fi
done
echo "[$(date '+%F %T')] sb_ab 完成"
