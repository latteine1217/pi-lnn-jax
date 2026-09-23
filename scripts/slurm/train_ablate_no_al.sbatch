#!/bin/bash
#SBATCH --job-name=jax_ablate_no_al
#SBATCH --output=logs/jax_ablate_no_al_%j.out
#SBATCH --error=logs/jax_ablate_no_al_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: #4 divergence ablation — headline config seed42，但 --ablate_no_al 關閉 AL
#       continuity 約束（total loss 無 al_cont_term → continuity 完全無約束）。訓練後 eval。
# Why:  對照 headline div_ratio 0.81%，量化 AL penalty 對 divergence 的效果（(b) statement）。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] PILNJAX_DATA_ROOT=$PILNJAX_DATA_ROOT 內無 data/" >&2; exit 4; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CFG=configs/exp_ablate_no_al.toml
echo "[ablate-no-al] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}  cfg=$CFG"
uv run python -c "import jax; print(f'JAX {jax.__version__} {jax.devices()}')"
uv run python -c "import jax, jax.numpy as jnp; from jax import lax; lax.conv_general_dilated(jnp.ones((1,8,8,1)), jnp.ones((3,3,1,1)), (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC')).block_until_ready(); print('[preflight] cuDNN OK')" \
  || { echo '[ERR] cuDNN preflight 失敗 → head node 跑 setup_gpu_venv.sh' >&2; exit 5; }

# ── 訓練（--ablate_no_al 覆蓋 --use_al → use_al=False → 無 continuity）──
uv run python -u train_kolmogorov.py \
  --config "$CFG" \
  --arch liquid --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
  --use_gradnorm --use_al --ablate_no_al \
  --use_time_marching --tm_start_frac 0.1 --tm_warmup_steps 2000 --tm_ramp_steps 6000 \
  --eval_every 0 --save_every 1000 --log_every 500 --refine_optimizer none
TRAIN_RC=$?
echo "[$(date '+%F %T')] train rc=${TRAIN_RC}"

# ── eval（產出 div_ratio 對照）──
if [ "${TRAIN_RC}" = "0" ]; then
  uv run python -u scripts/evaluate_exp245.py \
    --protocol follow_training \
    --config "$CFG" --ckpt latest \
    --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
    || echo "[WARN] eval 失敗；ckpt 保留可手動 retry"
fi
exit "${TRAIN_RC}"
