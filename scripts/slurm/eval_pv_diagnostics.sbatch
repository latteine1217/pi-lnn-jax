#!/bin/bash
#SBATCH --job-name=jax_eval_pv_diag
#SBATCH --output=logs/jax_eval_pv_diag_%j.out
#SBATCH --error=logs/jax_eval_pv_diag_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   對 placement-variance campaign 的 10 個 LES+oracle checkpoint 跑 eval-only（不重訓），
#   重新產出 metrics.json，補入新加的 div_ratio / kf_amp_ratio 診斷（compute_metrics 擴充）。
# Why:
#   §6 tab:main_metrics 的 Divergence ratio + k_f amplitude 兩列原值 provenance 已失、
#   外部重算驗證失敗；必須在 eval 內部（同 DNS 場/grid/striding）算才正確。eval 是快速
#   forward-only，10 個 ckpt 一 job 迴圈跑完。KE/u/v/ω 為確定性 eval，重跑值不變。

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

echo "[eval-pv-diag] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
uv run python -c "import jax; print(f'JAX {jax.__version__} {jax.devices()}')"
# preflight: cuDNN compile smoke
uv run python -c "import jax, jax.numpy as jnp; from jax import lax; lax.conv_general_dilated(jnp.ones((1,8,8,1)), jnp.ones((3,3,1,1)), (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC')).block_until_ready(); print('[preflight] cuDNN OK')" \
  || { echo '[ERR] cuDNN preflight 失敗 → head node 跑 setup_gpu_venv.sh' >&2; exit 5; }

CONFIGS=(
  configs/exp_pv_les_s42.toml  configs/exp_pv_les_s1.toml  configs/exp_pv_les_s2.toml
  configs/exp_pv_les_s3.toml   configs/exp_pv_les_s4.toml
  configs/exp_pv_oracle_s42.toml configs/exp_pv_oracle_s1.toml configs/exp_pv_oracle_s2.toml
  configs/exp_pv_oracle_s3.toml  configs/exp_pv_oracle_s4.toml
)

rc_all=0
for cfg in "${CONFIGS[@]}"; do
  echo "======================================================================"
  echo "[$(date '+%F %T')] eval $cfg"
  uv run python -u scripts/evaluate_exp245.py \
    --protocol follow_training \
    --config "$cfg" --ckpt latest \
    --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
    || { echo "[WARN] eval 失敗: $cfg"; rc_all=1; }
done
echo "[$(date '+%F %T')] eval-pv-diag 全部完成 rc_all=$rc_all"
exit "$rc_all"
