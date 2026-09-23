#!/bin/bash
#SBATCH --job-name=jax_eval_exp301_ens
#SBATCH --output=logs/eval_exp301_enstrophy_%j.out
#SBATCH --error=logs/eval_exp301_enstrophy_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   對 EXP-301 已跑完的 5 個 Re=1000 checkpoint 做 eval-only 重跑，取得新增的
#   enstrophy_rel_err（compute_metrics 新指標，commit 73dc811）。輸出寫獨立子目錄
#   final_eval_enstrophy，**不覆寫**原 final_eval/metrics.json（原檔即基準，兼作 KE 可重現性對照）。
#
# Why:
#   tab:re1e6「Enstrophy rel-err」列的 Re=1000 需 5-seed 新值（人審決）。eval-only 便宜
#   （每 seed ~5min），reuse 現有 checkpoint 不重訓。
#
# 對齊訓練（eval-script-discipline 紅線）：
#   --optimizer/--base_optimizer/--soap_precondition_frequency 與訓練一致（orbax restore 嚴格）；
#   --time-stride 1 對齊 Re=1000 訓練的 time_strides=[1]（預設 2 會用一半時間解析度評，錯）。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] PILNJAX_DATA_ROOT=$PILNJAX_DATA_ROOT 無 data/" >&2; exit 4; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node uv sync" >&2; exit 3; }
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[eval-ens] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HEAD=$(git rev-parse --short HEAD)"
nvidia-smi -L 2>/dev/null || echo "[WARN] no nvidia-smi"

# 5 seed：config 與其 artifacts_dir（seed42 無 _s 後綴）
CONFIGS=(
  "configs/exp_301_b3_re1000_les_T100.toml"
  "configs/exp_301_b3_re1000_les_T100_s1.toml"
  "configs/exp_301_b3_re1000_les_T100_s2.toml"
  "configs/exp_301_b3_re1000_les_T100_s3.toml"
  "configs/exp_301_b3_re1000_les_T100_s4.toml"
)

rc_all=0
for cfg in "${CONFIGS[@]}"; do
  echo "======================================================================"
  echo "[eval-ens] $(date '+%T') eval $cfg"
  uv run python -u scripts/evaluate_exp245.py \
    --protocol follow_training \
    --config "$cfg" \
    --ckpt latest \
    --optimizer schedule_free \
    --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --time-stride 1 \
    --output-subdir final_eval_enstrophy \
    || { echo "[WARN] eval 失敗 rc=$? cfg=$cfg"; rc_all=1; }
done

echo "[eval-ens] $(date '+%F %T') done rc_all=${rc_all}"
exit "${rc_all}"
