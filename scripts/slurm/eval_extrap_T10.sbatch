#!/bin/bash
#SBATCH --job-name=eval_extrap_T10
#SBATCH --output=logs/eval_extrap_T10_%j.out
#SBATCH --error=logs/eval_extrap_T10_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   EXP-530（時間外推）的正式評估：sensor 只餵 t∈[0,5]，query 走延長版 DNS 的
#   完整格點 t∈[0,10]，然後畫 error-vs-lead-time 曲線與兩條參考線。
#
# 協定宣告：--protocol sensor_time_independent --time-stride 2
#   sensor 時間軸（[0,5]）與 query 格點（[0,10]）本來就不同一段，不做對齊。
#   stride=2 同時是兩者的取樣率：sensor 101 幀（與訓練相同）、query 201 幀
#   @Δt=0.05 覆蓋 [0,10]。
#
# 前提（不成立就不該跑）：
#   1. 訓練已完成，ckpt 在 configs/eval_530_extrap_T10.toml 的 artifacts_dir。
#   2. 延長版 DNS 已通過軌跡一致性驗證（gen_dns_extended.sbatch 的第 3 步）。
#      config 的 dns_paths 指向被採納的那一臂——判 FAIL 的那一臂不可用來評估。
#   建議用 --dependency=afterok:<train_jobid>:<dns_jobid> 送，讓前提由 slurm 保證。
#
# GPU：ckpt 是 GPU 寫的，orbax 在 CPU-only node 推不出 sharding（見 CLAUDE.md）。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="${CONFIG:-configs/eval_530_extrap_T10.toml}"
OUT_SUBDIR="${OUT_SUBDIR:-extrap_T10}"
T_DATA_END="${T_DATA_END:-5.0}"
# branch 自迴歸（EXP-533）：必須與訓練該 ckpt 時的設定一致，否則是在評另一個方法。
AUTOREG_DT="${AUTOREG_DT:-0}"
AUTOREG_ROUNDS="${AUTOREG_ROUNDS:-1}"
AR_ARGS=()
if [ "${AUTOREG_DT}" != "0" ]; then
  AR_ARGS=(--autoreg-pseudo-dt "${AUTOREG_DT}" --autoreg-rounds "${AUTOREG_ROUNDS}")
  echo "[eval_extrap] autoreg dt=${AUTOREG_DT} rounds=${AUTOREG_ROUNDS}"
fi

echo "======================================================================"
echo "[eval_extrap] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "[eval_extrap] config=$CONFIG  out_subdir=$OUT_SUBDIR  t_data_end=$T_DATA_END"
echo "======================================================================"

uv run python -u scripts/evaluate_exp245.py \
  --protocol sensor_time_independent \
  --time-stride 2 \
  --config "$CONFIG" \
  --ckpt latest \
  --arch liquid \
  --optimizer schedule_free \
  --base_optimizer soap \
  --soap_precondition_frequency 2 \
  --output-subdir "$OUT_SUBDIR" \
  "${AR_ARGS[@]}" \
  --export-arrays \
  --export-fields \
  --no-plot

# eval 用哪份 DNS 由 config 決定；此處原樣取回同一個值，避免兩邊各寫一次而漂移。
DNS_PATH=$(uv run python -c "
import tomllib, pathlib
from pi_lnn_jax.data import _resolve_data_path
cfg = tomllib.loads(pathlib.Path('$CONFIG').read_text())
print(_resolve_data_path(cfg['train']['dns_paths'][0]))
")
ART_DIR=$(uv run python -c "
import tomllib, pathlib
print(tomllib.loads(pathlib.Path('$CONFIG').read_text())['train']['artifacts_dir'])
")

uv run python -u scripts/plot_extrapolation_leadtime.py \
  --series "${ART_DIR}/${OUT_SUBDIR}/series.npz" \
  --fields "${ART_DIR}/${OUT_SUBDIR}/fields.npz" \
  --dns "$DNS_PATH" \
  --t-data-end "$T_DATA_END"

echo "[eval_extrap] done $(date '+%F %T')"
