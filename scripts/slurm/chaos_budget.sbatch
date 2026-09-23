#!/bin/bash
#SBATCH --job-name=chaos_budget
#SBATCH --output=logs/chaos_budget_%j.out
#SBATCH --error=logs/chaos_budget_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   把模型在 t₀ 的重建場交給真實求解器積到 t_end，量「起點誤差相同時，任何正確的
#   外推方式最好能到哪裡」（混沌下限），外加一條求解器保真度控制組。
#
# Why:
#   EXP-530/531 的外推在 τ≈1 之後輸給「凍住不動」。這支決定那是模型離開軌跡
#   （方法問題）還是混沌放大（物理上限）——兩者的下一步完全相反。
#
# 前提：該臂的 eval 已跑過且帶 --export-fields（需要 fields.npz）。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ARM="${ARM:-exp530_b3_extrap_T10}"
SUBDIR="${SUBDIR:-extrap_T10}"
DNS="${DNS:-data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T10_dt2p5e4_si100_ds4_cuda.npy}"
T0="${T0:-5.0}"
T_END="${T_END:-10.0}"

FIELDS="artifacts/kolmogorov/${ARM}/${SUBDIR}/fields.npz"
[ -f "$DNS" ] || { echo "[ERR] DNS 不存在：$DNS" >&2; exit 2; }
[ -f "$FIELDS" ] || { echo "[ERR] 缺 $FIELDS（eval 需帶 --export-fields）" >&2; exit 3; }

echo "[chaos_budget] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[chaos_budget] arm=$ARM  t0=$T0  t_end=$T_END"

uv run python -u scripts/diag_chaos_budget.py \
  --dns "$DNS" \
  --fields "$FIELDS" \
  --t0 "$T0" --t-end "$T_END" \
  --out-dir "artifacts/kolmogorov/${ARM}/${SUBDIR}"

echo "[chaos_budget] done $(date '+%F %T')"
