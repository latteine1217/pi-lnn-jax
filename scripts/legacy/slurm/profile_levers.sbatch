#!/bin/bash
#SBATCH --job-name=jax_levers
#SBATCH --output=logs/levers_%j.out
#SBATCH --error=logs/levers_%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# 優化槓桿天花板：SOAP freq sweep + sensor mini-batch（穩態 ms/step，正確推進 opt_state）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "[levers] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo
echo "===== 槓桿天花板測量 ====="
# 基準（無 SOAP）
uv run python -m bench.profile_levers --optimizer adam                              || echo "[WARN] adam"
# SOAP freq sweep（full sensor）
uv run python -m bench.profile_levers --optimizer soap --soap-freq 2                || echo "[WARN] soap f2"
uv run python -m bench.profile_levers --optimizer soap --soap-freq 10               || echo "[WARN] soap f10"
uv run python -m bench.profile_levers --optimizer soap --soap-freq 1000000          || echo "[WARN] soap finf"
# sensor mini-batch 疊加（freq=10）
uv run python -m bench.profile_levers --optimizer soap --soap-freq 10 --sensor-batch 2000 || echo "[WARN] soap f10 sb2000"
# sensor mini-batch 對 adam（隔離 backward 省幅，不含 SOAP）
uv run python -m bench.profile_levers --optimizer adam --sensor-batch 2000          || echo "[WARN] adam sb2000"

echo
echo "[$(date '+%F %T')] levers 完成"
