#!/bin/bash
#SBATCH --job-name=jax_cyl_notm
#SBATCH --output=logs/cyl_notm_%j.out
#SBATCH --error=logs/cyl_notm_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# Cylinder isolation：FourierEmbs 修復 + 關 time_marching（排除 confound）+ over-energy 空間定位診斷。
#   只跑 parity 臂 collo=64。對照：FourierEmbs+time_marching=61.6%(4130)、無FourierEmbs無tm=38.2%。
#   eval 新增 freestream vs wake KE 分區 + inflow band u（查 over-energy 在哪、BC 是否釘住 freestream）。
# 資料：data/cylinder_v1.npz。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

echo "[cyl_notm] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

uv run python -u train_cylinder.py --config configs/exp_cyl_cexp002_notm.toml --n_collo 64 --steps 10000

echo
echo "[$(date '+%F %T')] cyl_notm 完成"
