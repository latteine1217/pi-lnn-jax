#!/bin/bash
#SBATCH --job-name=ctrl_cyl
#SBATCH --output=logs/ctrl_cyl_%j.out
#SBATCH --error=logs/ctrl_cyl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: config 驅動 train_cylinder.py 的 controlled_cylinder case（moving no-slip + physics ×0.3）。
# Why : RealPDEBench controlled cylinder（受迫振動 bluff body）sparse-sensor + PDE-residual 重建，
#       論文方向 B 光譜中間點。半資源 8CPU/1GPU/48G；--time 12h 保證跑完。判據 KE。
#       前置：先跑 dump_controlled_cyl.sbatch 產出 data/controlled_v2.npz。
#
# Env override:
#   CONFIG  預設 configs/exp_controlled_cyl_v2.toml
#   STEPS   覆蓋 iterations（smoke gate 用，如 STEPS=200）
#   SEED    覆蓋 seed（multi-seed ablation 用，如 SEED=1）
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PILNN_FOF_CHUNK="${PILNN_FOF_CHUNK:-256}"
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

CONFIG="${CONFIG:-configs/exp_controlled_cyl_v2.toml}"
[ -f "$CONFIG" ] || { echo "[ERR] $CONFIG 不存在" >&2; exit 4; }
STEPS_ARG=""
[ -n "${STEPS:-}" ] && STEPS_ARG="--steps ${STEPS}"
SEED_ARG=""
[ -n "${SEED:-}" ] && SEED_ARG="--seed ${SEED}"

echo "[ctrl_cyl] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} CONFIG=$CONFIG STEPS=${STEPS:-<config>} SEED=${SEED:-<config>} HEAD=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## train_cylinder.py controlled_cylinder ########"
uv run python -u train_cylinder.py --config "$CONFIG" ${STEPS_ARG} ${SEED_ARG} || echo "[WARN] rc=$?"
echo; echo "[$(date '+%F %T')] ctrl_cyl 完成"
