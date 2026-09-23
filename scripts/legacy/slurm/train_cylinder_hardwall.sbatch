#!/bin/bash
#SBATCH --job-name=jax_cyl_hw
#SBATCH --output=logs/cyl_hw_%j.out
#SBATCH --error=logs/cyl_hw_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# 半資源配置：2×8 CPU / 2×48 GB / 2×1 GPU = 16/96/2 ≤ 48/108/2 ✓
#
# Hard-wall BC A/B：基底=v3（geo+BC+vector，無 W_O，16k）。
#   HARD=0：soft slip loss（= v3，對照，已知 v≈0.616）
#   HARD=1：slip 改 hard ansatz v=φ(y)·v_net，強制 wall v=0，停用 slip soft loss
# DNS evidence：上下為 slip wall（u≠0,v≈0.025）；hard ansatz 強制 slip 理論 BC v=0。
# HARD 由 env 傳（submit 用 --export=ALL,HARD=1）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

HARD="${HARD:-0}"
STEPS="${STEPS:-16000}"
HW_FLAG=""
[ "$HARD" = "1" ] && HW_FLAG="--hard_wall"

echo "[cyl_hw] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HARD=$HARD STEPS=$STEPS"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector，HARD=$HARD（其餘對齊 v3）########"
uv run python -u train_cylinder_v1.py \
  --mode geo --steps "$STEPS" \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  $HW_FLAG \
  || echo "[WARN] HARD=$HARD rc=$?"

echo
echo "[$(date '+%F %T')] cyl_hw (HARD=$HARD) 完成"
