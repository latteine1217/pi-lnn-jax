#!/bin/bash
#SBATCH --job-name=jax_cyl_body
#SBATCH --output=logs/cyl_body_%j.out
#SBATCH --error=logs/cyl_body_%j.err
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
# body 幾何處理 A/B/C：基底=v3+ALM(ρ=0.3)（KE 主判據最佳 0.420）。
#   BODY=soft：body no-slip soft loss（= 對照，4143 KE 0.420）
#   BODY=hard：body hard gate u,v=clip(φ/scale,0,1)·net（pi-lnn Sukumar 2022）
#   BODY=brinkman：沉浸邊界 χ·u/η penalization（body 內外連續，移除 mask）
# 動機：body 用 soft point penalty 最弱，卻是物理鐵律（固體 no-slip）。測 hard/penaliz 是否改善。
# BODY 由 env 傳（submit 用 --export=ALL,BODY=hard）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

BODY="${BODY:-soft}"
STEPS="${STEPS:-16000}"
case "$BODY" in
  soft)     BODY_FLAGS="" ;;
  hard)     BODY_FLAGS="--body_hard --body_hard_scale 0.1" ;;
  brinkman) BODY_FLAGS="--brinkman --brinkman_eta 1e-2" ;;
  *) echo "[ERR] BODY 須為 soft/hard/brinkman" >&2; exit 5 ;;
esac
# 基底 = v3 + ALM(ρ=0.3)
AL_FLAGS="--use_al --al_rho 0.3 --al_freeze_start 5000 --al_freeze_end 15000 --al_update_freq 100"

echo "[cyl_body] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} BODY=$BODY STEPS=$STEPS"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector + ALM(ρ=0.3)，BODY=$BODY ########"
uv run python -u train_cylinder_v1.py \
  --mode geo --steps "$STEPS" \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  $AL_FLAGS $BODY_FLAGS \
  || echo "[WARN] BODY=$BODY rc=$?"

echo
echo "[$(date '+%F %T')] cyl_body (BODY=$BODY) 完成"
