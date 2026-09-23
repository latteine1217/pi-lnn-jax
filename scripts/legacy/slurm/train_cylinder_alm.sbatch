#!/bin/bash
#SBATCH --job-name=jax_cyl_alm
#SBATCH --output=logs/cyl_alm_%j.out
#SBATCH --error=logs/cyl_alm_%j.err
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
# ALM continuity A/B：基底=v3（geo+BC+vector，soft slip，16k）。
#   AL=0：continuity 等權併入 phys（= v3 對照，已知 KE 0.539 / div L2 720）
#   AL=1：continuity 走 ALM（λ·C+ρ/2·C²）。ρ=1，λ 前 5000 凍結（純 penalty 暖機）、
#         5000–15000 每 100 步 dual ascent、15000 後凍結收斂。
# 動機：div L2=720 = continuity 嚴重違反（現權重僅 w_phys×1/3≈0.003）；ALM λ 自適應強制 ∇·u→0。
# AL 由 env 傳（submit 用 --export=ALL,AL=1）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

AL="${AL:-0}"
STEPS="${STEPS:-16000}"
RHO="${RHO:-1.0}"      # ALM penalty ρ（sweep 用 --export=ALL,AL=1,RHO=0.3）
AL_FLAGS=""
[ "$AL" = "1" ] && AL_FLAGS="--use_al --al_rho $RHO --al_freeze_start 5000 --al_freeze_end 15000 --al_update_freq 100"

echo "[cyl_alm] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} AL=$AL RHO=$RHO STEPS=$STEPS"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector，AL=$AL（其餘對齊 v3）########"
uv run python -u train_cylinder_v1.py \
  --mode geo --steps "$STEPS" \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  $AL_FLAGS \
  || echo "[WARN] AL=$AL rc=$?"

echo
echo "[$(date '+%F %T')] cyl_alm (AL=$AL) 完成"
