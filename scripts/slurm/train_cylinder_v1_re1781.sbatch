#!/bin/bash
#SBATCH --job-name=cyl_v1_re1781
#SBATCH --output=logs/cyl_v1_re1781_%j.out
#SBATCH --error=logs/cyl_v1_re1781_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: 後來重做的 geometry-aware cylinder baseline（scripts/legacy/train_cylinder_v1.py
#       --mode geo + BC + vector attn + ALM ρ=0.3；K-sweep 基底 v3+ALM），Re=1781，
#       在 #1 folx anisotropic + C1 nu 量綱（u_inf·D/Re, D=30mm）修正後重跑。判據 KE。
# Why : train_cylinder_v1.py 已移到 scripts/legacy/，原 train_cylinder_ksweep.sbatch 指令路徑
#       過時；此檔修正路徑。DATA 預設 K=100 wake（cylinder_re1781.npz）；boundary K400 best
#       用 DATA=data/cylinder_re1781_bnd.npz（大 K 需 DATA_CHUNKS 降記憶體）。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PILNN_FOF_CHUNK="${PILNN_FOF_CHUNK:-256}"
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

DATA="${DATA:-data/cylinder_re1781.npz}"
STEPS="${STEPS:-16000}"
N_COLLO="${N_COLLO:-512}"
DATA_CHUNKS="${DATA_CHUNKS:-1}"
SEED="${SEED:-42}"
[ -f "$DATA" ] || { echo "[ERR] $DATA 不存在" >&2; exit 4; }

echo "[cyl_v1_re1781] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} DATA=$DATA STEPS=$STEPS SEED=$SEED HEAD=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector + ALM(ρ=0.3)，DATA=$DATA（#1 folx + C1 nu 修正後）########"
uv run python -u scripts/legacy/train_cylinder_v1.py \
  --data "$DATA" --mode geo --steps "$STEPS" --seed "$SEED" \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 --n_collo "$N_COLLO" --data_chunks "$DATA_CHUNKS" \
  --use_al --al_rho 0.3 --al_freeze_start 5000 --al_freeze_end 15000 --al_update_freq 100 \
  || echo "[WARN] rc=$?"
echo; echo "[$(date '+%F %T')] cyl_v1_re1781 完成"
