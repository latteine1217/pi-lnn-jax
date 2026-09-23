#!/bin/bash
#SBATCH --job-name=jax_cyl_v3
#SBATCH --output=logs/cyl_v3_%j.out
#SBATCH --error=logs/cyl_v3_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# 半資源配置（允許兩個 v3 job 同節點並行）：
#   2 × 1  GPU = 2 ≤ 2 (acmt20) ✓
#   2 × 8  CPU = 16 ≤ 48       ✓
#   2 × 48 GB  = 96 ≤ 108      ✓
#
# v3：geo + BC + PTv2 vector attention + 16k 步 + bc_body_w=4.0
# 測「attention_kind=vector 在有 BC 的情況下是否帶來額外改善」。
# 對比 v2 geo+BC（scalar，u=0.291 v=0.684）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

echo "[cyl_v3] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

STEPS=16000
echo; echo "######## geo + BC + vector attention (v3) ########"
uv run python -u train_cylinder_v1.py \
  --mode geo --steps $STEPS \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  || echo "[WARN] v3 rc=$?"

echo
echo "[$(date '+%F %T')] cyl_v3 完成"
