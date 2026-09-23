#!/bin/bash
#SBATCH --job-name=jax_cyl_v4
#SBATCH --output=logs/cyl_v4_%j.out
#SBATCH --error=logs/cyl_v4_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# 半資源配置：2×8 CPU / 2×48 GB / 2×1 GPU = 16/96/2 ≤ 48/108/2 ✓
# v4 真 MHA（geo + BC + vector + W_O） 50k 步。
# 測 W_O 在更長訓練下能否超越 v3（16k: W_O=0.616 vs noW_O=0.594）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

echo "[cyl_v4] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector + W_O (v4, 50k steps) ########"
uv run python -u train_cylinder_v1.py \
  --mode geo \
  --steps 50000 \
  --attention_kind vector \
  --use_mha_out_proj \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  || echo "[WARN] v4 rc=$?"

echo
echo "[$(date '+%F %T')] cyl_v4 完成"
