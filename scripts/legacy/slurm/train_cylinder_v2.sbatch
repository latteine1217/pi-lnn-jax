#!/bin/bash
#SBATCH --job-name=jax_cyl_v2
#SBATCH --output=logs/cyl_v2_%j.out
#SBATCH --error=logs/cyl_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Cylinder v2 A/B：geo (無 BC，= v1 geo) vs geo+BC (inflow/no-slip/slip)。
# 關鍵測試：BC losses 是否修好 v1 的「v 場 0.84 平滑懶解」問題。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

echo "[cyl_v2] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

STEPS=8000
echo; echo "######## geo (no BC, = v1 baseline-for-v2) ########"
uv run python -u train_cylinder_v1.py --mode geo --steps $STEPS --w_phys 0.01 \
  || echo "[WARN] geo-noBC rc=$?"
echo; echo "######## geo + BC (inflow/no-slip/slip) ########"
uv run python -u train_cylinder_v1.py --mode geo --steps $STEPS --w_phys 0.01 \
  --use_bc --w_bc 0.1 --bc_body_w 2.0 --bc_n 128 \
  || echo "[WARN] geo+BC rc=$?"

echo
echo "[$(date '+%F %T')] cyl_v2 完成"
