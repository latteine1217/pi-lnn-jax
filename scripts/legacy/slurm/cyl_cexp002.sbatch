#!/bin/bash
#SBATCH --job-name=jax_cyl_cexp002
#SBATCH --output=logs/cyl_cexp002_%j.out
#SBATCH --error=logs/cyl_cexp002_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# Cylinder CEXP-002（train_cylinder.py：GradNorm 4-task + SOAP/ScheduleFree + denorm + soft BC）。
#   Arm A (n_collo=64)   → parity：弱 physics，預期 ke_pred/ke_ref≈1、KE rel-err 顯著 < 41%（目標 ~3.5%）
#   Arm B (n_collo=1024) → failure：GradNorm 放大強 physics，驗「over-energy 是否 GradNorm-mediated」
# 判據用 KE / vorticity（train_cylinder.py eval 已輸出），非 field-L2。
# 資料：data/cylinder_v1.npz（dump_cylinder_v1.py 產）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在（先跑 dump_cylinder_v1.py）" >&2; exit 4; }

echo "[cyl_cexp002] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

CONFIG=configs/exp_cyl_cexp002.toml
STEPS=10000
for NC in 64 1024; do
  echo
  echo "######################## n_collo=${NC} ########################"
  uv run python -u train_cylinder.py --config "$CONFIG" --n_collo "$NC" --steps $STEPS \
    || echo "[WARN] n_collo=$NC rc=$?"
done

echo
echo "[$(date '+%F %T')] cyl_cexp002 完成"
