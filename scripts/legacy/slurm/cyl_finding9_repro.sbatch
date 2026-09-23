#!/bin/bash
#SBATCH --job-name=jax_cyl_f9
#SBATCH --output=logs/cyl_f9_%j.out
#SBATCH --error=logs/cyl_f9_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# Finding #9 對比複現（baseline path，非 geo）：兩臂只差 collocation 密度。
#   Arm A (n_collo=64)   → parity：弱 physics，預期 ke_pred/ke_ref≈1、KE rel-err 低
#   Arm B (n_collo=1024) → failure：強 physics（對應 CEXP-030），預期 over-energy（ke ratio≫1）
# 判據用 KE / vorticity（train_cylinder_v1.py eval 已輸出），非 field-L2。
# 偏離說明：harness 為 Adam + 固定權重（非 CEXP-002 的 SOAP+GradNorm），確認的是「機制」非字面數字。
# 資料：data/cylinder_v1.npz（dump_cylinder_v1.py 產）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在（先跑 dump_cylinder_v1.py）" >&2; exit 4; }

echo "[cyl_f9] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

STEPS=10000   # 對齊 CEXP-002 的 10k iter
for NC in 64 1024; do
  echo
  echo "######################## baseline n_collo=${NC} ########################"
  uv run python -u train_cylinder_v1.py --mode baseline --use_bc \
    --n_collo "$NC" --w_phys 0.01 --steps $STEPS \
    || echo "[WARN] n_collo=$NC rc=$?"
done

echo
echo "[$(date '+%F %T')] cyl_f9 完成"
