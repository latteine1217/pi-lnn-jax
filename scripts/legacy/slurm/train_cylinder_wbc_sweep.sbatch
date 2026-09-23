#!/bin/bash
#SBATCH --job-name=jax_cyl_wbc
#SBATCH --output=logs/cyl_wbc_%j.out
#SBATCH --error=logs/cyl_wbc_%j.err
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
# w_bc 提權 sweep：基底 = v3 最佳（geo + BC + vector，無 W_O，16k）。
# 動機（POC §4.13 後續）：v4 50k 時 BC 有效貢獻僅 1%（w_bc=0.1×bc 4.7e-4 = 4.7e-5
# vs data 1e-3），BC 幾乎沒發揮。提 w_bc 測「BC 提權能否壓低 field-L2」。
# w_bc 由 env 傳入（submit 用 --export=ALL,W_BC=0.5）；其餘 flag 全部固定對齊 v3。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }
[ -f data/cylinder_v1.npz ] || { echo "[ERR] data/cylinder_v1.npz 不存在" >&2; exit 4; }

W_BC="${W_BC:?需用 --export=ALL,W_BC=<值> 傳入}"
STEPS="${STEPS:-16000}"

echo "[cyl_wbc] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} W_BC=$W_BC STEPS=$STEPS"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector，w_bc=$W_BC（其餘對齊 v3）########"
uv run python -u train_cylinder_v1.py \
  --mode geo --steps "$STEPS" \
  --attention_kind vector \
  --use_bc --w_bc "$W_BC" --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 \
  || echo "[WARN] wbc=$W_BC rc=$?"

echo
echo "[$(date '+%F %T')] cyl_wbc (W_BC=$W_BC) 完成"
