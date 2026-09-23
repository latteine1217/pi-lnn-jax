#!/bin/bash
#SBATCH --job-name=jax_forcing_readout
#SBATCH --output=logs/jax_forcing_readout_%j.out
#SBATCH --error=logs/jax_forcing_readout_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   讀出 forcing 三臂訓練完的 learned (A, k_f)，回答 forcing 是否可辨識。
#
# Why GPU node:
#   ckpt 由 GPU 寫出，head node 的 CPU JAX 上 orbax 推不出 sharding。計算本身
#   只有一次 model.apply，秒級。
#
# Why --truth-* 寫死在這裡而非腳本預設:
#   真值是 DNS 的性質，不是腳本的性質。腳本端必填是刻意的——猜錯會把「塌回初值」
#   誤讀成「收斂到真值」。此處 A=0.1 / k_f=2.0 取自 ForcingPrior 的 dataclass
#   預設，主基線 exp_main5_s42 未覆寫它們，即以此真值訓練並達到 ~5.7% KE。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "======================================================================"
echo "[FORCING-READOUT] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  truth: A=0.1  k_f=2.0"
echo "======================================================================"

for arm in A kf both; do
  echo ""
  echo "---------- arm: ${arm} ----------"
  uv run python -u scripts/diag_forcing_recovery.py \
    --config "configs/exp_forcing_${arm}.toml" \
    --arch liquid \
    --truth-A 0.1 \
    --truth-kf 2.0 \
    --output "artifacts/diag/forcing_recovery_${arm}.json"
done

echo ""
echo "[$(date '+%F %T')] FORCING-READOUT 完成"
