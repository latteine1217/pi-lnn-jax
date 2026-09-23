#!/bin/bash
#SBATCH --job-name=jax_td1_xiter
#SBATCH --output=logs/jax_td1_xiter_%j.out
#SBATCH --error=logs/jax_td1_xiter_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   TD-1 的量測：同一批 ckpt 分別以 y-iterate（既有 final_eval）與 x-iterate
#   （Polyak 平均）評估，量 headline 指標的差。
#
# Why 這批 ckpt:
#   TD-1 的 recipe 說「任一 exp245_b3 ckpt」，但它的判準是以 σ_training(0.18pp)
#   為單位寫死的——單點量不出「是否小於 seed 散布」。改用 cleanab_b3_s*
#   五個 seed：provenance 乾淨（code_dirty=false、同一 revision d7a7470），
#   且 y 版 final_eval 已存在，直接可比。
#
# Why GPU node:
#   ckpt 由 GPU 寫出，head node 的 CPU JAX 上 orbax 推不出 sharding。
#
# Protocol:
#   follow_training —— 與產出 y 版 final_eval 的那次一致。兩側協定不同就白量。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax-cleanab}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
# compute node 離線且 .venv 已備妥，`uv run` 的預設同步永遠不可能成功——
# 只要有人新增依賴（如 c295e60 的 pytest-xdist），每個 job 就會在同步階段掛掉
# 而不是在自己的邏輯上。實測 job 5507 即為此。
export UV_NO_SYNC=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

echo "======================================================================"
echo "[TD1-XITER] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  repo: $(pwd)  @ $(git rev-parse --short HEAD)  porcelain=$(git status --porcelain | wc -l)"
echo "======================================================================"

# 臂可切換：y→x 的偏移若是**系統性且各臂相同**，臂間比較不受影響；
# 只量一個臂答不了這件事，而稿上 CfC 在積分 KE 的貢獻僅 0.15pp，與偏移同級。
ARM="${TD1_ARM:-b3}"
ARCH="${TD1_ARCH:-liquid}"
echo "  arm=${ARM}  arch=${ARCH}"

for s in 42 1 2 3 4; do
  echo ""
  echo "---------- ${ARM} seed ${s} ----------"
  uv run python -u scripts/evaluate_exp245.py \
    --config "configs/exp_cleanab_${ARM}_s${s}.toml" \
    --ckpt latest \
    --arch "${ARCH}" \
    --optimizer schedule_free \
    --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --protocol follow_training \
    --iterate x \
    --output-subdir final_eval_xiter
done

echo ""
echo "[$(date '+%F %T')] TD1-XITER 完成"
