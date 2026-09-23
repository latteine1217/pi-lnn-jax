#!/bin/bash
#SBATCH --job-name=jax_td1_k200
#SBATCH --output=logs/jax_td1_k200_%j.out
#SBATCH --error=logs/jax_td1_k200_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   TD-1 的第二半：K=200 兩臂的 y-vs-x。K=200 是「accepted」那一半所在
#   （thesis tab:pinn_budget_comparison，p=0.0021），K=100 的量測答不了它。
#
# Why 兩版都自己跑:
#   ksweep_k200_* 的既有 final_eval **早於 uv_rel_err 這個鍵**，拿它當 y 基準
#   會把 code 世代差混進 Δ。同一支 code、同一次 job 內跑 y 與 x，Δ 才純粹是 iterate。
#   K=100 的 y-control 一併補上——那批的 y 產於 d7a7470、x 產於 51ed1dc，
#   雖然兩側 fingerprint_verified=true（建構相同）且評估數學未變，
#   但把論證換成量測比較踏實。
#
# Why artifacts_dir 指向主 repo:
#   K=200 的 ckpt 在 ~/pi-lnn-jax；本 job 仍跑在乾淨 worktree 的 venv 內。
#   輸出寫到新的子目錄，不覆寫任何既有產物。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax-cleanab}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export UV_NO_SYNC=1        # 見 TD-27：離線 node 上的同步永遠不可能成功
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

MAIN="$HOME/pi-lnn-jax/artifacts/kolmogorov"

echo "======================================================================"
echo "[TD1-K200] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  repo: $(pwd) @ $(git rev-parse --short HEAD)  porcelain=$(git status --porcelain | wc -l)"
echo "======================================================================"

run() {   # run <config> <arch> <artifacts_dir> <iterate> <subdir>
  uv run python -u scripts/evaluate_exp245.py \
    --config "$1" --arch "$2" --artifacts_dir "$3" \
    --ckpt latest --optimizer schedule_free --base_optimizer soap \
    --soap_precondition_frequency 2 --protocol follow_training \
    --iterate "$4" --output-subdir "$5"
}

for s in 42 1 2 3 4; do
  for it in y x; do
    echo ""; echo "---------- K200 b3 s${s} iterate=${it} ----------"
    run "configs/exp_ksweep_k200_s${s}.toml" liquid "${MAIN}/ksweep_k200_s${s}" "$it" "td1_${it}"
    echo ""; echo "---------- K200 pinn s${s} iterate=${it} ----------"
    run "configs/exp_k200_pinn_s${s}.toml" pinn "${MAIN}/k200_pinn_s${s}" "$it" "td1_${it}"
  done
done

echo ""
echo "===== K=100 y-control（同一支 code 重跑 y，用來確認先前的 Δ 未被 code 世代污染）====="
for s in 42 1 2 3 4; do
  run "configs/exp_cleanab_b3_s${s}.toml"   liquid "artifacts/kolmogorov/cleanab_b3_s${s}"   y "final_eval_yctrl"
  run "configs/exp_cleanab_pinn_s${s}.toml" pinn   "artifacts/kolmogorov/cleanab_pinn_s${s}" y "final_eval_yctrl"
done

echo ""
echo "[$(date '+%F %T')] TD1-K200 完成"
