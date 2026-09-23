#!/bin/bash
#SBATCH --job-name=jax_probe_perf
#SBATCH --output=logs/probe_perf_%j.out
#SBATCH --error=logs/probe_perf_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   量兩件事的實際效益（純量測，不改任何 baseline artifact）：
#     A. folx wrap stop_gradient（PILNN_WRAP_STOPGRAD=0/1）對訓練 s/step 的影響。
#     B. eval chunk 預算（PILNN_EVAL_CHUNK_BUDGET）對 eval 吞吐與 OOM 邊界的影響。
#
# Why:
#   EXP-302 實測 eval 8h10m > 訓練 6h26m；且每個 Kolmogorov run 的 stderr 都有
#   "round not in registry"（folx 退回算完整 Hessian）。兩者都可能是可回收的成本。
#
# 安全性：
#   - A 用短訓練（600 步）只讀 s/step，artifacts 寫到 /tmp，不碰正式產出。
#   - B 用既有 EXP-302 seed42 checkpoint 做 forward-only eval，且 --time-stride 20
#     （只取 ~6 frames）僅為量吞吐；**不覆寫** final_eval（輸出到 _probe 子目錄）。
#     分塊對結果 bit-identical，故此處只比時間，正確性另由全量 eval 逐位比對驗收。

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

echo "=========================================================="
echo "[probe] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HEAD=$(git rev-parse --short HEAD)"
nvidia-smi -L 2>/dev/null || true
echo "=========================================================="

CFG_302=configs/exp_302_b3_re1e6_K200_les_T50.toml

# ── A. folx wrap A/B：短訓練量 s/step ────────────────────────────────
for sg in 0 1; do
  echo ""
  echo "### [A] PILNN_WRAP_STOPGRAD=${sg} — 600 步訓練 ###"
  PILNN_WRAP_STOPGRAD=$sg \
  ARTIFACTS_DIR=/tmp/probe_folx_sg${sg} \
  uv run python -u train_kolmogorov.py \
    --config "$CFG_302" \
    --arch liquid --optimizer schedule_free --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --use_gradnorm --use_al --use_time_marching \
    --tm_start_frac 0.1 --tm_warmup_steps 2000 --tm_ramp_steps 6000 \
    --eval_every 0 --save_every 100000 --log_every 200 \
    --refine_optimizer none \
    --iterations 600 \
    --artifacts_dir /tmp/probe_folx_sg${sg} > /tmp/probe_a_sg${sg}.log 2>&1 || true
  echo "  [A] sg=${sg} 最後幾筆 step 計時："
  grep -E "^ +[0-9]+ " /tmp/probe_a_sg${sg}.log 2>/dev/null | tail -3 | sed 's/^/    /' || true
  echo "  [A] sg=${sg} folx 'round not in registry' 次數: $(grep -c 'round not in registry' /tmp/probe_a_sg${sg}.log 2>/dev/null || echo 0)"
done

# ── B. eval chunk 掃描：forward-only 吞吐與 OOM 邊界 ──────────────────
for budget in 200000 400000 800000 1600000; do
  echo ""
  echo "### [B] PILNN_EVAL_CHUNK_BUDGET=${budget} (K=200 → chunk=$((budget/200))) ###"
  start=$(date +%s)
  if PILNN_EVAL_CHUNK_BUDGET=$budget \
     uv run python -u scripts/evaluate_exp245.py \
       --protocol fixed_grid \
       --protocol-reason "效能量測（folx chunk 掃描），非指標產出；固定格點才能比不同 chunk 的耗時" \
       --config "$CFG_302" --ckpt latest \
       --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
       --time-stride 20 \
       --output-subdir final_eval_probe_b${budget} > /tmp/probe_b_${budget}.log 2>&1
  then
    end=$(date +%s)
    echo "  OK   wall=$((end-start))s   $(grep -oE '\[eval\] wall = [0-9.]+s' /tmp/probe_b_${budget}.log 2>/dev/null | tail -1 || true)"
  else
    echo "  FAIL (可能 OOM) — 見 /tmp/probe_b_${budget}.log 末尾："
    tail -5 /tmp/probe_b_${budget}.log | sed 's/^/    /'
  fi
done

echo ""
echo "[probe] $(date '+%F %T') done"
