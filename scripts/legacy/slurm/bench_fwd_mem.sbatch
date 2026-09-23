#!/bin/bash
#SBATCH --job-name=jax_bench_fwd_mem
#SBATCH --output=logs/bench_fwd_mem_%j.out
#SBATCH --error=logs/bench_fwd_mem_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   三方 AD 模式 GPU peak-memory + step wall 對照：
#     fof forward-over-forward (現行 production, pi_lnn_jax/autodiff.py)
#     for forward-over-reverse (使用者原始研究命題)
#     ror reverse-over-reverse (舊 baseline 代表)
#   非訓練 one-off bench（skill §2.5 standalone sbatch 模式），仍 --partition=r740。
#
# Why 每組獨立 process：
#   JAX peak_bytes_in_use 是 process 累計峰值、無 reset；要公平比三變體
#   必須各跑獨立 process。下方迴圈對 matrix 逐一 `uv run python -m bench.gpu_peak_mem`。
#
# 前置（head node）：
#   cd ~/pi-lnn-jax && git pull && uv sync --python 3.12
#   （.venv 需含 jax[cuda12]；既有訓練 template 同此假設）
#
# 送出：sbatch scripts/slurm/bench_fwd_mem.sbatch
# 結果：logs/bench_fwd_mem_<jobid>.out（含對照表）+ logs/bench_fwd_mem_<jobid>.jsonl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH" >&2; exit 2; }
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

# GPU + 按需配置（量 peak-mem 必需，否則 JAX 預先吃掉 75% 顯存）
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs
RESULTS="logs/bench_fwd_mem_${SLURM_JOB_ID:-local}.jsonl"
: > "$RESULTS"

echo "======================================================================"
echo "[bench_fwd_mem] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "  results: $RESULTS"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}')"

# ── matrix：dtype × N_collo × mode × variant，各獨立 process ──
for DTYPE in f32 f64; do
  for N in 256 1024 4096; do
    for MODE in residual grad; do
      for VARIANT in fof for ror; do
        uv run python -m bench.gpu_peak_mem \
          --variant "$VARIANT" --mode "$MODE" --dtype "$DTYPE" --n-collo "$N" \
          >> "$RESULTS" \
          || echo "[WARN] 失敗: variant=$VARIANT mode=$MODE dtype=$DTYPE N=$N (rc=$?)"
      done
    done
  done
done

echo
echo "[bench_fwd_mem] 聚合對照表："
uv run python -m bench.summarize_mem "$RESULTS"

echo
echo "[$(date '+%F %T')] bench_fwd_mem 完成；raw JSONL=$RESULTS"
