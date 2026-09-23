#!/bin/bash
#SBATCH --job-name=jax_bench_ng
#SBATCH --output=logs/bench_ng_%j.out
#SBATCH --error=logs/bench_ng_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# JAX Natural-Gradient (Gauss-Newton) feasibility spike on GPU。
# 非訓練 one-off bench（skill §2.5）。correctness + 三方案效能（A1/A2，A3 LM 在 correctness 內 cross-check）。
# 每個 perf 配置獨立 process（乾淨量 peak-mem）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

mkdir -p logs
RESULTS="logs/bench_ng_${SLURM_JOB_ID:-local}.jsonl"
: > "$RESULTS"

echo "======================================================================"
echo "[bench_ng] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo
echo "===== correctness (analytic GN + real loss↓ + A1/A2/LM direction) ====="
uv run python -m bench.ng_spike --mode correctness || echo "[WARN] correctness rc=$?"

echo
echo "===== perf: A1 (materialized jacrev) / A2 (matrix-free CG) × N ====="
for VARIANT in A1 A2; do
  for N in 128 256 512; do
    uv run python -m bench.ng_spike --mode perf --variant "$VARIANT" --n-collo "$N" \
      >> "$RESULTS" 2>&1 \
      || echo "[WARN] FAILED variant=$VARIANT N=$N rc=$? (可能 OOM)"
  done
done

echo
echo "[bench_ng] raw=$RESULTS"
echo "=== perf JSONL ==="
grep -h '^{' "$RESULTS" 2>/dev/null || echo "(no json rows)"
echo "[$(date '+%F %T')] bench_ng 完成"
