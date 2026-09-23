#!/bin/bash
#SBATCH --job-name=jax_folxfix
#SBATCH --output=logs/folxfix_%j.out
#SBATCH --error=logs/folxfix_%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# folx 修復 benchmark：tile+stack fallback（修復前 e48af58）vs _repeat3+concat（修復後 origin/main）。
# 量 EXP-245 規模 fof physics step 的 ms/step + peak-mem。各獨立 process 乾淨量峰值記憶體。

set -uo pipefail
cd "$HOME/pi-lnn-jax"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false

BEFORE=e48af58           # tile + stack（folx fallback full hessian）
git fetch origin -q
AFTER=$(git rev-parse origin/main)   # _repeat3 + concat（稀疏 jacobian）

echo "[bench_folx_fix] $(date '+%F %T') node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null | head -1
echo "before=$BEFORE  after=$AFTER"

# 建兩個 detached worktree（共用主 .venv），不干擾其他正在跑的 worktree
for pair in "before:$BEFORE" "after:$AFTER"; do
  tag=${pair%%:*}; ref=${pair##*:}
  WT="$HOME/pilnn-bench-$tag"
  git worktree remove "$WT" --force 2>/dev/null || true
  git worktree add --detach "$WT" "$ref" 2>&1 | tail -1
  ln -sfn "$HOME/pi-lnn-jax/.venv" "$WT/.venv"
done
# bench script 從 after（修復後 commit 才有）複製給兩邊用（程式碼固定，差異只在 pi_lnn_jax）
cp "$HOME/pilnn-bench-after/scripts/bench_folx.py" /tmp/bench_folx_fix.py

for tag in before after; do
  WT="$HOME/pilnn-bench-$tag"
  echo; echo "######## $tag ($(git -C "$WT" log --oneline -1)) ########"
  (cd "$WT" && PYTHONPATH=. .venv/bin/python /tmp/bench_folx_fix.py 2>&1 \
     | grep -E "backend|RESULT|Error|Traceback|not in registry" | head -8)
done

# 清理 worktree
for tag in before after; do
  git worktree remove "$HOME/pilnn-bench-$tag" --force 2>/dev/null || true
done
echo; echo "[$(date '+%F %T')] bench_folx_fix 完成"
