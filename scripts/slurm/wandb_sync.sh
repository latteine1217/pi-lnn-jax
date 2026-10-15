#!/usr/bin/env bash
# 上傳 W&B offline run —— 在 lab-server **head node** 跑。
#
# Why: compute node 沒有對外 DNS（job 6029 實測 api.wandb.ai / pypi 皆解析逾時），
#   訓練時 pi_lnn_jax/tracking.py 預設 offline，把 run 寫進
#   <artifacts_dir>/wandb/offline-run-*（Kolmogorov）或 $WANDB_DIR/wandb/（cylinder）。
#   head node 連得到 W&B，由這裡上傳。認證用 head node 的 ~/.netrc（wandb login）。
#
# 用法（在要上傳的 worktree 根目錄）：
#   scripts/slurm/wandb_sync.sh                         # 上傳 artifacts/ 下全部 offline run
#   scripts/slurm/wandb_sync.sh artifacts/exp_x         # 只上傳指定目錄
#   LIVE=1 scripts/slurm/wandb_sync.sh artifacts/exp_x  # 訓練中邊寫邊傳，run 結束才返回
#                                                       # （建議放 tmux；job 異常死亡時會一直等，Ctrl-C 即可）
#
# 正式評估（evaluate_exp245 --wandb）是同一個 run 的第二個片段，在訓練結束後才出現——
# LIVE 跟完訓練後，job 結束時再跑一次（不帶 LIVE）。已上傳的片段重傳不會產生重複列
# （2026-09-28 實測）。
#
# 用 `wandb beta sync`：舊版 `wandb sync` 讀到仍在寫入的 .wandb 會報
# "transactionlog: error getting next record: EOF" 而中止；beta 版支援 --live。
# 用 `uv run --no-sync`：裸 `uv run` 會先 sync 環境，可能動到 out-of-band 的 CUDA wheels
# （見 CLAUDE.md §6）。
set -euo pipefail
# 非互動 ssh 不讀 ~/.zshrc/.bashrc，uv 不在 PATH（同 sbatch template 的處理）
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

roots=("$@")
[ ${#roots[@]} -gt 0 ] || roots=(artifacts)

runs=()
while IFS= read -r d; do runs+=("$d"); done < <(
  find "${roots[@]}" -type d -name 'offline-run-*' 2>/dev/null | sort)
if [ ${#runs[@]} -eq 0 ]; then
  echo "[wandb_sync] ${roots[*]} 下找不到 offline-run-*" >&2
  exit 1
fi

live=()
[ -n "${LIVE:-}" ] && live=(--live)
echo "[wandb_sync] $(date '+%F %T') 上傳 ${#runs[@]} 個 run${LIVE:+（live）}"
uv run --no-sync wandb beta sync "${live[@]}" "${runs[@]}"
