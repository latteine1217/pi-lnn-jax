#!/usr/bin/env bash
# scripts/slurm/submit_exp.sh
#
# What:
#   把 train_exp.sbatch.tmpl 中的 {EXP_ID} / {CONFIG_PATH} sed 替換後 sbatch 送出。
#   對齊 pi-lnn 的 submit_exp.sh，差異只在預設搜尋路徑（configs/exp_<ID>_*.toml 同樣慣例）。
#
# Why:
#   #SBATCH 行不展開 shell 變數，必須事先用 sed 替換 placeholder 才能讓
#   job-name / log file 帶 EXP_ID。
#
# Usage:
#   scripts/slurm/submit_exp.sh <EXP_ID> [CONFIG_PATH]
#     <EXP_ID>      實驗編號（純 ID，如 245；slurm job-name 變成 jax_exp_245）
#     [CONFIG_PATH] 不給則自動 glob configs/exp_<EXP_ID>_*.toml
#
# Example:
#   scripts/slurm/submit_exp.sh 245
#   scripts/slurm/submit_exp.sh 245 configs/exp_245_b3_les_T50.toml
#   DRY=1 scripts/slurm/submit_exp.sh 245   # 只生成 sbatch，不送出
#
# Output:
#   - 生成的 sbatch 落在 logs/sbatch/jax_train_exp_<ID>_<ts>.sbatch (gitignored)
#   - slurm stdout/stderr：logs/jax_exp_<ID>_<jobid>.{out,err}

set -euo pipefail

EXP_ID="${1:?usage: $0 <EXP_ID> [CONFIG_PATH]}"
CONFIG="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# TMPL 可由環境覆寫（如 multi-Re：TMPL=scripts/slurm/train_exp_multire.sbatch.tmpl）；預設標準 template
TMPL="${TMPL:-scripts/slurm/train_exp.sbatch.tmpl}"
[ -f "$TMPL" ] || { echo "[ERR] template 不存在：$TMPL" >&2; exit 1; }

# 自動找 config（若未提供）
if [ -z "$CONFIG" ]; then
  shopt -s nullglob
  cand=( configs/exp_${EXP_ID}_*.toml )
  shopt -u nullglob
  case "${#cand[@]}" in
    0)
      echo "[ERR] 找不到 configs/exp_${EXP_ID}_*.toml；請以第二個參數明示 config 路徑" >&2
      exit 1
      ;;
    1)
      CONFIG="${cand[0]}"
      ;;
    *)
      echo "[ERR] 對 EXP_ID=${EXP_ID} 找到多個 config，請明示：" >&2
      printf '  %s\n' "${cand[@]}" >&2
      exit 1
      ;;
  esac
fi

[ -f "$CONFIG" ] || { echo "[ERR] config 不存在：$CONFIG" >&2; exit 1; }

# logs/ 必須先存在（slurm 不會幫忙建）
mkdir -p logs logs/sbatch

ts=$(date +%Y%m%d_%H%M%S)
SBATCH_FILE="logs/sbatch/jax_train_exp_${EXP_ID}_${ts}.sbatch"
sed -e "s|{EXP_ID}|${EXP_ID}|g" \
    -e "s|{CONFIG_PATH}|${CONFIG}|g" \
    "$TMPL" > "$SBATCH_FILE"
chmod +x "$SBATCH_FILE"

echo "[submit_exp] EXP_ID    = ${EXP_ID}"
echo "[submit_exp] CONFIG    = ${CONFIG}"
echo "[submit_exp] sbatch    = ${SBATCH_FILE}"
echo "[submit_exp] log out   = logs/jax_exp_${EXP_ID}_<jobid>.out"
echo "[submit_exp] log err   = logs/jax_exp_${EXP_ID}_<jobid>.err"
echo

# Dry-run 模式：DRY=1 scripts/slurm/submit_exp.sh ...
if [ "${DRY:-0}" = "1" ]; then
  echo "[submit_exp] DRY=1；只生成 sbatch 不送出"
  echo "--- generated sbatch ---"
  cat "$SBATCH_FILE"
  exit 0
fi

sbatch "$SBATCH_FILE"
