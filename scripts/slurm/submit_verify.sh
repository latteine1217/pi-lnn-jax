#!/usr/bin/env bash
# scripts/slurm/submit_verify.sh — bit-identical 驗收 job 的提交入口（lab-server head node）
#
# What: 依 case 查出拓撲（HEAD/BASE worktree、基準分支、template、config），
#       做完前置（fetch、checkout、清掉會擋住 checkout 的未追蹤證據檔），
#       解析出 EXPECT_BASE / EXPECT_HEAD，再交給 submit_exp.sh 送出。
#
# Why 要這一層: 規則寫在 CLAUDE.md §7.1 還不夠——手組提交指令在 2026-08-01
#       那一輪實際犯了三次錯：main 前進了但 EXPECT_HEAD 還填舊 sha（job 4779/4780
#       送出後得取消）、忘了先清未追蹤的證據檔（checkout 失敗兩次）、
#       每次都要重查哪個 case 配哪個 worktree。
#       **手打的 sha 尤其糟**：EXPECT_BASE 是斷言的依據，手打等於拿記憶當真實來源。
#
# 承重的設計點：EXPECT_BASE 取自**基準分支的 ref**（origin/verify/…），
#       不是取自 BASE worktree 自己的 HEAD。那個斷言存在的目的正是抓
#       「BASE_WT 指向錯的樹」；若從該 worktree 自己讀出來再拿去比它自己，
#       斷言就變成恆真。分支是宣告，worktree 是待驗之物——兩者不可混用。
#       `tests/test_submit_verify.py` 釘住這一點。
#
# Usage（lab-server head node）:
#   scripts/slurm/submit_verify.sh kolmo            # Kolmogorov fresh-run 對拍
#   scripts/slurm/submit_verify.sh cyl              # cylinder fresh-run 對拍
#   scripts/slurm/submit_verify.sh ckpt             # 舊 ckpt 相容（驗收層 B）
#   REF=origin/some-branch scripts/slurm/submit_verify.sh kolmo   # 測別的 ref
#   DRY=1 scripts/slurm/submit_verify.sh kolmo      # 只印計畫，不送出
#
# 從本機用：ssh lab-server 'scripts/slurm/submit_verify.sh kolmo'（先確認已 push）

set -euo pipefail

CASE="${1:?usage: $0 <kolmo|cyl|ckpt> [EXP_ID]}"
EXP_ID="${2:-}"
REF="${REF:-origin/main}"
ROOT="${PILNJAX_REPO:-$HOME/pi-lnn-jax}"

# ─── case 拓撲：唯一一份宣告 ─────────────────────────────────────────────
# 欄位：HEAD worktree | BASE worktree | 基準分支 | template | config | 預設 EXP_ID
case "$CASE" in
  kolmo)
    HEAD_WT="$HOME/pi-lnn-jax-ledger"; BASE_WT="$HOME/pi-lnn-jax-base"
    BASE_BRANCH="verify/base-28b007a"
    TMPL="scripts/slurm/verify_cpu_ab.sbatch.tmpl"
    CFG="configs/_ledger_single_re.toml"; : "${EXP_ID:=cpuab}" ;;
  cyl)
    HEAD_WT="$HOME/pi-lnn-jax-cyl"; BASE_WT="$HOME/pi-lnn-jax-basecyl"
    BASE_BRANCH="verify/base-cyl-f2d63ff"
    TMPL="scripts/slurm/verify_cyl_cpu_ab.sbatch.tmpl"
    CFG="configs/exp_cyl_cexp002_notm.toml"; : "${EXP_ID:=cylab}" ;;
  ckpt)
    HEAD_WT="$HOME/pi-lnn-jax-ledger"; BASE_WT="$HOME/pi-lnn-jax-base"
    BASE_BRANCH="verify/base-28b007a"
    TMPL="scripts/slurm/verify_ckpt_compat.sbatch.tmpl"
    CFG="configs/_ledger_single_re.toml"; : "${EXP_ID:=ckptab}" ;;
  *) echo "[ERR] 未知 case：$CASE（可用：kolmo / cyl / ckpt）" >&2; exit 1 ;;
esac

[ -d "$ROOT/.git" ] || { echo "[ERR] repo 不存在：$ROOT" >&2; exit 2; }
for wt in "$HEAD_WT" "$BASE_WT"; do
  [ -d "$wt" ] || { echo "[ERR] worktree 不存在：$wt" >&2; exit 3; }
done

# ─── 前置：同步 ref、清掉會擋住 checkout 的未追蹤產物 ─────────────────────
git -C "$ROOT" fetch origin --quiet

# 證據檔一旦被提交進 main，worktree 內的未追蹤同名副本就會擋住 checkout。
# 只刪這個目錄下的 *.json——它們是可從 job log 重生的產物，且已提交者在 repo 內。
EV="$HEAD_WT/knowledge/superpowers/evidence"
case "$EV" in
  */knowledge/superpowers/evidence) rm -f "$EV"/*.json 2>/dev/null || true ;;
  *) echo "[ERR] 拒絕清理非預期路徑：$EV" >&2; exit 9 ;;
esac

git -C "$HEAD_WT" checkout --detach "$REF" --quiet

# ─── 解析兩個 sha ────────────────────────────────────────────────────────
# EXPECT_BASE 取自**分支 ref**（宣告），不是 BASE worktree 的 HEAD（待驗之物）。
EXPECT_BASE="$(git -C "$ROOT" rev-parse --short "origin/$BASE_BRANCH")"
EXPECT_HEAD="$(git -C "$HEAD_WT" rev-parse --short HEAD)"
BASE_ACTUAL="$(git -C "$BASE_WT" rev-parse --short HEAD)"

echo "======================================================================"
echo "[submit_verify] case=$CASE  EXP_ID=$EXP_ID"
echo "  HEAD worktree : $HEAD_WT  @ $EXPECT_HEAD  ($REF)"
echo "  BASE worktree : $BASE_WT  @ $BASE_ACTUAL"
echo "  基準分支      : $BASE_BRANCH @ $EXPECT_BASE  ← EXPECT_BASE 的來源"
echo "  template      : $TMPL"
echo "  config        : $CFG"
if [ "$BASE_ACTUAL" != "$EXPECT_BASE" ]; then
  echo "  [WARN] BASE worktree 落後於基準分支——job 內的 commit 斷言會擋下它。"
  echo "         先在該 worktree 跑：git merge --ff-only origin/$BASE_BRANCH"
fi
echo "======================================================================"

[ -n "${DRY:-}" ] && { echo "[DRY] 未送出。"; exit 0; }

cd "$HEAD_WT"
BASE_WT="$BASE_WT" \
PILNJAX_ROOT="${PILNJAX_ROOT:-$HOME/pi-lnn-jax-multire}" \
PILNN_ROOT="${PILNN_ROOT:-$HOME/pi-lnn-jax-main}" \
PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}" \
CYL_DATA_SRC="${CYL_DATA_SRC:-$HOME/pi-lnn-jax/data}" \
EXPECT_BASE="$EXPECT_BASE" EXPECT_HEAD="$EXPECT_HEAD" \
TMPL="$TMPL" \
  scripts/slurm/submit_exp.sh "$EXP_ID" "$CFG"
