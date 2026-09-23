#!/bin/bash
#SBATCH --job-name=jax_adapter_ctrl
#SBATCH --output=logs/jax_adapter_ctrl_%j.out
#SBATCH --error=logs/jax_adapter_ctrl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   verify_adapter_ab.sbatch（job 5425）缺的 determinism control。
#   在 HEAD 側**同一份程式碼**再跑一次，量出 GPU 上同碼兩跑的噪音底，
#   再拿它去判讀 5425 的 BASE vs HEAD 差異。
#
# Why 必須有這一段：
#   CLAUDE.md §7.1 第 2 點——GPU 上同碼兩跑 params_digest 就不同（job 4729）。
#   沒有這個底，「BASE 與 HEAD 差 2.4e-4」無法區分「adapter 改了行為」與
#   「GPU 本來就這樣」。5425 因為漏了它，跑完沒有產生可用的判定。
#
# ── 事前登錄的判準（跑之前寫死）────────────────────────────────────────
#   令 D_ab   = |ab_premig − ab_main|（5425 已測，evaluate_exp245 最大相對差
#               約 1e-4；dump_cp_fields pred_u 2.390e-04 / pred_v 3.008e-04）
#   令 D_ctrl = |ab_main − ab_main2|（本 job 量）
#
#   (A) D_ctrl 與 D_ab 同量級（比值在 [1/10, 10]）
#       → adapter 造成的差異與 GPU 非決定性無法區分。等價成立，記錄兩個量級。
#   (B) D_ctrl 顯著小於 D_ab（低一個數量級以上）
#       → 差異不是噪音。adapter 改變了行為，FAIL，回去查。
#   (C) D_ctrl ≈ 0（GPU 其實可重現）
#       → 5425 的差異全部是真的。FAIL。
#
#   先寫死判準再看數字：事後挑一個能讓結果好看的門檻，等於沒有判準。
#
# 用法（head node）：sbatch scripts/slurm/verify_adapter_control.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

HEAD_WT="$HOME/pi-lnn-jax"
ART="$HOME/pi-lnn-jax/artifacts/kolmogorov"
CFG="$HOME/pi-lnn-jax/configs/exp_ksweep_k200_s42.toml"
EV="$HOME/pi-lnn-jax/knowledge/superpowers/evidence"

export PYTHONUNBUFFERED=1
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

HEAD_SHA="$(git -C "$HEAD_WT" rev-parse --short HEAD)"
echo "======================================================================"
echo "[adapter-ctrl] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "  HEAD = $HEAD_WT @ $HEAD_SHA  （與 5425 的 HEAD 側同一份程式碼）"
echo "======================================================================"

# 5425 的產物必須還在，否則沒有東西可比
for f in "$ART/ksweep_k200_s42/ab_main/metric_artifact.json" \
         "$ART/ksweep_k200_s42/ab_premig/metric_artifact.json" \
         "$HOME/pi-lnn-jax/artifacts/cp/cp_ab_main.npz"; do
  [ -f "$f" ] || { echo "[ERR] 缺 5425 的產物：$f" >&2; exit 4; }
done

echo ""
echo "--- [main2] evaluate_exp245（同碼第二跑）---"
PYTHONPATH=. uv run python -u scripts/evaluate_exp245.py \
  --protocol follow_training --config "$CFG" --ckpt latest --arch liquid \
  --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
  --artifacts_dir "$ART/ksweep_k200_s42" --output-subdir ab_main2 --no-plot \
  || { echo "[ERR] main2 evaluate 失敗" >&2; exit 1; }

echo ""
echo "--- [main2] dump_cp_fields（同碼第二跑）---"
PYTHONPATH=. uv run python -u scripts/dump_cp_fields.py \
  --config "$CFG" --ckpt 20000 \
  --artifacts_dir "$ART/ksweep_k200_s42" \
  --out "$HOME/pi-lnn-jax/artifacts/cp/cp_ab_main2.npz" \
  || { echo "[ERR] main2 dump 失敗" >&2; exit 1; }

echo ""
echo "=== determinism control 判讀 ==="
PYTHONPATH=. uv run python - <<PY
import json, os, numpy as np, sys
sys.path.insert(0, "scripts")
from _common.ab_compare import metrics_projection_diff

A = "$ART/ksweep_k200_s42"
def load(p): return json.load(open(p))

# ---- (i) evaluate_exp245：噪音底 vs 待判差異，用同一支比對器 ----
ctrl = metrics_projection_diff(load(f"{A}/ab_main/metric_artifact.json"),
                               load(f"{A}/ab_main2/metric_artifact.json"))
ab   = metrics_projection_diff(load(f"{A}/ab_premig/metric_artifact.json"),
                               load(f"{A}/ab_main/metric_artifact.json"))
print(f"  evaluate_exp245  同碼兩跑不一致鍵數 = {len(ctrl)}")
print(f"  evaluate_exp245  BASE vs HEAD  不一致鍵數 = {len(ab)}")

# ---- (ii) dump_cp_fields：逐陣列的兩個量級 ----
b  = np.load("$HOME/pi-lnn-jax/artifacts/cp/cp_ab_premig.npz")
h1 = np.load("$HOME/pi-lnn-jax/artifacts/cp/cp_ab_main.npz")
h2 = np.load("$HOME/pi-lnn-jax/artifacts/cp/cp_ab_main2.npz")
rows, verdicts = [], []
for k in sorted(set(b.files) & set(h1.files) & set(h2.files)):
    if b[k].dtype.kind not in "fc":
        continue
    d_ab   = float(np.nanmax(np.abs(b[k].astype(np.float64)  - h1[k].astype(np.float64))))
    d_ctrl = float(np.nanmax(np.abs(h1[k].astype(np.float64) - h2[k].astype(np.float64))))
    if d_ab == 0.0 and d_ctrl == 0.0:
        v = "兩者皆逐位元相同"
    elif d_ctrl == 0.0:
        v = "(C) 同碼可重現而 BASE/HEAD 不同 → 真差異"
    else:
        r = d_ab / d_ctrl
        v = ("(A) 同量級，與噪音無法區分" if 0.1 <= r <= 10
             else "(B) D_ab 明顯較大 → 真差異" if r > 10
             else "D_ab 反而較小（噪音主導）")
    rows.append({"key": k, "d_ab": d_ab, "d_ctrl": d_ctrl, "verdict": v})
    verdicts.append(v)
    print(f"  {k:<22s} D_ab={d_ab:.3e}  D_ctrl={d_ctrl:.3e}  → {v}")

real = [r for r in rows if r["verdict"].startswith(("(B)", "(C)"))]
ok = not real
json.dump({"job_id": os.environ.get("SLURM_JOB_ID",""), "head_sha": "$HEAD_SHA",
           "eval245_ctrl_mismatch_keys": len(ctrl),
           "eval245_ab_mismatch_keys": len(ab),
           "dump_cp_rows": rows,
           "real_differences": real, "ok": ok},
          open("$EV/adapter_ctrl_${SLURM_JOB_ID:-manual}.json", "w"),
          indent=2, ensure_ascii=False)
print("")
print("  === 全部落在噪音底內：adapter 等價成立 ===" if ok
      else "  *** 有超出噪音底的差異，見上表 ***")
PY

echo ""
echo "[adapter-ctrl] $(date '+%F %T') 結束"
