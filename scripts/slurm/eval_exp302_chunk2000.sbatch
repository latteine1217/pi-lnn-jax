#!/bin/bash
#SBATCH --job-name=jax_eval302_c2k
#SBATCH --output=logs/eval_exp302_c2000_%j.out
#SBATCH --error=logs/eval_exp302_c2000_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   1. 驗收 gate：對 seed42 用 PILNN_EVAL_CHUNK_BUDGET=400000 (chunk=2000) 重跑 eval 到獨立
#      子目錄 final_eval_c2000，與既有 final_eval（chunk=1000 金標準）逐值比對。
#   2. 僅在 gate PASS 時，補跑 s1/s2 的 eval（同 chunk=2000）到各自 final_eval（原本無）。
#
# Why:
#   chunk 加大是純記憶體換算、演算法逐點獨立 → 結果應與 chunk=1000 一致（分塊 bit-identical
#   claim）。但 XLA 對不同 shape 可能編不同 kernel → 容許 ULP 級差異。gate 判定：
#   相對差 < 1e-9 = bit-identical；< 1e-6 = ULP 可接受；>= 1e-6 = 結構性 bug，停手。
#
# 對齊金標準（eval-script-discipline）：
#   --time-stride 1（seed42 金標準即以 EVAL_TIME_STRIDE=1 產出，T_eval=101）；
#   optimizer flags 對齊訓練（orbax restore 嚴格）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[eval302-c2k] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} HEAD=$(git rev-parse --short HEAD)"

BASE=artifacts/kolmogorov/exp302_b3_re1e6_K200_les_T50_jax_50k
CFG42=configs/exp_302_b3_re1e6_K200_les_T50.toml
COMMON_EVAL="--ckpt latest --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 --time-stride 1"

# ── 1. 驗收：seed42 chunk=2000 → final_eval_c2000 ────────────────────
echo ""
echo "=== [GATE] seed42 chunk=2000 重跑 ==="
t0=$(date +%s)
PILNN_EVAL_CHUNK_BUDGET=400000 \
uv run python -u scripts/evaluate_exp245.py --config "$CFG42" $COMMON_EVAL \
  --protocol follow_training \
  --output-subdir final_eval_c2000
rc=$?
echo "  seed42 c2000 eval rc=${rc}  wall=$(( $(date +%s) - t0 ))s"
[ "$rc" = "0" ] || { echo "[ABORT] seed42 c2000 eval 失敗"; exit 1; }

# ── 2. 逐值比對 gate ─────────────────────────────────────────────────
echo ""
echo "=== [GATE] 逐值比對 final_eval_c2000 vs final_eval（金標準）==="
uv run python - "$BASE/final_eval/metrics.json" "$BASE/final_eval_c2000/metrics.json" <<'PY'
import json, sys, math
gold = json.load(open(sys.argv[1])); test = json.load(open(sys.argv[2]))
max_rel = 0.0; max_abs = 0.0; nbad = 0; ncmp = 0
def walk(a, b, path=""):
    global max_rel, max_abs, nbad, ncmp
    if isinstance(a, dict):
        for k in a:
            if k in ("config","ckpt_dir"): continue  # 路徑字串，忽略
            if k in b: walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f"{path}[{i}]")
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        ncmp += 1
        d = abs(a-b); r = d/(abs(a)+1e-30)
        if d>max_abs: max_abs=d
        if r>max_rel: max_rel=r
        if r>=1e-6 and d>=1e-12:
            nbad += 1
            if nbad<=8: print(f"  DIFF {path}: gold={a!r} test={b!r} rel={r:.2e}")
walk(gold, test)
print(f"  比較 {ncmp} 個數值：max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
if max_rel < 1e-9:
    print("  VERDICT: PASS (bit-identical, rel<1e-9)"); sys.exit(0)
elif max_rel < 1e-6:
    print("  VERDICT: PASS (ULP 級, rel<1e-6，遠低於論文 2 位精度)"); sys.exit(0)
else:
    print(f"  VERDICT: FAIL — {nbad} 個值 rel>=1e-6，疑結構性差異，停手"); sys.exit(2)
PY
gate=$?
echo "  gate exit=${gate}"
[ "$gate" = "0" ] || { echo "[ABORT] 驗收未過，不補 s1/s2 eval"; exit 2; }

# ── 3. gate 通過 → 補 s1/s2 eval（chunk=2000）到各自 final_eval ──────
for s in 1 2; do
  echo ""
  echo "=== [FILL] s${s} chunk=2000 → final_eval ==="
  t0=$(date +%s)
  PILNN_EVAL_CHUNK_BUDGET=400000 \
  uv run python -u scripts/evaluate_exp245.py \
    --protocol follow_training \
    --config "configs/exp_302_b3_re1e6_K200_les_T50_s${s}.toml" $COMMON_EVAL \
    --output-subdir final_eval
  echo "  s${s} eval rc=$?  wall=$(( $(date +%s) - t0 ))s"
done

echo ""
echo "[eval302-c2k] $(date '+%F %T') done"
