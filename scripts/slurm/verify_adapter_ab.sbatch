#!/bin/bash
#SBATCH --job-name=jax_adapter_ab
#SBATCH --output=logs/jax_adapter_ab_%j.out
#SBATCH --error=logs/jax_adapter_ab_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   ADR 0006（reference_params_for adapter）的 Outstanding 驗收。對同一個
#   checkpoint 跑「遷移前」與「遷移後」兩份程式碼，逐鍵比對產出。
#
# Why 用 before/after 而非比對既有 artifact：
#   ADR 原本提議 dump_cp_fields 對照 artifacts/cp/seed42.npz（2026-06-04）。
#   但產生它的 artifacts/kolmogorov/exp245_seed42 已不存在，只能用「名字像」的
#   目錄頂替——而 ARCH 是 sbatch env、不進 config 也不進 summary.json，
#   目錄名相同不代表同一個模型。同一 ckpt 上跑兩份程式碼沒有這個推論缺口。
#
# Why dump_cp_fields 也用 ksweep_k200_s42（job 5424 的教訓）：
#   原本指向 exp245_b3_les_T50_jax_20k，該 ckpt 含 query_decoder.mha_out_proj
#   （2026-06-23 移除的參數），verify_params_tree 正確地擋下它——舊架構世代的
#   ckpt 無法以現行 code 還原。改用 evaluate_exp245 這一側已證明可還原的同一個
#   ckpt，兩個產生端因此也共用同一個輸入。
#
# 兩側：
#   BASE = ~/pi-lnn-jax-premig  @ fd44c70（adapter 之前；ckpt.py 無 reference_params_for）
#   HEAD = ~/pi-lnn-jax         @ main   （adapter 之後）
#
# ── 事前登錄的預期（ADR 要求 pre-register，跑之前寫死在這裡）────────────
#   (1) evaluate_exp245：metric_artifact.json 的每一個投影鍵在兩側相同，
#       容差 1e-12（ab_compare.py metrics 的預設）。
#   (2) dump_cp_fields：九個共用陣列逐位元相同。HEAD 端會多一個
#       ckpt_provenance 鍵——那是 adapter 之後才有的欄位，**預期出現，不算差異**。
#   (3) evaluate_multi_re：**本次無法驗**。lab-server 上沒有任何 multi-Re
#       checkpoint（rg 過 artifacts/kolmogorov/，只有 crossre_* 的單 Re 產物）。
#       不以「另找一個像的 ckpt」頂替——那會讓通過與否失去意義。此項留為 open。
#
#   任一鍵不符即為 FAIL：adapter 宣稱的是等價，不是近似。
#
# 用法（head node）：sbatch scripts/slurm/verify_adapter_ab.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

BASE_WT="$HOME/pi-lnn-jax-premig"
HEAD_WT="$HOME/pi-lnn-jax"
ART="$HOME/pi-lnn-jax/artifacts/kolmogorov"
CFG_EVAL="$HOME/pi-lnn-jax/configs/exp_ksweep_k200_s42.toml"
EV="$HOME/pi-lnn-jax/knowledge/superpowers/evidence"

export PYTHONUNBUFFERED=1
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BASE_SHA="$(git -C "$BASE_WT" rev-parse --short HEAD)"
HEAD_SHA="$(git -C "$HEAD_WT" rev-parse --short HEAD)"

echo "======================================================================"
echo "[adapter-ab] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "  BASE = $BASE_WT @ $BASE_SHA"
echo "  HEAD = $HEAD_WT @ $HEAD_SHA"
echo "======================================================================"

# 斷言兩側確實是遷移前 / 遷移後，否則整場對拍沒有意義（會「通過」但什麼都沒驗）
b_has=$(grep -c 'def reference_params_for' "$BASE_WT/pi_lnn_jax/ckpt.py" || true)
h_has=$(grep -c 'def reference_params_for' "$HEAD_WT/pi_lnn_jax/ckpt.py" || true)
echo "[guard] reference_params_for: BASE=$b_has HEAD=$h_has"
if [ "$b_has" != "0" ] || [ "$h_has" != "1" ]; then
  echo "[ERR] 兩側不是「遷移前 vs 遷移後」——停手" >&2; exit 4
fi

run_side () {  # $1=worktree  $2=tag
  local wt="$1" tag="$2"
  echo ""
  echo "--- [$tag] evaluate_exp245 → ab_$tag ---"
  ( cd "$wt" && PYTHONPATH=. uv run python -u scripts/evaluate_exp245.py \
      --protocol follow_training \
      --config "$CFG_EVAL" \
      --ckpt latest \
      --arch liquid \
      --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2 \
      --artifacts_dir "$ART/ksweep_k200_s42" \
      --output-subdir "ab_$tag" \
      --no-plot ) || { echo "[ERR] [$tag] evaluate_exp245 失敗" >&2; return 1; }

  echo ""
  echo "--- [$tag] dump_cp_fields → cp_ab_$tag.npz ---"
  ( cd "$wt" && PYTHONPATH=. uv run python -u scripts/dump_cp_fields.py \
      --config "$CFG_EVAL" \
      --ckpt 20000 \
      --artifacts_dir "$ART/ksweep_k200_s42" \
      --out "$HOME/pi-lnn-jax/artifacts/cp/cp_ab_$tag.npz" ) \
    || { echo "[ERR] [$tag] dump_cp_fields 失敗" >&2; return 1; }
}

run_side "$BASE_WT" premig || exit 1
run_side "$HEAD_WT" main   || exit 1

RC=0

echo ""
echo "=== (1) evaluate_exp245 投影逐鍵比對 ==="
PYTHONPATH=. uv run python scripts/_common/ab_compare.py metrics \
  --base "$ART/ksweep_k200_s42/ab_premig/metric_artifact.json" \
  --head "$ART/ksweep_k200_s42/ab_main/metric_artifact.json" \
  --base-sha "$BASE_SHA" --head-sha "$HEAD_SHA" \
  --job-id "${SLURM_JOB_ID:-?}" --node "${SLURMD_NODENAME:-?}" \
  --evidence "$EV/adapter_ab_eval245_${SLURM_JOB_ID:-manual}.json" || RC=1

echo ""
echo "=== (2) dump_cp_fields 陣列比對 ==="
PYTHONPATH=. uv run python - <<PY || RC=1
import numpy as np, sys, json, os
b = np.load("$HOME/pi-lnn-jax/artifacts/cp/cp_ab_premig.npz")
h = np.load("$HOME/pi-lnn-jax/artifacts/cp/cp_ab_main.npz")
bk, hk = set(b.files), set(h.files)
shared = sorted(bk & hk)
extra_h, extra_b = sorted(hk - bk), sorted(bk - hk)
print(f"  BASE keys={len(bk)}  HEAD keys={len(hk)}  shared={len(shared)}")
print(f"  HEAD 多出: {extra_h}   BASE 多出: {extra_b}")
bad = []
for k in shared:
    x, y = b[k], h[k]
    if x.shape != y.shape:
        bad.append(f"{k}: shape {x.shape} vs {y.shape}"); continue
    if x.dtype.kind in "fc":
        if not np.array_equal(x, y, equal_nan=True):
            bad.append(f"{k}: 逐位元不同，max|d|={np.nanmax(np.abs(x-y)):.3e}")
    elif not np.array_equal(x, y):
        bad.append(f"{k}: 不同")
# 事前登錄：HEAD 端多 ckpt_provenance 是預期的，不算差異；其他多出的鍵算
unexpected = [k for k in extra_h if k != "ckpt_provenance"]
if unexpected: bad.append(f"HEAD 出現未登錄的新鍵: {unexpected}")
if extra_b:    bad.append(f"BASE 有而 HEAD 缺: {extra_b}")
ok = not bad
json.dump({"job_id": os.environ.get("SLURM_JOB_ID",""),
           "base_sha": "$BASE_SHA", "head_sha": "$HEAD_SHA",
           "shared_keys": shared, "head_only": extra_h, "base_only": extra_b,
           "problems": bad, "ok": ok},
          open("$EV/adapter_ab_dumpcp_${SLURM_JOB_ID:-manual}.json", "w"),
          indent=2, ensure_ascii=False)
for p in bad: print("    -", p)
print("  === 共用陣列逐位元一致 ===" if ok else "  *** 不一致 ***")
sys.exit(0 if ok else 2)
PY

echo ""
echo "=== (3) evaluate_multi_re ==="
echo "  未執行：lab-server 上無 multi-Re checkpoint（事前登錄第 3 點）。此項留為 open。"

echo ""
echo "[adapter-ab] $(date '+%F %T') 結束 RC=$RC"
exit $RC
