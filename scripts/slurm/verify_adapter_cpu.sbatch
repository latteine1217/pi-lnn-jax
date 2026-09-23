#!/bin/bash
#SBATCH --job-name=jax_adapter_cpu
#SBATCH --output=logs/jax_adapter_cpu_%j.out
#SBATCH --error=logs/jax_adapter_cpu_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=r620
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# What:
#   ADR 0006 adapter 的**逐位元**驗收。jobs 5425/5452 在 GPU 上只能證明
#   「差異與平台噪音無法區分」（pred_u 2.390e-04，同碼兩跑得到同一個數）。
#   CLAUDE.md §7.1 第 2 點：逐位元判準只在 CPU 上成立。本 job 走 CPU。
#
# Why ckpt 在本 job 內於 CPU 上現產（job 5459 的教訓）：
#   第一版直接餵 ksweep_k200_s42，33 秒就死在
#   「sharding passed to deserialization ... Got None」——GPU 寫的 ckpt 在
#   CPU JAX 上 orbax 推不出 sharding。verify_ckpt_compat.sbatch.tmpl 早就
#   遇過同一件事，它的解法是自己在 CPU 上跑幾步生成 ckpt；此處照做，
#   不另造機制。
#
# ⚠️ 代價要講清楚：因此比對跑在 fixture 規模的模型上（_ledger_single_re.toml，
#   d_model=32），不是論文的 K=200 checkpoint。adapter 要驗的是「模板建構
#   路徑是否等價」，那條路徑不隨模型大小改變；但這仍是比 GPU 那輪更窄的
#   證據，不得被複述為「在論文 checkpoint 上逐位元相同」。
#
# Why 三側同一個 job：
#   §7.1 同一點也說 CPU 上 arm64 與 x86_64 的 digest 不同（job 4740）。
#   三側跑在同一個 sbatch 內就必定同節點同架構。
#
# ── 事前登錄的判準（跑之前寫死）────────────────────────────────────────
#   (0) determinism control 先跑：main vs main2 必須逐位元相同。
#       不相同 → CPU 在此也非決定性，逐位元問題在這台機器上無法回答。
#       據實報 INCONCLUSIVE 並停手，不改判準遷就結果。
#   (1) (0) 成立才判：premig vs main 每個共用陣列逐位元相同，且
#       metric_artifact.json 每個投影鍵相同（容差 1e-12）。
#       任一處不同 = FAIL，且是真差異（噪音底已由 (0) 證明為零）。
#   (2) HEAD 側可多出 ckpt_provenance——adapter 之後才有的欄位，預期出現。
#       其他新增鍵算差異。
#   (3) evaluate_multi_re 不在範圍：lab-server 無 multi-Re checkpoint。
#
# 用法（head node）：sbatch scripts/slurm/verify_adapter_cpu.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

BASE_WT="$HOME/pi-lnn-jax-premig"
HEAD_WT="$HOME/pi-lnn-jax"
CFG="$HOME/pi-lnn-jax/configs/_ledger_single_re.toml"
ART="$HOME/pi-lnn-jax/artifacts/adapter_cpu"
CP="$HOME/pi-lnn-jax/artifacts/cp"
EV="$HOME/pi-lnn-jax/knowledge/superpowers/evidence"
GEN_STEPS="${GEN_STEPS:-10}"

export PYTHONUNBUFFERED=1
# fixture config 的 sensor/DNS 在大檔池與 multire 根下，不在任一 worktree 內。
# 解析序 PILNJAX_ROOT → PILNJAX_DATA_ROOT，與 verify_cpu_ab.sbatch.tmpl 一致
# （job 5471 缺這一行，train 在讀 sensor JSON 時就 FileNotFoundError）。
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PILNJAX_ROOT="${PILNJAX_ROOT:-$HOME/pi-lnn-jax-multire}"
export PILNN_ROOT="${PILNN_ROOT:-$HOME/pi-lnn-jax-main}"   # 基準側舊 data.py 只認這個
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cpu

BASE_SHA="$(git -C "$BASE_WT" rev-parse --short HEAD)"
HEAD_SHA="$(git -C "$HEAD_WT" rev-parse --short HEAD)"

echo "======================================================================"
echo "[adapter-cpu] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "  arch  = $(uname -m)   （§7.1：CPU digest 不可跨架構比）"
echo "  BASE  = $BASE_WT @ $BASE_SHA"
echo "  HEAD  = $HEAD_WT @ $HEAD_SHA"
echo "  ckpt  = 本 job 於 CPU 現產（$GEN_STEPS 步，fixture 規模）"
echo "======================================================================"
uv run python -c "import jax; print('JAX', jax.__version__, jax.devices())"

b_has=$(grep -c 'def reference_params_for' "$BASE_WT/pi_lnn_jax/ckpt.py" || true)
h_has=$(grep -c 'def reference_params_for' "$HEAD_WT/pi_lnn_jax/ckpt.py" || true)
echo "[guard] reference_params_for: BASE=$b_has HEAD=$h_has"
if [ "$b_has" != "0" ] || [ "$h_has" != "1" ]; then
  echo "[ERR] 兩側不是「遷移前 vs 遷移後」——停手" >&2; exit 4
fi

# ── (a) 在 CPU 上現產一份 ckpt ─────────────────────────────────────────
rm -rf "$ART"
echo ""; echo "── (a) CPU 上跑 ${GEN_STEPS} 步產生 ckpt ──"
PYTHONPATH=. uv run python -u train_kolmogorov.py \
  --config "$CFG" --steps "$GEN_STEPS" --save_every "$GEN_STEPS" \
  --artifacts_dir "$ART" --arch liquid \
  || { echo "[ERR] ckpt 生成失敗" >&2; exit 1; }
ls "$ART/checkpoints" || { echo "[ERR] 沒有產出 checkpoints" >&2; exit 1; }

run_side () {  # $1=worktree  $2=tag
  local wt="$1" tag="$2" t0=$SECONDS
  echo ""
  echo "--- [$tag] evaluate_exp245 → cpu_$tag ---"
  ( cd "$wt" && PYTHONPATH=. uv run python -u scripts/evaluate_exp245.py \
      --protocol follow_training --config "$CFG" --ckpt latest --arch liquid \
      --artifacts_dir "$ART" --output-subdir "cpu_$tag" --no-plot ) \
    || { echo "[ERR] [$tag] evaluate_exp245 失敗" >&2; return 1; }
  echo "--- [$tag] dump_cp_fields → cp_cpu_$tag.npz ---"
  ( cd "$wt" && PYTHONPATH=. uv run python -u scripts/dump_cp_fields.py \
      --config "$CFG" --ckpt "$GEN_STEPS" \
      --artifacts_dir "$ART" --out "$CP/cp_cpu_$tag.npz" ) \
    || { echo "[ERR] [$tag] dump_cp_fields 失敗" >&2; return 1; }
  echo "[timing] $tag 用時 $((SECONDS - t0)) s"
}

run_side "$HEAD_WT" main   || exit 1
run_side "$HEAD_WT" main2  || exit 1
run_side "$BASE_WT" premig || exit 1

echo ""
echo "=== 判讀（先 (0) 後 (1)）==="
PYTHONPATH=. uv run python - <<PY
import json, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from _common.ab_compare import metrics_projection_diff

A, CP = "$ART", "$CP"
load = lambda p: json.load(open(p))

def arrays_equal(p, q):
    a, b = np.load(p), np.load(q)
    ak, bk = set(a.files), set(b.files)
    probs = []
    for k in sorted(ak & bk):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            probs.append(f"{k}: shape {x.shape} vs {y.shape}")
        elif x.tobytes() != y.tobytes():
            d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            probs.append(f"{k}: 逐位元不同 max|d|={d:.6e}")
    return (not probs), probs, sorted(bk - ak), sorted(ak - bk)

ctrl_keys = metrics_projection_diff(load(f"{A}/cpu_main/metric_artifact.json"),
                                    load(f"{A}/cpu_main2/metric_artifact.json"))
ctrl_ok, ctrl_probs, _, _ = arrays_equal(f"{CP}/cp_cpu_main.npz", f"{CP}/cp_cpu_main2.npz")
det = (not ctrl_keys) and ctrl_ok
print(f"  (0) 同碼兩跑：投影不一致鍵={len(ctrl_keys)}  陣列問題={len(ctrl_probs)}")
for p in (ctrl_probs[:5] + ctrl_keys[:5]): print("      -", p)
print("      → CPU 在此節點可重現" if det else "      → CPU 在此也非決定性")

verdict, ab_keys, ab_probs, extra = None, [], [], []
if not det:
    verdict = "INCONCLUSIVE：噪音底非零，逐位元問題在此機器上無法回答"
else:
    ab_keys = metrics_projection_diff(load(f"{A}/cpu_premig/metric_artifact.json"),
                                      load(f"{A}/cpu_main/metric_artifact.json"))
    ab_ok, ab_probs, head_only, base_only = arrays_equal(
        f"{CP}/cp_cpu_premig.npz", f"{CP}/cp_cpu_main.npz")
    extra = [k for k in head_only if k != "ckpt_provenance"] + base_only
    print(f"  (1) premig vs main：投影不一致鍵={len(ab_keys)}  陣列問題={len(ab_probs)}")
    print(f"      HEAD 多出={head_only}  BASE 多出={base_only}")
    for p in (ab_probs[:5] + ab_keys[:5]): print("      -", p)
    verdict = ("PASS：逐位元相同（fixture 規模，非論文 checkpoint）"
               if (ab_ok and not ab_keys and not extra)
               else "FAIL：噪音底為零而仍有差異 → 真差異")

print("")
print(f"  === {verdict} ===")
json.dump({"job_id": os.environ.get("SLURM_JOB_ID",""),
           "node": os.environ.get("SLURMD_NODENAME",""), "arch": os.uname().machine,
           "platform": "cpu", "ckpt": "fixture _ledger_single_re, $GEN_STEPS steps, CPU-generated",
           "base_sha": "$BASE_SHA", "head_sha": "$HEAD_SHA",
           "determinism_control_clean": bool(det),
           "control_problems": ctrl_probs + ctrl_keys,
           "ab_problems": ab_probs + ab_keys, "unexpected_keys": extra,
           "verdict": verdict},
          open(f"$EV/adapter_cpu_${SLURM_JOB_ID:-manual}.json", "w"),
          indent=2, ensure_ascii=False)
sys.exit(0 if verdict.startswith("PASS") else 2)
PY
RC=$?

echo ""
echo "=== evaluate_multi_re ==="
echo "  不在範圍：lab-server 無 multi-Re checkpoint（事前登錄第 3 點）。"
echo "[adapter-cpu] $(date '+%F %T') 結束 RC=$RC"
exit $RC
