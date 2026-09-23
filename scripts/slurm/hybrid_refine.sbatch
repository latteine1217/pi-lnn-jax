#!/bin/bash
#SBATCH --job-name=hybrid_ref
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hybrid_ref_%j.out
#SBATCH --error=logs/hybrid_ref_%j.err
#
# EXP-514：learned 初始 + 迭代精修，量「單次前向 vs 迭代」的分界。
#
# 為什麼是這個問題：EXP-512 顯示 oracle 數字是（問題＋優化器＋預算）的性質；
#   EXP-513 顯示中頻佔渦度範數 64%、線性化框架不適用。兩者共同指向一個比
#   「感測 vs 模型」更站得住的主張——中頻可能只能由迭代／全域方法恢復，
#   任何 amortized 單次前向算子都做不到。這是方法類別的陳述。
#
# 為什麼值得做：中頻只佔速度能量 4.4%，但因相對誤差達 57.73%，它貢獻了 headline
#   平方誤差的 65%。完全修好可讓 uv 從 15.03% 降到 8.85%（6.18pp）——是主數字上
#   最大的單一槓桿，不是支線。
#
# 對照組已存在（EXP-512 Stage 3a，job 5345）：同 nf=20、非鏈式、maxiter=200，
#   零起點得 E_mid=34.87%。本實驗僅換初始猜測，是乾淨的單變因對照。
#
# 驗證階梯（腳本內建，不過關即 SystemExit）：
#   gate1 curl round-trip   ω0=i(k_x v̂−k_y û) 須逐位元反轉 vorticity_to_velocity
#                           （測試場帶限——實場在 Nyquist 的往返損失是結構性的，
#                             真實損失由 gate3 在原始資料上量並報告）
#   gate2 重現已發表數字     由 fields.npz 算出的 learned mid 須落在 57.73% 附近
#   gate3 div-free 投影損失  以 ω0 為控制變數丟掉多少模型輸出（報告，不 gate）
#
# 事前登錄見 knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md。
#
# 環境變數：SEED（main5 的 seed，預設 s42）／MAXITER／TRACE_EVERY／REPRO_TOL／OUT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
SEED="${SEED:-s42}"
FIELDS=artifacts/kolmogorov/main5_${SEED}/final_eval/fields.npz
SENSORS=data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
MAXITER="${MAXITER:-200}"
TRACE_EVERY="${TRACE_EVERY:-5}"
REPRO_TOL="${REPRO_TOL:-0.05}"
OUT="${OUT:-artifacts/identifiability/hybrid_refine_K100_${SEED}_iter${MAXITER}.json}"
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] SEED=$SEED MAXITER=$MAXITER TRACE_EVERY=$TRACE_EVERY OUT=$OUT"
[ -f "$FIELDS" ] || { echo "MISSING FIELDS $FIELDS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
[ -f "$OUT" ] && { echo "REFUSE overwrite existing $OUT"; exit 3; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/diag_hybrid_refine.py \
  --fields "$FIELDS" --sensor-json "$SENSORS" \
  --n-frames 20 --maxiter "$MAXITER" --trace-every "$TRACE_EVERY" \
  --repro-tol "$REPRO_TOL" \
  --out "$OUT"
echo "[done] exit=$?"
