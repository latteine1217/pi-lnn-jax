#!/bin/bash
#SBATCH --job-name=4dvar_budget
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/4dvar_budget_%j.out
#SBATCH --error=logs/4dvar_budget_%j.err
#
# EXP-512 Stage 3b'：臂 Z 逐字設定，僅 maxiter ×10。測 H-A（預算受限）。
#
# 為什麼是這個組態（對事前登錄 S3-B 的細化，理由見實驗紀錄）：
#   事前登錄的 S3-B 寫「冷啟動 + maxiter 2000」。但 Stage 3a（job 5345）證明
#   **鏈式 warm-start 有幫助而非污染**——冷啟動 J_C = 22.9 × J_Z，E 從 16.63% 惡化到
#   34.87%，且 nit=200/200、success=False。把預算加在冷啟動上沒有意義；
#   該加在產生 ledger 數字 16.63% 的那個組態上，也就是臂 Z 的鏈式 sweep。
#
# 與臂 Z（job 5028）的唯一變因 = **maxiter 200 → 2000**（maxfun 隨之 4000 → 40000）。
#   DNS / sensor / --n-frames-list / --init-mode / ftol / gtol 逐字相同。
#
# 事前登錄判準（不變）見 knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md
#   「EXP-512 Stage 3」。參考值：J* = 2.18e-07、J_Z = 5.94e-04、E_Z = 16.63%。
#   **總 falsifier**：若 J 落在 [J_Z/2, 2J_Z] 且 E 變動 <2pp ⟹ 預算不是主因，
#   停止調參，回到方法設計（Gauss-Newton／incremental 4D-Var／換控制變數）。
#
# 時限：臂 Z 為 1h33m @ maxiter=200。×10 預算上限約 15h ⟹ 設 48h（不自估、留餘裕）。
#
# 資料路徑（2026-08-21 實測 cmp 確認與臂 Z 所用複本逐位元相同）：
#   DNS     走大檔池 PILNJAX_DATA_ROOT。
#   SENSORS 走 **repo 內版控路徑**。
#   ⚠️ 不可照抄 fourdvar_identify.sbatch 的 SENSORS——它指向
#      $PILNJAX_DATA_ROOT/data/kolmogorov_sensors/，該路徑不存在（sensor 已入庫版控），
#      該檔目前無法執行。本檔不修改它（其存在目的為保存產生已發表數字的指令）。
#
# 環境變數：
#   MAXITER    L-BFGS-B 上限（預設 2000）
#   INIT_MODE  冷啟動初始猜測（預設 zero = 臂 Z）
#   OUT        輸出 JSON（預設依上述命名）
#
# 用法：
#   sbatch scripts/slurm/fourdvar_sweep_budget.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS=$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy
SENSORS=data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
MAXITER="${MAXITER:-2000}"
INIT_MODE="${INIT_MODE:-zero}"
OUT="${OUT:-artifacts/identifiability/fourdvar_sweep_K100_${INIT_MODE}_iter${MAXITER}.json}"
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] INIT_MODE=$INIT_MODE MAXITER=$MAXITER OUT=$OUT"
[ -f "$DNS" ] || { echo "MISSING DNS $DNS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
[ -f "$OUT" ] && { echo "REFUSE overwrite existing $OUT"; exit 3; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/fourdvar_identifiability.py \
  --dns "$DNS" --sensor-json "$SENSORS" \
  --n-frames-list 3,6,10,15,20 --maxiter "$MAXITER" \
  --init-mode "$INIT_MODE" \
  --out "$OUT"
echo "[done] exit=$?"
