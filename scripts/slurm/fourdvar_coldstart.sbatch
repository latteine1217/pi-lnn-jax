#!/bin/bash
#SBATCH --job-name=4dvar_cold
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/4dvar_cold_%j.out
#SBATCH --error=logs/4dvar_cold_%j.err
#
# EXP-512 Stage 3a：冷啟動於 nf=20，測「warm-start 鏈式污染」(H-B)。
#
# 為什麼：Stage 1（job 5133）證實兩臂都遠離可達地板——nf=20 時
#   J_A/J* = 501、J_Z/J* = 2722，而 model error 只佔 0.2%（E* = 0.05%）。
#   其中 H-B 有直接證據：nf=3 時 J_A = 2.44e-12 < J* = 8.83e-11，
#   **優化器找到的點比真值更貼合那 3 幀觀測**，而該點正是 nf=6 的 warm start。
#   整條鏈可能建在那個偽解的 basin 上。
#
# 與臂 Z 的唯一變因 = **初始猜測**（鏈下來的解 vs 零場）：
#   臂 Z   : --n-frames-list 3,6,10,15,20  （incremental warm-start）
#   本臂   : --n-frames 20                 （單次路徑，w0_init_field=None = 純冷啟動）
#   其餘（DNS / sensor / init-mode / maxiter / ftol / gtol）逐字相同。
#
# 順帶回收 Stage 1 之後才有的 nit / success 埋點 → 同時判別 H-A（預算）與 H-D（梯度品質）。
#
# 事前登錄的判讀表見 knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md
#   「事前登錄：4D-Var 優化受限的歸因與修復（EXP-512 Stage 3）」。
#   參考值：J* = 2.18e-07、J_Z = 5.94e-04、E_Z = 16.63%。
#
# ⚠️ 不修改 fourdvar_identify.sbatch——該檔的存在目的是逐字保存產生已發表數字的指令。
# ⚠️ 單次路徑寫出的產物是 **dict**（sweep 才是 list），讀取腳本須兩者都能處理。
#
# 環境變數：
#   INIT_MODE  冷啟動初始猜測（預設 zero = 對照臂 Z）
#   MAXITER    L-BFGS-B 上限（預設 200 = 與臂 Z 相同；Stage 3b 用 2000）
#   NFRAMES    window（預設 20）
#   OUT        輸出 JSON（預設依上述三者命名）
#
# 用法：
#   sbatch scripts/slurm/fourdvar_coldstart.sbatch                  # Stage 3a
#   MAXITER=2000 sbatch scripts/slurm/fourdvar_coldstart.sbatch     # Stage 3b（條件性）
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
# 資料來源（2026-08-21 實測確認可比性）：
#   DNS     走大檔池 PILNJAX_DATA_ROOT；與臂 Z（job 5028）用的舊路徑複本 cmp **逐位元相同**。
#   SENSORS 走 **repo 內版控路徑**；與臂 Z 用的舊路徑複本 cmp **逐位元相同**。
#   ⚠️ 不可照抄 fourdvar_identify.sbatch 的 SENSORS——它指向
#      $PILNJAX_DATA_ROOT/data/kolmogorov_sensors/，該路徑**不存在**（sensor 已入庫版控）。
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS=$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy
SENSORS=data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
INIT_MODE="${INIT_MODE:-zero}"
MAXITER="${MAXITER:-200}"
NFRAMES="${NFRAMES:-20}"
OUT="${OUT:-artifacts/identifiability/fourdvar_cold_K100_nf${NFRAMES}_${INIT_MODE}_iter${MAXITER}.json}"
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] INIT_MODE=$INIT_MODE MAXITER=$MAXITER NFRAMES=$NFRAMES OUT=$OUT"
[ -f "$DNS" ] || { echo "MISSING DNS $DNS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
[ -f "$OUT" ] && { echo "REFUSE overwrite existing $OUT"; exit 3; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/fourdvar_identifiability.py \
  --dns "$DNS" --sensor-json "$SENSORS" \
  --n-frames "$NFRAMES" --maxiter "$MAXITER" \
  --init-mode "$INIT_MODE" \
  --out "$OUT"
echo "[done] exit=$?"
