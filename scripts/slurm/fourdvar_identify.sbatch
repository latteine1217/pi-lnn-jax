#!/bin/bash
#SBATCH --job-name=4dvar_ident
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/4dvar_ident_%j.out
#SBATCH --error=logs/4dvar_ident_%j.err
#
# NS 4D-Var mid-band 可識別性 oracle。
#
# 納管理由：產出 `fourdvar_main5_K100.json`（gap 48.9pp，被多處引用）的這支 sbatch
# 原本只存在於 lab-server、從未進 git——那個數字的確切執行條件因此不可稽核。
# 本檔即該指令，參數與原版逐字相同，只多出 INIT_MODE / OUT 兩個可覆寫變數。
#
# 環境變數：
#   INIT_MODE   cold-start 初始猜測：dns_lowband（預設 = 原版行為）| zero
#   OUT         輸出 JSON 路徑（預設隨 INIT_MODE 命名，避免覆寫既有結果）
#
# 用法：
#   sbatch scripts/slurm/fourdvar_identify.sbatch                    # 原版
#   INIT_MODE=zero sbatch scripts/slurm/fourdvar_identify.sbatch     # 控制組
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS=$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy
SENSORS=$PILNJAX_DATA_ROOT/data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
INIT_MODE="${INIT_MODE:-dns_lowband}"
if [ "$INIT_MODE" = "dns_lowband" ]; then
  OUT="${OUT:-artifacts/identifiability/fourdvar_main5_K100.json}"
else
  OUT="${OUT:-artifacts/identifiability/fourdvar_main5_K100_init_${INIT_MODE}.json}"
fi
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] INIT_MODE=$INIT_MODE OUT=$OUT"
[ -f "$DNS" ] || { echo "MISSING DNS $DNS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/fourdvar_identifiability.py \
  --dns "$DNS" --sensor-json "$SENSORS" \
  --n-frames-list 3,6,10,15,20 --maxiter 200 \
  --init-mode "$INIT_MODE" \
  --out "$OUT"
echo "[done] exit=$?"
