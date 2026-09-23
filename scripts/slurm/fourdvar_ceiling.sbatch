#!/bin/bash
#SBATCH --job-name=4dvar_ceiling
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/4dvar_ceiling_%j.out
#SBATCH --error=logs/4dvar_ceiling_%j.err
#
# EXP-512 Stage 1：4D-Var 反演的天花板控制組（不優化，只求值）。
#
# 要量的兩個數（從未量過，而沒有它們 E_oracle_mid 無法判讀）：
#   J*  = cost(ω₀_true) 在真實 DNS 觀測上 → **model-error 地板**
#         （前向 IF-RK4 ≠ 產生 DNS 的 ETDRK4；gate3 只驗到 t=4，本窗跑到 t=5）
#         任何 cost_final 低於 J* 都不可能。
#   E*  = 由 ω₀_true 前向積分後的 bands.mid.uv_rel → **oracle 天花板**
#         任何反演都不可能優於它。
#
# 動機（2026-08-20 稽核）：nf=20 時兩臂 cost_final 差 5.4×
#   （zero 5.94e-4 vs dns_lowband 1.09e-4）。cost 的最小值不隨初始猜測改變，
#   故至少一臂未達全域最小——而較高的那臂正是被引用為誠實 headline 的 16.63%。
#   事前登錄判準見 knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md
#   的「事前登錄：4D-Var 反演的天花板控制組與 twin test（EXP-512）」。
#
# 與 fourdvar_identify.sbatch 的關係：**除 --eval-at 外參數逐字相同**
#   （同 DNS、同 sensor、同 --n-frames-list 3,6,10,15,20、同 --maxiter 200），
#   否則天花板與被比較的數字不可比。
#
# 環境變數：
#   EVAL_AT   評估點：dns_true（預設，= J*/E*）| dns_lowband | zero
#             後兩者給既有兩臂的「起點 cost」作脈絡，非必要。
#   OUT       輸出 JSON 路徑（預設隨 EVAL_AT 命名）
#
# 用法：
#   sbatch scripts/slurm/fourdvar_ceiling.sbatch                      # J* / E*（必跑）
#   EVAL_AT=zero        sbatch scripts/slurm/fourdvar_ceiling.sbatch  # 脈絡（選）
#   EVAL_AT=dns_lowband sbatch scripts/slurm/fourdvar_ceiling.sbatch  # 脈絡（選）
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS=$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy
SENSORS=$PILNJAX_DATA_ROOT/data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
EVAL_AT="${EVAL_AT:-dns_true}"
OUT="${OUT:-artifacts/identifiability/fourdvar_ceiling_K100_${EVAL_AT}.json}"
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] EVAL_AT=$EVAL_AT OUT=$OUT"
[ -f "$DNS" ] || { echo "MISSING DNS $DNS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
[ -f "$OUT" ] && { echo "REFUSE overwrite existing $OUT"; exit 3; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/fourdvar_identifiability.py \
  --dns "$DNS" --sensor-json "$SENSORS" \
  --n-frames-list 3,6,10,15,20 --maxiter 200 \
  --eval-at "$EVAL_AT" \
  --out "$OUT"
echo "[done] exit=$?"
