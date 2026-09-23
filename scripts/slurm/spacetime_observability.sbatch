#!/bin/bash
#SBATCH --job-name=st_observ
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/st_observ_%j.out
#SBATCH --error=logs/st_observ_%j.err
#
# EXP-513：時空觀測算子 Φ 的奇異譜（optimizer-free 可識別性）。
#
# 為什麼換方向：EXP-512 證明 4D-Var oracle 的數字是（問題＋優化器＋預算）的性質
#   ——加預算讓 E_oracle_mid 從 16.63% 惡化到 22.27%。要問「中頻可不可識別」，
#   必須拿掉優化器。Φ 的 frame 下界即 dynamical sampling 的 β_{K,T}，與優化器無關。
#
# 根因：控制變數選錯。整個 ω0 對 4000 個觀測欠定 16.4×，但 k≤16（整個目標頻帶）
#   只有 796 個自由度、其實過定 5.03×；欠定全來自 sensor 看不到的 k>16。
#
# 驗證階梯（腳本內建，不過關即 SystemExit，不得讀後續數字）：
#   gate1 adjoint       ⟨Φv,u⟩ = ⟨v,Φᵀu⟩ 至 float64 精度
#   gate2 linearization ε→0 時線性化誤差 →0（先驗：λ_max=0.135、t=5 ⟹ 放大僅 ≈2×）
#   gate3 spectrum      low(96 dof) / mid(700 dof) 的奇異譜與有效秩 r(η)
#
# 事前登錄的判讀表（含反方向列）見
#   knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md 的
#   「事前登錄：時空觀測算子的奇異譜（EXP-513）」。
#
# 資料路徑與 fourdvar_sweep_budget.sbatch 相同（已 cmp 確認與臂 Z 所用複本逐位元相同）：
#   DNS 走大檔池、SENSORS 走 repo 版控路徑。
#   ⚠️ 不可照抄 fourdvar_identify.sbatch 的 SENSORS——該路徑不存在。
#
# 環境變數：
#   CHUNK    vmap 批次（預設 25；記憶體不足時調小）
#   GATE     adjoint | linearization | spectrum | all（預設 all）
#   OUT      輸出 JSON
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export UV_OFFLINE=1
export JAX_PLATFORMS=cuda
export PYTHONPATH=.
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS=$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy
SENSORS=data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json
CHUNK="${CHUNK:-25}"
GATE="${GATE:-all}"
OUT="${OUT:-artifacts/identifiability/spacetime_observability_K100_nf20.json}"
mkdir -p artifacts/identifiability logs
echo "[env] host=$(hostname) date=$(date) commit=$(git rev-parse --short HEAD)"
echo "[cfg] CHUNK=$CHUNK GATE=$GATE OUT=$OUT"
[ -f "$DNS" ] || { echo "MISSING DNS $DNS"; exit 2; }
[ -f "$SENSORS" ] || { echo "MISSING SENSORS $SENSORS"; exit 2; }
[ -f "$OUT" ] && { echo "REFUSE overwrite existing $OUT"; exit 3; }
.venv/bin/python -c "import jax; print(\"[jax] devices\", jax.devices())"
.venv/bin/python scripts/diag_spacetime_observability.py \
  --dns "$DNS" --sensor-json "$SENSORS" \
  --n-frames 20 --chunk "$CHUNK" --gate "$GATE" \
  --out "$OUT"
echo "[done] exit=$?"
