#!/bin/bash
#SBATCH --job-name=dns_ext_T10
#SBATCH --output=logs/dns_ext_T10_%j.out
#SBATCH --error=logs/dns_ext_T10_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   重跑 Kolmogorov Re=10000 DNS 到 T_end=10（N=1024 → stride-4 降採樣成 N=256），
#   參數逐項對齊既有 T=5 資料檔的 config，然後驗證共同時間窗 [0,5] 是否為同一條軌跡。
#
# Why:
#   時間外推實驗（EXP-530）需要 t∈[0,10] 的真值，但既有的 T=20 檔**不是**同一條
#   軌跡（native N=256 生成，第 0 幀相對差 1.48），所以同 seed 重跑並驗證前半段。
#
#   ⚠️ 2026-09-11 更正：原註寫「T=5 的來源檔 N=1024 已不存在於任何一台機器」，
#   那是只搜 lab-server 得到的結論。該檔在 **home-gpu** 上：
#   ~/gi_test_re10000/kolmogorov_dns_fp64_etdrk4_Re10000_N1024_T5_dt2p5e4_si100_seed42_icspectral.npy
#   （6.3 GB，同批還有 N=128/256/512）。它存的是 stored-cadence 的場序列而非求解器
#   內部狀態，所以「從 t=5 續跑」是否可行仍未驗（以末幀當 IC 重啟不會是逐位元續行）；
#   重跑路線因此保留，但**不要再以「檔案不存在」為理由**。生成器機器分工見
#   knowledge 的 generator-machine-map 記載：在 lab-server 搜不到是常態，不是證據。
#
#   Re=10⁴ 是混沌流，backend/機器不同造成的 round-off 會被指數放大，所以本 job
#   最後一步的驗證不是形式檢查：它決定延長版能不能當 eval 真值（判定與意義見
#   scripts/verify_dns_extension.py）。
#
# 兩臂（同一份 sbatch，靠 CLI 覆寫）：
#   GPU 臂（預設，快、~小時級）：backend=torch-cuda
#     sbatch scripts/slurm/gen_dns_extended.sbatch
#   CPU 臂（慢、但 backend 與原始檔相同，重現機率最高）：
#     sbatch --partition=r620 --gres=none --cpus-per-task=4 \
#            --export=ALL,DNS_BACKEND=numpy scripts/slurm/gen_dns_extended.sbatch
#
# Notes:
#   - 生成器住在 pi-lnn（PyTorch repo）的 tools/dns_generator/，本 job 只呼叫它。
#   - N=1024 中間檔約 13 GB；降採樣後 N=256 約 845 MB。
#   - 不覆寫既有檔：目標檔已存在就跳過該步（重跑本 job 只會補做缺的步驟）。

set -euo pipefail

JAX_ROOT="${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
cd "$JAX_ROOT"

export PYTHONUNBUFFERED=1
PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
DNS_BACKEND="${DNS_BACKEND:-torch-cuda}"
T_END="${T_END:-10.0}"
TAG="${DNS_BACKEND##*-}"          # torch-cuda → cuda；numpy → numpy

GEN="$JAX_ROOT/scripts/dns_generator/generate_kolmogorov_dns_fp64.py"
DOWN="$JAX_ROOT/scripts/dns_generator/downsample_dns.py"
PY_JAX="$JAX_ROOT/.venv/bin/python"

for f in "$GEN" "$DOWN" "$PY_JAX"; do   # PY_DNS 由下方閘門檢查（此處尚未定義）
  [ -e "$f" ] || { echo "[ERR] 缺檔：$f" >&2; exit 2; }
done

# PY_DNS 必須由提交端指定，且**不可**指向本專案的 .venv。
# Why（2026-08-21 實測，非推測）：
#   1. lab-server driver 是 560.35.05（支援到 CUDA 12.6）。torch 的 cu13 wheel 在
#      compute node 上 torch.cuda.is_available() == False（job 5149）→ 生成器落到
#      torch-cpu，比 torch-cuda 慢數十倍，而且是**靜默**的。
#   2. 更糟：裝 torch 會把 nvidia-cudnn-cu13 帶進同一個 venv，蓋掉 JAX 的 cu12 路徑，
#      整個 worktree 的 JAX GPU 當場失效（RET_CHECK dnn_support != nullptr）。
#      job 5138-5147 十個訓練/eval 因此全滅。修復要跑 setup_gpu_venv.sh。
#   結論：DNS 一律在 home-gpu 生成，不在 lab-server。這裡只留一道明確的閘門。
PY_DNS="${PY_DNS:-}"
[ -n "$PY_DNS" ] || {
  echo "[ERR] PY_DNS 未設。DNS 生成不在 lab-server 跑——請在 home-gpu 執行，" >&2
  echo "      或指定一個獨立於本專案 .venv、且 torch 拿得到 CUDA 的 python：" >&2
  echo "        PY_DNS=/path/to/dns-venv/bin/python sbatch <this>" >&2
  echo "      切勿指向 $JAX_ROOT/.venv —— 見本檔上方註解。" >&2
  exit 4; }
"$PY_DNS" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch 看不到 CUDA（生成器會靜默降級到 numpy backend）')
" || { echo "[ERR] PY_DNS=$PY_DNS 的 torch 拿不到 CUDA" >&2; exit 4; }

# 參考軌跡（sensor 的來源）。解析順序同 data._resolve_data_path：本 repo → 大檔池。
# 缺它就沒有可驗的對象，直接停。
REF_REL="data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"
REF=""
for root in "$JAX_ROOT" "$PILNJAX_DATA_ROOT"; do
  [ -f "$root/$REF_REL" ] && { REF="$root/$REF_REL"; break; }
done
[ -n "$REF" ] || { echo "[ERR] 參考 DNS 不存在於 $JAX_ROOT 或 $PILNJAX_DATA_ROOT：$REF_REL" >&2; exit 3; }

RAW="$PILNJAX_DATA_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N1024_T10_dt2p5e4_si100_${TAG}.npy"
DS4="$JAX_ROOT/data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T10_dt2p5e4_si100_ds4_${TAG}.npy"

echo "======================================================================"
echo "[dns_ext] $(date '+%F %T')  job=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "[dns_ext] backend=$DNS_BACKEND  T_end=$T_END"
echo "[dns_ext] raw=$RAW"
echo "[dns_ext] ds4=$DS4"
echo "[dns_ext] ref=$REF"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "======================================================================"

# ── 1. 生成 N=1024 DNS ────────────────────────────────────────────────
# 參數逐項對齊 T=5 檔的 config：nu/A/k_f/dt/save_interval/integrator/dealias/seed/ic_mode。
# target-initial-* 三個鍵沿用生成器預設（T=5 檔的 config 記錄值與預設完全相同）。
if [ -f "$RAW" ]; then
  echo "[dns_ext] N=1024 檔已存在，跳過生成"
else
  "$PY_DNS" "$GEN" \
    --N 1024 --L 1.0 --nu 1e-4 --A 0.1 --k_f 2 \
    --dt 2.5e-4 --T_end "$T_END" --save_interval 100 \
    --integrator etdrk4 --dealias-mode 3/2 --seed 42 \
    --ic_mode band_limited_random \
    --backend "$DNS_BACKEND" \
    --output "$RAW"
fi

# ── 2. stride-4 降採樣到 N=256（與既有資料集同一條產線）──────────────
if [ -f "$DS4" ]; then
  echo "[dns_ext] ds4 檔已存在，跳過降採樣"
else
  "$PY_DNS" "$DOWN" --src "$RAW" --stride 4 --out "$DS4"
fi

# ── 3. 軌跡一致性驗證（本步決定它能不能當 eval 真值）──────────────────
echo "[dns_ext] 驗證共同時間窗 [0,5] 是否同一條軌跡"
"$PY_JAX" scripts/verify_dns_extension.py --extended "$DS4" --reference "$REF"
echo "[dns_ext] done $(date '+%F %T')"
