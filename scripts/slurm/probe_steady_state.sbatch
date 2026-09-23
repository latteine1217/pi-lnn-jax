#!/bin/bash
#SBATCH --job-name=probe_steady
#SBATCH --output=logs/probe_steady_%j.out
#SBATCH --error=logs/probe_steady_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   把 Kolmogorov Re=10000 跑到 T_end=30（N=256、粗存），只為回答一個問題：
#   **這個系統會不會進入統計穩態？** 看 enstrophy/KE 時序有沒有平台且帶波動。
#
# Why:
#   時間外推的結論目前綁在一條衰減暫態上（enstrophy 142→8、有效模態 4.2→1.5）。
#   要證明結論可推廣，就得在統計穩態窗重做一次。但那要 N=1024 DNS + 四臂重訓
#   （~半天）——而它有一個先決風險：**這個 attractor 可能是近層流或週期解**，
#   那樣外推變得平庸地容易，測試在反方向失效。
#
#   本 job 先用便宜的 N=256 回答那個先決問題。N=256 在此流場的保真度已驗過
#   （chaos_budget 的控制組：從 DNS 場積分 5 個時間單位，對 DNS 偏離 1.7e-5）。
#   IC realization 與主軌跡不同（native N=256），但「系統會不會進穩態」是系統
#   層級的性質，不取決於 IC realization。
#
# 判準（事前寫死）：
#   取 t∈[20,30] 這段，若 enstrophy 的線性趨勢斜率 |slope|·10 < 其標準差 → 視為平台；
#   且標準差/均值 > 0.05 → 仍有波動（非靜止解）。兩者皆成立才叫統計穩態。
#   任一不成立就**不要**投入 N=1024 + 四臂重訓。

set -euo pipefail

JAX_ROOT="${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
cd "$JAX_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
# uv 不在 compute node 的預設 PATH（漏這行的話生成跑完才在判定那步掛掉）
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
GEN="$JAX_ROOT/scripts/dns_generator/generate_kolmogorov_dns_fp64.py"
[ -f "$GEN" ] || { echo "[ERR] 缺生成器：$GEN" >&2; exit 2; }

OUT="${OUT:-$PILNJAX_DATA_ROOT/data/dns/probe_kolmogorov_Re10000_N256_T30_si400.npy}"
T_END="${T_END:-30.0}"
# 判定時窗必須跟著 T_END 走：寫死 20-30 會讓 T_END=100 的 job 判在早期段（踩過一次）
WIN_LO="${WIN_LO:-20}"
WIN_HI="${WIN_HI:-30}"

echo "[probe] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[probe] T_end=$T_END  out=$OUT"

if [ -f "$OUT" ]; then
  echo "[probe] 檔案已存在，跳過生成"
else
  # PY_DNS 的閘門只在**真的要生成**時才適用：純判定（檔案已存在）不碰 torch，
  # 擋在這條分支之外會讓「重跑判定」也需要一個 CUDA python（踩過一次，job 5173）。
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

  "$PY_DNS" "$GEN" \
    --N 256 --L 1.0 --nu 1e-4 --A 0.1 --k_f 2 \
    --dt 2.5e-4 --T_end "$T_END" --save_interval 400 \
    --integrator etdrk4 --dealias-mode 3/2 --seed 42 \
    --ic_mode band_limited_random --backend torch-cuda --output "$OUT"
fi

# 判定寫在版控的腳本裡，不寫回 heredoc（本機測得到，job 內也跑同一份）
uv run python -u scripts/diag_steady_state.py --dns "$OUT" --window "$WIN_LO" "$WIN_HI"
echo "[probe] done $(date '+%F %T')"
