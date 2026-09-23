#!/bin/bash
#SBATCH --job-name=cyl_run
#SBATCH --output=logs/cyl_run_%j.out
#SBATCH --error=logs/cyl_run_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What: config 驅動 train_cylinder.py CEXP-002 重訓（#1 folx anisotropic 修復後）。
# Why : cylinder Lx≠Ly（0.32/0.17）下，舊 folx 把 collapsed laplacian 全塞 d2x、d2y=0，
#       黏性項 d2x/Lx²+d2y/Ly² 退化成 (uxx+uyy)/Lx²（~20× off）。修正後驗證 KE 是否
#       從歷史 486% 大幅改善。半資源 8CPU/1GPU/48G；--time 12h 保證跑完。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# UV_NO_SYNC：compute node 離線 + .venv 已備妥（AGENTS.md §6），`uv run` 的預設
# 同步在該前提下永遠只能失敗或無事可做。少了它，任何人新增依賴都會讓本 job 在
# 訓練開始前就掛（TD-27；實測 job 5515）。
export UV_NO_SYNC=1
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
# fof collocation 分塊：vector attention 中間張量 ∝N×K×hidden，序列處理降峰值
export PILNN_FOF_CHUNK="${PILNN_FOF_CHUNK:-256}"
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

CONFIG="${CONFIG:-configs/exp_cyl_cexp002.toml}"
[ -f "$CONFIG" ] || { echo "[ERR] $CONFIG 不存在" >&2; exit 4; }

echo "[cyl_folxfix] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} CONFIG=$CONFIG HEAD=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## train_cylinder.py CEXP-002（#1 folx anisotropic 修復後）########"
# SEED 可選：未設時走 config 的 seed（向後相容）。
EXTRA=()
[ -n "${SEED:-}" ] && EXTRA+=(--seed "$SEED")
# EXTRA_ARGS：原樣附加到 train_cylinder.py（例：--soap_precondition_frequency 2）。
# 刻意不加引號，讓多個旗標展開；未設時什麼都不加，既有呼叫端行為不變。
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=(${EXTRA_ARGS})

# 不要用 `|| echo`：那會把失敗吞成成功退出，slurm 的 State 顯示 COMPLETED 而
# 實際只跑了幾秒（實測 job 5528——缺資料檔，7 秒掛掉，sacct 卻是 COMPLETED，
# 只有 Elapsed 洩漏了真相）。保留 exit code，讓失敗就是失敗。
uv run python -u train_cylinder.py --config "$CONFIG" "${EXTRA[@]}"
rc=$?
if [ $rc -ne 0 ]; then echo "[ERR] train_cylinder.py rc=$rc"; fi
echo; echo "[$(date '+%F %T')] cyl_folxfix 結束 rc=$rc"
exit $rc
