#!/bin/bash
#SBATCH --job-name=jax_profile_ncollo
#SBATCH --output=logs/jax_profile_ncollo_%j.out
#SBATCH --error=logs/jax_profile_ncollo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   per-step timing profile — 掃 collocation 點數 (n_collo)，量 forward-Laplacian
#   的邊際成本。序列跑 4 段短訓練，每段固定 n_collo，讀 progress log 的 `wall` 欄
#   得 per-step time = wall / log_every。
#
# Why:
#   回答「訓練 per-step 成本裡 forward-Laplacian 佔多少」——ROM/降維方向可行性的前置量測。
#   零 code / 零 config 改動：train_kolmogorov.py 本來就印 `wall` 與 `n_co` 欄
#   (train_kolmogorov.py:1245,1368,1376)，n_collo 由 --n_collo_start/--n_collo_end CLI 覆寫。
#   線性擬合 t(N) ≈ t_fixed + c·N：c = Laplacian 每點邊際成本，t_fixed = forward+data+IO 固定成本。
#
# Notes:
#   - n_collo 一變會觸發 JAX recompile → 必須「每段固定 N、段間不 ramp」(start==end)。
#   - 每段 s=1 的 wall 吸收 JIT compile（超大，捨棄）；取 s>=200 的穩態區間平均。
#   - 訓練 flag 對齊 EXP-245 (B3 full-arch) 以反映真實訓練成本結構；只縮 steps、
#     關 eval/refine/ckpt、固定 n_collo、導到 job-scoped throwaway artifacts。
#   - 驗證內建：每段 log 的 `n_co` 欄必須等於該段設定的 N，否則 CLI flag 未生效 → 該段作廢。
#   - 這是 one-off profile，不走 submit_exp.sh（單段 template）；手動 sbatch。

set -euo pipefail

# ─── 1. 進到 repo root ────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

# ─── 2. 環境（對齊 train_exp.sbatch.tmpl）────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] PILNJAX_DATA_ROOT=$PILNJAX_DATA_ROOT 內無 data/ 目錄" >&2; exit 4; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH；先在 lab-server 裝 uv" >&2; exit 2; }
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；請先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="configs/exp_245_b3_les_T50.toml"
[ -f "$CONFIG" ] || { echo "[ERR] config 不存在：$CONFIG" >&2; exit 6; }

# ─── 3. 環境摘要 ──────────────────────────────────────────────────────
echo "======================================================================"
echo "[JAX-PROFILE-ncollo] $(date '+%F %T')  slurm_job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
echo "  repo:      $(pwd)"
echo "  config:    ${CONFIG}"
echo "  cuda vis:  ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  cpus:      ${SLURM_CPUS_PER_TASK:-?}"
echo "  mem:       ${SLURM_MEM_PER_NODE:-?} MB"
echo "  JAX_PLATFORMS=${JAX_PLATFORMS}  preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
uv --version
uv run python -c "import jax; print(f'JAX {jax.__version__}  devices={jax.devices()}')"

# ─── 3.5 preflight：cuDNN compile smoke（防 cuDNN drift 白跑）─────────────
echo "[preflight] cuDNN compile smoke..."
uv run python -c "import jax, jax.numpy as jnp; from jax import lax; lax.conv_general_dilated(jnp.ones((1,8,8,1)), jnp.ones((3,3,1,1)), (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC')).block_until_ready(); print('[preflight] cuDNN OK')" \
  || { echo '[ERR] cuDNN/GPU preflight 失敗（多半 nvidia-cudnn-cu12 漂到 9.23.1.3）→ head node 跑 scripts/slurm/setup_gpu_venv.sh 重建 venv（pin 9.23.0.39）後重送' >&2; exit 5; }

# ─── 4. profile 主體：掃 n_collo，每段固定 N ────────────────────────────
# throwaway artifacts（job-scoped，不污染正式 artifacts）
PROFILE_ART="/tmp/pilnn_profile_${SLURM_JOB_ID:-manual}"
STEPS=500          # 每段步數：s=1 吸收 compile，取 s>=200 穩態
LOG_EVERY=100      # per-step = wall / LOG_EVERY；多個穩態採樣點降噪

echo ""
echo "###### profile 開始：STEPS=${STEPS}  LOG_EVERY=${LOG_EVERY}  artifacts=${PROFILE_ART} ######"
for N in 1024 256 64 16; do
  echo ""
  echo "====================== profile n_collo=${N} ======================"
  # 訓練 flag 對齊 EXP-245，除了：--steps 縮短、--n_collo_start/end 固定、
  # --eval_every 0 / --save_every 0 / --refine_optimizer none 去掉非 per-step 成本。
  uv run python -u train_kolmogorov.py \
    --config "${CONFIG}" \
    --arch liquid \
    --optimizer schedule_free \
    --base_optimizer soap \
    --soap_precondition_frequency 2 \
    --use_gradnorm \
    --use_al \
    --use_time_marching \
    --tm_start_frac 0.1 \
    --tm_warmup_steps 2000 \
    --tm_ramp_steps 6000 \
    --n_collo_start "${N}" \
    --n_collo_end "${N}" \
    --steps "${STEPS}" \
    --log_every "${LOG_EVERY}" \
    --eval_every 0 \
    --save_every 0 \
    --refine_optimizer none \
    --artifacts_dir "${PROFILE_ART}/n${N}" \
    || { echo "[profile] n_collo=${N} 段失敗（rc=$?）；續跑下一段"; }
done

echo ""
echo "###### profile 結束 $(date '+%F %T') ######"
echo "判讀：每段取 s>=200 的 (wall/${LOG_EVERY}) 平均 = per-step t(N)；擬合 t(N)=t_fixed+c·N。"
echo "     驗證：每段 n_co 欄必須 == 該段 N。"
