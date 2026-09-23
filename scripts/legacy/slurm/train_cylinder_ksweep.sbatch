#!/bin/bash
#SBATCH --job-name=jax_cyl_k
#SBATCH --output=logs/cyl_k_%j.out
#SBATCH --error=logs/cyl_k_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# 半資源配置：2×8 CPU / 2×48 GB / 2×1 GPU = 16/96/2 ≤ 48/108/2 ✓
#
# Sensor 數 (K) sweep：基底=v3+ALM(ρ=0.3)（KE 最佳 0.420，body soft）。
# 測 v 退步是否為 K=100 資訊論天花板——QR pivot K=100/200/400 補 wake 點，看 v 是否改善。
# DATA 由 env 傳（cylinder_v1=K100 / cylinder_K200 / cylinder_K400）。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
# fof collocation 分塊：vector attention 大 K（200/400）下中間張量 ∝N×K×hidden 會 OOM；
# lax.map 序列處理每 256 點，峰值降到 ∝CHUNK×K×hidden。可保完整 N_collo=1024。
export PILNN_FOF_CHUNK="${PILNN_FOF_CHUNK:-256}"
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在" >&2; exit 3; }

DATA="${DATA:-data/cylinder_v1.npz}"
STEPS="${STEPS:-16000}"
# fof full-hessian over N×K 隨 K 爆記憶體；增 K 時減 collocation N 平衡（K sweep 統一 N 才公平）
N_COLLO="${N_COLLO:-512}"
[ -f "$DATA" ] || { echo "[ERR] $DATA 不存在" >&2; exit 4; }

AL_FLAGS="--use_al --al_rho 0.3 --al_freeze_start 5000 --al_freeze_end 15000 --al_update_freq 100"
FS_FLAG=""
[ "${FREESTREAM:-0}" = "1" ] && FS_FLAG="--freestream_ref"
FED_FLAG=""
[ -n "${FED:-}" ] && FED_FLAG="--fourier_embed_dim $FED"
# DUMP=<npz>：eval 時存第一個 val 時刻的 pred/dns 場供畫重建圖（無 ckpt，重現用同 seed 重跑）。
# 注意：boundary K400（vector attn 大 K）需 PILNN_FOF_CHUNK + DATA_CHUNKS 降記憶體，否則 step0 OOM。
DUMP_FLAG=""
[ -n "${DUMP:-}" ] && DUMP_FLAG="--dump_fields $DUMP"

echo "[cyl_k] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} DATA=$DATA STEPS=$STEPS FS=${FREESTREAM:-0}"
nvidia-smi -L 2>/dev/null | head -1
uv run python -c "import jax; print(f'JAX {jax.__version__} backend={jax.default_backend()}')"

echo; echo "######## geo + BC + vector + ALM(ρ=0.3)，DATA=$DATA ########"
uv run python -u train_cylinder_v1.py \
  --data "$DATA" --mode geo --steps "$STEPS" \
  --attention_kind vector \
  --use_bc --w_bc 0.1 --bc_body_w 4.0 --bc_n 128 \
  --w_phys 0.01 --data_chunks "${DATA_CHUNKS:-1}" --n_collo "$N_COLLO" \
  $AL_FLAGS $FS_FLAG $FED_FLAG $DUMP_FLAG \
  || echo "[WARN] DATA=$DATA rc=$?"

echo
echo "[$(date '+%F %T')] cyl_k (DATA=$DATA) 完成"
