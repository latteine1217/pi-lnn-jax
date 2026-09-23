#!/bin/bash
#SBATCH --job-name=jax_gnt_cov
#SBATCH --output=logs/jax_gnt_cov_%j.out
#SBATCH --error=logs/jax_gnt_cov_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   在**固定 ckpt** 上掃 GradNorm probe 的 data 項截斷長度 n0，量 G_data 怎麼變。
#   不重訓、不改任何 state。
#
# Why:
#   台帳原記「權重的 K 依賴主要是 grad_accum 依賴（69%）」。讀 assembly.py:725-729 後
#   那個除法站不住：n0 = n_collo/M **與 K 無關**，而攤平是 time-major，所以
#     - M 決定 probe 拿到幾個點（n0）
#     - K 決定那 n0 個點橫跨幾個時刻（n0/K）
#   已有的兩個比較（M 效應 @K=100、K 效應 @M=4）**走的是同一個機制**，拆成兩個因子
#   再比大小沒有意義。而且 2x2 缺 (K=200, M=1) 那一格——那是唯一能看到「K 作為感測
#   預算」的量測。
#
#   本 job 補兩件事：
#     1. (K=200, M=1) 全量 20,200 點，填滿 2x2。
#     2. **時間涵蓋配對**：K=100 @ n0=256 與 K=200 @ n0=512 的涵蓋同為 2.56 個時刻。
#        若配對後 G_data 相近 → 控制量是涵蓋，K 本身不進 probe；若仍差 → K 有效果，
#        現行敘述要再改一次。三對配對（2.56 / 5.12 / 10.24）互相檢核。
#
# 判讀：n_probe 與 probe_time_coverage 已寫進每份 JSON，判讀時必須配著看。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1 PYTHONPATH=".:scripts"
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -d "$PILNJAX_DATA_ROOT/data" ] || { echo "[ERR] 無 data/" >&2; exit 4; }
[ -x .venv/bin/python ] || { echo "[ERR] 無 .venv" >&2; exit 3; }

OUT=artifacts/diag/gnt_coverage
mkdir -p "$OUT"
# K=200 用 ksweep_k200_s42（LES-QR，headline 那條線）。
# **不要**用 exp245k200_b3_les_T50_jax_20k：它是 4-task GradNorm 時代訓的
# （gradnorm_state.task_names 磁碟上 4 個），現行 cont_gradnorm=False 建 3 個，
# orbax 樹不合而還原失敗（job 5705 實測）。不得用 partial_restore 繞過。
K200_CFG=configs/exp_ksweep_k200_s42.toml

echo "=============================================================="
echo "[gnt_cov] $(date '+%F %T') job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "=============================================================="

# cell <tag> <config> <grad_accum> <n_collo|-> [artifacts_dir]
cell () {
  tag="$1"; cfg="$2"; M="$3"; nc="$4"; ad="${5:-}"
  extra=""
  [ "$nc" != "-" ] && extra="$extra --n_collo_end $nc"
  [ -n "$ad" ]     && extra="$extra --artifacts_dir $ad"
  echo ""; echo "######## $tag  (M=$M n_collo=$nc) ########"
  uv run python -u scripts/diag_gradnorm_target.py \
      --config "$cfg" --resume latest --grad_accum "$M" \
      --out "$OUT/$tag.json" $extra \
    || echo "[WARN] $tag 失敗（rc=$?）——續跑其餘格"
}

# ── K=100（ckpt 在本 worktree）────────────────────────────────────────
# SKIP_K100=1 跳過 K=100 那半（job 5705 已完成，產物在 $OUT）
if [ "${SKIP_K100:-0}" != "1" ]; then
cell k100_M1_full  configs/exp_pv_les_s42.toml    1 -
cell k100_n0_256   configs/exp_pv_les_s42.toml    4 1024
cell k100_n0_512   configs/exp_pv_les_s42.toml    4 2048
cell k100_n0_1024  configs/exp_pv_les_s42.toml    4 4096
cell k100_n0_2048  configs/exp_pv_les_s42.toml    4 8192
fi

# ── K=200（ckpt 在 multire worktree）─────────────────────────────────
cell k200_M1_full  "$K200_CFG" 1 -
cell k200_n0_256   "$K200_CFG" 4 1024
cell k200_n0_512   "$K200_CFG" 4 2048
cell k200_n0_1024  "$K200_CFG" 4 4096
cell k200_n0_2048  "$K200_CFG" 4 8192

echo ""; echo "[$(date '+%F %T')] 完成 → $OUT"
ls -1 "$OUT"
