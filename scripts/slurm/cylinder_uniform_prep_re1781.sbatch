#!/bin/bash
#SBATCH --job-name=cyl_unif_prep
#SBATCH --output=logs/cyl_unif_prep_%j.out
#SBATCH --error=logs/cyl_unif_prep_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# What: 完全均勻（farthest-point quasi-uniform）placement 資料準備（CPU）：從 Re=1781 shard
#       產 K=400 uniform sensors + dump cylinder_re1781_uniform.npz。
# Why : placement ablation 第 4 臂（uniform vs wake / boundary / SDF）；測 cross-attention 對
#       均勻相對位置的假設 + Nyquist 均勻取樣。依賴鏈同 cylinder_sdf_prep_re1781（整條鏈已在本專案）。
set -uo pipefail
# 2026-08-21：qrpivot / CylinderDataset / Arrow I/O 已移植進本專案
# （scripts/_common/），不再 cd 到 pi-lnn 借它的 venv。
SELF="${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
ND=/home/junyi/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical
PY=$SELF/.venv/bin/python
cd "$SELF"
export PYTHONPATH=.

RE=1781
SHARD="$ND/data-00020-of-00092.arrow"
[ -f "$SHARD" ] || { echo "[ERR] $SHARD 不存在" >&2; exit 4; }
echo "[cyl_unif_prep] $(date '+%F %T') job=${SLURM_JOB_ID:-?} RE=$RE SHARD=$SHARD"

# 1) uniform placement（K=400 farthest-point on full fluid，無 bias）
$PY "$SELF/scripts/gen_sensors_uniform.py" --shards "$SHARD" --K 400 \
  --out data/cylinder_sensors \
  || { echo "[ERR] gen uniform sensors"; exit 5; }
STEM="sensors_uniform_K400_cylinder_Re${RE}"
[ -f "data/cylinder_sensors/${STEM}.json" ] || { echo "[ERR] ${STEM}.json 未生成"; exit 6; }

# 2) dump 訓練 npz（sensor coords/values + body + DNS eval 場 + normalization）
$PY "$SELF/scripts/legacy/dump_cylinder_v1.py" --K 400 --sensor_stem "$STEM" \
  --arrow "$SHARD" --re "$RE" --out "$SELF/data/cylinder_re1781_uniform.npz" \
  || { echo "[ERR] dump"; exit 7; }

echo "=== uniform prep 完成 ==="
ls -la "$SELF/data/cylinder_re1781_uniform.npz"
