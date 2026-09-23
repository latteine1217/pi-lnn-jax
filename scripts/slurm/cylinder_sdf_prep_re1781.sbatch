#!/bin/bash
#SBATCH --job-name=cyl_sdf_prep
#SBATCH --output=logs/cyl_sdf_prep_%j.out
#SBATCH --error=logs/cyl_sdf_prep_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# What: SDF 等值線 placement 資料準備（CPU）：從 Re=1781 shard（data-00020, sim_id=1781）產
#       K=400 SDF-isocontour sensors（8 條線性等值線 farthest-point）+ dump 訓練 npz
#       cylinder_re1781_sdf.npz。
# Why : A 實驗（SDF placement vs wake / boundary K400）。2026-08-21 起整條鏈都在本專案：
#       qrpivot / CylinderDataset / Arrow I/O 移植到 scripts/_common/，sensor 檔落
#       本專案 data/cylinder_sensors（已 gitignore），dump 由此讀。
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
echo "[cyl_sdf_prep] $(date '+%F %T') job=${SLURM_JOB_ID:-?} RE=$RE SHARD=$SHARD"

# 1) SDF isocontour placement（K=400，8 條線性等值線）
$PY "$SELF/scripts/gen_sensors_sdf_isocontour.py" --shards "$SHARD" --K 400 \
  --n-levels 8 --out data/cylinder_sensors \
  || { echo "[ERR] gen SDF sensors"; exit 5; }
STEM="sensors_sdf_iso8_K400_cylinder_Re${RE}"
[ -f "data/cylinder_sensors/${STEM}.json" ] || { echo "[ERR] ${STEM}.json 未生成"; exit 6; }

# 2) dump 訓練 npz（sensor coords/values + body + DNS eval 場 + normalization）
$PY "$SELF/scripts/legacy/dump_cylinder_v1.py" --K 400 --sensor_stem "$STEM" \
  --arrow "$SHARD" --re "$RE" --out "$SELF/data/cylinder_re1781_sdf.npz" \
  || { echo "[ERR] dump"; exit 7; }

echo "=== SDF prep 完成 ==="
ls -la "$SELF/data/cylinder_re1781_sdf.npz"
