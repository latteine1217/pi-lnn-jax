#!/bin/bash
#SBATCH --job-name=cyl_resweep_prep
#SBATCH --output=logs/resweep_prep_%j.out
#SBATCH --error=logs/resweep_prep_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Re-sweep 資料準備（CPU）：每個 Re 從已下載 shard 產 boundary K400 sensors + dump 訓練 npz。
# 前置：shard 由 head node hf download 落地於 RealPDEBench/.../numerical/（compute node 離線無網路）。
# 依賴鏈：2026-08-21 起 sensor gen + CylinderDataset 都在本專案（scripts/_common/）；
#         dump 產物仍落 $JAX（pilnn-cyl）。
set -uo pipefail
# 2026-08-21：qrpivot / CylinderDataset / Arrow I/O 已移植進本專案
# （scripts/_common/），不再 cd 到 pi-lnn 借它的 venv。
SELF="${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
JAX=/home/junyi/pilnn-cyl
ND=/home/junyi/RealPDEBench/data/realpdebench/cylinder/hf_dataset/numerical
PY=$SELF/.venv/bin/python
cd "$SELF"
export PYTHONPATH=.

# Re:shardIdx（sorted 每 5 個，到 Re~5000）。PAIRS 由 env 覆寫可換 scope。
PAIRS="${PAIRS:-2343:25 2906:30 3375:35 4031:40 4500:45 4968:50}"
echo "[prep] $(date '+%F %T') job=${SLURM_JOB_ID:-?} PAIRS=$PAIRS"

for pair in $PAIRS; do
  RE=${pair%%:*}; IDX=${pair##*:}
  SHARD=$(printf "$ND/data-%05d-of-00092.arrow" "$IDX")
  echo; echo "######## Re=$RE shard=$IDX ########"
  [ -f "$SHARD" ] || { echo "[SKIP] $SHARD 不存在"; continue; }
  # 1) boundary K400 placement（uniform + 邊界層 + wake QR）
  $PY scripts/gen_sensors_boundary.py --shards "$SHARD" --K 400 \
    --n-uniform 120 --n-boundary 120 --bl-thresh 0.03 --out data/cylinder_sensors \
    || { echo "[ERR] gen Re=$RE"; continue; }
  STEM="sensors_bnd120u120b_K400_cylinder_Re${RE}"
  [ -f "data/cylinder_sensors/${STEM}.json" ] || \
    STEM=$(basename "$(ls data/cylinder_sensors/sensors_bnd*K400_cylinder_Re${RE}.json 2>/dev/null | head -1)" .json)
  echo "  stem=$STEM"
  # 2) dump 訓練 npz（sensors + body + DNS eval 場 → pilnn-cyl/data）
  $PY "$JAX/scripts/dump_cylinder_v1.py" --K 400 --sensor_stem "$STEM" \
    --arrow "$SHARD" --re "$RE" --out "$JAX/data/cylinder_re${RE}_bnd.npz" \
    || { echo "[ERR] dump Re=$RE"; continue; }
  echo "  [OK] Re=$RE -> data/cylinder_re${RE}_bnd.npz"
done

echo; echo "=== resweep prep 完成 ==="
ls -la "$JAX"/data/cylinder_re*_bnd.npz 2>/dev/null
