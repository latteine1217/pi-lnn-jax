#!/bin/bash
#SBATCH --job-name=dump_ctrl_cyl
#SBATCH --output=logs/dump_ctrl_cyl_%j.out
#SBATCH --error=logs/dump_ctrl_cyl_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#
# What: RealPDEBench controlled-cylinder shard → data/controlled_v2.npz。
#       2026-08-21 起 CylinderDataset / Arrow I/O 已移植進本專案，用本專案 venv 跑
#       與 cylinder QR sensor placement；振幅由 pi_lnn_jax.controlled.infer_trajectory 反推。
# Why : sim_params=[Re,control_freq] 無振幅、z 幾何靜態 → 需從場反推 osc_amp/phase。
#       純 numpy/CPU（無需 GPU）。dump 印出的 reconstructed amp/phase 即 §4 驗證階梯 step-2
#       的軌跡反推 sanity 證據（預期 amp ~0.05-0.1 normalized）。
#
# Env override:
#   ARROW  已 stage 的 controlled shard 路徑（預設見下）
#   OUT    輸出 npz（預設 ~/pi-lnn-jax/data/controlled_v2.npz）
#   STEM   借用的 cylinder sensor placement stem
set -uo pipefail

JAXREPO="${JAXREPO:-$HOME/pi-lnn-jax}"
ARROW="${ARROW:-$JAXREPO/data/controlled_arrow/data-00002-of-00096.arrow}"
OUT="${OUT:-$JAXREPO/data/controlled_v2.npz}"
STEM="${STEM:-sensors_qrpivot_K100_cylinder_Re10031}"

[ -x "$JAXREPO/.venv/bin/python" ] || { echo "[ERR] $JAXREPO/.venv 不存在" >&2; exit 3; }
[ -f "$ARROW" ] || { echo "[ERR] arrow shard 不存在：$ARROW（先 stage）" >&2; exit 4; }

cd "$JAXREPO"
export PYTHONPATH=.

echo "[dump_ctrl_cyl] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "  ARROW=$ARROW"
echo "  OUT=$OUT  STEM=$STEM"

.venv/bin/python "$JAXREPO/scripts/dump_controlled_cylinder_v2.py" \
  --arrow "$ARROW" --out "$OUT" --sensor-stem "$STEM"

echo "[$(date '+%F %T')] dump_ctrl_cyl 完成 → $OUT"
