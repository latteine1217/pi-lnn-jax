#!/bin/bash
#SBATCH --job-name=flow_snapshots
#SBATCH --output=logs/flow_snapshots_%j.out
#SBATCH --error=logs/flow_snapshots_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=r620
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#
# What: 畫渦度場快照面板（CPU-only；DNS 檔可能數 GB，記憶體開大）。
# Why:  這條軌跡的 enstrophy 跨越 144→5.4→67，看數字難以理解「評估窗長什麼樣」。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cpu

DNS="${DNS:?需指定 DNS=...}"
TIMES="${TIMES:-0 5 10 20 40 60 80 100}"
OUT_DIR="${OUT_DIR:-artifacts/kolmogorov/_snapshots}"
STEM="${STEM:-flow_snapshots}"
SENSOR_ARG=()
[ -n "${SENSOR_JSON:-}" ] && SENSOR_ARG=(--sensor-json "$SENSOR_JSON")

echo "[snapshots] $(date '+%F %T') dns=$DNS times=$TIMES"
uv run python -u scripts/plot_flow_snapshots.py \
  --dns "$DNS" --times $TIMES "${SENSOR_ARG[@]}" --out-dir "$OUT_DIR" --stem "$STEM"
