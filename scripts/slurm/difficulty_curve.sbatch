#!/bin/bash
#SBATCH --job-name=difficulty_curve
#SBATCH --output=logs/difficulty_curve_%j.out
#SBATCH --error=logs/difficulty_curve_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=r620
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# What:
#   量「從 K 個點重建這個流場，隨時間變得多容易」——與模型無關的難度尺。
#
# Why:
#   時間外推的 τ 與流場難度在本軌跡上完全糾纏（衰減暫態，越晚越低維）。
#   有了這條曲線，任何「模型晚期比較好」的觀察才能先除掉難度成分。
#
# CPU-only：training-free 散點內插 + FFT，不需要 GPU。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
export JAX_PLATFORMS=cpu

DNS="${DNS:-data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T10_dt2p5e4_si100_ds4_cuda.npy}"
SENSOR_JSON="${SENSOR_JSON:-data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json}"
OUT_DIR="${OUT_DIR:-artifacts/kolmogorov/_difficulty}"

echo "[difficulty] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
uv run python -u scripts/diag_difficulty_curve.py \
  --dns "$DNS" --sensor-json "$SENSOR_JSON" --time-stride 2 --out-dir "$OUT_DIR"
echo "[difficulty] done $(date '+%F %T')"
