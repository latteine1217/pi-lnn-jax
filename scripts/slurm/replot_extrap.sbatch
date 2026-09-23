#!/bin/bash
#SBATCH --job-name=replot_extrap
#SBATCH --output=logs/replot_extrap_%j.out
#SBATCH --error=logs/replot_extrap_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=r620
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# What:
#   用現版 scripts/plot_extrapolation_leadtime.py 重畫既有 eval 產物的 lead-time
#   圖與 JSON。不重跑評估——吃的是 series.npz / fields.npz（已在 artifacts 裡）。
#
# Why:
#   圖面與判讀量會演進（撤下無依據的參考線、換判準），但重跑一次評估要 40 分鐘 GPU。
#   這支只讀既有陣列 + DNS，CPU 一分鐘。
#
# 用法（ARMS 以空白分隔 artifacts 子目錄名；DNS 可覆寫）：
#   sbatch scripts/slurm/replot_extrap.sbatch
#   ARMS="exp530_b3_extrap_T10" sbatch scripts/slurm/replot_extrap.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONUNBUFFERED=1

ARMS="${ARMS:-exp530_b3_extrap_T10 exp531_b3_extrap_T10_collo2x}"
SUBDIR="${SUBDIR:-extrap_T10}"
T_DATA_END="${T_DATA_END:-5.0}"
DNS="${DNS:-data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T10_dt2p5e4_si100_ds4_cuda.npy}"

[ -f "$DNS" ] || { echo "[ERR] DNS 不存在：$DNS" >&2; exit 2; }

for arm in $ARMS; do
  d="artifacts/kolmogorov/${arm}/${SUBDIR}"
  [ -f "$d/series.npz" ] || { echo "[ERR] 缺 $d/series.npz" >&2; exit 3; }
  echo "=== $arm ==="
  # chaos_budget.json 有就帶上（物理下限入圖）；沒有就照舊，不 fail
  cb_arg=()
  [ -f "$d/chaos_budget.json" ] && cb_arg=(--chaos-budget "$d/chaos_budget.json")
  .venv/bin/python scripts/plot_extrapolation_leadtime.py \
    --series "$d/series.npz" \
    --fields "$d/fields.npz" \
    --dns "$DNS" \
    "${cb_arg[@]}" \
    --t-data-end "$T_DATA_END"
done
