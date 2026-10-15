#!/usr/bin/env bash
# 把壓力表的 run 清單展開成最終 sbatch，並備妥 lab-server job 目錄的本機鏡像。
# **只產生檔案，不送 job、不碰 lab-server。**
#
# What:
#   STAGE_DIR/
#     eval_pressure_limit.sbatch          ← eval_pressure_limit.sbatch.tmpl 展開
#     scripts/eval_pressure_limit.py      ← 評估腳本（sha256 會寫進每份輸出）
#     scripts/aggregate_pressure_limit.py ← 事後聚合用
#     logs/ out/                          ← slurm 不會自己建 --output 的目錄
#   整個目錄原樣 rsync 到 lab-server 的 JOB_DIR，就是 job 需要的全部。
#
# Why 清單寫在這裡而不是 template:
#   template 只描述「怎麼跑一個 run」；清單是這次重產的實驗設計，每一列的路徑都在
#   2026-09-30 以 ssh 唯讀查證過（summary.json 的 arch=liquid、config_path，
#   metrics.json 的 ckpt_dir／ckpt_step=20000／config／uv_rel_err）。改清單前先重查。
#
# Usage:
#   scripts/slurm/render_eval_pressure_limit.sh STAGE_DIR
#   ssh lab-server mkdir -p jobs/pressure_eval_20260930   # ~/jobs 目前不存在
#   rsync -a STAGE_DIR/ lab-server:jobs/pressure_eval_20260930/
#   # 之後在 lab-server head node：cd ~/jobs/pressure_eval_20260930 && sbatch eval_pressure_limit.sbatch
set -euo pipefail

STAGE="${1:?用法：$0 STAGE_DIR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(dirname "${HERE}")"
TMPL="${HERE}/eval_pressure_limit.sbatch.tmpl"
JOB_DIR="${JOB_DIR:-/home/junyi/jobs/pressure_eval_20260930}"
REPO="${REPO:-/home/junyi/pi-lnn-jax}"
A="${REPO}/artifacts/kolmogorov"
C="${REPO}/configs"

# arm|artifacts_dir|config|expect_metrics
RUNS=(
  # K=100 主線（AL on，μ=0.1），既有評估 = eval_v2_chunk400k
  "K100_al|${A}/pv_les_s42|${C}/exp_pv_les_s42.toml|${A}/pv_les_s42/eval_v2_chunk400k/metrics.json"
  "K100_al|${A}/pv_les_s1|${C}/exp_pv_les_s1.toml|${A}/pv_les_s1/eval_v2_chunk400k/metrics.json"
  "K100_al|${A}/pv_les_s2|${C}/exp_pv_les_s2.toml|${A}/pv_les_s2/eval_v2_chunk400k/metrics.json"
  "K100_al|${A}/pv_les_s3|${C}/exp_pv_les_s3.toml|${A}/pv_les_s3/eval_v2_chunk400k/metrics.json"
  "K100_al|${A}/pv_les_s4|${C}/exp_pv_les_s4.toml|${A}/pv_les_s4/eval_v2_chunk400k/metrics.json"
  # K=200 主線
  "K200_al|${A}/ksweep_k200_s42|${C}/exp_ksweep_k200_s42.toml|${A}/ksweep_k200_s42/eval_v2_chunk400k/metrics.json"
  "K200_al|${A}/ksweep_k200_s1|${C}/exp_ksweep_k200_s1.toml|${A}/ksweep_k200_s1/eval_v2_chunk400k/metrics.json"
  "K200_al|${A}/ksweep_k200_s2|${C}/exp_ksweep_k200_s2.toml|${A}/ksweep_k200_s2/eval_v2_chunk400k/metrics.json"
  "K200_al|${A}/ksweep_k200_s3|${C}/exp_ksweep_k200_s3.toml|${A}/ksweep_k200_s3/eval_v2_chunk400k/metrics.json"
  "K200_al|${A}/ksweep_k200_s4|${C}/exp_ksweep_k200_s4.toml|${A}/ksweep_k200_s4/eval_v2_chunk400k/metrics.json"
  # K=200 no-AL（job 5704，--ablate_no_al），唯一的評估 = final_eval（2026-09-12，含 uv_rel_err）
  "K200_noal|${A}/k200_noal|${C}/exp_k200noal_b3.toml|${A}/k200_noal/final_eval/metrics.json"
  # K=100 no-AL：tab:al_strength 的 K=100 AL-off 臂（--ablate_no_al；config 與 exp_pv_les_s42 只差
  # artifacts_dir），final_eval 的 uv_rel_err = 0.27273 即稿上 27.27%。不用 ablate_no_al_s42：
  # 它唯一的評估（2026-07-04）早於 uv_rel_err，沒有指紋可對。
  "K100_noal|${A}/alrhooff_v2|${C}/exp_alrhooff_v2.toml|${A}/alrhooff_v2/final_eval/metrics.json"
)

[ -f "${TMPL}" ] || { echo "[ERR] 找不到 ${TMPL}" >&2; exit 2; }
mkdir -p "${STAGE}/scripts/_common" "${STAGE}/logs" "${STAGE}/out"
cp "${SCRIPTS}/eval_pressure_limit.py" "${SCRIPTS}/aggregate_pressure_limit.py" "${STAGE}/scripts/"
# eval_pressure_limit 從 _common 匯入壓力與梯度的定義；漏複製會在 import 時失敗
cp "${SCRIPTS}/_common/__init__.py" "${SCRIPTS}/_common/periodic_fields.py" "${STAGE}/scripts/_common/"

OUT="${STAGE}/eval_pressure_limit.sbatch"
p_job='@JOB_DIR@'; p_repo='@REPO@'
while IFS= read -r line || [ -n "${line}" ]; do
  if [ "${line}" = "@RUN_TABLE@" ]; then
    printf '  "%s"\n' "${RUNS[@]}"
    continue
  fi
  line="${line//"${p_job}"/${JOB_DIR}}"
  line="${line//"${p_repo}"/${REPO}}"
  printf '%s\n' "${line}"
done < "${TMPL}" > "${OUT}"

if grep -n '@[A-Z_]*@' "${OUT}"; then
  echo "[ERR] ${OUT} 仍有未展開的 placeholder" >&2
  exit 3
fi
bash -n "${OUT}"

echo "[render] ${OUT}（${#RUNS[@]} runs）"
echo "[render] eval_pressure_limit.py sha256 = $(shasum -a 256 "${STAGE}/scripts/eval_pressure_limit.py" | cut -d' ' -f1)"
echo "[render] 下一步（人工）：ssh lab-server mkdir -p ${JOB_DIR#/home/junyi/} && rsync -a ${STAGE}/ lab-server:${JOB_DIR#/home/junyi/}/"
