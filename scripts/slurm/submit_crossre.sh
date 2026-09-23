#!/usr/bin/env bash
# submit_crossre.sh — 送出 cross-Re × sensor-budget campaign 的 15 個訓練 job。
#
# 用途：
#   Re ∈ {100, 500, 1000, 10000, 1e6} × K ∈ {10, 50, 100}，FPS placement，
#   對應 thesis 的 cross-Re sensor-budget 圖。config 由
#   scripts/gen_crossre_configs.py 生成（先跑它，本腳本只負責提交）。
#
# 為什麼不直接迴圈 submit_exp.sh：
#   每個 case 的 config 不同（Re × K），需逐一挑選。
#
# 2026-08-06：本腳本原本用 sed 從 config 撈 time_strides 再以 EVAL_TIME_STRIDE
#   傳下去，理由是「eval 必須與訓練落在同一時間格點，否則五條線不能並列」。
#   那個判斷是對的，但實作在 shell 裡——只有本腳本這條路徑受保護，其他
#   呼叫端不受保護（實測 12 份既有產物因此不符，見 technical-debt TD-4）。
#   現在改由 evaluation_protocol 的 follow_training 模式負責：eval 端自己讀
#   config 的 time_strides，CLI 若不一致即失敗。sed 因此刪除。
#
# 用法：
#   scripts/slurm/submit_crossre.sh                  # 送全部 15 個
#   DRY=1 scripts/slurm/submit_crossre.sh            # 只生成 sbatch，不送出
#   RE=100,500 scripts/slurm/submit_crossre.sh       # 只送指定 Re
#   K=10 scripts/slurm/submit_crossre.sh             # 只送指定 K
#   RE=1e6 DRY=1 scripts/slurm/submit_crossre.sh     # 檢查 Re=1e6 那組
#
# 注意：Re=1e6 的 DNS 與 sensor 不在本機開發機上，只在 lab-server；
#       送出前先用 DRY=1 確認該處檔案存在。
set -euo pipefail

cd "$(dirname "$0")/../.."

RE_LIST="${RE:-100,500,1000,10000,1e6}"
K_LIST="${K:-10,50,100}"

submitted=0
skipped=0

for re in ${RE_LIST//,/ }; do
  for k in ${K_LIST//,/ }; do
    config="configs/exp_crossre_re${re}_k${k}.toml"
    if [ ! -f "$config" ]; then
      echo "[skip] $config 不存在（先跑 scripts/gen_crossre_configs.py）" >&2
      skipped=$((skipped + 1))
      continue
    fi

    echo "=== Re=${re} K=${k} (eval protocol=follow_training) ==="
    EVAL_PROTOCOL=follow_training \
      scripts/slurm/submit_exp.sh "crossre_re${re}_k${k}" "$config"
    submitted=$((submitted + 1))
  done
done

echo
echo "[submit_crossre] 提交 ${submitted} 個，略過 ${skipped} 個"
if [ "${DRY:-0}" = "1" ]; then
  echo "[submit_crossre] DRY=1；以上皆未實際送出"
fi
