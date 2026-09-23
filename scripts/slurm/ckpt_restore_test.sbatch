#!/bin/bash
#SBATCH --job-name=jax_ckpt_test
#SBATCH --output=logs/ckpt_restore_test_%j.out
#SBATCH --error=logs/ckpt_restore_test_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#
# What: 驗證「刪掉 orbax 舊 step 後，最後一個 ckpt 仍可 restore」。
#   在一個非關鍵 Re=1000 測試 run 上：restore latest（前）→ 刪舊 step → restore latest（後）。
#   兩次都成功且 latest step 不變 = 安全，可大量刪。此 job 只動這一個測試 run。

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PYTHONPATH="${PYTHONPATH:-}:." UV_OFFLINE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

CK=artifacts/kolmogorov/exp301_b3_re1000_les_T100_jax_20k_s2/checkpoints
echo "[ckpt-test] $(date '+%F %T') target=$CK"
echo "  steps 現況: $(ls "$CK" | tr '\n' ' ')"

echo "=== [1] 刪除前 restore latest ==="
.venv/bin/python - "$CK" <<'PY'
import sys
from pi_lnn_jax.ckpt import CheckpointManager
m=CheckpointManager(directory=sys.argv[1], max_to_keep=10)
s=m.latest_step(); r=m.restore()
n=len(r) if hasattr(r,"__len__") else "?"
print(f"  BEFORE: all_steps={m.all_steps()} latest={s} restore_ok={r is not None} top_len={n}")
PY
rc1=$?

echo "=== [2] 刪除舊 step（保留 max）==="
keep=$(ls "$CK" | grep -E '^[0-9]+$' | sort -n | tail -1)
for s in $(ls "$CK" | grep -E '^[0-9]+$' | sort -n | head -n -1); do
  echo "  rm $CK/$s"; rm -rf "$CK/$s"
done
echo "  保留: $keep ; 現況: $(ls "$CK" | tr '\n' ' ')"

echo "=== [3] 刪除後 restore latest ==="
.venv/bin/python - "$CK" <<'PY'
import sys
from pi_lnn_jax.ckpt import CheckpointManager
m=CheckpointManager(directory=sys.argv[1], max_to_keep=10)
s=m.latest_step(); r=m.restore()
n=len(r) if hasattr(r,"__len__") else "?"
print(f"  AFTER:  all_steps={m.all_steps()} latest={s} restore_ok={r is not None} top_len={n}")
PY
rc3=$?

echo ""
if [ "$rc1" = 0 ] && [ "$rc3" = 0 ]; then
  echo "[ckpt-test] VERDICT: PASS — 刪舊 step 後 latest 仍可 restore。可安全大量刪。"
else
  echo "[ckpt-test] VERDICT: FAIL (rc1=$rc1 rc3=$rc3) — 停手，勿大量刪。"
fi
