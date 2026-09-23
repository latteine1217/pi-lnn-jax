#!/bin/bash
# Foil 2D-compatibility gate — run on lab-server HEAD NODE.
#
# WHY head node (not sbatch/r740): this step (a) DOWNLOADS one Foil numerical
# shard, which needs internet — only the head node has it (r740 is UV_OFFLINE);
# (b) runs a read-only numpy divergence diagnostic (seconds, no GPU, no autograd).
# Neither is training. Per lab-server-slurm policy this is head-node prep + check.
#
# GOAL: decide whether Foil (3D WaterLily sim, spanwise slice) is 2D
# divergence-free enough for our 2D NS residual. Compare Foil's relative
# divergence r against the 2D-generated cylinder baseline (0.020-0.068).
#   r < 0.15  -> PDE-residual route valid for Foil; proceed to pipeline.
#   r > 0.15  -> Foil is 3D-contaminated; PDE-residual invalid, rethink.
#
# Results are written to logs/foil_2d_gate.{log,json} — cat them yourself; do NOT
# trust any chat-relayed summary if the session shows injection.
#
# Prereq: `huggingface-cli login` once (RealPDEBench is CC BY-NC; may need to
# accept the license on the HF dataset page first).
#
# Usage:
#   cd ~/pi-lnn-jax && git pull        # get scripts/check_2d_divergence.py
#   bash scripts/run_foil_2d_gate.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

DATA_ROOT="${DATA_ROOT:-$HOME/RealPDEBench/data/realpdebench}"
mkdir -p "$DATA_ROOT" logs

echo "[1/3] locating + downloading one Foil numerical shard (head node, needs internet)"
TARGET=$(uv run --with huggingface_hub python - "$DATA_ROOT" <<'PY'
import os, sys
from huggingface_hub import list_repo_files, hf_hub_download
repo = "AI4Science-WestlakeU/RealPDEBench"
data_root = sys.argv[1]
files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
               if f.startswith("foil/hf_dataset/numerical/") and f.endswith(".arrow"))
if not files:
    sys.exit("no foil numerical shards found (license accepted? logged in?)")
target = files[0]
p = hf_hub_download(repo, target, repo_type="dataset", local_dir=data_root)
print(p)
PY
)
echo "    downloaded: $TARGET"

echo "[2/3] running 2D-compatibility gate on Foil shard"
# also re-run the cylinder baseline in the SAME clean invocation if present,
# so Foil is compared against a freshly-computed 2D reference (not a relayed number).
CYL="$DATA_ROOT/cylinder/hf_dataset/numerical/data-00020-of-00092.arrow"
CTRL=$(ls "$DATA_ROOT"/controlled_cylinder/hf_dataset/numerical/*.arrow 2>/dev/null | head -1 || true)
ARGS=("$TARGET")
[ -f "$CYL" ] && ARGS+=("$CYL")
[ -n "${CTRL:-}" ] && ARGS+=("$CTRL")   # also re-check controlled (local read failed under injection)

uv run --with pyarrow python scripts/check_2d_divergence.py \
    --arrow "${ARGS[@]}" \
    --out logs/foil_2d_gate.json 2>&1 | tee logs/foil_2d_gate.log

echo "[3/3] done. Inspect results yourself (do not trust relayed summaries):"
echo "    cat logs/foil_2d_gate.log"
echo "    cat logs/foil_2d_gate.json"
echo "Decision: Foil r < 0.15 (near cylinder 0.02-0.068) -> 2D-OK, proceed to Foil pipeline."
