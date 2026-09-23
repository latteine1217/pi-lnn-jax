#!/usr/bin/env bash
# scripts/slurm/setup_gpu_venv.sh
#
# What:
#   準備（或修復）GPU 訓練用 .venv 的「唯一正規入口」。在 lab-server head node 跑
#   （需 internet）。每個 worktree 各自 .venv，故每個要跑 GPU 的 worktree 各跑一次。
#
# Why:
#   GPU 的 jax[cuda12] 刻意 out-of-band（不在 uv.lock，保護 macOS sync）。但裸的
#   `uv pip install "jax[cuda12]==0.10.1"` 對 nvidia-cudnn-cu12 無上界 → 會解析到最新
#   patch（9.23.1.3），與 jaxlib 0.10.1 不相容：job 在 XLA 編譯時丟
#   `RET_CHECK failure ... dnn_support != nullptr` 秒級失敗。
#   本 script 把 cuDNN pin 到已知良好的 9.23.0.39，並寫在同一條 install 讓 uv 一起解析，
#   根除「每次 reinstall 又重抓壞版本」。對齊 CLAUDE.md <cudaJax>。
#
# Usage:
#   cd ~/<worktree> && scripts/slurm/setup_gpu_venv.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH（head node 先裝 uv）" >&2; exit 2; }

CUDNN_PIN="nvidia-cudnn-cu12==9.23.0.39"   # 已知良好；9.23.1.3 與 jaxlib 0.10.1 不相容

echo "[1/3] uv sync（CPU 依賴 + Python 3.12）..."
uv sync --python 3.12

echo "[2/3] GPU jax plugin（out-of-band）+ pin cuDNN：${CUDNN_PIN}"
# 同一條 install：uv 一起解析 → cuDNN 被釘住，不會抓到最新壞版本
uv pip install "jax[cuda12]==0.10.1" "${CUDNN_PIN}"

echo "[3/3] 驗證 cuDNN 版本..."
ls -d .venv/lib/python3.12/site-packages/nvidia_cudnn_cu12-* 2>/dev/null | xargs -n1 basename \
  || { echo "[ERR] 找不到 nvidia-cudnn-cu12，安裝可能失敗" >&2; exit 3; }
uv run python -c "import jax; print('JAX', jax.__version__)" 2>/dev/null \
  || echo "[note] head node 無 GPU，import 檢查略過；真正 GPU 編譯驗證由 sbatch preflight 把關"

echo "[ok] GPU venv ready（cuDNN pinned 9.23.0.39）。"
