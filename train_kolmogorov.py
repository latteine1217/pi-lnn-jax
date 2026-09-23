"""Kolmogorov flow 訓練入口 —— 只有 CLI 進入點，實作全在 `pi_lnn_jax.pipeline.kolmogorov`。

先讀 `pi_lnn_jax/pipeline/kolmogorov/__init__.py` 的鏈圖拿全景（每個節點標明實作在
哪個檔），要細節再往 `config.py` / `assembly.py` / `run.py` 下鑽。
兩案的全景與「什麼共用、什麼刻意不共用」在 `pi_lnn_jax/pipeline/__init__.py`。

檔名與 CLI 旗標是對外介面，不得更動（`scripts/slurm/*.tmpl` 依賴）；
旗標清單以 `--help` 為準，正式提交走 `scripts/slurm/submit_exp.sh`。
"""
from __future__ import annotations

from pi_lnn_jax.pipeline.kolmogorov import build_context, resolve_inputs, run_training


def main() -> None:
    """CLI + TOML → 建構期依賴 → 訓練。

    本函式刻意不含任何邏輯（連 artifacts 目錄推導都不做——那屬 build_context，
    見 `TrainingContext.ckpt_dir`）：入口一旦開始做事，「讀 pipeline 就懂全鏈」
    的保證就破了，而且會長出第二份與 pipeline 打架的推導。

    `resolve_inputs(argv=None)` 讀 `sys.argv`，與 `run.replay_schedule` 走同一條
    解析路徑，不在此手動重組 —— 兩份會漂移，漂移後 replay oracle 會安靜失效。
    """
    resolved = resolve_inputs()    # CLI + TOML → typed config + provenance sidecar
    ctx = build_context(resolved.config)  # 建構期：資料/模型/loss/optimizer/step_fn/ckpt
    run_training(ctx)              # 執行期：init → restore → loop → refine → finalize


if __name__ == "__main__":
    main()
