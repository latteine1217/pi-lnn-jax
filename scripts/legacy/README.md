# scripts/legacy/

歸檔的歷史腳本。**非死碼刪除**——以 `git mv` 移入此處保留可重現性與來源歷史。
要復原任一檔，`git mv scripts/legacy/<path> <原位置>` 即可。

歸檔日期：2026-06-20

## 群組

- **A. 版本化重複 (`*.py`)**
  - `train_cylinder_v1.py` — cylinder 可行性探針（Adam + 固定權重），已由根目錄 `train_cylinder.py`（CEXP-002）取代。
  - `dump_cylinder_v1.py` — v1 資料 dump。
  - `generate_picon_architecture_imagegen.py` — codex image_gen 架構圖（舊方法），產物 `architecture_imagegen.png` 論文未引用；deterministic 版 `scripts/generate_architecture.py` 為現用。

- **B. cylinder sbatch（`slurm/`）** — `train_cylinder_*.sbatch`、`cyl_*.sbatch`。
  cylinder 已定為誠實 negative result（論文仍引用結論）；整條訓練/提交工作流凍結於此。

- **C. re10000 迭代 sbatch（`slurm/`）** — `train_re10000_T20_*.sbatch`。
  multi-Re Kolmogorov 早期手寫的單體版本（v1/v2/v3 + pin/gauge/inputonly/pw 等變體），
  已由 `scripts/slurm/train_exp_multire.sbatch.tmpl` + `submit_exp.sh` 取代。

- **D. bench/profile sbatch（`slurm/`）** — `bench_*.sbatch`、`profile_*.sbatch`。
  一次性效能量測腳本。

## 未納入歸檔（刻意保留）

- `scripts/generate_architecture.py` — deterministic 論文架構圖生成器，仍是被追蹤的論文資產來源。
- 所有 `scripts/eval_*` / `plot_*` / `train_kolmogorov.py` / `train_cylinder.py` 與 `slurm/*.tmpl` — 活躍核心。
