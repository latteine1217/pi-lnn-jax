# scripts/ — 功能分類地圖

> 檔案**不做目錄重組**（sbatch template 與文件引用相對路徑，移動的破壞面 > 收益）；
> 本頁提供分類導引。訓練主入口在 repo root（`train_kolmogorov.py`、`train_cylinder.py`）。
>
> **寫或改腳本前先讀 [`CLAUDE.md`](CLAUDE.md)** — 共用件清單（別重寫已存在的東西）、
> 兩層落點原則、新增腳本規則與已知陷阱。

## 評估（high-risk，紅線見 AGENTS.md §7 / skill `eval-script-discipline`）

⚠️ 下列七支的 `--protocol` **必填、無預設**（`follow_training` / `fixed_grid` /
`sensor_time_independent`）。取樣旗標也不再有數值預設：給了是斷言，沒給由協定決定。
理由與三種模式各自的必要性見 `docs/adr/0003-evaluation-protocol.md`。

| 檔案 | 用途 |
|---|---|
| `evaluate_exp245.py` | 主 Kolmogorov checkpoint 評估（E(t) MAPE 等 headline 指標） |
| `evaluate_multi_re.py` | 多-Re manifest 評估（train/held-out split） |
| `evaluate_baselines.py` | classical baselines（interp/gappy-POD）鏡像評估；gappy 預設 validation 選模態 |
| `eval_gappy_cross_re.py` | cross-Re gappy（pooled train-Re basis，公平設定） |
| `ke_timeseries_errors.py` | KE 時序誤差 |
| `backfill_ksweep_metrics.py` | K-sweep 指標回填 |
| `compare_baselines.py` | 四方對照合成（PI-CON/gappy/SHRED/interp） |
| `aggregate_pv_campaign.py` | placement-variance campaign 聚合（σ_training/σ_placement） |
| `cost_accuracy.py` | Tier 0.2 cost-vs-accuracy 曲線（sbatch: `slurm/eval_cost_accuracy.sbatch.tmpl`） |
| `classical_baselines_fair.py` | tab:fair 的 trig-LSQ/RBF/IDW baselines（sbatch: `slurm/eval_tabfair.sbatch.tmpl`） |

## 訓練（baseline）

| 檔案 | 用途 |
|---|---|
| `train_baseline_shred.py` | SHRED baseline 訓練+評估（GPU，lab-server） |

## 診斷

| 檔案 | 用途 |
|---|---|
| `diag_observability.py` | K-placement 可觀測性（κ、r_99、MI 資訊量層） |
| `diag_pressure_observability.py` | 壓力場可觀測性 |
| `diag_subspace_transfer.py` | 跨-Re 子空間轉移診斷 |
| `diag_offgrid_query.py` | 在 sensor 時戳**之間**查詢，量「continuous-time」宣稱的代價（三個評估協定都做不到 off-grid，見該檔 docstring） |
| `diag_spatial_emb_bandwidth.py` | 訓練後 `LearnableFourierEmb` 的可達諧波階（Jacobi–Anger）；量 trunk 基底頻寬是否涵蓋 k≥16。⚠️ 不要用它做「感測極限 vs 架構」的二分歸因——該問法已於 2026-09-01 撤回（見 `knowledge/research/negative-results.md`） |
| `run_cp_analysis.py` | conformal prediction 分析（§6.6 / Appendix E） |
| `dump_cp_fields.py` | CP 場 dump |
| `bench_folx.py` | folx AD benchmark |
| `plot_extrapolation_leadtime.py` | 單臂外推 lead-time 曲線 + persistence／rel-L2=1／√2 參考線；曲線與判讀量落 JSON（sbatch: `slurm/replot_extrap.sbatch`） |
| `plot_extrap_arms.py` | 跨臂合成圖：baseline 多 seed 畫成雜訊帶、其餘臂畫線，併報 win_fraction（只吃各臂 JSON，不需 DNS） |
| `diag_steady_state.py` | 這條 DNS 有沒有進入統計穩態（平台 + 仍在波動，雙向判定）；投入穩態窗實驗前的閘門（sbatch: `slurm/probe_steady_state.sbatch`） |
| `diag_difficulty_curve.py` | 與模型無關的難度尺：逐時 training-free 內插重建誤差 + 三個流場複雜度指標（sbatch: `slurm/difficulty_curve.sbatch`） |
| `diag_chaos_budget.py` | 混沌下限：把模型在 t₀ 的重建場交給真實求解器積分，量「起點誤差相同時任何正確外推的最好結果」＋求解器保真度控制組（sbatch: `slurm/chaos_budget.sbatch`） |
| `verify_dns_extension.py` | 延長版 DNS 與既有 DNS 是否同一條軌跡（時間外推實驗的真值前提；判 PASS/FAIL 並落 `.verify.json`） |

## Sensor 生成

| 檔案 | 用途 |
|---|---|
| `gen_sensors_uniform.py` | uniform / FPS space-filling placement（placement-uniformity 判別實驗用） |
| `gen_sensors_boundary.py` | cylinder boundary-biased placement |
| `gen_sensors_sdf_isocontour.py` | cylinder SDF 等值線 placement |
| `gen_sensors_re10000_T20.py` | Re=10⁴ 主 case sensor 集 |

上列前三支**不再跨 repo**：pi-lnn 依賴已於 2026-08-21（`bf08ff2`）消除，QR-pivot 的
cylinder 實作已移植為 `_common/qrpivot_cylinder.py`，在本 repo env 直接執行即可
（該檔用 `pyarrow` 做 Arrow I/O，已在 `uv sync` 的依賴內）。用法見各檔 docstring 與
[`CLAUDE.md`](CLAUDE.md) §4。共用件在 `_common/cylinder_sensors.py` 與
`_common/qrpivot_cylinder.py`。

## 繪圖（論文 figures）

`plot_ksweep.py`、`plot_energy_spectrum.py`、`plot_observability_figures.py`、
`plot_placement_kappa_mi.py`、`plot_nyquist_recoverability.py`、
`plot_forward_cfd_anisotropy.py`、`plot_exp_snap_drop.py` 走 `journal_style.py`。

same-IC LES 那三支（2026-09-17 補寫，收投稿前稽核的 `A2-M05`／`A2-M06`／`A2-N06`；
重建配方見 [`docs/paper-figure-provenance.md`](../docs/paper-figure-provenance.md) §3.2）：
`plot_les_sameic_predictability.py`、`plot_dns_les_vorticity_compare.py`、
`plot_les_sensor_placement.py`。三支都吃大檔池的全場（要設 `PILNJAX_DATA_ROOT`），
算式與前提檢查由 `tests/test_plot_les_sameic_figures.py` 釘住。

尚未接上 `journal_style` 的四支（風格不一致，碰到時順手接）：
`plot_cylinder_recon.py`、`plot_cylinder_re_sweep.py`、`plot_multi_re_crp_compare.py`、
`plot_multi_re_generalization.py`。`generate_architecture.py`（架構示意圖）另計。

## _common/

只有腳本需要、不屬於 `pi_lnn_jax` 函式庫的共用件：

- `cylinder_sensors.py` — cylinder placement 幾何推導 + sensor set 輸出 schema。
- `sameic_les_dns.py` — same-IC LES／DNS 場對的載入與前提檢查（時間軸逐點相同、
  網格同形、`t=0` 的場一致）。**只檢查不修正**：兩支求解器的速度號約定相反時要修
  生成端，在繪圖端自動翻號會把「讀錯檔」也一起修掉。
- `ab_compare.py` — 重構前後對拍的判定（ledger / commit 斷言 / summary / 評估報表）。
  兩支 `slurm/verify_*_cpu_ab.sbatch.tmpl` 共用；也可獨立當 CLI 跑。
  行為由 `tests/test_ab_compare.py` 釘住——判定寫在 sbatch 的 heredoc 裡就永遠測不到。

落點原則與完整共用件清單見 [`CLAUDE.md`](CLAUDE.md)。

## slurm/

- `submit_exp.sh` — **一般 job 的正規提交入口**（`DRY=1` 先看生成的 sbatch；`TMPL=` 可覆寫 template）
- `submit_verify.sh` — **驗收 job 的提交入口**（`kolmo` / `cyl` / `ckpt`）。
  查拓撲、做前置（fetch／切 worktree／清未追蹤證據檔）、解析兩個 sha，再轉給 `submit_exp.sh`。
  **不要手組驗收指令**——手打的 `EXPECT_BASE` 是拿記憶當斷言依據，實際白跑過一輪。
- `setup_gpu_venv.sh` — GPU venv 唯一準備入口（cuDNN pin 9.23.0.39，見 skill `gpu-venv-bringup`）
- `*.sbatch.tmpl` — 參數化 template（train_exp / train_exp_multire / train_shred / eval_baselines / eval_gappy_cross_re）
- **驗收 template**（`pipeline/` 的 bit-identical 對拍；規則見 [`../CLAUDE.md`](../CLAUDE.md) §7.1）：
  `verify_cpu_ab.sbatch.tmpl`（Kolmogorov fresh run）、`verify_cyl_cpu_ab.sbatch.tmpl`（cylinder）、
  `verify_ckpt_compat.sbatch.tmpl`（舊 ckpt 相容 / 驗收層 B）。
  - **編排**（前置檢查／清空／determinism／弱路徑門檻／對拍呼叫）在 `_verify_common.sh`，
    由各模板 `source`。新增一段驗收面只改那一處，不必逐支模板套。
  - **判定**在 `_common/ab_compare.py`。兩者都**不要寫回 heredoc**——那裡的程式碼
    只在 job 內執行、本機測試永遠碰不到，而那正是要拿它來背書 bit-identical 的地方。
  - 各模板只留該案專屬的東西：資料根／symlink、`record()`、fixture→config 對照。
    兩案的錄製方式本就不同（一案 ledger 直接落在 `--artifacts_dir`，另一案沒有該旗標、
    ledger 固定落點後需搬移），硬收進共用層只會長出 `if case ==` 分支。
- `gen_dns_extended.sbatch` — 重跑 Kolmogorov DNS 到更長 T_end（N=1024→ds4）並驗證共同
  時間窗是否同一條軌跡。兩臂靠 CLI 覆寫：預設 GPU（`torch-cuda`，快）／
  `--partition=r620 --gres=none --export=ALL,DNS_BACKEND=numpy`（backend 與原始檔相同）。
- 其餘 `*.sbatch` — 一次性 job 的固定版（保留作歷史記錄）

## legacy/

已淘汰入口（保留可重現性）：`train_cylinder_v1.py`（含已否證的 `--mode geo`，勿作新工作起點）、
`dump_cylinder_v1.py`、`generate_picon_architecture_imagegen.py`。
