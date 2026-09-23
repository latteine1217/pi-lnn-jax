# configs/ — 實驗 config 地圖

> 本目錄的 TOML 是**實驗記錄的一部分**：每個 config 對應一個（批）已跑或進行中的
> 訓練/評估 run，是論文數字與 schema 預設值決策的可重現性依據。
> **原則：不刪除已跑過的 config**（即使 repo 內無文字引用——config 由
> `scripts/slurm/submit_exp.sh EXP_ID CONFIG_PATH` 從 CLI 傳入，本來就不會被程式碼引用）。
> Schema 與驗證見 `pi_lnn_jax/config.py`；合併順序：CLI override → TOML → schema default。

## 命名慣例

```
exp_<campaign>_<variant>[_s<training-seed>|_p<placement-seed>][_v<iteration>].toml
```

- `_s{42,1,2,3,4}`：training seed（42 為歷史主 seed；5-seed 集合 = {42,1,2,3,4}）
- `_p{42,1,2,3,4}`：placement seed（僅 random placement 臂使用）
- `_v1/_v2/_v3`：同一實驗的版本迭代（v 越大越新）
- `eval_*`：純評估 config（與訓練 config 配對）

## Campaign 分群

> **本表不是全集。** 目錄現有 432 份 TOML，本表只列與論文數字有對應、或其結論被
> `config.py` 的 schema 預設引用的群組。未列於此的 campaign（2×2 消融 `exp_abl_b*` /
> `exp_k200_abl_b*`、主線 `exp_main5_*`、CVT 佈點 `exp_pv_cvt_*`、噪聲 `exp_noisen*`、
> PINN 對照 `exp_k100_pinn*` / `exp_k200_pinn*`、10k 預算 `exp_abl10k_b*`、跨-Re
> `exp_crossre_*` / `exp_301_*` / `exp_302_*`、間歇時鐘 `exp_gap*`、AL 強度 `exp_alrho*`
> 等）一樣適用「不刪除已跑過的 config」原則。完整實驗敘事見
> `knowledge/experiments/experiment-index.md`；要反查某個 artifact 由哪份 config 產生，
> 用 `rg artifacts_dir configs/`。

| 群組 | 檔案 | 狀態 | 用途 / 論文對應 |
|---|---|---|---|
| **exp_245_b{1,2,3}** + `pinn_large` | 5 | 已定案 | 架構消融 B1/B2/B3 + 容量匹配 Standard PINN（512-wide）。論文 §6.2 `tab:ablation`（5-seed）。**`exp_245_pinn_large.toml` 是 PINN 對照的生成 config，不可刪。** ⚠️ 該 claim 已於 2026-09-03 改判為**條件式**：accepted 僅在 $K{=}200$，$K{=}100$ 為 not established（$p{=}0.24$）。舊的「1.9×」出自 10 000-iteration 的 PINN 值，不可與 20k 的 B3 並列——見 `knowledge/research/claims-ledger.md`。 |
| **exp_245_b3_les_T50_liquidtau** | 1 | weak claim | Liquid-τ on/off ablation（§6.2 `tab:liquidtau`，單 seed，claim 標 weak）。 |
| **exp_245k{50,200,400,800}** | 4 | 已定案 | K-sweep 的 K≠100 臂（K=100 = b3 本體）。§7.1 `tab:kscaling`（單 seed sweep）。 |
| **exp_ksweep_k{50,200,400,800}_s{...}** | 20 | 已定案 | K-sweep 5-seed 重跑（fig:ksweep error bar 版）。 |
| **exp_pv_{les,oracle}_s{...}** | 10 | 已定案 | Placement-variance campaign：同 placement × 5 training seeds（σ_training）。§7.4。jobs 4407–4421。 |
| **exp_pv_random_p{...}** | 5 | 已定案 | 5 個 random placement seed（σ_placement）。§7.4。 |
| **exp_pv_spacefill_s{...}** | 5 | 已定案 | GAP-1：FPS space-filling placement JAX 同軸 5-seed（jobs 4471–4475）。⚠️ 結論**指標相依**：KE 軸上 FPS 追平 oracle，主指標上 FPS 是五臂中較差的一端。判讀準則見 `knowledge/experiments/kolmogorov-placement-uniformity-2026-07-06.md`。 |
| **exp_re10000_T20_*** | 14 | 歷史迭代 | Re=10⁴ 主 case 的早期版本迭代（v1→v3、pin/gauge/inputonly/pw 變體、lra、random placement 對照、rho 掃描）。v3 系為 exp_245 系前身。保留作 lineage。 |
| **exp_510_modmlp / 512_signed** | 2 | 已否證 | 架構 ablation：modmlp 無效益、signed_mean 全面退步（div 暴增 3.2×）。結論已寫入 schema 預設。<br>（511_lra 已於 2026-08-03 連同 lra.py 移除——實測 LRA 與 GradNorm 數值等價，那不是對照臂而是同一個 controller 換名字。） |
| **exp_52{0-5}（rwf/τ-scale 掃描）** | 6 | 已定案 | rwf on/off × cfc_tau_mod_scale {0.5,1.0,2.0} 掃描。結論=**rwf on + τ on + scale 0.5**（`config.py` 註解引用 exp_521-524 為預設值依據，**不可刪**）。⚠️ 舊記的「rwf off」已被 n=5 判別推翻（主指標 −0.119 pp, $t=-4.49$，五 seed 範圍不重疊）；schema 預設自 2026-08-29 起為 `use_rwf = true`，2026-09-03 與 `models.py` 兩案對齊。 |
| **exp_ablate_no_al** | 1 | 已定案 | Augmented-Lagrangian on/off 消融。 |
| **exp_cyl_cexp002{,_notm,_re1781}** | 3 | 正面結果 + 邊界 | Cylinder wake。⚠️ 2026-08-28 裁決（mike）已由 feasibility boundary **升為正面結果**：$K{=}400$ 全域覆蓋下成立（KE 5.92 ± 0.92%, n=5），$K{=}100$ 欠感測仍失敗。勿再依「誠實 negative」的舊定位引用。 |
| **exp_multi_re_{poc,train5,train5_crp}** | 3 | 已否證 | 跨-Re 泛化（5train/3test）。held-out 不轉移（claims-ledger rejected）；CRP/re-conditioning 是錯槓桿。 |
| **eval_multi_re_train5{,_crp}** | 2 | 工具 | 上列的評估 config（`scripts/evaluate_multi_re.py`、baselines eval 重用）。 |
| **exp_snap_st{1,2,4,8}_s{42,1,2}** | 12 | 已定案 | 實驗1：data-snapshot 數量 sweep。訓練 `time_strides`∈{1,2,4,8}→T∈{201,101,51,26}，n=3 seeds。派生自 `exp_245_b3_les_T50.toml`（只改 time_strides/seed/artifacts_dir）。eval 端 `evaluate_exp245.py --time-stride 2` 固定格點（勿設 `EVAL_TIME_STRIDE`），train stride 為唯一變因。 |
| **exp_drop_b0_s{42,1,2}** | 3 | 已定案 | 實驗2 sensor-dropout robustness 的 B0 vanilla baseline（用 `ARCH=vanilla` 跑）。B3 對照重用 `exp_snap_st2_s{seed}`。dropout 機制：`pi_lnn_jax/sensor_dropout.py` zero-mask；train-time `--sensor_dropout_rate`（denoising，mask 輸入監督真值）、eval-time `evaluate_exp245.py --sensor_dropout_rate/realizations`。 |
| **mini_smoke** | 1 | 工具 | 最小 smoke config（僅 lab-server Slurm 用；本地不跑 smoke）。 |

## 新增 config 時

1. 從最接近的 baseline config 複製（主線 = `exp_245_b3_les_T50.toml`），diff 面越小越好。
2. 只改假設卡明確指出的參數；其餘（optimizer、lr、warmup、loss weighting）與 baseline 一致。
3. sensor / DNS 路徑逐一確認存在（eval silent-fail 的主要源頭）。
4. `DRY=1 scripts/slurm/submit_exp.sh <EXP_ID> <CONFIG>` 檢查生成的 sbatch。
5. 跑完後在 `knowledge/experiments/` 記錄（模板 `knowledge/_templates/experiment-record.md`），並回填本表。
