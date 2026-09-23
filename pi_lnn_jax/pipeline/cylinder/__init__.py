"""Cylinder wake（CEXP-002 / controlled_cylinder）訓練流水線 —— 這個 case 的對外協定與完整鏈圖。

與 `pi_lnn_jax.pipeline.kolmogorov` 是**對等**的兩個 case：協定形狀相同、內部實作
不共用（RNG 形狀、迴圈骨架、eval 都不同；強行共用只會長出 `if is_cylinder:`）。
兩案的分工與「什麼共用、什麼刻意不共用」見上層 `pipeline/__init__.py`。

    resolve_inputs(argv=None) -> ResolutionResult  cylinder/config.py
    build_context(config)     -> TrainingContext   cylinder/assembly.py
    run_training(ctx)         -> TrainResult       cylinder/run.py

這個 case 在做什麼（CEXP-002「物理正確性」trainer，非 `scripts/legacy/train_cylinder_v1.py`
那支 Adam + 固定權重的可行性探針）：
  - inter-task 權重用 GradNorm（data / ns_u / ns_v 三 task，ref 子樹=`temporal_encoder`）；
    continuity 不進 weighted task，純由 Augmented Lagrangian 約束。
  - optimizer 走 schedule_free + SOAP，CPU 時 `optimizers.py` 自動 fallback。
  - physics 用 `ns_residuals(A=0, k_f=0, Lx/Ly 物理尺度化, return_per_point=True)`，
    body mask 逐項 reduce。
  - wall BC（inflow / body no-slip / slip）為固定權重，置於 GradNorm 之外。
  - 1-shot 實驗：**禁止 resume**、無 checkpoint round-trip、不落任何 artifact。
  - 判生死用 KE（kinetic energy）為主判據，field-L2 / vorticity 為輔助。

讀下面這張圖即可掌握整條鏈：每個節點都標明**實作在哪個檔**，要細節才往下跳。
`train_cylinder.py` 只是 CLI 進入點，本身不含邏輯。

═══════════════════════════════════════════════════════════════════════════
CLI argv + TOML
  └─ resolve_inputs(argv=None)                     cylinder/config.py
       ├─ parse CLI + resume/backend guards        cylinder/config.py
       ├─ load_config()                            pi_lnn_jax/config.py
       ├─ merge_config_layers()                    pi_lnn_jax/config.py
       └─ build CylinderEffectiveConfig            cylinder/config.py
       → ResolutionResult(config, provenance)；優先序：default < TOML < CLI

建構期 —— 只組裝依賴，不含任何 runtime decision
（無 RNG／取樣／state 初始化；邊界由 tests/test_cylinder_assembly.py 以 AST 守）
  └─ build_context(config)                         cylinder/assembly.py
       ├─ 啟動 banner                              cylinder/assembly.py（印在載資料之前；stdout 順序即既有行為）
       ├─ np.load(config.data.npz_path)             cylinder/assembly.py（唯一資料入口；未設 → sys.exit(6)）
       │    spec §6【不修 1】刻意繞過 `_resolve_data_path` —— 改走專案解析會把
       │    「找不到就炸」變成「往別處找」
       │    → sv[K,T,C] → sv_TKC[T,K,C]（CfC 走時間軸）／sp[K,2]／st[T] 物理時間
       │    → RE_NORM / st0 / T_total / tm_end / nu=u_inf·D/Re
       │    → n_sensor_q = min(config 值, T*K)     ← 迴圈與 replay 只准讀這個夾過的值
       │    → t_q_full_np[T*K]（host；time-marching 的 valid 篩選用）
       ├─ CylinderGeometry | OscillatingGeometry   pi_lnn_jax/boundary.py（`is_controlled` 決定）
       ├─ LiquidOperator(**CFG, torch_style_init=True)   pi_lnn_jax/models.py
       │    spec §6【不修 2】刻意繞過 `model_factory.build_model`（factory 預設不同）
       ├─ make_ns_residual_fn(model)               pi_lnn_jax/physics.py → ns_residuals
       ├─ _data_loss / _ns_components              cylinder/assembly.py（per-task closure，各自重算 h=encode）
       ├─ REF_PATH=("temporal_encoder",) + _get_subtree   cylinder/assembly.py
       │    spec §6【不修 3】與 Kolmogorov 的 `trunk_out` 不同，兩者都是各自的既有行為
       ├─ compute_task_grad_norms = jit(...)       cylinder/assembly.py → ctx.grad_norm_fn
       ├─ loss_fn                                  cylinder/assembly.py
       │    ├─ data（sensor minibatch + t_early 前期加權）
       │    ├─ NS body-masked：fluid_mask | fluid_mask_moving        pi_lnn_jax/boundary.py
       │    ├─ wall BC 固定權重：wall_bc_loss | wall_bc_loss_moving  pi_lnn_jax/boundary.py
       │    └─ continuity：λ·c + ½ρc²（AL 項，不進 weighted task）
       ├─ build_optimizer                          pi_lnn_jax/optimizers.py → tx / opt_info
       └─ step_fn = jit(value_and_grad(loss_fn) → tx.update)         cylinder/assembly.py
       → TrainingContext（只有依賴；不持有 params / opt_state；**無 CheckpointManager**）

執行期 —— state 以 TrainingState 顯式傳遞，流程是一條直線、不用 class 包
（**沒有** Kolmogorov 的 restore / refine：cylinder 完全沒有 checkpoint）
  └─ run_training(ctx)                             cylinder/run.py
       ├─ initialize   spec §5.1：**單一** RandomState(seed) 連抽 init_xy / init_t
       │                 → model.init(PRNGKey(seed))
       │                 → gradnorm_init（3 task：data / ns_u / ns_v）→ al_init
       │                 → GradNorm ref 子樹 probe（jax.grad(ctx.data_loss_fn)，吃真實 params）
       │                 → tx.init
       │                → TrainingState（刻意無 `step` 欄位）
       ├─ run_loop     _open_loop_streams  spec §5.2：rk=PRNGKey(seed+1)、
       │                 np_rng=RandomState(seed+7)（只給 time-marching 的受限 sensor 取樣）
       │                 兩條都由 seed **重新建構**，不由 initialize 的 stream 衍生
       │                for s in range(steps + 1)  ← **含第 0 步**（與 Kolmogorov 不同）
       │                  _plan_step（host-side 決策全集，spec §5.3 順序，**無條件**）：
       │                    split(rk,4)→k1/k2/k3 → split(k1,3)→ks
       │                      ⚠ `k3` **無論如何都要抽**：time-marching 路徑不用它，但
       │                        「只在需要時才 split」會改變 rk 後續狀態 → 全鏈分岔
       │                    → t_max（time-marching 線性展開）→ cx/cy/ct = uniform(ks[0..2])
       │                    → sq_idx：use_tm ? np_rng.choice(valid) : choice(k3)
       │                    → wall BC(k2)：**is_controlled**（不是 use_tm）決定 moving / 靜態
       │                    → trig_gradnorm / trig_al（比 Kolmogorov 多一個 s > 0 條件）
       │                  然後：step_fn → gradnorm_step → al_update → log
       │                    → NaN fail-fast（break）
       │                ledger 尾列：params/opt_state digest + argv + config_path + _data_meta
       └─ finalize     DNS grid 分塊重建（EVAL_CHUNK=2048）→ compute_metrics 逐 eval 時刻
                       → KE 主判據 + freestream/wake 分區 over-energy + inflow band 報表
       → TrainResult(state, eval_metrics)   eval_metrics 是**逐時刻序列**，不是聚合值

診斷（不在 production 路徑上）
  └─ replay_plans(golden_rows)                     cylinder/run.py → [(step, _StepPlan)]
       從尾列 `argv` 重解 config（replay 端沒有第二個 config 來源），再走
       `build_context`；錄製當時的 npz 不在本機時改用尾列 `data_meta` 重建最小 ctx
       （`_ctx_from_data_meta`）。兩條路徑的驗證強度不同，實際走哪條印在 stdout
       （`REPLAY_CTX_SOURCE_*`）。與 run_loop 共用同一份 `_open_loop_streams` /
       `_plan_step`（兩份會漂移）。
     replay_schedule(golden_rows)                  cylinder/run.py → ledger 列
       只是把每個 plan 交給 `_ledger_fields` 攤平；需要原始陣列（例如鑑別 1-ULP
       跨平台差異）時用上面的 `replay_plans`。
       ledger 本身在 pipeline/_ledger.py，由 PILNN_RNG_LEDGER 環境變數 gate。
═══════════════════════════════════════════════════════════════════════════

Leaf 模組速查（「這個功能實作在哪」，皆位於 pi_lnn_jax/）
  輸入面 cylinder/config.py 呼叫：
    config.py          resolve_config（兩案共用 engine；case policy 與 typed config 不共用）
  建構期 cylinder/assembly.py 呼叫：
    models.py          LiquidOperator —— **直建**，刻意不走 model_factory
    boundary.py        CylinderGeometry / OscillatingGeometry、
                       fluid_mask(_moving)、wall_bc_loss(_moving)
    physics.py         make_ns_residual_fn
    optimizers.py      build_optimizer / is_soap_available
  執行期 cylinder/run.py 呼叫：
    losses.py          gradnorm_init|step|weights、al_init|update
    boundary.py        sample_wall_bc / sample_wall_bc_moving（每步 BC 取樣）、
                       body_center_at（eval 逐時刻重算 body 遮罩）
    models.py          LiquidOperator.encode / decode_query（eval 分塊重建）
    evaluate.py        compute_metrics（逐 eval 時刻；**不是** evaluate_against_dns）

  cylinder **不用** 的 leaf（用到就代表接錯了）：ckpt.py（無 checkpoint）、
  curriculum.py（time-marching 自帶，見 `_plan_step`）、data.py（直接 np.load）、
  model_factory.py、causal.py、sensor_dropout.py、lra.py、refiners.py。

改動前必讀：
  cylinder/run.py 模組 docstring           —— RNG 消費不變量（bit-identical 契約的承重面，
                                              有論文證據綁著：paper/tmlr-format/sections/
                                              07b_cylinder_feasibility.tex、appendix/C_cylinder.tex）
  cylinder/assembly.py 模組 docstring      —— 建構期／執行期的硬性邊界、spec §6 三條不修
  cylinder/config.py 模組與 ADR —— case policy 與相容邊界
  knowledge/superpowers/specs/2026-07-30-pipeline-wave2-cylinder-design.md —— 設計文件（wave 2）

怎麼證明改動沒破壞契約（本機禁跑 training，只能走這條）：
  tests/test_cylinder_replay.py  重播 host-side 決策序列比對 golden fixture，含
                                 「k3 無條件消費」的 c1/c2 對照證明；另行為級釘住
                                 §5.1 的 RandomState 連抽與 §5.2 的 seed+1 / seed+7。
                                 （`model.init` 的 PRNGKey 只由 fixture 的 params
                                 digest 覆蓋 —— 那要 lab-server 重錄，本機測不到。）
  tests/test_cylinder_ledger.py  ledger 欄位定義與 `_data_meta` 的不變量。
  tests/test_cylinder_assembly.py 建構期/執行期邊界（AST）。
  改到 RNG 就不是本機測試能收尾的：需在 lab-server 以
  `scripts/slurm/record_cyl_ledger.sbatch.tmpl` 重錄 fixture 並做 CPU A/B 對照
  （驗收流程見 spec §7）。

診斷開關：`PILNN_RNG_LEDGER=1` → 在 `config.run.artifacts_dir` 落 rng_ledger.json
（cylinder 本身不落任何 artifact，該目錄只在 gate 內讀）。
"""
from __future__ import annotations

from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext, build_context
from pi_lnn_jax.pipeline.cylinder.config import CylinderEffectiveConfig, resolve_inputs
from pi_lnn_jax.pipeline.cylinder.run import TrainResult, run_training

#: 對外只露協定三件組與其型別，與 kolmogorov 同形——兩案是對等 peer，
#: 匯出面不對稱會讓讀者以為協定不只三件。
#: 各階段（initialize / run_loop / finalize）與診斷用的 replay_schedule 都住在
#: `run.py`，要用就從那裡取；不在 package 層再露一次，免得呼叫端繞過協定。
__all__ = [
    "CylinderEffectiveConfig",
    "TrainResult",
    "TrainingContext",
    "build_context",
    "resolve_inputs",
    "run_training",
]
