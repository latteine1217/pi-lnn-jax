"""Kolmogorov flow 訓練流水線 —— 這個 case 的對外協定與完整鏈圖。

與 `pi_lnn_jax.pipeline.cylinder` 是**對等**的兩個 case：協定形狀相同、內部實作
不共用。兩案的分工與「什麼共用、什麼刻意不共用」見上層 `pipeline/__init__.py`。

    resolve_inputs(argv=None) -> ResolutionResult  kolmogorov/config.py
    build_context(config)     -> TrainingContext   kolmogorov/assembly.py
    run_training(ctx)         -> TrainResult       kolmogorov/run.py

讀下面這張圖即可掌握整條鏈：每個節點都標明**實作在哪個檔**，要細節才往下跳。
`train_kolmogorov.py` 只是 CLI 進入點，本身不含邏輯。

═══════════════════════════════════════════════════════════════════════════
CLI argv + TOML
  └─ resolve_inputs(argv=None)                     kolmogorov/config.py
       ├─ parse CLI policy                         kolmogorov/config.py
       ├─ load_config()                            pi_lnn_jax/config.py
       ├─ merge_config_layers()                    pi_lnn_jax/config.py
       └─ build KolmogorovEffectiveConfig          kolmogorov/config.py
       → ResolutionResult(config, provenance)；優先序：default < TOML < CLI

建構期 —— 只組裝依賴，不含任何 runtime decision
（無 RNG split／取樣／state 初始化；邊界由 tests/test_pipeline_assembly.py 以 AST 守）
  └─ build_context(config)                         kolmogorov/assembly.py
       ├─ artifacts_dir / ckpt_dir                 kolmogorov/assembly.py（產物落點的唯一推導點）
       ├─ _load_datasets → _make_re_batch          kolmogorov/assembly.py → per-Re ReBatch
       ├─ _load_dns_for_eval + 時間軸對齊檢查      kolmogorov/assembly.py → dns_u/v/t（訓練中監控用）
       ├─ _build_crp_interp / re_norm_scale_of     kolmogorov/assembly.py（連續-Re 物理的 norm_stats 內插表）
       ├─ build_model(arch, model_kwargs, K)       pi_lnn_jax/model_factory.py
       ├─ make_ns_residual_fn                      pi_lnn_jax/physics.py → ns_fn / poisson_fn
       │  make_ns_residual_fn_baseline             pi_lnn_jax/physics.py → ns_fn_baseline（B0/PINN 路徑）
       ├─ _build_loss_fn                           kolmogorov/assembly.py（data + NS + Poisson + AL + causal + CRP）
       ├─ build_optimizer                          pi_lnn_jax/optimizers.py → tx / opt_info
       ├─ _build_grad_norm_fn                      kolmogorov/assembly.py（weighting 關閉時為 None）
       ├─ step_fn = jit(value_and_grad(loss_fn) → tx.update)          kolmogorov/assembly.py
       │    └─ grad_accum_chunks>1 → accumulate_grads                 pi_lnn_jax/optimizers.py
       └─ CheckpointManager(ckpt_dir)              pi_lnn_jax/ckpt.py
       → TrainingContext（只有依賴；不持有 params / opt_state）

執行期 —— state 以 TrainingState 顯式傳遞，流程是一條直線、不用 class 包
  └─ run_training(ctx)                             kolmogorov/run.py
       ├─ initialize  model.init → tx.init → gradnorm|al init       主 RNG #0-#1
       ├─ restore     僅 --resume_step：ckpt 還原 + 續跑 sanity check
       │                （用 rng_check 而非 rng → **不推進主流**）
       ├─ run_loop    _open_loop_streams: rar_init / rng_collo / rng_re / rng_crp
       │                                                             主 RNG #2-#5
       │                每步 _plan_step（host-side 決策全集，spec §5.4 順序）：
       │                  抽 Re → CRP re_norm' → 課程(n_collo, w_phys, t_upper)
       │                  → sensor mini-batch → dropout mask
       │                  → collocation(RAR | uniform) → 四個觸發旗標
       │                然後：step_fn → weighting update → AL update → log
       │                  → NaN fail-fast → ckpt save → mid-eval
       ├─ refine      lbfgs | lm | gn | none       kolmogorov/run.py + pi_lnn_jax/refiners.py
       └─ finalize    final eval → final ckpt → summary.json / eval_history.json
       → TrainResult(state, final_results, eval_history, summary, summary_path,
                     history_path)

診斷（不在 production 路徑上）
  └─ replay_schedule(golden_rows)                  kolmogorov/run.py
       重播 host-side 決策序列，不呼叫 step_fn、不做 forward/gradient，
       與 run_loop 共用同一份 `_plan_step` / `_ledger_fields`（兩份會漂移）。
       ledger 本身在 pipeline/_ledger.py，由 PILNN_RNG_LEDGER 環境變數 gate。
═══════════════════════════════════════════════════════════════════════════

Leaf 模組速查（「這個功能實作在哪」，皆位於 pi_lnn_jax/）
  建構期 kolmogorov/assembly.py 呼叫：
    data.py            load_multi_re_sensors / load_dns（+ *_from_path）
    model_factory.py   build_model —— 訓練端與 eval 端共用的唯一入口
    physics.py         make_ns_residual_fn / make_ns_residual_fn_baseline
    causal.py          causal_weights（use_causal 時進 loss）
    sensor_dropout.py  apply_sensor_dropout（loss 內套 mask）
    losses.py          al_constraint_value（loss 內算 AL 懲罰）
    optimizers.py      build_optimizer / accumulate_grads
    ckpt.py            CheckpointManager
  執行期 kolmogorov/run.py 呼叫：
    curriculum.py      physics_weight_at_step / time_marching_t_max /
                       rar_init / rar_sample
    losses.py          gradnorm_init|step|weights、al_init|update
    sensor_dropout.py  make_keep_mask（每步重抽的 dropout mask）
    ckpt.py            TrainState（on-disk 投影，_to_ckpt_state 唯一入口）
    evaluate.py        evaluate_against_dns（mid-eval + final eval）
    refiners.py        lbfgs_refine / lm_refine / gn_refine

改動前必讀：
  kolmogorov/run.py 模組 docstring           —— RNG 消費不變量（bit-identical 契約的承重面）
  kolmogorov/assembly.py 模組 docstring      —— 建構期／執行期的硬性邊界
  kolmogorov/config.py 模組與 ADR —— effective-config policy 與相容邊界
  knowledge/superpowers/specs/2026-07-29-training-pipeline-design.md —— 設計文件（wave 1）

診斷開關：`PILNN_RNG_LEDGER=1` → 在 artifacts_dir 落 rng_ledger.json。

本檔只 re-export 三段協定用到的六個名字；階段函式（`initialize` / `restore` /
`run_loop` / `refine` / `finalize`）與 `replay_schedule` 直接從
`pi_lnn_jax.pipeline.kolmogorov.run` 匯入。
"""
from __future__ import annotations

from pi_lnn_jax.pipeline.kolmogorov.assembly import TrainingContext, build_context
from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovEffectiveConfig, resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.run import TrainResult, run_training

__all__ = [
    "KolmogorovEffectiveConfig",
    "TrainResult",
    "TrainingContext",
    "build_context",
    "resolve_inputs",
    "run_training",
]
