"""Training pipeline —— 集中化接線層。兩個**對等**的 case，各自擁有完整的一條鏈。

本檔是路標，不是鏈圖：先在這裡挑出你要的 case，鏈圖在該 case 自己的 `__init__.py`。
兩張圖不放在這裡並排，是因為它們沒有共用的骨架可以合併（見下方「刻意不共用」），
並排只會讓讀者以為有。

═══════════════════════════════════════════════════════════════════════════
我要改哪一案？
───────────────────────────────────────────────────────────────────────────
  Kolmogorov flow（多-Re、time-marching、RAR、checkpoint / resume / refine）
    入口 train_kolmogorov.py  →  鏈圖 pi_lnn_jax/pipeline/kolmogorov/__init__.py

  Cylinder wake（CEXP-002 / controlled_cylinder；1-shot、wall BC、無 checkpoint）
    入口 train_cylinder.py    →  鏈圖 pi_lnn_jax/pipeline/cylinder/__init__.py
═══════════════════════════════════════════════════════════════════════════

共用的是**協定形狀**（兩案都實作，簽名同形；spec wave2 §4）：

    resolve_inputs(argv=None) -> ResolutionResult  CLI + TOML → typed config + provenance
    build_context(config)     -> TrainingContext   建構期：只組裝依賴
    run_training(ctx)         -> TrainResult       執行期：跑完並回結果

呼叫端因此看到同一套 Interface 形狀，只是 import 路徑不同：

    from pi_lnn_jax.pipeline.kolmogorov import resolve_inputs, build_context, run_training
    from pi_lnn_jax.pipeline.cylinder   import resolve_inputs, build_context, run_training

**統一的是形狀與共用元件，不是共用的迴圈碼。** 六個名字在兩案是同名不同物：
`TrainingContext` 的欄位集不同、`TrainResult` 的欄位集不同、`resolve_inputs` 解析的
是兩套不同的旗標。跨案傳遞這些物件沒有意義，型別系統也不會擋你。

真正共用的東西（改這些會同時影響兩案）
───────────────────────────────────────────────────────────────────────────
  pipeline/_ledger.py   RNG/schedule ledger —— 兩案共用的 first-divergence oracle。
                        由 PILNN_RNG_LEDGER 環境變數 gate，預設關閉、production 零 overhead。
                        紀律：只記錄既有計算結果，不得新增 RNG split / 取樣 / 陣列重算。

  5 個 leaf module（兩案都直接呼叫，皆位於 pi_lnn_jax/）：
    config.py      resolve_config —— TOML schema、ordered-layer engine 與 provenance
    physics.py     make_ns_residual_fn —— NS residual
    losses.py      GradNorm（gradnorm_init|step|weights）、AL（al_init|update）
    optimizers.py  build_optimizer —— schedule_free / SOAP / fallback
    evaluate.py    Kolmogorov 用 evaluate_against_dns、cylinder 用 compute_metrics
  各案另有自己的 leaf（Kolmogorov: ckpt/curriculum/data/refiners/sensor_dropout/
  causal/model_factory；cylinder: boundary/models 直建），清單見各案 `__init__.py`。

  建構期／執行期的邊界紀律 —— 兩案同一條規則，各有一份 AST 測試守：
    建構期（`assembly.py`）只組裝依賴：**不得**出現 RNG split、取樣或任何 state
    初始化；`model.init` / `tx.init` / `gradnorm_init` / `al_init` 一律在執行期。
    `TrainingContext` 因此不持有 params / opt_state。
      Kolmogorov: tests/test_pipeline_assembly.py
      cylinder:   tests/test_cylinder_assembly.py

刻意**不共用**的東西（想合併之前先讀這段）
───────────────────────────────────────────────────────────────────────────
  `run_loop` / `_plan_step` / `build_context` 本體 / eval —— 兩案各一份。

  不是還沒抽，是抽不動：兩案的 **RNG 形狀**與**迴圈骨架**本身就不同，而兩者都被
  bit-identical 契約釘死（各有 golden fixture 與論文證據背書）：

    RNG 初始化   Kolmogorov 是兩個各自建構的 RandomState（同 seed，故 init_t 不是
                 init_xy 的續抽）；cylinder 是**一個**生成器連抽兩次。
    迴圈 stream  Kolmogorov 的 rng_collo / rng_re / rng_crp 由主流 split 衍生；
                 cylinder 的 PRNGKey(seed+1) / RandomState(seed+7) 由 seed 重新建構。
    每步消費     Kolmogorov 是條件式 split；cylinder 是**無條件** 4-way + 3-way
                 （`k3` 在 time-marching 路徑上抽出但不使用——仍必須抽）。
    迴圈範圍     Kolmogorov `range(start+1, steps+1)`；cylinder `range(steps + 1)`（含第 0 步）。
    階段         Kolmogorov 有 restore / refine；cylinder 沒有（無 checkpoint）。

  把這些合進一個 `run_loop`，只能靠 `if is_cylinder:` 分支去保住兩邊的既有行為——
  那不是統一，是把兩份邏輯壓進同一個函式再用旗標拆開，讀者要同時載入兩案才敢改一案。
  平行結構讓每一案可獨立推理，代價是兩份迴圈碼；這個取捨是刻意的
  （knowledge/superpowers/specs/2026-07-30-pipeline-wave2-cylinder-design.md §1、§2、§4）。

  等到出現**第三個** case、且共同骨架由三份真實實作浮現出來，再談抽取；
  預先設計介面正是 wave 1 決定延後這個接縫的理由。

搬移確實改變了的一件事：**標準輸出的行序**
───────────────────────────────────────────────────────────────────────────
  建構期的訊息現在全部提前——`build_context` 一次跑完才輪到 `run_training`，
  而單體版是邊建邊跑。實例：Kolmogorov 的 `Model: … params=…` 與 `Optimizer: …`
  在 `28b007a` 分別位於 980 與 1030 行（Model 先）；現在 Optimizer 屬建構期、
  Model 屬執行期，順序**顛倒**。

  **決定（2026-07-31）：接受，不還原。** 還原等於把建構期的訊息延後到執行期才印，
  也就是在 `build_context` 裡留一份「待印清單」交給 `run_training`——為了 log 行序
  在兩個階段之間長出一條狀態通道，代價大於收益。這是純顯示層變更：無腳本解析這些行，
  落盤的 summary 與 ledger 皆不受影響，bit-identical 契約也不涵蓋 stdout。

  代價要說清楚：「把新舊 log 逐行 diff」這個廉價的健全性檢查因此失效。
  要對拍請用 `scripts/slurm/verify_*_cpu_ab.sbatch.tmpl`——它比的是 ledger、
  params/opt digest 與評估產物，本來就不看行序。
═══════════════════════════════════════════════════════════════════════════

本檔頂層 re-export 的六個名字是 **Kolmogorov 的**（wave 1 只有一案時留下的相容別名）。
名字同形會撞，故 repo 內一律走 case-qualified 路徑；cylinder 只能從
`pi_lnn_jax.pipeline.cylinder` 匯入，這裡不提供、也不會提供 cylinder 的同名別名。
"""
from __future__ import annotations

from pi_lnn_jax.pipeline.kolmogorov import (
    KolmogorovEffectiveConfig,
    TrainResult,
    TrainingContext,
    build_context,
    resolve_inputs,
    run_training,
)

__all__ = [
    "KolmogorovEffectiveConfig",
    "TrainResult",
    "TrainingContext",
    "build_context",
    "resolve_inputs",
    "run_training",
]
