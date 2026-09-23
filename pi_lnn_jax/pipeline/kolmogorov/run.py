"""執行期 —— `TrainingContext` → 訓練 → `TrainResult`。

What: `train_kolmogorov.main()` 的執行期段落（model.init / resume / 主迴圈 /
      refinement / final eval + 落盤）逐字搬入，拆成 module-level 函式，
      state 以 `TrainingState` 顯式傳遞。**不用 class 包裝流程**——
      流程只有一條直線，物件化只會把「誰改了哪個欄位」藏起來。

Why 這裡是整個重構的最高風險面：bit-identical 契約的破口幾乎都在 RNG
     消費時序。以下表格照抄 spec §5，是本檔的承重不變量；任何改動（包含
     「順手清理」）都必須先對照這張表。

═══════════════════════════════════════════════════════════════════════════
spec §5.1 主 stream `rng`
───────────────────────────────────────────────────────────────────────────
| # | 動作                                              | 條件                |
|---|---------------------------------------------------|---------------------|
| 0 | `rng = PRNGKey(eff["seed"])`                       | —                   |
| 1 | `rng, rng_init = split(rng)` → `model.init`        | 恆                  |
| — | `rng_check, sub_check = split(rng)`                | **只在 resume**；賦值給
|   |                                                   | `rng_check` 而非 `rng`
|   |                                                   | → **不推進主流**。此
|   |                                                   | 中立性必須保留      |
| 2 | `rng, rng_rar = split(rng)` → `rar_init(randint())`| 恆                  |
| 3 | `rng, rng_collo = split(rng)`                      | 恆                  |
| 4 | `rng, rng_re = split(rng)`                         | 恆                  |
| 5 | `rng, rng_crp = split(rng)`                        | **僅 `use_continuous_re_physics`**；
|   |                                                   | 旗標關閉時主流不推進 |
| 6 | `rng, sub = split(rng)` → refine 固定 collocation  | 僅 refine 啟用      |

spec §5.2 獨立 NumPy stream（易被「順手清理」破壞）
───────────────────────────────────────────────────────────────────────────
    init_xy = np.random.RandomState(eff["seed"]).uniform(0, 1, (8, 2))
    init_t  = np.random.RandomState(eff["seed"]).uniform(0, T_total, (8,))

**兩次各自建構新的 `RandomState`，用同一個 seed。** 因此 `init_t` **不是**
`init_xy` 的續抽，兩者都取自各自生成器的第一批亂數。合併成單一
`RandomState` 會改變 `init_t` 的值。

spec §5.3 獨立 refine stream
───────────────────────────────────────────────────────────────────────────
    rng_refine = PRNGKey(eff["seed"] + 7919)

spec §5.4 迴圈內消費順序（條件性是承重的）
───────────────────────────────────────────────────────────────────────────
每步依序，且**每一項的條件都必須保留**——條件不成立時不得消耗任何 key：

1. `rng_re, sub_re = split(rng_re)`     → `re_idx`      僅 `n_datasets > 1`
2. `rng_crp, sub_crp = split(rng_crp)`  → `re_norm_p`   僅 CRP 啟用
3. `rng_collo, sub_sq = split(rng_collo)`   → `sensor_idx`   僅 `n_sensor_query > 0`
4. `rng_collo, sub_drop = split(rng_collo)` → dropout mask   僅 `sensor_dropout_rate > 0`
5. `rng_collo, sub = split(rng_collo)` → `split(sub, 3)` → `cx, cy, ct`   恆
   （第 5 點的 split **無條件**發生，RAR 觸發時 `sub` 被丟棄不用——
    這正是 RAR 步與非 RAR 步之間 `rng_collo` 仍然對齊的原因。）
═══════════════════════════════════════════════════════════════════════════

Step planning（`_plan_step`）刻意留在本檔而非抽成獨立 pure function：
它依賴 current `params`（RAR 依 residual 選點）與 `rar_state`，從來就不是
純函式。`run_loop` 與 `replay_schedule` **呼叫同一個 `_plan_step`**，不另寫
一份——兩份會漂移，而漂移後 replay 會安靜地不再測真正在跑的東西
（`4792b93` 治的正是這種病）。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.ckpt import TrainState
from pi_lnn_jax.curriculum import (
    physics_weight_at_step,
    rar_init,
    rar_sample,
    time_marching_t_max,
)
from pi_lnn_jax.evaluate import evaluate_against_dns
from pi_lnn_jax.losses import (
    al_init,
    al_update,
    gradnorm_init,
    gradnorm_step,
    gradnorm_weights,
)
from pi_lnn_jax.model_factory import model_fingerprint
from pi_lnn_jax.pipeline._ledger import digest, get_ledger, params_digest
from pi_lnn_jax.pipeline.kolmogorov.assembly import (
    _CRP_STAT_KEYS,  # 與建構期的 _build_crp_interp 共用同一組 stat key，不另抄一份
    ReBatch,
    TrainingContext,
    build_context,
    effective_sensor_query_points,
    re_norm_scale_of,
)
from pi_lnn_jax.pipeline.kolmogorov.config import (
    KolmogorovEffectiveConfig,
    resolve_inputs,
)
from pi_lnn_jax.sensor_dropout import make_keep_mask


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class TrainingState(NamedTuple):
    """訓練期的可變狀態（in-memory）。

    與 `ckpt.TrainState`（on-disk）是**不同型別**，欄位也不同：本型別多帶
    `rng_collo` / `rng_re` / `rng_crp` / `rar_state` / `task_weights`，
    而 ckpt 只存其中一條 rng（見 `ckpt.py:14-17` 自陳的非 bit-deterministic
    resume 限制）。落盤一律走 `_to_ckpt_state` 顯式投影，不得直接互轉。
    """
    params: Any
    opt_state: Any
    step: int
    rng: Any
    rng_collo: Any
    rng_re: Any
    rng_crp: Any
    gn_state: Any
    lra_state: Any
    al_state: Any
    rar_state: Any
    task_weights: Any


class TrainResult(NamedTuple):
    """`finalize` 的產物 —— 落盤後的摘要，供入口腳本或呼叫端使用。"""
    state: TrainingState
    final_results: Any
    eval_history: list
    summary: dict
    summary_path: Path
    history_path: Path


@dataclass
class RunJournal:
    """跨階段累積的**可觀測性**紀錄；不參與任何數值決策。

    Why 不塞進 `TrainingState`：這些是 run log（wall time / eval 歷史 /
    最後一次 log 的 metrics），與 optimizer 軌跡無關，塞進 state 會讓
    「state 決定數值」這個讀法失效。
    """
    n_params: int = 0
    start_step: int = 0
    train_wall: float = 0.0
    eval_history: list = field(default_factory=list)
    last_metrics: dict = field(default_factory=dict)


def _to_ckpt_state(
    *, params, opt_state, step: int, rng_key, gn_state, al_state, lra_state,
) -> TrainState:
    """`TrainingState` → `ckpt.TrainState` 的**唯一**投影點。

    `ckpt.TrainState` 的 NamedTuple 欄位不得增刪或重排——orbax restore 比對
    tree structure，少一個 leaf 就讓舊 ckpt 還原不了（`ckpt.py:60-63` 的
    brdr_state 遷移記錄即此教訓）。
    """
    return TrainState(
        params=params, opt_state=opt_state,
        step=jnp.int32(step), rng_key=rng_key,
        gradnorm_state=gn_state, al_state=al_state,
        lra_state=lra_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_training(ctx: TrainingContext) -> TrainResult:
    """initialize → restore → run_loop → refine → finalize。"""
    journal = RunJournal()
    state = initialize(ctx, journal)
    state = restore(ctx, state)
    state = run_loop(ctx, state, journal)
    state = refine(ctx, state)
    return finalize(ctx, state, journal)


# ─────────────────────────────────────────────────────────────────────────────
# 建構單步訓練計畫的輔助函式（construction-of-a-step helpers）
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum: n_collo schedule (Python scalar; 不需要 jit-able)
# ─────────────────────────────────────────────────────────────────────────────

def _n_collo_at_step(step: int, start: int, end: int, ramp: int) -> int:
    """線性 ramp n_collo（與 curriculum.physics_points_at_step 同公式但 inline）。"""
    if start == end or ramp <= 0:
        return end
    progress = min(step / max(ramp, 1), 1.0)
    return int(round(start + (end - start) * progress))


# ─────────────────────────────────────────────────────────────────────────────
# 連續-Re 物理正則：per-step 取樣的 ReBatch builder（執行期）
# ─────────────────────────────────────────────────────────────────────────────

def _make_re_batch_crp(
    active, re_norm_p: float, interp: dict,
    re_norm_scale: float, dtype: Any = jnp.float32,
):
    """以 sampled re_norm' 造 ReBatch：sensors 沿用 active，re_norm/nu/norm_stats 換成
    sampled 值（norm_stats 線性內插，nu' = 1/scale^re_norm'）。"""
    xs = interp["re_norm_sorted"]
    stats = {k: float(np.interp(re_norm_p, xs, interp[k])) for k in _CRP_STAT_KEYS}
    re_p = float(re_norm_scale) ** float(re_norm_p)
    return active._replace(
        re_norm=jnp.asarray(re_norm_p, dtype=dtype),
        nu=jnp.asarray(1.0 / re_p, dtype=dtype),
        u_mean=jnp.asarray(stats["u_mean"], dtype=dtype),
        u_std=jnp.asarray(stats["u_std"], dtype=dtype),
        v_mean=jnp.asarray(stats["v_mean"], dtype=dtype),
        v_std=jnp.asarray(stats["v_std"], dtype=dtype),
        p_mean=jnp.asarray(stats["p_mean"], dtype=dtype),
        p_std=jnp.asarray(stats["p_std"], dtype=dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resume continuity guard
# ─────────────────────────────────────────────────────────────────────────────

def _previous_sensor_loss(artifacts_dir) -> float | None:
    """前一輪 summary.json 記的最後 sensor loss；拿不到就回 None。

    回 None 的常見情形都不是錯誤，所以這裡不 raise：首次訓練沒有前一輪；
    crash/timeout 後 resume 時前一輪沒跑到 finalize，summary.json 不存在；
    artifacts_dir 未設定（測試 stub）。
    呼叫端會把 None 印成「continuity 未驗證」——不靜默當成通過。

    壞掉或殘缺的 summary 同樣回 None：這個守衛是診斷輔助，不該讓一份讀不懂的
    產物擋下整輪 resume。
    """
    if artifacts_dir is None:
        return None
    path = Path(artifacts_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text()).get("last_train_metrics", {}).get("sensor")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def deterministic_sensor_idx(n_total: int, n_take: int):
    """在攤平後的 `T*K` sensor 網格上取 `n_take` 個**系統性均勻**點。

    兩個用途共用同一個定義，因為它們必須算同一個量：`finalize` 落下參考值、
    `_resume_sanity_check` 重算它。任一側改了取樣方式，比較就失去意義。

    為什麼不是隨機抽：（1）不消耗 RNG，不動 §7.1 的 ledger 不變量；
    （2）攤平是 time-major，`linspace` 在時間軸上均勻，而取前綴只會看到最早的
    `n_take/K` 個時刻；（3）`index mod K` 會循環，不會鎖定 QR-pivot 的前幾個
    高重要度感測器。
    """
    # `n_take` 可能是 None（`n_sensor_query` 未設 = 不做 mini-batch）或 0（同義）。
    # 兩者都代表「用全量」，與 loss_fn 對 `sensor_idx=None` 的既有語意一致。
    if not n_take:
        return None
    n = min(int(n_take), int(n_total))
    if n <= 0 or n >= int(n_total):
        return None                      # None = 全量（小 K 下本來就跑得動）
    return jnp.linspace(0, int(n_total) - 1, n).astype(jnp.int32)


def _previous_resume_reference(artifacts_dir) -> float | None:
    """前一輪 `resume_reference.json` 落下的確定性參考 sensor loss；拿不到就回 None。

    與 `_previous_sensor_loss` 分開，因為兩者的**可比性不同**：這個是用
    `deterministic_sensor_idx` 算的，resume 端能逐點重現；那個是隨機 mini-batch，
    只能當參考數字不能當判準。混用兩者正是這個守衛先前假警報的來源。
    """
    if artifacts_dir is None:            # 測試 stub / 未設定，與 _previous_sensor_loss 同
        return None
    path = Path(artifacts_dir) / "resume_reference.json"
    if not path.exists():
        return None
    try:
        ref = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(ref, dict):
        return None
    v = ref.get("sensor_loss")
    return float(v) if v is not None else None


def _resume_sanity_check(
    loss_fn,
    params,
    cx,
    cy,
    ct,
    task_weights,
    al_lambda,
    w_phys_now,
    last_logged_loss: float | None,
    re_batch=None,
    re_batch_crp=None,
    sensor_idx=None,
    reference_sensor_loss: float | None = None,
) -> dict:
    """Resume 後跑一次 forward，比對與 ckpt 落盤前最後 logged loss 偏差。

    Wave 5: loss_fn 需 re_batch runtime arg；caller 傳 d0 的 ReBatch。
    CRP（use_continuous_re_physics）開啟時 loss_fn 無條件解引用 re_batch_crp，
    caller 須傳非 None（零擾動 proxy 即可，本 check 只驗 forward 可跑性）。

    What: 重算當前 (params + task_weights + al_lambda + curriculum w_phys) 的 sensor + total
    loss，若 sensor loss 與 `last_logged_loss` 偏差 > 1e-3（相對）→ 印 warning。
    Why: 即使 ckpt round-trip bit-equal，curriculum / task_weights / al_lambda 三者也必須
         與落盤前一致；任一漂移皆是 silent regression 線索。

    `last_logged_loss` 取自**前一輪 summary.json 的 `last_train_metrics["sensor"]`**。
    兩個限制寫在這裡，因為它們決定這個守衛什麼時候真的在守：
      1. 比 sensor 不比 total——本函式重抽 collocation 點，total 的 physics 項
         必然不同（見下方比較處）。
      2. summary.json 在 finalize 才寫，所以 crash/timeout 後的 resume 沒有前一份
         可讀。那時 `last_logged_loss=None`，函式會印出「未驗證」而非靜默通過。
    """
    if re_batch is None:
        raise ValueError("Wave 5 _resume_sanity_check 需要 re_batch arg")
    # sensor_idx 不再寫死 None。傳 None 等於拿**全量** T*K 網格過一次 decoder，
    # 而訓練走的是 mini-batch：K=200 是 20200 對 2000，cross-attention 的中介張量
    # 隨之放大，在 r740 上直接 RESOURCE_EXHAUSTED（還原本身成功，炸在還原之後的
    # 這個 check）。這正是「訓練中斷可以續跑」從未被驗證過的原因。
    (total, (sl, mu, mv, c, pr, _al_c)) = loss_fn(
        params, cx, cy, ct, task_weights, al_lambda, w_phys_now, re_batch, 0.0,
        sensor_idx, re_batch_crp,
    )
    info = {
        "sensor_loss_resumed": float(sl),
        "total_loss_resumed": float(total),
        "mom_u_resumed": float(mu),
        "mom_v_resumed": float(mv),
        "cont_resumed": float(c),
    }
    # 比 sensor 而非 total：本函式重抽了 collocation 點（cx/cy/ct 來自新的 RNG
    # split），所以 total 裡的 physics 項與上次記錄的**本來就不同**——拿 total
    # 比對在任何容忍度下都不是良置的比較。sensor loss 只依賴 params 與 sensor
    # 資料，是 resume 前後唯一該相等的量。
    if reference_sensor_loss is not None and reference_sensor_loss > 0:
        # 首選：`finalize` 用**同一組確定性索引**算下的參考值。兩側算同一個量，
        # 所以 1e-3 這個容忍度是有意義的——剩下的只有 float 噪聲與還原保真度。
        rel = abs(float(sl) - reference_sensor_loss) / abs(reference_sensor_loss)
        info["rel_drift_vs_reference"] = rel
        info["reference_sensor_loss"] = reference_sensor_loss
        if rel > 1e-3:
            print(
                f"[WARN] resume continuity: sensor={float(sl):.4e} vs reference"
                f"={reference_sensor_loss:.4e} (rel drift {rel:.4f}) — 高於 1e-3 容忍度。"
                "兩側用同一組確定性索引，所以這個偏差指向還原本身。",
                flush=True,
            )
    elif last_logged_loss is not None and last_logged_loss > 0:
        # 退路（2026-09-14 之前的 ckpt 沒有參考值）：`last_logged_loss` 是訓練時
        # **隨機 mini-batch** 算的，與這裡的確定性子取樣是同一期望值的兩個估計量。
        # 實測 n=2000 的相對標準誤約 3.3%，是 1e-3 的 **33 倍**——套那個容忍度會
        # 對每一次正常 resume 都發警告。假警報與靜默通過一樣沒有守衛效果，
        # 所以這條路徑只報數字、不判定。
        rel = abs(float(sl) - last_logged_loss) / max(abs(last_logged_loss), 1e-8)
        info["rel_drift_vs_last_log"] = rel
        info["last_logged_loss"] = last_logged_loss
        print(f"  [resume] 無確定性參考值（舊 ckpt）。sensor={float(sl):.4e} vs "
              f"上次 mini-batch 記錄={last_logged_loss:.4e}（相對差 {rel:.3f}）——"
              "兩者抽樣方式不同，此數字**不構成判定**（mini-batch 噪聲約 3%）。",
              flush=True)
    else:
        # 說出口。靜默跳過與「檢查通過」在輸出上無法區分，而這個守衛先前正是
        # 因為呼叫端寫死 None 而整段沒在執行。
        print("  [resume] 無前一輪的 sensor loss 可比對 → continuity 未驗證",
              flush=True)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# RNG stream 開場（spec §5.1）—— run_loop 與 replay_schedule 共用
# ─────────────────────────────────────────────────────────────────────────────

def _main_rng(seed: int):
    """spec §5.1 #0/#1：主 stream 起手 + `model.init` 用的 sub key。"""
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    return rng, rng_init


def _open_loop_streams(use_continuous_re_physics: bool, rng):
    """spec §5.1 #2-#5：迴圈前的四次（CRP 關時三次）split。

    順序與條件即規格；`rng_crp` 只在 `use_continuous_re_physics` 為真時 split，
    旗標關閉時主流不推進——這是為了讓 flag-off 的下游 RNG 流保持 bit-identical。
    """
    rng, rng_rar = jax.random.split(rng)
    rar_state = rar_init(seed=int(jax.random.randint(rng_rar, (), 0, 2**31 - 1)))
    rng, rng_collo = jax.random.split(rng)
    rng, rng_re = jax.random.split(rng)
    rng_crp = None
    if use_continuous_re_physics:
        rng, rng_crp = jax.random.split(rng)
    return rng, rar_state, rng_collo, rng_re, rng_crp


# ─────────────────────────────────────────────────────────────────────────────
# Step planning（spec §5.4）—— run_loop 與 replay_schedule 的唯一一份
# ─────────────────────────────────────────────────────────────────────────────

class _StepPlan(NamedTuple):
    """一個 training step 的 host-side 決策全集（不含 forward/gradient）。"""
    re_idx: int
    re_batch: Any
    t_upper: float
    re_norm_p: float | None
    re_batch_crp: Any
    n_collo_now: int
    w_phys_now: float
    sensor_idx: Any
    sensor_input_mask: Any
    sensor_cut_idx: Any
    cx: Any
    cy: Any
    ct: Any
    rar_triggered: bool
    rng_collo: Any
    rng_re: Any
    rng_crp: Any
    rar_state: Any
    trig_weighting: bool
    trig_al: bool
    trig_ckpt: bool
    trig_eval: bool


def _plan_step(
    ctx: TrainingContext,
    s: int,
    *,
    params,
    rng_collo,
    rng_re,
    rng_crp,
    rar_state,
    n_datasets: int,
    sensor_dropout_rate: float,
    rar_residual_fn,
) -> _StepPlan:
    """算出第 s 步的所有 host-side 決策，並回傳推進後的 RNG streams。

    這不是 pure function：RAR 分支依 current `params` 的 residual 選點，且會
    推進 `rar_state`。刻意不假裝它是——把它模型化成純函式只會逼出一個
    假的介面。
    """
    config = ctx.config
    run = config.run
    loss = config.loss
    curriculum = config.curriculum

    # 1. Wave 5: random sample Re index per step (multi-Re 解耦 closure bug 後的正確做法)
    if n_datasets > 1:
        rng_re, sub_re = jax.random.split(rng_re)
        re_idx = int(jax.random.randint(sub_re, (), 0, n_datasets))
    else:
        re_idx = 0
    re_batch = ctx.re_batches[re_idx]
    t_min_now = ctx.re_t_min_host[re_idx]
    t_max_now = ctx.re_t_max_host[re_idx]
    # 連續-Re 物理：抽 re_norm' ∈ [min,max]；旗標 off 時傳 active 當 dummy（loss_fn 不用）
    if loss.use_continuous_re_physics:
        rng_crp, sub_crp = jax.random.split(rng_crp)
        re_norm_p = float(jax.random.uniform(
            sub_crp, (), minval=ctx.crp_interp["re_norm_min"],
            maxval=ctx.crp_interp["re_norm_max"]))
        re_batch_crp = _make_re_batch_crp(
            re_batch, re_norm_p, ctx.crp_interp, ctx.crp_re_norm_scale)
    else:
        re_norm_p = None
        re_batch_crp = re_batch
    # 2. curriculum schedules
    n_collo_now = _n_collo_at_step(
        s, curriculum.n_collo_start, curriculum.n_collo_end, curriculum.n_collo_ramp,
    )
    w_phys_now = physics_weight_at_step(
        s, loss.physics_weight, loss.physics_warmup_steps, loss.physics_ramp_steps,
    )
    # physics 時窗：預設等於 data span；`physics_t_max>0` 時延伸到資料之外
    # （該段只有 PDE residual、無 data loss）。0.0 → 表達式與舊版逐字相同。
    t_phys_max = curriculum.physics_t_max if curriculum.physics_t_max > 0.0 else t_max_now
    if curriculum.use_time_marching:
        # #2: ramp ceiling 用 physics 時窗（預設 = data span t_max_now = sensor_time[-1]）
        #     而非 model T_total，否則 T_total≠時窗時 ramp 比例失準、physics 漏掉部分時間窗。
        t_upper = time_marching_t_max(
            s, t_phys_max,
            start_frac=curriculum.tm_start_frac,
            warmup_steps=curriculum.tm_warmup_steps,
            ramp_steps=curriculum.tm_ramp_steps,
        )
        # #3: clamp 進 [t_min_now, t_phys_max]，避免 t_min_now>0 時 t_upper<t_min → uniform 反向範圍。
        t_upper = max(min(t_upper, t_phys_max), t_min_now)
    else:
        t_upper = t_phys_max
    # 3a. sensor mini-batch index（shape 固定 → jit 不 retrace）
    n_sq = ctx.n_sensor_query
    if n_sq and n_sq > 0:
        T_rb = re_batch.sensor_vals.shape[0]
        K_rb = re_batch.sensor_vals.shape[1]
        n_total_sq = T_rb * K_rb
        rng_collo, sub_sq = jax.random.split(rng_collo)
        sensor_idx = jax.random.choice(sub_sq, n_total_sq, shape=(n_sq,), replace=False)
    else:
        sensor_idx = None
    # 3a'. train-time sensor dropout mask（denoising；每 step 重抽，shape 固定 [K] → 不 retrace）
    if sensor_dropout_rate > 0.0:
        rng_collo, sub_drop = jax.random.split(rng_collo)
        sensor_input_mask = make_keep_mask(
            sub_drop, int(re_batch.sensor_vals.shape[1]), sensor_dropout_rate,
        )
    else:
        sensor_input_mask = None
    # 3a''. sensor 序列截斷（dt>0 的監督訊號）：抽 frac ~ U(min_frac, 1.0]，
    # decoder 只看得到前 round(frac·(T−1)) 幀。關閉時**不抽 RNG**，主流不受影響。
    # 傳 traced int（非 Python int）→ 不同 cut 值不會觸發 jit retrace。
    if curriculum.sensor_cut_min_frac > 0.0:
        rng_collo, sub_cut = jax.random.split(rng_collo)
        T_cut = int(re_batch.sensor_vals.shape[0])
        frac = jax.random.uniform(
            sub_cut, (), minval=curriculum.sensor_cut_min_frac, maxval=1.0)
        sensor_cut_idx = jnp.asarray(jnp.round(frac * (T_cut - 1)), jnp.int32)
    else:
        sensor_cut_idx = None

    # 3b. collocation sampling
    rng_collo, sub = jax.random.split(rng_collo)
    rar_triggered = (
        curriculum.rar_freq > 0
        and s > curriculum.rar_warmup
        and s % curriculum.rar_freq == 0
    )
    if rar_triggered:
        cx, cy, ct, rar_state = rar_sample(
            rar_state,
            lambda p_, xs_, ys_, ts_: rar_residual_fn(p_, xs_, ys_, ts_, re_batch),
            params,
            n_select=n_collo_now,
            pool_size=curriculum.rar_pool_size,
            t_min=t_min_now,
            t_max=t_upper,
            exploration_ratio=curriculum.rar_exploration_ratio,
        )
    else:
        keys = jax.random.split(sub, 3)
        cx = jax.random.uniform(keys[0], (n_collo_now,), minval=0.0, maxval=1.0)
        cy = jax.random.uniform(keys[1], (n_collo_now,), minval=0.0, maxval=1.0)
        ct = jax.random.uniform(
            keys[2], (n_collo_now,),
            minval=t_min_now, maxval=t_upper,
        )
    # 四個觸發條件在此算一次，ledger 與下方控制流共用同一個值。
    # Why 提上來：若 ledger 自行再算一次，兩份條件會漂移，ledger 就會
    # 安靜說謊（正是 4792b93 治的那種病）。純布林值、無 RNG/副作用，
    # 提前求值不改變任何行為。
    trig_weighting = s % loss.gradnorm_freq == 0
    al_past_warmup = s >= loss.al_warmup_steps
    trig_al = loss.use_al and s % loss.al_update_freq == 0 and al_past_warmup
    trig_ckpt = run.save_every > 0 and s % run.save_every == 0
    trig_eval = run.eval_every > 0 and s % run.eval_every == 0

    return _StepPlan(
        re_idx=re_idx, re_batch=re_batch,
        t_upper=t_upper,
        re_norm_p=re_norm_p, re_batch_crp=re_batch_crp,
        n_collo_now=n_collo_now, w_phys_now=w_phys_now,
        sensor_idx=sensor_idx, sensor_input_mask=sensor_input_mask,
        sensor_cut_idx=sensor_cut_idx,
        cx=cx, cy=cy, ct=ct, rar_triggered=rar_triggered,
        rng_collo=rng_collo, rng_re=rng_re, rng_crp=rng_crp, rar_state=rar_state,
        trig_weighting=trig_weighting, trig_al=trig_al,
        trig_ckpt=trig_ckpt, trig_eval=trig_eval,
    )


def _ledger_fields(ctx: TrainingContext, plan: _StepPlan) -> dict:
    """把 `_StepPlan` 攤成 ledger 欄位。

    `run_loop`（錄製）與 `replay_schedule`（重播）共用這一份，否則兩邊的
    欄位定義會漂移，比對就變成比兩份不同的東西。
    只讀既有已算好的值，不新增任何計算。
    """
    loss = ctx.config.loss
    return dict(
        re_idx=int(plan.re_idx),
        re_norm_p=(float(plan.re_norm_p)
                   if loss.use_continuous_re_physics else None),
        n_collo=int(plan.n_collo_now),
        w_phys=float(plan.w_phys_now),
        t_upper=float(plan.t_upper),
        sensor_idx=(digest(plan.sensor_idx) if plan.sensor_idx is not None else None),
        dropout_mask=(digest(plan.sensor_input_mask)
                      if plan.sensor_input_mask is not None else None),
        cx=digest(plan.cx), cy=digest(plan.cy), ct=digest(plan.ct),
        trig_weighting=bool(plan.trig_weighting),
        trig_al=bool(plan.trig_al),
        trig_ckpt=bool(plan.trig_ckpt),
        trig_eval=bool(plan.trig_eval),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: initialize（model.init / opt_state / weighting state / AL state）
# ─────────────────────────────────────────────────────────────────────────────

def initialize(ctx: TrainingContext, journal: RunJournal) -> TrainingState:
    """建 params / opt_state / weighting state；消耗主 RNG #0-#1（spec §5.1）。"""
    config = ctx.config
    run = config.run
    loss = config.loss
    re_batch_d0 = ctx.re_batches[0]  # 初始 active batch (init + sanity check 用)

    # ── 建模：model.init 消耗主 RNG（spec §5.1 第 1 項），故留在執行期 ──
    rng, rng_init = _main_rng(run.seed)
    init_xy = jnp.asarray(
        np.random.RandomState(run.seed).uniform(0, 1, (8, 2)).astype(np.float32)
    )
    init_t = jnp.asarray(
        np.random.RandomState(run.seed).uniform(0, ctx.T_total, (8,)).astype(np.float32)
    )
    params = ctx.model.init(
        rng_init,
        re_batch_d0.sensor_vals, re_batch_d0.sensor_pos,
        re_batch_d0.re_norm, re_batch_d0.sensor_time,
        init_xy, init_t,
    )
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nModel: {ctx.model_name}  params={n_params:,}")
    # 初始化階段的 params digest。尾列的 `params_digest` 記的是**訓練後**的
    # 參數（在 finalize 寫），無法用來檢查「建構 + 初始化」這一段；
    # 故在此另記一份，由 `Ledger.note` 帶到尾列。
    ledger = get_ledger()
    if ledger is not None:
        ledger.note(init_params_digest=params_digest(params))

    # ── Optimizer state（tx 由 assembly 建好；opt_state 需要 params，屬執行期）──
    opt_state = ctx.tx.init(params)

    # ── Inter-task weighting state（GradNorm | off）──
    # grad-norm 函式與 gn_ref_path 由 assembly 提供；此處只造 state。
    gn_state = None
    # lra_state 永遠是 None：LRA controller 已於 2026-08-03 移除（與 GradNorm 數值等價，
    # 非真對照）。欄位保留是 ckpt 格式相容性墓碑——實測移除 TrainState 欄位會讓
    # 既有 ckpt 全部 restore 失敗（orbax 連 None 也存 metadata entry）。見 ckpt.py。
    lra_state = None
    if loss.use_gradnorm:
        # startup probe：ref path 在 params 樹內 resolve 不到時 _get_subtree 會 silent
        # fallback 到 full params（含 Fourier kernel 梯度放大路徑）——這裡印一次讓它可稽核。
        try:
            _sub = params["params"]
            for _k in ctx.gn_ref_path:
                _sub = _sub[_k]
            _ref_resolved = True
        except (KeyError, TypeError):
            _ref_resolved = False
        print(f"  GradNorm ref-subtree={ctx.gn_ref_path} → "
              f"{'resolved' if _ref_resolved else 'FULL_PARAMS(fallback)'}")
    if loss.use_gradnorm:
        if loss.cont_gradnorm:
            # continuity 為第四個 task；init 權重沿用 physics 的 0.057
            _w0 = loss.gradnorm_init_weights or [1.0, 0.057, 0.057]
            _w0 = list(_w0) + [0.057] if len(_w0) == 3 else list(_w0)
            gn_state = gradnorm_init(_w0, task_names=("data", "ns_u", "ns_v", "cont"))
        else:
            gn_state = gradnorm_init(loss.gradnorm_init_weights or [1.0, 0.057, 0.057])
        task_weights = gradnorm_weights(gn_state)
    else:
        _n_tasks = 4 if loss.cont_gradnorm else 3
        task_weights = jnp.ones((_n_tasks,), dtype=jnp.float32)

    # AL 固定啟用
    al_state = al_init(
        init_lambda=0.0, rho=loss.al_rho, lambda_clip=loss.al_lambda_clip,
    )

    journal.n_params = int(n_params)
    return TrainingState(
        params=params, opt_state=opt_state, step=0, rng=rng,
        rng_collo=None, rng_re=None, rng_crp=None,
        gn_state=gn_state, lra_state=lra_state, al_state=al_state,
        rar_state=None, task_weights=task_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: restore（optional resume + continuity guard）
# ─────────────────────────────────────────────────────────────────────────────


def assert_al_statics_match(al_state, loss) -> None:
    """ckpt 還原的 AL 靜態超參必須與當前 config 一致，否則大聲失敗。

    `ALState` 的 `rho` / `lambda_clip` / `ema_momentum` 是 NamedTuple 欄位，也就是
    pytree leaf——orbax 會序列化它們，restore 於是**用 ckpt 的值蓋掉當前 config**。
    改了 `al_rho` 再 resume，跑的還是舊 ρ，而 `summary.json` 記的是 config 的值：
    兩者不一致時沒有任何地方會發現。不靜默採用任一側。

    `ema_momentum` 不在比對範圍：它沒有對應的 config 鍵（寫死在 `al_init` 預設 0.5），
    無從判斷「不符」。
    """
    if al_state is None:
        return
    drift = [
        (name, float(getattr(al_state, field)), float(want))
        for name, field, want in (("al_rho", "rho", loss.al_rho),
                                  ("al_lambda_clip", "lambda_clip", loss.al_lambda_clip))
        if float(getattr(al_state, field)) != float(want)
    ]
    if drift:
        raise ValueError(
            "ckpt 的 AL 靜態超參與當前 config 不符，restore 會靜默採用 ckpt 的值："
            + "；".join(f"{k}: ckpt={c} vs config={v}" for k, c, v in drift)
            + "。要沿用 ckpt 請把 config 改成相同值；要換新值請從頭訓練。"
        )


def restore(ctx: TrainingContext, state: TrainingState) -> TrainingState:
    """`--resume_step` 指定時從 ckpt 還原，並跑一次 continuity sanity check。

    Sanity check 用 `rng_check, sub_check = split(rng)` —— 賦值給 `rng_check`
    而**非** `rng`，主流不被推進（spec §5.1）。這個中立性是規格，不是筆誤。
    """
    config = ctx.config
    run = config.run
    loss = config.loss
    curriculum = config.curriculum
    if run.resume_step is None:
        return state

    params = state.params
    opt_state = state.opt_state
    gn_state = state.gn_state
    al_state = state.al_state
    lra_state = state.lra_state
    rng = state.rng
    task_weights = state.task_weights
    ckpt_mgr = ctx.ckpt_mgr
    ckpt_dir = ctx.ckpt_dir
    re_batch_d0 = ctx.re_batches[0]

    ref_state = _to_ckpt_state(
        params=params, opt_state=opt_state, step=0, rng_key=rng,
        gn_state=gn_state, al_state=al_state, lra_state=lra_state,
    )
    target_step = (
        ckpt_mgr.latest_step() if run.resume_step == "latest"
        else int(run.resume_step)
    )
    if target_step is None:
        raise FileNotFoundError(f"無 ckpt 可 resume；ckpt dir={ckpt_dir}")
    restored = ckpt_mgr.restore(target_step, reference_state=ref_state)
    params = restored.params
    opt_state = restored.opt_state
    gn_state = restored.gradnorm_state
    al_state = restored.al_state
    # ALState 的 rho / lambda_clip / ema_momentum 是 NamedTuple 欄位，也就是 pytree
    # leaf——orbax 會序列化它們，restore 於是**用 ckpt 的值蓋掉當前 config**。改了
    # al_rho 再 resume，跑的還是舊 ρ，而 summary.json 記的是 config 的值：兩者不一致
    # 時沒有任何地方會發現。不靜默採用任一側，直接讓它爆。
    assert_al_statics_match(al_state, loss)
    lra_state = restored.lra_state
    rng = restored.rng_key
    start_step = int(restored.step)
    if loss.use_gradnorm and gn_state is not None:
        task_weights = gradnorm_weights(gn_state)
        # 從 3-task ckpt resume 到 cont_gradnorm（或反向）時長度會不符。JAX 對越界索引
        # 是靜默 clamp——task_weights[3] 會回 w_ns_v，continuity 拿到別的 task 的權重，
        # 直到下一次 gradnorm_step 才因 broadcast 失敗而崩。這裡先擋下來。
        _n_expected = 4 if loss.cont_gradnorm else 3
        if task_weights.shape[0] != _n_expected:
            raise ValueError(
                f"ckpt 的 GradNorm 權重長度 {task_weights.shape[0]} 與 "
                f"loss.cont_gradnorm={loss.cont_gradnorm} 所需的 {_n_expected} 不符："
                "該 ckpt 是用另一種 task 佈局訓練的，不能直接 resume。"
            )
    print(f"\n[RESUME] restored step={start_step} from {ckpt_dir}")
    # Sanity check: recompute one forward
    rng_check, sub_check = jax.random.split(rng)
    n0 = _n_collo_at_step(
        start_step + 1, curriculum.n_collo_start, curriculum.n_collo_end,
        curriculum.n_collo_ramp,
    )
    kc = jax.random.split(sub_check, 3)
    cx0 = jax.random.uniform(kc[0], (n0,), minval=0.0, maxval=1.0)
    cy0 = jax.random.uniform(kc[1], (n0,), minval=0.0, maxval=1.0)
    d0_st = np.asarray(re_batch_d0.sensor_time)
    ct0 = jax.random.uniform(
        kc[2], (n0,),
        minval=float(d0_st[0]), maxval=float(d0_st[-1]),
    )
    w_phys_now0 = physics_weight_at_step(
        max(start_step, 1), loss.physics_weight,
        loss.physics_warmup_steps, loss.physics_ramp_steps,
    )
    sanity = _resume_sanity_check(
        ctx.loss_fn, params, cx0, cy0, ct0,
        task_weights, al_state.lambda_, w_phys_now0,
        last_logged_loss=_previous_sensor_loss(ctx.artifacts_dir),
        sensor_idx=deterministic_sensor_idx(
            re_batch_d0.sensor_vals.shape[0] * re_batch_d0.sensor_vals.shape[1],
            ctx.n_sensor_query),
        reference_sensor_loss=_previous_resume_reference(ctx.artifacts_dir),
        re_batch=re_batch_d0,
        # CRP 開啟時 loss_fn 需要非 None 的 crp batch，此處以 d0 batch 作零擾動
        # proxy（re_norm'=re_norm），僅驗 forward 可跑。
        # 原始碼給的理由是「crp_interp 尚未建構」——單體版確實在 sanity check 之後
        # 才建（`28b007a:1229` vs `:1165`）。該前提在新架構下**已不成立**：
        # crp_interp 由 build_context 一次建好，restore 時 `ctx.crp_interp` 就在手邊。
        # 仍不改用它——那會改變 sanity check 印出的數字，屬行為變更。
        re_batch_crp=(re_batch_d0 if loss.use_continuous_re_physics else None),
    )
    print(f"  sensor_loss after resume = {sanity['sensor_loss_resumed']:.4e}")

    return state._replace(
        params=params, opt_state=opt_state, step=start_step, rng=rng,
        gn_state=gn_state, lra_state=lra_state, al_state=al_state,
        task_weights=task_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: run_loop（主訓練迴圈）
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(ctx: TrainingContext, state: TrainingState,
             journal: RunJournal) -> TrainingState:
    """主訓練迴圈。RNG 開場 #2-#5 與每步的 §5.4 消費順序皆在此。"""
    config = ctx.config
    run = config.run
    loss = config.loss
    curriculum = config.curriculum
    ledger = get_ledger()
    ckpt_dir = ctx.ckpt_dir
    datasets = ctx.datasets
    n_datasets = len(datasets)
    d0 = datasets[0]
    re_batch_d0 = ctx.re_batches[0]
    model = ctx.model
    ns_fn = ctx.ns_fn
    step_fn = ctx.step_fn
    ckpt_mgr = ctx.ckpt_mgr
    compute_grad_norms = ctx.grad_norm_fn
    params = state.params
    opt_state = state.opt_state
    gn_state = state.gn_state
    lra_state = state.lra_state
    al_state = state.al_state
    task_weights = state.task_weights
    start_step = int(state.step)
    journal.start_step = start_step

    # ── RAR state + 迴圈用的四條 stream（spec §5.1 #2-#5）──
    rng, rar_state, rng_collo, rng_re, rng_crp = _open_loop_streams(
        loss.use_continuous_re_physics, state.rng,
    )

    # 為 RAR 寫一個簡化 per-point residual fn（用 mom_u² + mom_v² + cont²）
    # 與 train loss 共用 ns_fn 的同一個 (A, k_f) 動態取值
    LiquidClass = model.__class__

    @jax.jit
    def _rar_residual_per_point(params_, xs, ys, ts, re_batch):
        """Return per-point |residual| L2，shape [N]。

        **必須 jit。** 未 jit 時這個函式在 eager 模式逐 op 執行完整的 encode +
        folx 二階殘差；實測 `rar_freq=1`、pool=4096 下訓練慢 12 倍以上（job 5710：
        18 分鐘走不完 500 步，同時開跑的對照臂已到 5500 步）。RAR 原本只有 CLI 旗標
        且全庫無 config 開啟，所以這條路徑從沒被實際跑過，慢在哪也就沒人發現。

        `re_batch` 是 NamedTuple of arrays（合法 pytree），形狀在單一 dataset 下固定，
        故只會 trace 一次；multi-Re 切換也只是同形狀的不同值，不觸發重編譯。

        Wave 5: 接受 re_batch 為 runtime arg（multi-Re 切換時用當前 Re 算）。
        用 return_per_point=True 取真 per-point 殘差，否則 score 對所有點同分 →
        top_k 退化成取首 n_top 個（隨機池下等同隨機，無 residual-adaptive 效果）。
        """
        h_states = model.apply(params_,
                                re_batch.sensor_vals, re_batch.sensor_pos,
                                re_batch.re_norm, re_batch.sensor_time,
                                method=LiquidClass.encode)
        A_f, k_f_f = model.apply(params_, method=LiquidClass.get_forcing)
        _mu, _mv, _c, mom_u_pp, mom_v_pp, cont_pp = ns_fn(
            params_, h_states, xs, ys, ts, A_f, k_f_f,
            re_batch.sensor_pos, re_batch.sensor_time,
            re_batch.nu, re_batch.u_mean, re_batch.u_std,
            re_batch.v_mean, re_batch.v_std, re_batch.p_mean, re_batch.p_std,
            return_per_point=True,
        )
        return jnp.sqrt(mom_u_pp ** 2 + mom_v_pp ** 2 + cont_pp ** 2)

    # ── 主訓練 loop ──
    total_steps = int(run.steps)
    eval_history = journal.eval_history
    print("\n" + "=" * 80)
    print(f"=== Training: step {start_step+1} → {total_steps} ===")
    header = (
        f"{'step':>6s} {'total':>11s} {'sensor':>11s} {'mom_u':>11s} "
        f"{'mom_v':>11s} {'cont':>11s} {'poisson':>10s} "
        f"{'C_AL':>11s} {'w_d/u/v/c':>22s} {'λ_AL':>7s} {'w_ph':>7s} {'n_co':>5s} {'wall':>7s}"
    )
    print(header)
    print("-" * len(header))

    t_train_start = time.time()
    last_log_t = t_train_start
    last_metrics: dict[str, float] = {}

    # 實驗2 train-time sensor dropout（Python float；0=停用，下游 RNG 流 bit-identical）
    sensor_dropout_rate = float(curriculum.sensor_dropout_rate)
    if sensor_dropout_rate > 0.0:
        print(f"[dropout] train-time sensor dropout rate={sensor_dropout_rate} "
              f"(denoising：mask 輸入、監督真值；每 step 重抽)")

    for s in range(start_step + 1, total_steps + 1):
        plan = _plan_step(
            ctx, s,
            params=params,
            rng_collo=rng_collo, rng_re=rng_re, rng_crp=rng_crp,
            rar_state=rar_state,
            n_datasets=n_datasets,
            sensor_dropout_rate=sensor_dropout_rate,
            rar_residual_fn=_rar_residual_per_point,
        )
        rng_collo = plan.rng_collo
        rng_re = plan.rng_re
        rng_crp = plan.rng_crp
        rar_state = plan.rar_state

        if ledger is not None:
            ledger.record(s, **_ledger_fields(ctx, plan))

        # 4. main step
        params, opt_state, total, sl, mu, mv, c, pr, al_c = step_fn(
            params, opt_state, plan.cx, plan.cy, plan.ct, task_weights,
            al_state.lambda_, plan.w_phys_now, plan.re_batch,
            loss.causal_eps if loss.use_causal else 0.0,
            plan.sensor_idx, plan.re_batch_crp, plan.sensor_input_mask,
            plan.sensor_cut_idx,
        )
        # 5. Inter-task weight updates（GradNorm 消耗 gradient norms）
        if loss.use_gradnorm and compute_grad_norms is not None and plan.trig_weighting:
            task_grads = compute_grad_norms(
                params, plan.cx, plan.cy, plan.ct, plan.re_batch,
                loss.causal_eps if loss.use_causal else 0.0,
            )
            gn_state = gradnorm_step(
                gn_state, task_grads,
                ema_momentum=loss.gradnorm_ema_momentum,
                min_weight=loss.gradnorm_min,
                max_weight=loss.gradnorm_max,
            )
            task_weights = gradnorm_weights(gn_state)
        # AL update 用獨立 freq；al_warmup_steps 內 lambda 保持 0（只走 ρ·C²/2 暖機）
        if plan.trig_al:
            lambda_min = (
                -loss.al_lambda_clip if loss.al_constraint_mode == "signed_mean" else 0.0
            )
            al_state = al_update(al_state, al_c, lambda_min=lambda_min)
        # 6. log
        if s % run.log_every == 0 or s == total_steps or s == start_step + 1:
            wall = time.time() - last_log_t
            last_log_t = time.time()
            w_str = "/".join(f"{float(w):.2f}" for w in task_weights)
            print(
                f"{s:>6d} {float(total):>11.4e} {float(sl):>11.4e} "
                f"{float(mu):>11.4e} {float(mv):>11.4e} {float(c):>11.4e} "
                f"{float(pr):>10.3e} {float(al_c):>11.4e} {w_str:>22s} "
                f"{float(al_state.lambda_):>7.3f} {plan.w_phys_now:>7.4f} "
                f"{plan.n_collo_now:>5d} {wall:>7.2f}",
                flush=True,
            )
            last_metrics = {
                "step": s,
                "total": float(total),
                "sensor": float(sl),
                "mom_u": float(mu),
                "mom_v": float(mv),
                "cont": float(c),
                "al_constraint": float(al_c),
                "poisson": float(pr),
            }
        # 7. fail-fast
        if not np.isfinite(float(total)):
            print(f"[FATAL] loss NaN/Inf at step {s} — abort", flush=True)
            sys.exit(2)
        # 8. ckpt save
        if plan.trig_ckpt:
            ckpt_state = _to_ckpt_state(
                params=params, opt_state=opt_state, step=s, rng_key=rng_collo,
                gn_state=gn_state, al_state=al_state, lra_state=lra_state,
            )
            wrote = ckpt_mgr.save(s, ckpt_state, force=True)
            if wrote:
                print(f"  [ckpt] saved step={s} → {ckpt_dir}", flush=True)
        # 9. eval (Wave 5: POC 階段只 eval d0 的 DNS；multi-Re 各 Re eval 留給 evaluate_multi_re)
        if plan.trig_eval:
            print(f"  --- mid-eval at step {s} ---", flush=True)
            r = evaluate_against_dns(
                model, params,
                re_batch_d0.sensor_vals, re_batch_d0.sensor_pos,
                re_batch_d0.re_norm, re_batch_d0.sensor_time,
                d0["norm_stats"],
                ctx.dns_u, ctx.dns_v, ctx.dns_t, verbose=True,
            )
            eval_history.append({"step": s, "results": r})

    train_wall = time.time() - t_train_start
    print("-" * len(header))
    print(f"Total training wall: {train_wall:.2f}s "
          f"({train_wall / max(1, total_steps - start_step):.3f}s/step avg)")

    journal.train_wall = train_wall
    journal.last_metrics = last_metrics
    return state._replace(
        params=params, opt_state=opt_state, step=total_steps, rng=rng,
        rng_collo=rng_collo, rng_re=rng_re, rng_crp=rng_crp,
        gn_state=gn_state, lra_state=lra_state, al_state=al_state,
        rar_state=rar_state, task_weights=task_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: refine（Adam → LBFGS / LM / GN）
# ─────────────────────────────────────────────────────────────────────────────

def refine(ctx: TrainingContext, state: TrainingState) -> TrainingState:
    """Wave 5 refinement phase；`--refine_optimizer none` 時原樣返回。

    ⚠ 本函式只換 `params` 不換 `opt_state`（末行的 `_replace`）。schedule_free 下
    x 由 `x = (y − (1−b1)z)/b1` 重建，y 換了而 z 沒換就不再屬於同一步，x-iterate
    的重建與 resume 都會失真（b1=0.9 → 位移被放大 1/0.9 且方向錯）。**這條路跑過。** `scripts/legacy/slurm/train_re10000_T20_{les_qr,random}_v{2,3}.sbatch`
    四支都是 `--optimizer schedule_free` 配 `--refine_optimizer {gn,lbfgs}`，而
    `knowledge/research/negative-results.md` 的「Late LBFGS refinement 幾乎中性」
    就出自該路徑。先前這段寫「三支 template 都硬寫 none，故此路從未執行」是錯的
    ——只看了 `scripts/slurm/` 的現行 template，沒看 `scripts/legacy/slurm/`。

    後果有兩層：那筆 negative result 是經由本函式現在判定為不健全的實作取得的，
    「中性」多了一個競爭解釋；而那四支 legacy 腳本現在會硬失敗，無法原樣重跑。
    下方 fail-fast 仍保留——它擋的是真缺陷，但它不是「保護一條沒人走過的路」。
    """
    config = ctx.config
    run = config.run
    loss = config.loss
    curriculum = config.curriculum
    refinement = config.refinement
    refine_choice = refinement.optimizer
    if refine_choice == "none":
        return state
    if run.optimizer == "schedule_free":
        raise NotImplementedError(
            f"refine_optimizer={refine_choice} 與 optimizer=schedule_free 不相容："
            "本函式只換 params 不換 opt_state，schedule_free 的 y 與 z 會不再屬於同一步，"
            "x-iterate 重建與 resume 都會失真。要用 refinement 請改用非 schedule_free 的"
            "optimizer，或先讓 refine 一併重建 opt_state（需走 §7.1 對拍）。"
        )

    from pi_lnn_jax.refiners import (
        gn_refine, lbfgs_refine, lm_refine, make_pinn_residual_vector_fn,
    )

    datasets = ctx.datasets
    d0 = datasets[0]
    model = ctx.model
    ns_fn = ctx.ns_fn
    re_batch_d0 = ctx.re_batches[0]
    T, K = d0["sensor_vals"].shape[0], d0["sensor_vals"].shape[1]
    params = state.params
    rng = state.rng

    if refine_choice == "gn":
        print("\n" + "=" * 80)
        print(f"=== Refinement phase: GN matrix-free "
              f"(max {refinement.steps} outer iters, "
              f"lr={refinement.gn_lr}, cg={refinement.gn_cg_iters}, "
              f"damping={refinement.gn_damping}) ===")
    else:
        print("\n" + "=" * 80)
        print(f"=== Refinement phase: {refine_choice.upper()} "
              f"(max {refinement.steps} optimistix iters, "
              f"rtol={refinement.rtol}, atol={refinement.atol}) ===")
    # Wave 5: refine phase 對 d0 進行（multi-Re refine 不在 POC scope）
    refine_re_batch = re_batch_d0
    refine_st = np.asarray(refine_re_batch.sensor_time)
    refine_re_value = float(datasets[0]["re_value"])
    refine_re_norm = float(datasets[0]["re_norm"])
    refine_norm_stats = datasets[0]["norm_stats"]
    # Fix one collocation batch (refiners are deterministic full-batch)
    rng, sub = jax.random.split(rng)
    keys = jax.random.split(sub, 3)
    n_collo_refine = (
        int(refinement.n_collo) if refinement.n_collo is not None
        else int(curriculum.n_collo_end)
    )
    print(f"  refine n_collo: {n_collo_refine} "
          f"(override default n_collo_end={curriculum.n_collo_end})")
    cx_fix = jax.random.uniform(keys[0], (n_collo_refine,), minval=0.0, maxval=1.0)
    cy_fix = jax.random.uniform(keys[1], (n_collo_refine,), minval=0.0, maxval=1.0)
    ct_fix = jax.random.uniform(keys[2], (n_collo_refine,),
                                 minval=float(refine_st[0]),
                                 maxval=float(refine_st[-1]))
    # Loss-before 探測延後到 mb_lbfgs_loss 定義後（refine_choice 各分支）
    refine_start = time.time()
    if refine_choice == "lbfgs":
        # 對齊 pi-lnn src/pi_con/training.py:298 LBFGS pattern:
        #   outer loop, 每 step sample mini-batch sensor + collocation,
        #   inner LBFGS max_iter=20 + strong-wolfe line search (optimistix LBFGS)
        # 解 OOM root cause: 原本 reuse main loss_fn closure 帶 full-T sensor query
        # (T*K=10100 點) → cross-attn 中間 tensor 3GB. 改 sub-sample T_sub=10 → 1000 點.
        T_sub = int(refinement.lbfgs_t_subsample)
        n_outer = int(refinement.steps)
        max_iter_inner = int(refinement.lbfgs_max_iter)
        history_len = int(refinement.lbfgs_history)
        print(f"  LBFGS (pi-lnn style): outer={n_outer}  max_iter/outer={max_iter_inner}  "
              f"history={history_len}  T_sub={T_sub}  K={K}  n_collo={n_collo_refine}")

        LO_class = model.__class__
        w_phys_refine = float(loss.physics_weight)

        # Wave 5: refine 用 d0 的 ReBatch；ns_fn 簽名也對應新 runtime args
        rsv = refine_re_batch.sensor_vals
        rsp = refine_re_batch.sensor_pos
        rst = refine_re_batch.sensor_time
        rrn = refine_re_batch.re_norm

        def mb_lbfgs_loss(p, args_tuple):
            """Mini-batch loss for LBFGS refine (對齊 pi-lnn closure pattern)。
            args_tuple = (t_idx, cx, cy, ct)
            Wave 5: ns_fn 取 refine_re_batch 的 sensor/nu/norm_stats。
            """
            t_idx_mb, cx_mb, cy_mb, ct_mb = args_tuple
            st_mb = rst[t_idx_mb]              # [T_sub]
            sv_mb_target = rsv[t_idx_mb]       # [T_sub, K, 2]
            xy_mb = jnp.broadcast_to(rsp[None], (T_sub, K, 2)).reshape(T_sub * K, 2)
            t_mb = jnp.broadcast_to(st_mb[:, None], (T_sub, K)).reshape(T_sub * K)
            # TODO: refine 路徑尚未支援 3-channel(uvp)；此處寫死 2，開 refine + uvp 前需改用 C，否則 silent 丟棄 p
            target_mb = sv_mb_target.reshape(T_sub * K, 2)

            h_states = model.apply(p, rsv, rsp, rrn, rst, method=LO_class.encode)
            pred = model.apply(p, xy_mb, t_mb, h_states, rst, rsp,
                                method=LO_class.decode_query)
            sensor_loss = jnp.mean((pred[:, :2] - target_mb) ** 2)
            A_f, k_f_f = model.apply(p, method=LO_class.get_forcing)
            mom_u, mom_v, cont = ns_fn(
                p, h_states, cx_mb, cy_mb, ct_mb, A_f, k_f_f,
                rsp, rst,
                refine_re_batch.nu, refine_re_batch.u_mean, refine_re_batch.u_std,
                refine_re_batch.v_mean, refine_re_batch.v_std, refine_re_batch.p_mean, refine_re_batch.p_std,
            )
            total = (
                loss.data_weight * sensor_loss
                + w_phys_refine * (mom_u + mom_v + cont)
            )
            return total

        rng_refine = jax.random.PRNGKey(run.seed + 7919)
        n_steps_taken = 0
        for outer in range(n_outer):
            rng_refine, sub_o = jax.random.split(rng_refine)
            kk = jax.random.split(sub_o, 4)
            t_idx = jax.random.choice(kk[0], T, (T_sub,), replace=False)
            cx_o = jax.random.uniform(kk[1], (n_collo_refine,), minval=0.0, maxval=1.0)
            cy_o = jax.random.uniform(kk[2], (n_collo_refine,), minval=0.0, maxval=1.0)
            ct_o = jax.random.uniform(kk[3], (n_collo_refine,),
                                      minval=float(refine_st[0]),
                                      maxval=float(refine_st[-1]))
            params, sol = lbfgs_refine(
                params, mb_lbfgs_loss, args=(t_idx, cx_o, cy_o, ct_o),
                max_steps=max_iter_inner,
                rtol=refinement.rtol, atol=refinement.atol,
                history_length=history_len,
            )
            n_steps_taken += int(getattr(sol.stats, "get", lambda *_: 0)("num_steps", 0)) if hasattr(sol, "stats") else 0
            if (outer + 1) % max(1, n_outer // 20) == 0:
                cur_loss = mb_lbfgs_loss(params, (t_idx, cx_o, cy_o, ct_o))
                print(f"  [refine] outer {outer+1:4d}/{n_outer}  loss={float(cur_loss):.4e}", flush=True)
    elif refine_choice == "lm":
        # LM uses residual vector (per-point)
        # NOTE: LM 對 vanilla Liquid arch; B0/B2 baseline 暫不支援
        # Wave 5: refine 對 d0 進行（multi-Re refine 待 GPU 階段）
        if run.arch != "liquid":
            raise NotImplementedError(
                "LM refinement 目前僅支援 --arch liquid（B0/B2 PDE path 走不同 codepath）"
            )
        T_d0 = refine_re_batch.sensor_vals.shape[0]
        K_d0 = refine_re_batch.sensor_vals.shape[1]
        xy_sq_d0 = jnp.broadcast_to(
            refine_re_batch.sensor_pos[None], (T_d0, K_d0, 2)
        ).reshape(T_d0 * K_d0, 2)
        t_sq_d0 = jnp.broadcast_to(
            refine_re_batch.sensor_time[:, None], (T_d0, K_d0)
        ).reshape(T_d0 * K_d0)
        tgt_d0 = refine_re_batch.sensor_vals.reshape(T_d0 * K_d0, 2)
        res_fn = make_pinn_residual_vector_fn(
            model,
            refine_re_batch.sensor_vals,
            refine_re_batch.sensor_pos,
            refine_re_batch.sensor_time,
            re_norm=refine_re_norm,
            norm_stats=refine_norm_stats,
            re_value=refine_re_value,
            xy_sensor_q=xy_sq_d0, t_sensor_q=t_sq_d0,
            sensor_target=tgt_d0,
            cx=cx_fix, cy=cy_fix, ct=ct_fix,
            w_data=loss.data_weight, w_phys=loss.physics_weight,
        )
        params, sol = lm_refine(
            params, res_fn, args=None,
            max_steps=refinement.steps,
            rtol=refinement.rtol, atol=refinement.atol,
        )
    elif refine_choice == "gn":
        # Matrix-free GN（移植自 sci-algorithm/natural_gradient.py）
        # jax.linearize 一次前向後重用線性算子，O(P) 記憶體，比 optimistix LM 省數倍 VRAM。
        if run.arch != "liquid":
            raise NotImplementedError("GN refinement 目前僅支援 --arch liquid")
        T_d0 = refine_re_batch.sensor_vals.shape[0]
        K_d0 = refine_re_batch.sensor_vals.shape[1]
        xy_sq_d0 = jnp.broadcast_to(
            refine_re_batch.sensor_pos[None], (T_d0, K_d0, 2)
        ).reshape(T_d0 * K_d0, 2)
        t_sq_d0 = jnp.broadcast_to(
            refine_re_batch.sensor_time[:, None], (T_d0, K_d0)
        ).reshape(T_d0 * K_d0)
        tgt_d0 = refine_re_batch.sensor_vals.reshape(T_d0 * K_d0, 2)
        res_fn = make_pinn_residual_vector_fn(
            model,
            refine_re_batch.sensor_vals,
            refine_re_batch.sensor_pos,
            refine_re_batch.sensor_time,
            re_norm=refine_re_norm,
            norm_stats=refine_norm_stats,
            re_value=refine_re_value,
            xy_sensor_q=xy_sq_d0, t_sensor_q=t_sq_d0,
            sensor_target=tgt_d0,
            cx=cx_fix, cy=cy_fix, ct=ct_fix,
            w_data=loss.data_weight, w_phys=loss.physics_weight,
        )
        params, _ = gn_refine(
            params, res_fn,
            max_steps=refinement.steps,
            lr=refinement.gn_lr,
            cg_iters=refinement.gn_cg_iters,
            damping=refinement.gn_damping,
            log_every=refinement.gn_log_every,
        )
    refine_wall = time.time() - refine_start
    print(f"  Refine done. wall={refine_wall:.2f}s")
    # LM 路徑仍有 total_before/sol；LBFGS 路徑進度已 inline 印過。
    # 兩條路皆走 final eval 量化效果，故此處不再做集中 reduction 報表。

    return state._replace(params=params, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
def _resume_reference_sensor_loss(ctx, params) -> float:
    """在確定性索引上算一次 sensor loss，供下一輪 resume 比對。

    collocation 傳零向量：本函式只取 aux 的 sensor 項，而它不依賴 cx/cy/ct
    （`_build_loss_fn` 的 data 項只吃 sensor 網格）。physics 仍會被算一次，
    那是為了不另建一條 loss 路徑——多一條就多一個會漂移的地方。
    """
    rb = ctx.re_batches[0]
    T, K = int(rb.sensor_vals.shape[0]), int(rb.sensor_vals.shape[1])
    n_collo = int(ctx.config.curriculum.n_collo_end)
    z = jnp.zeros((n_collo,))
    _, aux = ctx.loss_fn(
        params, z, z, z, jnp.ones((4,)), 0.0, 1.0, rb, 0.0,
        deterministic_sensor_idx(T * K, ctx.n_sensor_query), rb,
    )
    return float(aux[0])


# Phase 5: finalize（final eval / final ckpt / summary.json / eval_history.json）
# ─────────────────────────────────────────────────────────────────────────────

def finalize(ctx: TrainingContext, state: TrainingState,
             journal: RunJournal) -> TrainResult:
    """最終 eval + 落盤 + summary 列印。"""
    config = ctx.config
    run = config.run
    loss = config.loss
    ledger = get_ledger()
    artdir = ctx.artifacts_dir
    ckpt_dir = ctx.ckpt_dir
    datasets = ctx.datasets
    d0 = datasets[0]
    re_batch_d0 = ctx.re_batches[0]
    model = ctx.model
    ckpt_mgr = ctx.ckpt_mgr
    opt_info = ctx.opt_info
    params = state.params
    total_steps = int(run.steps)
    start_step = journal.start_step
    eval_history = journal.eval_history

    # ── Final eval (Wave 5: 對 d0 evaluation；multi-Re sweep 留給 evaluate_multi_re) ──
    print("\n" + "=" * 80)
    print("=== Final DNS evaluation (mid time slice, dataset[0]) ===")
    t_eval_start = time.time()
    final_results = evaluate_against_dns(
        model, params,
        re_batch_d0.sensor_vals, re_batch_d0.sensor_pos,
        re_batch_d0.re_norm, re_batch_d0.sensor_time,
        d0["norm_stats"],
        ctx.dns_u, ctx.dns_v, ctx.dns_t, verbose=True,
    )
    print(f"final eval wall: {time.time() - t_eval_start:.2f}s")
    eval_history.append({"step": total_steps, "results": final_results, "final": True})

    # ── Final ckpt save（避免覆寫 loop 內剛存的同 step）──
    existing_steps = set(ckpt_mgr.all_steps())
    if total_steps not in existing_steps:
        final_state = _to_ckpt_state(
            params=params, opt_state=state.opt_state,
            step=total_steps, rng_key=state.rng_collo,
            gn_state=state.gn_state, al_state=state.al_state,
            lra_state=state.lra_state,
        )
        ckpt_mgr.save(total_steps, final_state, force=True)
    final_ckpt_steps = ckpt_mgr.all_steps()

    # ── Save eval_history + run summary ──
    summary = {
        "config_path": str(Path(run.config_path).resolve()),
        "arch": run.arch,
        "optimizer": opt_info["name"],
        "steps_run": total_steps - start_step,
        "start_step": start_step,
        "end_step": total_steps,
        "train_wall_seconds": journal.train_wall,
        "final_metrics": final_results,
        "last_train_metrics": journal.last_metrics,
        "ckpt_steps": final_ckpt_steps,
        "ckpt_dir": str(ckpt_dir),
        "use_gradnorm": loss.use_gradnorm,
        "use_al": loss.use_al,
        "al_constraint_mode": loss.al_constraint_mode,
        "n_params": int(journal.n_params),
        "model_name": ctx.model_name,
        # eval 端據此擋下「用另一份 config 評這個 ckpt」。n_params 與 params 樹
        # 都分辨不出改 forward 卻不改樹的旗標（disable_cross_attention 兩側都是
        # 14546 個參數），所以必須記實際建構值。見 model_factory.model_fingerprint。
        "model_construction": model_fingerprint(ctx.model),
        # 訓練當時的正規化常數。它們是**推導出來的**——`data.py` 載入時對 strided
        # 後的子序列算 mean/std——不進 ckpt、不進參數樹，所以既有 ckpt 無從得知訓練
        # 用的是哪一組。eval 端重載 sensor 時若 stride 不同就拿到另一組，而反正規化
        # 直接用它（`evaluate.py:94`）：實測 stride 20 下 v_std 差 2.17%，stride 8 差
        # 0.11%。參數樹閘門與建構指紋都攔不到（兩者皆非參數也非建構值）。
        # 每個 dataset 各記一份（multi-Re 時每個 Re 有自己的常數）。
        "norm_stats": [{k: float(v) for k, v in d["norm_stats"].items()}
                       for d in datasets],
    }
    # cont_gradnorm 只在開啟時寫入：這一臂與 --ablate_no_al 臂的 summary 其餘欄位
    # 逐鍵相同（兩者 use_al 都是 false），沒有這個欄位就無法從產物分辨。
    # 無條件寫會讓 bit-identical A/B 對拍假紅——ab_compare.summary_diff 取兩邊鍵的
    # 聯集比對，而基準分支 verify/base-28b007a 沒有這個設定、也產不出這個鍵。
    # 關閉時省略，既有 config 的 summary 因此逐位元不變。
    if loss.cont_gradnorm:
        summary["cont_gradnorm"] = True
    # resume 的確定性參考值**刻意不放進 summary.json**。那個檔在 §7.1 的對拍裡
    # 被逐鍵比對（`ab_compare.summary_diff` 取兩邊鍵的聯集），而基準分支是重構前的
    # 程式、沒有 `deterministic_sensor_idx`——把它加進去等於把重構帶進基準，
    # 那是 §7.1 第 3 點明文禁止的。它也不是「兩側本來就該相同」的值，而是重構後
    # 才有的能力，所以走自己的檔。
    #
    # 對照：`norm_stats` 留在 summary.json，因為它**是**兩側該相同的值——兩邊算出
    # 不同的正規化常數正是該被對拍抓到的行為分歧。那一欄走 cherry-pick 同步基準分支。
    #
    # 順序有意義：**run 自己的產物先落盤，可有可無的診斷後落盤**。
    # 2026-09-18 之前是反過來的，而 `_resume_reference_sensor_loss` 在 K=400 上會
    # OOM（sensor decode 的 cross-attention 中介張量隨 K 線性放大），結果是
    # jobs 5824–5829 六支跑完 20000 步的訓練只剩 checkpoints，summary.json 與
    # eval_history.json 全部沒寫成——一個下一輪才會用到的參考值帶走了整輪的 provenance。
    summary_path = artdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    history_path = artdir / "eval_history.json"
    with open(history_path, "w") as f:
        json.dump(eval_history, f, indent=2, default=str)

    # 接住失敗並**寫下原因**。這與「用 except 吞掉錯誤」的分界在於包住的是什麼：
    # 這裡是一個純診斷、其缺席只讓下一輪 resume 少一個比對基準（`_resume_sanity_check`
    # 讀不到就退回舊行為並警告）；包住會產生結果的計算才是吞錯誤。
    ref: dict = {
        "n_sensor_query": int(ctx.n_sensor_query),
        "note": "deterministic_sensor_idx(T*K, n_sensor_query) 上的 sensor loss",
    }
    try:
        ref["sensor_loss"] = float(_resume_reference_sensor_loss(ctx, state.params))
    except Exception as exc:  # noqa: BLE001 —— 診斷失敗不得中止 finalize
        ref["sensor_loss"] = None
        ref["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"  [warn] resume 參考值算不出來，已記 unavailable_reason："
              f"{type(exc).__name__}. 本輪結果不受影響，下一輪 resume 會少一個比對基準。")
    (artdir / "resume_reference.json").write_text(json.dumps(ref, indent=2))

    if ledger is not None:
        # 尾列同時存下重播所需的 provenance：replay_schedule 必須能還原錄製當時的
        # config 與 CLI（argv/config_path），以及 `_plan_step` 的 data 衍生輸入
        # （data_meta）——兩者都不得在 replay 端用猜的。
        ledger.record(-1,
                      params_digest=params_digest(params),
                      opt_state_digest=params_digest(state.opt_state),
                      argv=sys.argv[1:],
                      config_path=str(Path(run.config_path).resolve()),
                      data_meta=_data_meta(ctx))
        ledger_path = artdir / "rng_ledger.json"
        ledger.dump(ledger_path)
        print(f"  rng_ledger   : {ledger_path}")

    print("\n" + "=" * 80)
    print("=== Training Summary ===")
    print(f"  arch          : {run.arch} ({ctx.model_name})")
    print(f"  optimizer     : {opt_info['name']}")
    print(f"  total steps   : {start_step} → {total_steps}")
    print(f"  train wall    : {journal.train_wall:.1f}s")
    print(f"  n_params      : {journal.n_params:,}")
    print(f"  final KE rel-err   : {final_results[0]['ke_rel_err']:.4f}")
    print(f"  final low_band err : {final_results[0]['low_band_rel_err']:.4f}")
    print(f"  final ω rel-err    : {final_results[0]['omega_rel_err']:.4f}")
    print(f"  final div pred/dns : {final_results[0]['div_pred_l2']:.2e} / "
          f"{final_results[0]['div_dns_l2']:.2e}")
    print(f"  ckpt steps saved   : {final_ckpt_steps}")
    print(f"  summary       : {summary_path}")
    print(f"  eval_history  : {history_path}")
    print("=" * 80)

    return TrainResult(
        state=state,
        final_results=final_results,
        eval_history=eval_history,
        summary=summary,
        summary_path=summary_path,
        history_path=history_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ledger replay（診斷/測試用；不呼叫 step_fn、不做 forward/gradient）
# ─────────────────────────────────────────────────────────────────────────────

#: RAR 步的 cx/cy/ct 佔位值。RAR 依 current params 的 residual 選點，host-side
#: 重播拿不到那一步的 params，**因此不冒充可重播**——用哨兵值明說「這格沒被驗證」，
#: 而不是塞一個看起來對的 digest 讓 oracle 安靜失效。
RAR_SENTINEL = "<rar:params-dependent>"

#: `replay_schedule` 建 ctx 走的兩條路——見該函式 docstring。字串本身即
#: 對外可見的觀測值（印在 stdout、被測試斷言），故獨立命名而非行內字面量，
#: 避免兩處拼字漂移。
REPLAY_CTX_SOURCE_BUILD_CONTEXT = "build_context"
REPLAY_CTX_SOURCE_DATA_META_FALLBACK = "data_meta fallback"


class ReplayError(RuntimeError):
    """replay 前提不成立（缺尾列 / 缺 argv / 錄製時開了 resume）。"""


def _replay_rar_residual_stub(params_, xs, ys, ts, re_batch):
    """replay 用的 per-point residual 佔位（全 0）。

    Why 佔位是安全的、而且是必要的：`rar_sample` 的 RNG 消費（`split(key, 7)`、
    pool 取樣、exploration 取樣）與 residual 值完全無關，因此用佔位仍能正確推進
    `rar_state`、讓後續步的 `rng_collo`/`rar_state` 對齊。唯一依賴 params 的是
    `top_k` 的選點——那一格由 `RAR_SENTINEL` 標記為未驗證，不參與比對。
    """
    return jnp.zeros((xs.shape[0],), dtype=jnp.float32)


def _tail_row(golden_rows: list[dict]) -> dict:
    tails = [r for r in golden_rows if r.get("step") == -1]
    if len(tails) != 1:
        raise ReplayError(f"ledger 尾列（step=-1）應恰有一列，實得 {len(tails)}")
    return tails[0]


def _missing_recorded_dataset(e: FileNotFoundError) -> bool:
    """True 若這個 FileNotFoundError 看起來是「錄製當時的資料檔不在本機」。

    Why 要辨別而非一律 fallback：資料檔不在本機是預期情境（改用尾列
    `data_meta` 重建最小 ctx）；但若哪天 `resolve_inputs` 悄悄改了 data root
    解析規則，也會丟 FileNotFoundError——那是 regression，必須讓它炸，不能被
    fallback 吞掉變成一個「看似通過」的 replay。
    """
    msg = str(e)
    return (".json" in msg or ".npy" in msg) and "/data/" in msg


# ─────────────────────────────────────────────────────────────────────────────
# data_meta：`_plan_step` 的 data 衍生輸入 —— 錄製端與重播端的唯一一份定義
# ─────────────────────────────────────────────────────────────────────────────

def _data_meta(ctx: TrainingContext) -> dict:
    """把 `_plan_step` 從 ctx 讀到的 **data 衍生量** 攤成 ledger 尾列欄位。

    Why: replay 端若沒有錄製當時的 sensor/DNS 檔，就只剩「手寫猜值」一途——
    猜錯雖會紅，但「調到綠」不構成證據。這些值在 `build_context` 期間全部
    已經算完，此處只是讀出來寫下（ledger 紀律：只記錄既有計算結果，不新增
    RNG split、sampling 或陣列重算）。

    收錄範圍恰為 `_plan_step` 讀到的 data 衍生量（哨兵測試
    `test_plan_step_ctx_reads_are_covered_by_data_meta` 盯住這個對應）：
      - `len(ctx.datasets)`                 → 選 Re index 的上界
      - `re_t_min_host` / `re_t_max_host`   → collocation 的時間範圍
      - 每個 Re 的 `sensor_vals.shape[:2]`  → sensor mini-batch 的 T*K、
                                              dropout mask 的 K
      - `crp_interp`                        → re_norm' 的取樣範圍 + norm_stats 內插表

    `ctx.crp_re_norm_scale` 刻意不收：它是 `data_kwargs.re_norm_scale`，
    replay 端從尾列 argv 重解 config 就有。把 config 也錄進來會多出一份
    可能與 config 打架的真相，違反「config 一律從 argv 還原」的優先序。
    """
    def _to_json(v):
        a = np.asarray(v, dtype=np.float64)
        return float(a) if a.ndim == 0 else [float(x) for x in a]

    return dict(
        n_datasets=len(ctx.datasets),
        re_t_min_host=[float(t) for t in ctx.re_t_min_host],
        re_t_max_host=[float(t) for t in ctx.re_t_max_host],
        # [[T, K], ...]；只取 shape，不搬任何 sensor 值
        sensor_tk=[[int(rb.sensor_vals.shape[0]), int(rb.sensor_vals.shape[1])]
                   for rb in ctx.re_batches],
        crp_interp={k: _to_json(v) for k, v in ctx.crp_interp.items()},
    )


def _blank_re_batch(T: int, K: int) -> ReBatch:
    """只帶 (T, K) 形狀的 `ReBatch` 佔位 —— **不含任何數值**。

    `_plan_step` 對 re_batch 的依賴只有 `sensor_vals.shape[0]` / `[1]`。故
    `sensor_vals` 用 `ShapeDtypeStruct` 明說「這裡只有形狀、沒有資料」，其餘
    欄位留 None：replay 若哪天真去讀值會立刻炸，而不是拿到一個看起來合理的
    假值。channel 維 C 未被 `_plan_step` 讀到，因此不記錄、也不假造——刻意
    留 rank-2，讓「有人開始讀 shape[2]」變成 IndexError 而非 silent wrong。
    """
    fields = dict.fromkeys(ReBatch._fields)
    fields["sensor_vals"] = jax.ShapeDtypeStruct((int(T), int(K)), jnp.float32)
    return ReBatch(**fields)


def _ctx_from_data_meta(
    config: KolmogorovEffectiveConfig, tail: dict, missing: FileNotFoundError,
) -> TrainingContext:
    """用尾列 `data_meta` 重建 `_plan_step` 所需的最小 ctx。

    只填 `_plan_step` 會讀的欄位（見 `_data_meta` docstring），其餘留 None：
    replay 若走到別的欄位就該炸，不該拿到一個看起來合理的預設值。
    `crp_re_norm_scale` 由 config 重解（與 `build_context` 共用
    `re_norm_scale_of`），不從 ledger 取——config 一律以尾列 argv 為準。
    """
    meta = tail.get("data_meta")
    if meta is None:
        raise ReplayError(
            "ledger 尾列缺 'data_meta'，且錄製當時的資料檔不在本機"
            f"（{missing}）—— replay 不得在本端猜 data 衍生量。"
            "此 fixture 錄製於 data_meta 之前，需在 lab-server 以現版 "
            "train_kolmogorov.py 重錄。"
        ) from missing
    n_datasets = int(meta["n_datasets"])
    t_min = [float(x) for x in meta["re_t_min_host"]]
    t_max = [float(x) for x in meta["re_t_max_host"]]
    tk = [(int(T), int(K)) for T, K in meta["sensor_tk"]]
    if not len(t_min) == len(t_max) == len(tk) == n_datasets:
        raise ReplayError(
            f"data_meta 自相矛盾：n_datasets={n_datasets} 但 per-Re 欄位長度為 "
            f"{len(t_min)}/{len(t_max)}/{len(tk)}"
        )
    # scalar 保持 float、per-Re 表回 numpy，與 _build_crp_interp 的產物同型
    crp_interp = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=np.float64))
        for k, v in meta["crp_interp"].items()
    }
    blank = TrainingContext(**dict.fromkeys(TrainingContext._fields))
    return blank._replace(
        config=config,
        datasets=[None] * n_datasets,   # 只有 len() 被讀到
        re_batches=[_blank_re_batch(T, K) for T, K in tk],
        re_t_min_host=t_min,
        re_t_max_host=t_max,
        crp_interp=crp_interp,
        crp_re_norm_scale=re_norm_scale_of(config.data),
        n_sensor_query=effective_sensor_query_points(
            config.curriculum.n_sensor_query_requested, tk[0][0] * tk[0][1],
        ),
    )


def replay_schedule(golden_rows: list[dict]) -> list[dict]:
    """重播 host-side 決策序列，不呼叫 step_fn。

    Why: bit-identical 最高風險是 RNG 消費時序（spec §5）。該序列完全由
    host 端決定，不需要 forward/gradient 即可重現 —— 因此可在本機以
    unit test 驗證，不違反「本機禁跑 training」。

    golden_rows 只用來取回錄製當時的 provenance（尾列），不參與比對：
      - `argv`      → **config 一律從這裡重解**，replay 端沒有第二個來源。
      - `data_meta` → `_plan_step` 的 data 衍生輸入（時間範圍、per-Re T/K、
        CRP 內插表）。錄製當時的資料檔在本機時走正式的 `build_context`；
        不在時改用 `data_meta` 重建最小 ctx，兩條路都不猜任何值。

    尾列缺 `data_meta` 且資料檔又不在本機 → `ReplayError`。這裡沒有「猜一個
    合理值」的退路，因為猜對猜錯都會產生一份看起來完整的 replay。

    這兩條路的驗證強度**不同**：`build_context` 重新從資料檔算出
    `_load_datasets` / DNS 時間軸對齊 / `_build_crp_interp` / `re_t_min_host`
    等衍生量，`data_meta` fallback 則是把錄製當時就算好的同一批數字原樣讀回來
    ——換言之，走 fallback 時這些量本身的正確性完全沒被本次 replay 覆蓋到。
    兩條路都是合法、刻意保留的（本機常態就是走 fallback），只是強度不同；
    故此處把實際走了哪條路印出來（`REPLAY_CTX_SOURCE_*`），讓讀 stdout 的人
    能一眼判斷這次 replay 覆蓋到哪一層，而不必臆測。
    """
    tail = _tail_row(golden_rows)
    argv = tail.get("argv")
    if argv is None:
        raise ReplayError(
            "ledger 尾列缺 'argv' —— replay 不得在本端猜 config，請補 Task 2 的尾列欄位並重錄 fixture"
        )
    resolved = resolve_inputs(list(argv))
    config = resolved.config
    if config.run.resume_step is not None:
        raise ReplayError("replay 不支援錄製時開啟 --resume_step 的 ledger（restore 路徑另有 rng_check 中立性）")
    try:
        ctx = build_context(config)
        ctx_source = REPLAY_CTX_SOURCE_BUILD_CONTEXT
    except FileNotFoundError as missing:
        if not _missing_recorded_dataset(missing):
            raise
        ctx = _ctx_from_data_meta(config, tail, missing)
        ctx_source = REPLAY_CTX_SOURCE_DATA_META_FALLBACK
    print(f"[replay_schedule] ctx source: {ctx_source}")

    n_datasets = len(ctx.datasets)
    sensor_dropout_rate = float(config.curriculum.sensor_dropout_rate)

    rng, _rng_init = _main_rng(config.run.seed)          # #0/#1；model.init 本身不重播
    rng, rar_state, rng_collo, rng_re, rng_crp = _open_loop_streams(
        config.loss.use_continuous_re_physics, rng,
    )

    rows: list[dict] = []
    for s in range(1, int(config.run.steps) + 1):
        plan = _plan_step(
            ctx, s,
            params=None,
            rng_collo=rng_collo, rng_re=rng_re, rng_crp=rng_crp,
            rar_state=rar_state,
            n_datasets=n_datasets,
            sensor_dropout_rate=sensor_dropout_rate,
            rar_residual_fn=_replay_rar_residual_stub,
        )
        rng_collo = plan.rng_collo
        rng_re = plan.rng_re
        rng_crp = plan.rng_crp
        rar_state = plan.rar_state
        row = {"step": s, **_ledger_fields(ctx, plan)}
        if plan.rar_triggered:
            row["cx"] = row["cy"] = row["ct"] = RAR_SENTINEL
        rows.append(row)
    return rows
