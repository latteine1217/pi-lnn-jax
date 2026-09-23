"""執行期 —— `TrainingContext` → 訓練 → `TrainResult`。

What: `train_cylinder.main()` 的執行期段落（params 初始化 / 訓練迴圈 /
      DNS grid 重建 eval）逐字搬入，拆成 module-level 函式，state 以
      `TrainingState` 顯式傳遞。**不用 class 包裝流程**——流程只有一條直線，
      物件化只會把「誰改了哪個欄位」藏起來（同 kolmogorov/run.py）。

Why 這裡是整個搬移的最高風險面：bit-identical 契約的破口幾乎都在 RNG
     消費時序，而 cylinder 有論文證據綁著（07b_cylinder_feasibility.tex、
     appendix/C_cylinder.tex）。以下表格照抄 spec §5，是本檔的承重不變量；
     任何改動（包含「順手清理」）都必須先對照這張表。

═══════════════════════════════════════════════════════════════════════════
spec §5.1 初始化（`initialize`）—— **單一** RandomState 連抽兩次
───────────────────────────────────────────────────────────────────────────
| # | 動作                                                  | 條件 |
|---|-------------------------------------------------------|------|
| 0 | `rng = np.random.RandomState(seed)`                   | 恆   |
|   | → `init_xy = rng.uniform(0, 1, (8, 2))`               | 恆   |
|   | → `init_t  = rng.uniform(st0, st0 + T_total, (8,))`   | 恆   |
| 1 | `params = model.init(jax.random.PRNGKey(seed), …)`    | 恆   |

**與 Kolmogorov 相反**：Kolmogorov 是兩個各自建構的 `RandomState`（同 seed），
因此它的 `init_t` 不是 `init_xy` 的續抽；cylinder 是**一個**生成器連續抽兩次，
`init_t` 正是 `init_xy` 的續抽。兩者都不得「統一」——各自都是承重的。

spec §5.2 迴圈 stream（`_open_loop_streams`）
───────────────────────────────────────────────────────────────────────────
| # | 動作                                              | 條件 |
|---|---------------------------------------------------|------|
| 2 | `rk = jax.random.PRNGKey(seed + 1)`  ← 是 `seed+1`，不是 `seed` | 恆 |
| 3 | `np_rng = np.random.RandomState(seed + 7)`         | 恆   |
|   | 只給 time-marching 的受限 sensor 取樣用            |      |

兩條迴圈 stream 各自**由 seed 重新建構**，不由 §5.1 的 init stream 衍生。
因此 `replay_schedule` 不需要重播 `initialize`（Kolmogorov 則必須，那邊的
`rng_collo`/`rng_re`/`rng_crp` 都是主流 split 出來的）。

spec §5.3 每步消費（`_plan_step`）—— **無條件**，與 Kolmogorov 的條件式不同
───────────────────────────────────────────────────────────────────────────
| # | 動作                                              | 條件 |
|---|---------------------------------------------------|------|
| 1 | `rk, k1, k2, k3 = jax.random.split(rk, 4)`        | **恆**（4-way） |
| 2 | `ks = jax.random.split(k1, 3)`                    | **恆**（3-way） |
| 3 | `cx/cy/ct = jax.random.uniform(ks[0..2], …)`      | 恆   |
| 4 | `sq_idx`：`use_tm` ? `np_rng.choice(valid, …)`    | 分支；但 `k3` 兩路 |
|   | : `jax.random.choice(k3, T*K, …)`                 | 都已在 #1 抽出   |
| 5 | wall BC：**`is_controlled`**（不是 `use_tm`）決定  | 分支；兩路都消費 |
|   | `sample_wall_bc_moving(k2, …)` / `sample_wall_bc(k2, …)` | `k2`      |

**關鍵**：`k3` 在 #1 就被抽出，`use_tm` 為真時它**不被使用但已消耗**。任何
「只在需要時才 split」的最佳化都會改變 `rk` 的後續狀態。這一條有兩份 golden
fixture 直接背書：`c1`/`c2` 同 config 同 seed、只差 `time_marching`，其 `cx`
digest 在全部 21 步相同（`rk` 同步推進），`sq_idx` 在全部 21 步不同（取樣來源
不同）。分支若被改成條件式 split，`c1`/`c2` 的 `cx` 會從第 1 步就分岔。

迴圈骨架與 Kolmogorov 的另兩處差異（同屬既有行為，不得「對齊」）：
  - 迴圈範圍是 `range(steps + 1)`（**含第 0 步**），非 `range(start+1, steps+1)`。
  - GradNorm / AL 觸發多一個 `s > 0` 條件。
═══════════════════════════════════════════════════════════════════════════

Step planning（`_plan_step`）刻意留在本檔而非抽成獨立 pure function：它讀
`ctx` 的資料衍生量、且 `np_rng` 是就地推進的有狀態物件，從來就不是純函式。
`run_loop` 與 `replay_schedule` **呼叫同一個 `_plan_step`**，不另寫一份——
兩份會漂移，而漂移後 replay 會安靜地不再測真正在跑的東西。

cylinder **沒有** `restore`／`refine`（完全沒有 checkpoint；CEXP-002 是 1-shot、
`--resume` 直接 fail-fast），故本檔只有 initialize → run_loop → finalize 三段。
也沒有 Kolmogorov 的 `RunJournal`：cylinder 不落任何 artifact，沒有跨階段需要
累積的觀測值，補一個空殼只會多一份沒人讀的狀態。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.boundary import (
    CylinderGeometry,
    OscillatingGeometry,
    body_center_at,
    sample_wall_bc,
    sample_wall_bc_moving,
)
from pi_lnn_jax.evaluate import compute_metrics
from pi_lnn_jax.losses import (
    al_init,
    al_update,
    gradnorm_init,
    gradnorm_step,
    gradnorm_weights,
)
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.pipeline._ledger import digest, get_ledger, params_digest
from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext, build_context
from pi_lnn_jax.pipeline.cylinder.config import CylinderEffectiveConfig, resolve_inputs

EVAL_CHUNK = 2048  # DNS grid 重建分塊，避免 attention 中間張量 OOM（同 v1）
LO = LiquidOperator


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class TrainingState(NamedTuple):
    """訓練期的可變狀態（in-memory）。

    刻意**沒有** `step` 欄位：cylinder 無 checkpoint、無 resume、也沒有以 step
    索引的產物，步數不是任何下游決策的輸入。NaN abort 由迴圈內的 `sys.exit(2)`
    當場終止並印出步數，不需要再存一份。
    """
    params: Any
    opt_state: Any
    rk: Any                # 迴圈 PRNG stream（PRNGKey(seed+1)）；initialize 後為 None
    gn_state: Any
    al_state: Any
    task_weights: Any


class TrainResult(NamedTuple):
    """`finalize` 的產物。

    `eval_metrics` 存的是**逐 eval 時刻的序列**，不是聚合值：印出去的
    `np.mean(...)` 就在 `finalize` 裡，此處若再算一次平均，同一個數字就有兩份
    來源可以漂移。呼叫端要摘要自己 reduce。
    """
    state: TrainingState
    eval_metrics: dict


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_training(ctx: TrainingContext) -> TrainResult:
    """initialize → run_loop → finalize（cylinder 無 restore / refine）。"""
    state = initialize(ctx)
    state = run_loop(ctx, state)
    return finalize(ctx, state)


# ─────────────────────────────────────────────────────────────────────────────
# RNG stream 開場（spec §5.2）—— run_loop 與 replay_schedule 共用
# ─────────────────────────────────────────────────────────────────────────────

def _open_loop_streams(seed: int):
    """spec §5.2 #2/#3：迴圈的兩條 stream。

    兩者都由 `seed` 直接建構（`+1` / `+7`），與 `initialize` 的 init stream
    互不相干——順序與偏移量即規格。
    """
    rk = jax.random.PRNGKey(seed + 1)
    np_rng = np.random.RandomState(seed + 7)     # time-marching sq_idx 受限取樣用
    return rk, np_rng


# ─────────────────────────────────────────────────────────────────────────────
# Step planning（spec §5.3）—— run_loop 與 replay_schedule 的唯一一份
# ─────────────────────────────────────────────────────────────────────────────

class _StepPlan(NamedTuple):
    """一個 training step 的 host-side 決策全集（不含 forward/gradient）。

    收錄範圍 = 迴圈在呼叫 `step_fn` 之前決定的全部東西，故 `_ledger_fields`
    只是把它攤平，不另外計算任何值。
    """
    t_max: float
    cx: Any
    cy: Any
    ct: Any
    sq_idx: Any
    bc_in: Any
    bc_body: Any
    bc_slip: Any
    bc_bvel: Any
    trig_gradnorm: bool
    trig_al: bool
    rk: Any                # 推進後的 PRNG stream（`np_rng` 是就地推進，不回傳）


def _plan_step(ctx: TrainingContext, s: int, *, rk, np_rng) -> _StepPlan:
    """算出第 s 步的所有 host-side 決策，並回傳推進後的 `rk`。

    這不是 pure function：`np_rng`（`RandomState`）在 time-marching 路徑上被
    就地推進。刻意不假裝它是——把它模型化成純函式只會逼出一個假的介面。

    函式開頭把 `ctx` / `eff` 攤回**與搬移前同名**的區域變數，下方每一行因此
    維持逐字搬移，可直接與搬移前的 `train_cylinder.py` 迴圈體 diff。
    """
    config = ctx.config
    run = config.run
    loss = config.loss
    curriculum = config.curriculum
    is_controlled = run.is_controlled
    n_collo, bc_n = run.n_collo, loss.bc_n
    gradnorm_freq, al_update_freq = loss.gradnorm_freq, loss.al_update_freq
    use_tm, tm_start = curriculum.use_time_marching, curriculum.time_marching_start
    tm_warmup = config.time_marching_warmup_steps
    # n_sensor_q 讀 ctx —— 那是被 `min(·, T*K)` 夾過的值；eff 裡的是未夾值（只供 banner）。
    n_sensor_q = ctx.n_sensor_q
    st0, T_total, tm_end = ctx.st0, ctx.T_total, ctx.tm_end
    K, T, t_q_full_np = ctx.K, ctx.T, ctx.t_q_full_np
    geom = ctx.geom

    rk, k1, k2, k3 = jax.random.split(rk, 4)
    ks = jax.random.split(k1, 3)
    # time-marching：t_max 從 tm_start 線性展開到 tm_end（warmup 步）；關閉則為全時間窗
    # 起點 clamp 到 ≥ st0：st0 > tm_start 時舊式 t_max < st0 會給 uniform 反向區間
    if use_tm:
        progress = min(s / max(tm_warmup, 1), 1.0)
        tm_lo = max(tm_start, st0)
        t_max = tm_lo + (tm_end - tm_lo) * progress
    else:
        t_max = st0 + T_total
    cx = jax.random.uniform(ks[0], (n_collo,), jnp.float32, 0.0, 1.0)
    cy = jax.random.uniform(ks[1], (n_collo,), jnp.float32, 0.0, 1.0)
    ct = jax.random.uniform(ks[2], (n_collo,), jnp.float32, st0, t_max)  # 限 ≤ t_max
    if use_tm:
        valid = np.nonzero(t_q_full_np <= t_max)[0]   # 只取 ≤ t_max 的 sensor query
        sq_idx = jnp.asarray(np_rng.choice(valid, n_sensor_q, replace=len(valid) < n_sensor_q))
    else:
        sq_idx = jax.random.choice(k3, T * K, (n_sensor_q,), replace=False)
    # controlled：moving BC 採物理時間窗 [st0, t_max]，附剛體壁速；cylinder：靜態 BC（壁速為 0，傳 dummy）
    if is_controlled:
        bc_in, bc_body, bc_slip, bc_bvel = sample_wall_bc_moving(k2, geom, bc_n, st0, t_max)
    else:
        bc_in, bc_body, bc_slip = sample_wall_bc(k2, geom, bc_n)
        bc_bvel = jnp.zeros((bc_n, 2), jnp.float32)
    # 兩個觸發條件在此算一次，ledger 與 run_loop 的控制流共用同一個值。
    # Why 收進 plan：若 ledger 自行再算一次，兩份條件會漂移，ledger 就會安靜說謊。
    # 純整數取模、無 RNG／副作用，提前求值不改變任何行為。
    trig_gradnorm = gradnorm_freq > 0 and s > 0 and s % gradnorm_freq == 0
    trig_al = al_update_freq > 0 and s > 0 and s % al_update_freq == 0

    return _StepPlan(
        t_max=t_max, cx=cx, cy=cy, ct=ct, sq_idx=sq_idx,
        bc_in=bc_in, bc_body=bc_body, bc_slip=bc_slip, bc_bvel=bc_bvel,
        trig_gradnorm=trig_gradnorm, trig_al=trig_al, rk=rk,
    )


def _ledger_fields(plan: _StepPlan) -> dict:
    """把 `_StepPlan` 攤成 ledger 欄位。

    `run_loop`（錄製）與 `replay_schedule`（重播）共用這一份，否則兩邊的欄位
    定義會漂移，比對就變成比兩份不同的東西。只讀既有已算好的值，不新增任何
    RNG split、取樣或陣列重算。

      - `t_max`      time-marching 時間課程；同時是 ct 的上界、time-marching
                     sensor 篩選門檻、moving BC 時間窗上界，以及 step_fn 的 t_bc_hi
      - `cx/cy/ct`   collocation 取樣（消耗 k1 → ks[0..2]）
      - `sq_idx`     sensor query minibatch（k3 或 np_rng 兩路，見 spec §5.3）
      - `bc_*`       wall BC 取樣（消耗 k2；bc_bvel 在靜態路徑為常數零，仍記錄以免兩路欄位不同形）
      - `trig_*`     GradNorm / AL 的觸發布林

    大陣列一律只存 deterministic digest，不把整份陣列搬進檔案（_ledger.py 紀律）。
    """
    return dict(
        t_max=float(plan.t_max),
        cx=digest(plan.cx), cy=digest(plan.cy), ct=digest(plan.ct),
        sq_idx=digest(plan.sq_idx),
        bc_in=digest(plan.bc_in), bc_body=digest(plan.bc_body),
        bc_slip=digest(plan.bc_slip), bc_bvel=digest(plan.bc_bvel),
        trig_gradnorm=bool(plan.trig_gradnorm),
        trig_al=bool(plan.trig_al),
    )


# ─────────────────────────────────────────────────────────────────────────────
# data_meta：`_plan_step` 的 data 衍生輸入 —— 錄製端與重播端的唯一一份定義
# ─────────────────────────────────────────────────────────────────────────────

def _data_meta(ctx: TrainingContext) -> dict:
    """把 `_plan_step` 從 ctx 讀到的**資料衍生量**攤成 ledger 尾列欄位。

    Why: replay 端若沒有錄製當時的 npz，就只剩「手寫猜值」一途；猜錯雖會紅，
    但「調到綠」不構成證據。這四個欄位是 `_plan_step` 實際讀到的資料衍生量的
    最小完備集：

      - `sensor_time` = `ctx.st`（[T] 物理時間）→ 一次覆蓋 `st0`(=st[0])、
        `tm_end`(=st[-1])、`T`(=len) 三個讀取點。
      - `n_sensors`   = `ctx.K` → 與上一項合成 `T*K`（非 time-marching 路徑
        `jax.random.choice` 的上界，以及 `n_sensor_q` 的 clamp 上限）。
      - `t_total`     = `ctx.T_total` → **不從 sensor_time 反推**。assembly 用的是
        `float(st[-1] - st[0])`，那是 float32 相減再放大到 float64；replay 端
        若用兩個 float64 端點相減會得到「精確差」，兩者可差 1 ulp，而它是
        非 time-marching 路徑 `t_max` 的來源 → ct 取樣區間、進而 digest 全歪。
      - `geom`        → wall BC 取樣點的座標直接由幾何決定
        （`sample_wall_bc` 讀 body_center/body_radius；`sample_wall_bc_moving`
        讀 body_radius_phys/Lx/Ly/amp/freq/phase/axis/base_center）。

    刻意不記 `t_q_full_np`（[T*K]）：time-marching 分支的
    `valid = np.nonzero(t_q_full_np <= t_max)` 看似需要整份陣列，但
    `t_q_full = broadcast(st[:,None], (T,K)).reshape(T*K)` 就是「st 每個元素重複
    K 次」，故 `sensor_time` + `n_sensors` 已能逐位元重建它。多記 T*K 份浮點數
    只是把同一個事實抄第二遍，還多一份會與 sensor_time 打架的真相。
    **重建時必須 `np.repeat(..., dtype=np.float32)`**：`t_q_full_np` 是 float32，
    `t_max` 是 Python float，NumPy 的 weak-scalar 提升（NEP 50）會把純量降成
    float32 再比較；用 float64 重建會在邊界 t_max 上選到不同的 `valid`
    （tests/test_cylinder_ledger.py 有存證）。

    config／CLI 衍生量（n_collo、bc_n、use_tm、gradnorm_freq、case…）一律不收：
    replay 端從尾列 argv 重解 config 就有，收進來會多一份可能打架的真相。
    `n_sensor_q` 雖被 `min(·, T*K)` 夾過，夾的上界由本表的 sensor_time/n_sensors
    給出，replay 端重算即可，故同樣不收。
    """
    return dict(
        sensor_time=[float(v) for v in np.asarray(ctx.st)],
        n_sensors=int(ctx.K),
        t_total=float(ctx.T_total),
        # geometry NamedTuple 全是純量／tuple，原樣攤平即可 JSON 序列化
        geom={k: ([float(z) for z in v] if isinstance(v, tuple) else float(v))
              for k, v in ctx.geom._asdict().items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: initialize（model.init / opt_state / GradNorm / AL state）
# ─────────────────────────────────────────────────────────────────────────────

def initialize(ctx: TrainingContext) -> TrainingState:
    """建 params / opt_state / GradNorm / AL state；消耗 spec §5.1 的兩條 init stream。"""
    run = ctx.config.run
    loss = ctx.config.loss
    seed = run.seed
    al_rho, al_lambda_clip = loss.al_rho, loss.al_lambda_clip
    n_sensor_q = ctx.n_sensor_q
    sv_TKC, sp, st, RE_NORM = ctx.sv_TKC, ctx.sp, ctx.st, ctx.re_norm
    st0, T_total = ctx.st0, ctx.T_total
    model = ctx.model

    # ── 參數初始化（spec §5.1：單一 RandomState 連抽 init_xy/init_t，接著 model.init
    #    消耗主 PRNGKey；三者都消耗隨機源 → 建構期不得碰，故留在這裡）──
    rng = np.random.RandomState(seed)
    init_xy = jnp.asarray(rng.uniform(0, 1, (8, 2)), jnp.float32)
    init_t = jnp.asarray(rng.uniform(st0, st0 + T_total, (8,)), jnp.float32)
    params = model.init(jax.random.PRNGKey(seed), sv_TKC, sp, RE_NORM, st, init_xy, init_t)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  params={n_params:,}")
    # 初始化階段的 params digest。尾列的 `params_digest` 記的是**訓練後**的
    # 參數（在 finalize 寫），無法用來檢查「建構 + 初始化」這一段；
    # 故在此另記一份，由 `Ledger.note` 帶到尾列。
    ledger = get_ledger()
    if ledger is not None:
        ledger.note(init_params_digest=params_digest(params))

    gn_state = gradnorm_init([1.0, 0.01, 0.01], ("data", "ns_u", "ns_v"))
    task_weights = gradnorm_weights(gn_state)
    # AL state：continuity 純由 AL 約束（固定啟用）
    al_state = al_init(init_lambda=0.0, rho=al_rho, lambda_clip=al_lambda_clip)

    # 首次更新印一行確認 ref 子樹（temporal_encoder 而非 full params）
    # probe 吃真實 params（jax.grad）→ 執行期；ref path 常數本身在 assembly。
    _probe = jax.grad(ctx.data_loss_fn)(params, jnp.arange(n_sensor_q))
    _is_temporal = ctx.get_subtree(_probe) is not _probe
    print(f"  GradNorm ref-subtree={ctx.gn_ref_path} → "
          f"{'temporal_encoder' if _is_temporal else 'FULL_PARAMS(fallback)'}")

    opt_state = ctx.tx.init(params)
    print(f"  Optimizer: {ctx.opt_info['name']}  lr_schedule={ctx.opt_info['lr_schedule']}  "
          f"fallback={ctx.opt_info['fallback_to']}")

    return TrainingState(
        params=params, opt_state=opt_state, rk=None,
        gn_state=gn_state, al_state=al_state, task_weights=task_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: run_loop（主訓練迴圈；含 ledger 尾列）
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(ctx: TrainingContext, state: TrainingState) -> TrainingState:
    """主訓練迴圈。RNG 開場 §5.2 與每步的 §5.3 消費順序皆在此（後者委由 `_plan_step`）。"""
    run = ctx.config.run
    loss = ctx.config.loss
    # 診斷用（PILNN_RNG_LEDGER）；關閉時為 None，下方所有記錄點皆跳過。
    ledger = get_ledger()
    steps = run.steps
    gradnorm_ema = loss.gradnorm_ema_momentum
    gradnorm_min, gradnorm_max = loss.gradnorm_min, loss.gradnorm_max
    step_fn = ctx.step_fn
    compute_task_grad_norms = ctx.grad_norm_fn

    params = state.params
    opt_state = state.opt_state
    gn_state = state.gn_state
    al_state = state.al_state
    task_weights = state.task_weights

    # ── 訓練 loop ──
    rk, np_rng = _open_loop_streams(run.seed)
    t0 = time.time()
    last = t0
    print(f"\n{'step':>6s} {'total':>11s} {'data':>11s} {'ns_u':>11s} "
          f"{'ns_v':>11s} {'cont':>11s} {'bc':>11s} {'w_d/u/v/c':>22s} {'t_max':>6s} {'wall':>7s}")
    for s in range(steps + 1):
        plan = _plan_step(ctx, s, rk=rk, np_rng=np_rng)
        rk = plan.rk

        if ledger is not None:
            ledger.record(s, **_ledger_fields(plan))

        params, opt_state, total, aux = step_fn(
            params, opt_state, plan.cx, plan.cy, plan.ct, task_weights, al_state.lambda_, plan.sq_idx,
            jnp.asarray(plan.t_max, jnp.float32), plan.bc_in, plan.bc_body, plan.bc_slip, plan.bc_bvel)

        # GradNorm 更新（每 gradnorm_freq 步）
        if plan.trig_gradnorm:
            tg = compute_task_grad_norms(params, plan.cx, plan.cy, plan.ct, plan.sq_idx)
            gn_state = gradnorm_step(gn_state, tg, ema_momentum=gradnorm_ema,
                                     min_weight=gradnorm_min, max_weight=gradnorm_max)
            task_weights = gradnorm_weights(gn_state)
        # AL dual update：用該步 continuity mse（aux[3]）更新 λ
        if plan.trig_al:
            al_state = al_update(al_state, aux[3])

        if s == 0 or (s + 1) % 500 == 0 or s == steps:
            dl, nu_l, nv_l, c_l, bl = aux
            wall = time.time() - last
            last = time.time()
            w_str = "/".join(f"{float(w):.2f}" for w in task_weights)
            print(f"{s:>6d} {float(total):>11.4e} {float(dl):>11.4e} "
                  f"{float(nu_l):>11.4e} {float(nv_l):>11.4e} {float(c_l):>11.4e} "
                  f"{float(bl):>11.4e} {w_str:>22s} {plan.t_max:>6.2f} {wall:>7.2f}")
        if not np.isfinite(float(total)):
            # fail-fast：非零 exit code，且**不**繼續 ledger 尾列與 eval。
            # Why 不用 break：break 會讓壞掉的 run 走完 finalize、印出 `ke_pred=nan`
            # 並以 rc=0 收場，Slurm 記成 COMPLETED，於是混進 multi-seed 聚合
            # （job 4745 實際發生過）。訊息與退出碼對齊 kolmogorov/run.py。
            print(f"[FATAL] loss NaN/Inf at step {s} — abort", flush=True)
            sys.exit(2)
    print(f"train wall: {time.time() - t0:.1f}s")

    if ledger is not None:
        # 尾列同時存下重播所需的 provenance：replay 必須能還原錄製當時的 config 與 CLI
        # （argv/config_path），以及迴圈的資料衍生輸入（data_meta）——兩者都不得在
        # replay 端用猜的。
        # Why 落在訓練迴圈之後、eval 之前：eval 不改 params/opt_state（只做 decode），
        # 兩處記到的值相同；但 cylinder 的 DNS grid 重建正是 OOM 高風險段（EVAL_CHUNK
        # 存在的理由），擺在它後面會讓一次 eval 失手賠掉整份錄製。
        ledger.record(-1,
                      params_digest=params_digest(params),
                      opt_state_digest=params_digest(opt_state),
                      argv=sys.argv[1:],
                      config_path=str(Path(run.config_path).resolve()),
                      data_meta=_data_meta(ctx))
        # cylinder 本身不落任何 artifact，故 artifacts_dir 只在 gate 內讀，
        # 由錄製用 config 指定（schema 預設 "artifacts"）。
        ledger_path = Path(run.artifacts_dir) / "rng_ledger.json"
        ledger.dump(ledger_path)
        print(f"  rng_ledger   : {ledger_path}")

    return state._replace(
        params=params, opt_state=opt_state, rk=rk,
        gn_state=gn_state, al_state=al_state, task_weights=task_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: finalize（DNS grid 重建 eval；KE 主判據）
# ─────────────────────────────────────────────────────────────────────────────

def finalize(ctx: TrainingContext, state: TrainingState) -> TrainResult:
    """DNS grid 重建 eval：分塊 recon → 逐 eval 時刻 metric → KE / over-energy 報表。"""
    is_controlled = ctx.config.run.is_controlled
    d = ctx.npz
    sv_TKC, sp, st, RE_NORM = ctx.sv_TKC, ctx.sp, ctx.st, ctx.re_norm
    obs_mean, obs_std = ctx.obs_mean, ctx.obs_std
    Lx, Ly, bcen, br = ctx.Lx, ctx.Ly, ctx.bcen, ctx.br
    geom, u_inf = ctx.geom, ctx.u_inf
    model = ctx.model
    params = state.params

    # ── DNS grid 重建 eval（KE 主判據）──
    dns_u = np.asarray(d["dns_u"])          # [n_eval,H,W] 物理
    dns_v = np.asarray(d["dns_v"])
    dns_x = np.asarray(d["dns_x"])          # normalized grid 軸
    dns_y = np.asarray(d["dns_y"])
    dns_t_idx = np.asarray(d["dns_t_idx"])
    Hn, Wn = dns_u.shape[1], dns_u.shape[2]
    gx, gy = np.meshgrid(dns_x, dns_y)                              # [H,W]
    gx = jnp.asarray(gx.reshape(-1), jnp.float32)
    gy = jnp.asarray(gy.reshape(-1), jnp.float32)
    xy_grid = jnp.stack([gx, gy], axis=-1)                         # [H*W, 2]
    h_eval = model.apply(params, sv_TKC, sp, RE_NORM, st, method=LO.encode)

    @jax.jit
    def recon_chunk(xy_chunk, tq_chunk):
        out = model.apply(params, xy_chunk, tq_chunk, h_eval, st, sp,
                          method=LO.decode_query)
        u = out[:, 0] * obs_std[0] + obs_mean[0]
        v = out[:, 1] * obs_std[1] + obs_mean[1]
        return u, v

    def recon(tq):
        n = xy_grid.shape[0]
        us, vs = [], []
        for off in range(0, n, EVAL_CHUNK):
            xc = xy_grid[off:off + EVAL_CHUNK]
            tc = jnp.full((xc.shape[0],), tq, dtype=jnp.float32)
            uc, vc = recon_chunk(xc, tc)
            jax.block_until_ready(uc)
            us.append(np.asarray(uc))
            vs.append(np.asarray(vc))
        return np.concatenate(us), np.concatenate(vs)

    # body 內遮罩（流體域 bool；所有判據共用，比照 v1 245）
    gx2 = np.asarray(gx).reshape(Hn, Wn)
    gy2 = np.asarray(gy).reshape(Hn, Wn)
    body_mask = (np.sqrt((gx2 - bcen[0]) ** 2 + (gy2 - bcen[1]) ** 2) - br) > 0
    # ── over-energy 定位（Finding #9 診斷）：freestream(無 sensor) vs wake(有 sensor) 分區 ──
    x_sensor_min = float(np.asarray(sp)[:, 0].min())     # sensor 最上游 x（≈0.18）
    free_mask = body_mask & (gx2 < x_sensor_min)          # 來流/freestream 區（無 sensor，只有 inflow BC）
    wake_mask = body_mask & (gx2 >= x_sensor_min)         # wake 區（有 sensor）
    inflow_band = gx2 < 0.02                              # x≈0 inflow 帶（檢查 BC 是否釘住 u=u_inf）

    def _ke_region(ke_p, ke_r, m):
        r = float(np.sum(ke_p * m) / (np.sum(ke_r * m) + 1e-12))
        e = float(np.linalg.norm((ke_p - ke_r) * m) / (np.linalg.norm(ke_r * m) + 1e-12))
        return r, e

    ke_ratios, ke_relerrs, omega_relerrs, div_l2s = [], [], [], []
    free_r, free_e, wake_r, wake_e, inflow_u = [], [], [], [], []
    for i, ti in enumerate(dns_t_idx):
        tq = float(st[ti])
        # controlled：body 逐時刻移動 → 每個 eval 時刻用 body_center_at 重算圓遮罩（取代靜態圓）
        if is_controlled:
            # 物理空間正圓判斷（與 body_sdf_moving 一致）：normalized 偏移經 Lx/Ly 還原
            cbx, cby = body_center_at(geom, tq)
            bmask_t = (np.sqrt(((gx2 - float(cbx)) * Lx) ** 2
                               + ((gy2 - float(cby)) * Ly) ** 2)
                       - float(d["body_radius_phys"])) > 0
            fmask_t = bmask_t & (gx2 < x_sensor_min)
            wmask_t = bmask_t & (gx2 >= x_sensor_min)
        else:
            bmask_t, fmask_t, wmask_t = body_mask, free_mask, wake_mask
        pu, pv = recon(tq)
        pu = pu.reshape(Hn, Wn)
        pv = pv.reshape(Hn, Wn)
        du, dv = dns_u[i], dns_v[i]
        mt = compute_metrics(
            pu, pv, du, dv, periodic=False,
            dns_x=dns_x, dns_y=dns_y, Lx=Lx, Ly=Ly, mask=bmask_t,
        )
        ke_ratios.append(mt["ke_pred_over_ref"])
        ke_relerrs.append(mt["ke_rel_err"])
        omega_relerrs.append(mt["omega_rel_err"])
        div_l2s.append(mt["div_pred_l2"])
        # 分區 KE + inflow u
        ke_p = 0.5 * (pu ** 2 + pv ** 2)
        ke_r = 0.5 * (du ** 2 + dv ** 2)
        fr, fe = _ke_region(ke_p, ke_r, fmask_t)
        free_r.append(fr)
        free_e.append(fe)
        wr, we = _ke_region(ke_p, ke_r, wmask_t)
        wake_r.append(wr)
        wake_e.append(we)
        inflow_u.append(float(np.mean(pu[inflow_band])))

    print("\n=== KE / vorticity (PRIMARY criterion) ===")
    print(f"  ({len(dns_t_idx)} val times)")
    print(f"  ke_pred/ke_ref = {np.mean(ke_ratios):.4f}   "
          f"KE rel-err = {np.mean(ke_relerrs):.4f}")
    print(f"  omega rel-err  = {np.mean(omega_relerrs):.4f}   "
          f"div L2 = {np.mean(div_l2s):.4e}")
    print(f"\n=== over-energy 定位（freestream x<{x_sensor_min:.3f} vs wake）===")
    if int(free_mask.sum()) == 0:
        print(f"  freestream: N/A（free_mask 為空：sensor 最上游 x={x_sensor_min:.3f} 達域邊界）")
    else:
        print(f"  freestream: ke_pred/ke_ref = {np.mean(free_r):.4f}   KE rel-err = {np.mean(free_e):.4f}")
    print(f"  wake:       ke_pred/ke_ref = {np.mean(wake_r):.4f}   KE rel-err = {np.mean(wake_e):.4f}")
    print(f"  inflow band (x<0.02) mean u_pred = {np.mean(inflow_u):.4f}  (u_inf={u_inf:.4f})")

    return TrainResult(
        state=state,
        eval_metrics=dict(
            dns_t_idx=[int(ti) for ti in dns_t_idx],
            ke_ratios=ke_ratios, ke_relerrs=ke_relerrs,
            omega_relerrs=omega_relerrs, div_l2s=div_l2s,
            free_r=free_r, free_e=free_e, wake_r=wake_r, wake_e=wake_e,
            inflow_u=inflow_u, x_sensor_min=x_sensor_min,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ledger replay（診斷/測試用；不呼叫 step_fn、不做 forward/gradient）
# ─────────────────────────────────────────────────────────────────────────────

#: `replay_schedule` 建 ctx 走的兩條路——見該函式 docstring。字串本身即對外可見的
#: 觀測值（印在 stdout、被測試斷言），故獨立命名而非行內字面量，避免兩處拼字漂移。
REPLAY_CTX_SOURCE_BUILD_CONTEXT = "build_context"
REPLAY_CTX_SOURCE_DATA_META_FALLBACK = "data_meta fallback"

#: `_data_meta` 的 geom 欄位可能是這兩種之一。判定靠**欄位集完全相等**
#: （`_geom_from_meta` 用 `set(raw) == set(cls._fields)`）——不是靠不相交：
#: 兩者其實共用 `Lx`/`Ly`/`u_inf`，各自獨有的是 `body_center`/`body_radius`
#: vs `base_center`/`body_radius_phys`/`amp`/`freq`/`phase`/`axis`。
#: 完整集合互不相同即足以唯一判定，故不必另外記一個 type tag；但**不可**
#: 改成「命中任一獨有欄位就認」——那對缺欄位的 meta 會猜錯型別而非拒絕。
_GEOM_TYPES = (CylinderGeometry, OscillatingGeometry)


class ReplayError(RuntimeError):
    """replay 前提不成立（缺尾列 / 缺 argv / 缺 data_meta / geom 欄位對不上）。"""


def _tail_row(golden_rows: list[dict]) -> dict:
    tails = [r for r in golden_rows if r.get("step") == -1]
    if len(tails) != 1:
        raise ReplayError(f"ledger 尾列（step=-1）應恰有一列，實得 {len(tails)}")
    return tails[0]


def _missing_recorded_dataset(
    e: FileNotFoundError, config: CylinderEffectiveConfig,
) -> bool:
    """True 若這個 FileNotFoundError 就是「錄製當時的 npz 不在本機」。

    Why 比對確切檔名而非關鍵字：資料檔不在本機是預期情境（改用尾列 `data_meta`
    重建最小 ctx）；但若哪天 `build_context` 因為別的檔案（config 內的其他路徑、
    未來新增的載入點）而丟 FileNotFoundError，那是 regression，必須讓它炸，不能被
    fallback 吞掉變成一個「看似通過」的 replay。cylinder 只有唯一一個資料入口
    （`np.load(eff["npz_path"])`，spec §6【不修 1】），所以這裡可以精確比對。
    """
    filename = getattr(e, "filename", None)
    npz_path = config.data.npz_path
    return filename is not None and npz_path is not None and str(filename) == str(npz_path)


def _geom_from_meta(meta: dict):
    """由 `data_meta["geom"]` 的欄位集還原 geometry NamedTuple。

    JSON 把 tuple 攤成 list，這裡轉回 tuple：`sample_wall_bc` 做
    `cx, cy = geom.body_center`、`sample_wall_bc_moving` 讀 `geom.axis`，
    型別不同雖然多半仍能跑，但 replay 的前提是「與錄製當時同一個物件」，
    不留下型別差異這種將來會咬人的分岔。
    """
    raw = meta["geom"]
    for cls in _GEOM_TYPES:
        if set(raw) == set(cls._fields):
            return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in raw.items()})
    raise ReplayError(
        f"data_meta 的 geom 欄位集 {sorted(raw)} 對不上任何已知幾何型別"
        f"（{'/'.join(c.__name__ for c in _GEOM_TYPES)}）—— 不得在此猜型別"
    )


def _ctx_from_data_meta(
    config: CylinderEffectiveConfig, tail: dict, missing: FileNotFoundError,
) -> TrainingContext:
    """用尾列 `data_meta` 重建 `_plan_step` 所需的最小 ctx。

    只填 `_plan_step` 會讀的欄位（見 `_data_meta` docstring），其餘留 None：
    replay 若走到別的欄位就該炸，不該拿到一個看起來合理的預設值。

    `n_sensor_q` 是唯一在此**重算**而非讀回的量：它 = `min(config 值, T*K)`，
    clamp 那一行在 `assembly.build_context`（`n_sensor_q = min(n_sensor_q, T * K)`）。
    config 部分一律以尾列 argv 為準、資料部分來自本表的 sensor_time/n_sensors，
    兩邊都不是猜的。若哪天 assembly 的 clamp 規則改了，這裡必須同步——
    `tests/test_cylinder_replay.py` 的 ctx-讀取哨兵會逼作者看到這個欄位。
    """
    meta = tail.get("data_meta")
    if meta is None:
        raise ReplayError(
            "ledger 尾列缺 'data_meta'，且錄製當時的資料檔不在本機"
            f"（{missing}）—— replay 不得在本端猜 data 衍生量。"
            "此 fixture 錄製於 data_meta 之前，需在 lab-server 以現版 "
            "train_cylinder.py 重錄。"
        ) from missing
    # sensor_time 錄的是 float32 值放大成的 float64，收回 float32 即逐位元還原
    st = np.asarray(meta["sensor_time"], dtype=np.float32)
    T = int(st.shape[0])
    K = int(meta["n_sensors"])
    blank = TrainingContext(**dict.fromkeys(TrainingContext._fields))
    return blank._replace(
        config=config,
        st0=float(st[0]),
        T_total=float(meta["t_total"]),   # 端點相減會差 1 ulp，故讀回不重算
        tm_end=float(st[-1]),
        K=K,
        T=T,
        # t_q_full = broadcast(st[:,None],(T,K)).reshape(T*K) = 每個元素重複 K 次；
        # dtype 必須維持 float32（NEP 50 弱純量提升，見 _data_meta docstring）
        t_q_full_np=np.repeat(st, K),
        n_sensor_q=min(config.run.n_sensor_query_requested, T * K),
        geom=_geom_from_meta(meta),
    )


def replay_plans(golden_rows: list[dict]) -> list[tuple[int, _StepPlan]]:
    """重播 host-side 決策序列，回傳逐步的 `(step, plan)`；不呼叫 step_fn。

    與 `replay_schedule` 的關係：後者只是把每個 plan 交給 `_ledger_fields` 攤平。
    分成兩層是因為 ledger 只存 digest，而診斷有時需要**原始陣列**——例如判定
    一個 digest 分岔到底是邏輯錯誤，還是跨平台浮點最後一位的差異
    （`tests/test_cylinder_replay.py` 的 1-ULP 鑑別即用本函式取陣列）。
    只有這一條重播路徑，兩層共用，不會漂移。

    行為契約與 `replay_schedule` docstring 相同（provenance 來源、兩條 ctx 路徑、
    fail-fast 條件），該文件是這兩個函式共同的規格。
    """
    tail = _tail_row(golden_rows)
    argv = tail.get("argv")
    if argv is None:
        raise ReplayError(
            "ledger 尾列缺 'argv' —— replay 不得在本端猜 config，請補尾列欄位並重錄 fixture"
        )
    resolved = resolve_inputs(list(argv))
    config = resolved.config
    try:
        ctx = build_context(config)
        ctx_source = REPLAY_CTX_SOURCE_BUILD_CONTEXT
    except FileNotFoundError as missing:
        if not _missing_recorded_dataset(missing, config):
            raise
        ctx = _ctx_from_data_meta(config, tail, missing)
        ctx_source = REPLAY_CTX_SOURCE_DATA_META_FALLBACK
    print(f"[replay_schedule] ctx source: {ctx_source}")

    rk, np_rng = _open_loop_streams(config.run.seed)

    plans: list[tuple[int, _StepPlan]] = []
    # 迴圈範圍與 run_loop 相同：`range(steps + 1)`，含第 0 步。
    for s in range(int(config.run.steps) + 1):
        plan = _plan_step(ctx, s, rk=rk, np_rng=np_rng)
        rk = plan.rk
        plans.append((s, plan))
    return plans


def replay_schedule(golden_rows: list[dict]) -> list[dict]:
    """重播 host-side 決策序列，不呼叫 step_fn。

    Why: bit-identical 最高風險是 RNG 消費時序（spec §5）。該序列完全由 host 端
    決定，不需要 forward/gradient 即可重現 —— 因此可在本機以 unit test 驗證，
    不違反「本機禁跑 training」。

    cylinder 的迴圈 stream（`PRNGKey(seed+1)` / `RandomState(seed+7)`）不由
    `initialize` 的 init stream 衍生，故此處不重播 `initialize`；它的兩條 init
    stream 由 `tests/test_cylinder_replay.py` 另外以行為級測試釘住。

    golden_rows 只用來取回錄製當時的 provenance（尾列），不參與比對：
      - `argv`      → **config 一律從這裡重解**，replay 端沒有第二個來源。
      - `data_meta` → `_plan_step` 的資料衍生輸入（時間軸、K/T、geometry）。
        錄製當時的 npz 在本機時走正式的 `build_context`；不在時改用 `data_meta`
        重建最小 ctx，兩條路都不猜任何值。

    尾列缺 `data_meta` 且 npz 又不在本機 → `ReplayError`。這裡沒有「猜一個合理值」
    的退路，因為猜對猜錯都會產生一份看起來完整的 replay。

    這兩條路的驗證強度**不同**：`build_context` 重新從 npz 算出 `st0`/`T_total`/
    `t_q_full_np`/`geom` 等衍生量，`data_meta` fallback 則是把錄製當時就算好的同一批
    數字原樣讀回來——換言之，走 fallback 時這些量本身的正確性沒被本次 replay 覆蓋到。
    兩條路都是合法、刻意保留的（本機常態就是走 fallback），只是強度不同；故此處把
    實際走了哪條路印出來（`REPLAY_CTX_SOURCE_*`），讓讀 stdout 的人能一眼判斷這次
    replay 覆蓋到哪一層，而不必臆測。
    """
    return [{"step": s, **_ledger_fields(plan)} for s, plan in replay_plans(golden_rows)]
