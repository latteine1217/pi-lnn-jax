"""Tests for `pi_lnn_jax.pipeline.kolmogorov.run` phase functions（非 replay 路徑）。

`test_pipeline_replay.py` 只覆蓋迴圈排程（`_plan_step` 消費的 `rng_collo` /
`rng_re` / `rng_crp`），其 module docstring 明言 `model.init` 本身不重播；
而且 `replay_schedule` 明文拒絕 resume 錄製的 ledger，也沒有任何 fixture
錄過 refine。本檔補的就是 replay 原理上碰不到的三個階段：

* `initialize()` —— spec §5.2 那組「易被順手清理破壞」的獨立 NumPy stream。
* `restore()`   —— resume sanity check 的 `rng_check` 中立性（spec §5.1）。
* `refine()`    —— 主流 split #6 的條件性與 spec §5.3 的獨立 refine stream。

不跑 training、不需要真實 model / 資料檔：這三個階段的 RNG 行為只取決於
它們**餵給下游的值**，故一律用只記錄呼叫參數的假 model / 假 refiner 頂掉
重量級依賴，換得可在本機 CPU 秒級跑完的行為級驗證。

每個測試都先做一次**鑑別力自檢**：斷言「規格正確形式」與「最可能被寫成的
壞形式」數值上確實不同。少了這一步，測試可能在某次重構後靜靜失去鑑別力，
卻永遠是綠的——那比沒有測試更危險。
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import pathlib
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.ckpt import TrainState
from pi_lnn_jax.pipeline.kolmogorov.assembly import ReBatch, TrainingContext
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.run import (
    RunJournal,
    TrainingState,
    _previous_sensor_loss,
    initialize,
    refine,
    restore,
)

_BASE_CONFIG = resolve_inputs([
    "--config", str(Path(__file__).resolve().parent.parent / "configs/_ledger_single_re.toml"),
]).config


class _RecordingModel:
    """假 model：只記錄 `model.init` 收到的 `init_xy` / `init_t`，不做任何運算。

    回傳空 dict 當 params——`initialize()` 只對它做
    `jax.tree_util.tree_leaves` 求 size 總和，空 pytree 合法且總和為 0。
    """

    def __init__(self):
        self.captured: dict[str, np.ndarray] = {}

    def init(self, rng_init, sensor_vals, sensor_pos, re_norm, sensor_time, init_xy, init_t):
        del rng_init, sensor_vals, sensor_pos, re_norm, sensor_time  # 本測試不關心
        self.captured["init_xy"] = np.asarray(init_xy)
        self.captured["init_t"] = np.asarray(init_t)
        return {}


class _NoopOptimizer:
    """假 optimizer：`initialize()` 只呼叫一次 `.init(params)` 拿 opt_state。"""

    def init(self, params):
        del params
        return None


def _stub_ctx(*, seed: int, T_total: float, model: _RecordingModel) -> TrainingContext:
    """組出 `initialize()` 執行所需的最小 `TrainingContext`。

    weighting_method="off" 且 use_gradnorm=False，跳過 GradNorm/LRA
    ref-subtree resolve 分支（那段需要真實 params 樹，與本測試無關）。
    """
    re_batch = ReBatch(
        sensor_vals=np.zeros((2, 2, 1), dtype=np.float32),
        sensor_pos=np.zeros((2, 2), dtype=np.float32),
        sensor_time=np.zeros((2,), dtype=np.float32),
        re_norm=np.asarray(1.0, dtype=np.float32),
        nu=np.asarray(1.0, dtype=np.float32),
        u_mean=np.asarray(0.0, dtype=np.float32),
        u_std=np.asarray(1.0, dtype=np.float32),
        v_mean=np.asarray(0.0, dtype=np.float32),
        v_std=np.asarray(1.0, dtype=np.float32),
        p_mean=np.asarray(0.0, dtype=np.float32),
        p_std=np.asarray(1.0, dtype=np.float32),
    )
    config = replace(
        _BASE_CONFIG,
        run=replace(_BASE_CONFIG.run, seed=seed),
        loss=replace(
            _BASE_CONFIG.loss,
            use_gradnorm=False,
            gradnorm_init_weights=None,
            al_rho=1.0,
            al_lambda_clip=10.0,
        ),
    )
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config,
        model=model,
        model_name="stub",
        re_batches=[re_batch],
        T_total=T_total,
        tx=_NoopOptimizer(),
    )


def test_initialize_init_t_is_not_a_continuation_of_init_xy_stream():
    """spec §5.2：`init_xy`/`init_t` 各自來自獨立建構的 `np.random.RandomState`。

    `initialize()`（`pi_lnn_jax/pipeline/kolmogorov/run.py`）目前是：

        init_xy = np.random.RandomState(eff["seed"]).uniform(0, 1, (8, 2))
        init_t  = np.random.RandomState(eff["seed"]).uniform(0, T_total, (8,))

    兩次**各自**建構新的 `RandomState`，同一個 seed。`init_t` 因此不是
    `init_xy` 的續抽，兩者都取自各自生成器的第一批亂數。若被「順手」改成
    共用一個生成器連續抽兩次（最自然的清理手法），`init_t` 的數值會悄悄
    改變，進而改變 `model.init` 的輸入、再進而改變下游每一個數字——且
    330 個既有測試全部維持綠燈，因為沒有任何測試盯著這條 invariant。

    選 behavioral 而非 AST 計數：直接呼叫真正的 `initialize()`（用假
    model/optimizer 頂掉重量級依賴），比對它實際餵給 `model.init` 的
    `init_t` 數值，而非只驗證原始碼結構。這樣「合併成單一生成器」的
    修改會在數值層面被抓到，不必依賴 AST pattern 能否涵蓋所有改法。

    `seed` / `T_total` 由本測試直接建構進 `ctx`，不經任何 config 解析——
    因此不是「猜一個配得剛好的預設值」，而是本測試自己控制、自己核對
    的輸入。
    """
    seed = 20260729
    T_total = 3.7  # 任意值：直接由本測試建構的 ctx.T_total 提供，非猜測 config 預設

    # 規格正確形式（spec §5.2）：兩個「各自建構」的 RandomState
    expected_init_xy = np.random.RandomState(seed).uniform(0, 1, (8, 2)).astype(np.float32)
    expected_init_t = np.random.RandomState(seed).uniform(0, T_total, (8,)).astype(np.float32)

    # 「順手合併」後的形式：單一 RandomState 連續抽樣兩次
    merged_rng = np.random.RandomState(seed)
    merged_init_xy = merged_rng.uniform(0, 1, (8, 2)).astype(np.float32)
    merged_init_t = merged_rng.uniform(0, T_total, (8,)).astype(np.float32)

    # 鑑別力自檢：兩形式的 init_t 必須不同，否則本測試測不出任何東西
    assert not np.allclose(expected_init_t, merged_init_t), (
        "獨立雙生成器與合併單生成器的 init_t 竟相同——測試本身已失去鑑別力"
    )
    # init_xy 兩形式必然相同（皆是各自新生成器的第一批抽樣），僅作為理解性斷言
    assert np.allclose(expected_init_xy, merged_init_xy)

    # 呼叫真正的 initialize()，證明它產出「正確形式」而非「合併形式」
    model = _RecordingModel()
    ctx = _stub_ctx(seed=seed, T_total=T_total, model=model)
    initialize(ctx, RunJournal())

    assert np.allclose(model.captured["init_xy"], expected_init_xy)
    assert np.allclose(model.captured["init_t"], expected_init_t), (
        "initialize() 的 init_t 與雙生成器規格值不符——RandomState 疑似被合併"
    )
    assert not np.allclose(model.captured["init_t"], merged_init_t), (
        "initialize() 的 init_t 與合併單生成器形式相同——兩個 RandomState 被合併了"
    )
    print("✓ initialize_init_t_is_not_a_continuation_of_init_xy_stream")


# ─────────────────────────────────────────────────────────────────────────────
# restore / refine 共用的最小替身
# ─────────────────────────────────────────────────────────────────────────────

def _key_eq(a, b) -> bool:
    """PRNGKey 逐 bit 比較。

    刻意不是 `allclose`：RNG key 只有「同一把」與「不同把」兩種狀態，
    近似相等在這裡沒有意義，而且會讓「被 split 推進過一次」這種差異
    有機會被容忍掉。
    """
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _re_batch(*, T: int, K: int, t0: float, t1: float) -> ReBatch:
    """restore / refine 測試共用的最小 ReBatch。

    `sensor_time` 必須嚴格遞增：兩個階段都拿 `sensor_time[0]` /
    `sensor_time[-1]` 當 collocation 時間座標的 minval/maxval，退化成同值
    會讓 `ct` 恆為常數，「時間座標是從哪把 key 抽的」也就測不出來了。
    """
    return ReBatch(
        sensor_vals=jnp.zeros((T, K, 2), dtype=jnp.float32),
        sensor_pos=jnp.zeros((K, 2), dtype=jnp.float32),
        sensor_time=jnp.linspace(t0, t1, T, dtype=jnp.float32),
        re_norm=jnp.asarray(1.0, dtype=jnp.float32),
        nu=jnp.asarray(1.0, dtype=jnp.float32),
        u_mean=jnp.asarray(0.0, dtype=jnp.float32),
        u_std=jnp.asarray(1.0, dtype=jnp.float32),
        v_mean=jnp.asarray(0.0, dtype=jnp.float32),
        v_std=jnp.asarray(1.0, dtype=jnp.float32),
        p_mean=jnp.asarray(0.0, dtype=jnp.float32),
        p_std=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _state(rng) -> TrainingState:
    """帶著指定主 stream 的最小 `TrainingState`（其餘欄位皆 None）。"""
    return TrainingState(**dict.fromkeys(TrainingState._fields))._replace(
        params={}, step=0, rng=rng, task_weights=jnp.ones((3,), dtype=jnp.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: restore —— resume sanity check 的主流中立性
# ─────────────────────────────────────────────────────────────────────────────

class _RecordingLossFn:
    """假 loss_fn：只記錄被呼叫次數與 collocation 座標，回傳固定值。

    真 loss_fn 需要 model / physics / 全套 norm stats；resume sanity check
    只把它的輸出印成一行診斷，數值本身不影響任何 RNG 決策，故可整個頂掉。
    """

    def __init__(self):
        self.calls = 0
        self.cx = None

    def __call__(self, params, cx, cy, ct, task_weights, al_lambda, w_phys_now,
                 re_batch, poisson_w, aux, re_batch_crp):
        del (params, cy, ct, task_weights, al_lambda, w_phys_now,
             re_batch, poisson_w, aux, re_batch_crp)
        self.calls += 1
        self.cx = np.asarray(cx)
        zeros = tuple(jnp.float32(0.0) for _ in range(6))
        return jnp.float32(1.0), zeros


class _StubCkptMgr:
    """假 CheckpointManager：`restore()` 一律回同一個預先組好的 TrainState。"""

    def __init__(self, restored: TrainState):
        self._restored = restored
        self.restore_calls = 0

    def latest_step(self) -> int:
        return int(self._restored.step)

    def restore(self, step, reference_state=None):
        del step, reference_state
        self.restore_calls += 1
        return self._restored


def _restore_ctx(*, ckpt_rng, ckpt_step: int, loss_fn,
                 artifacts_dir=None) -> TrainingContext:
    """組出 `restore()` 走完 resume 分支所需的最小 `TrainingContext`。

    weighting_method="off" 且 use_gradnorm=False，跳過 task_weights 重算；
    use_continuous_re_physics=False，跳過 CRP proxy batch。兩者都與本測試
    要守的 RNG invariant 無關。
    """
    restored = TrainState(
        params={}, opt_state=None, step=jnp.int32(ckpt_step), rng_key=ckpt_rng,
        gradnorm_state=None, lra_state=None,
        # rho / lambda_clip 是 ALState 的 pytree leaf，restore 會用 ckpt 的值蓋掉
        # config；`assert_al_statics_match` 因此要讀它們（見
        # tests/test_silent_failure_guards.py）。stub 必須帶上與 _BASE_CONFIG 相同的值，
        # 否則測到的會是那個 guard 而不是本檔要守的 RNG invariant。
        al_state=SimpleNamespace(lambda_=0.0,
                                 rho=_BASE_CONFIG.loss.al_rho,
                                 lambda_clip=_BASE_CONFIG.loss.al_lambda_clip),
    )
    config = replace(
        _BASE_CONFIG,
        run=replace(_BASE_CONFIG.run, resume_step=str(ckpt_step)),
        loss=replace(
            _BASE_CONFIG.loss,
            use_gradnorm=False,
            use_continuous_re_physics=False,
            physics_weight=1.0,
            physics_warmup_steps=0,
            physics_ramp_steps=0,
        ),
        curriculum=replace(
            _BASE_CONFIG.curriculum,
            n_collo_start=4,
            n_collo_end=4,
            n_collo_ramp=0,
        ),
    )
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config,
        re_batches=[_re_batch(T=2, K=2, t0=0.0, t1=1.0)],
        loss_fn=loss_fn,
        ckpt_mgr=_StubCkptMgr(restored),
        ckpt_dir="<stub-ckpt-dir>",
        artifacts_dir=artifacts_dir,
    )


def test_restore_does_not_advance_the_main_rng_stream():
    """spec §5.1：resume sanity check 的 split **不得**推進主 stream。

    `restore()`（`pi_lnn_jax/pipeline/kolmogorov/run.py`）目前是：

        rng_check, sub_check = jax.random.split(rng)

    賦值給 `rng_check` 而非 `rng`。看起來像筆誤（左值用完就丟），實際上是
    規格：sanity check 只是一次診斷用的 forward，不該讓 resume 後的主流與
    「同一個 ckpt 但沒跑 sanity check」的主流分岔。有人把它「修」成
    `rng, sub = split(rng)`，resume 後每一次後續抽樣都會整條位移——而
    resumed run 的 golden fixture 不存在（`replay_schedule` 明文拒絕 resume
    錄製的 ledger），所以既有測試一個都不會紅。

    斷言直接盯回傳的 `state.rng`：必須**就是** ckpt 裡那把 key，而不是它的
    任何衍生物。所有輸入（pre_rng / ckpt_rng / step）由本測試自行建構，
    不讀任何 config 預設值。
    """
    pre_rng = jax.random.PRNGKey(11)     # initialize() 交棒進來的主流
    ckpt_rng = jax.random.PRNGKey(9001)  # ckpt 內記錄的主流（restore 應原樣接手）
    advanced, _sub = jax.random.split(ckpt_rng)  # 「順手 tidy」後會得到的值

    # 鑑別力自檢：正確形式與壞形式必須不同，且 ckpt 那把不能剛好等於交棒進來那把
    assert not _key_eq(ckpt_rng, advanced), (
        "split 後的 key 竟等於原 key——本測試無法分辨主流有沒有被推進"
    )
    assert not _key_eq(ckpt_rng, pre_rng), (
        "ckpt key 與 pre-resume key 相同——無法分辨 restore 有沒有真的接手 ckpt 的 key"
    )

    loss_fn = _RecordingLossFn()
    ctx = _restore_ctx(ckpt_rng=ckpt_rng, ckpt_step=7, loss_fn=loss_fn)
    out = restore(ctx, _state(pre_rng))

    # 沒跑到 sanity check 就等於沒經過那個 split，測試會變成空轉
    assert loss_fn.calls == 1, "resume sanity check 沒被執行，本測試沒有覆蓋到那個 split"
    assert ctx.ckpt_mgr.restore_calls == 1

    assert _key_eq(out.rng, ckpt_rng), (
        "restore() 回傳的主 rng 不是 ckpt 的 rng_key——sanity check 的 split 疑似被改成推進主流"
    )
    assert not _key_eq(out.rng, advanced), (
        "restore() 回傳的主 rng 等於 split(ckpt_rng)[0]——主流被 sanity check 推進了一次"
    )
    assert int(out.step) == 7
    print("✓ restore_does_not_advance_the_main_rng_stream")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: refine —— split #6 的條件性與獨立 refine stream
# ─────────────────────────────────────────────────────────────────────────────

def _refine_ctx(*, args_kw: dict, eff_kw: dict, re_batch: ReBatch,
                model=None, ns_fn=None) -> TrainingContext:
    """組出 `refine()` 各分支所需的最小 `TrainingContext`。"""
    T, K = int(re_batch.sensor_vals.shape[0]), int(re_batch.sensor_vals.shape[1])
    dataset = {
        "sensor_vals": np.zeros((T, K, 2), dtype=np.float32),
        "re_value": 100.0,
        "re_norm": 0.5,
        "norm_stats": {},
    }
    run_map = {"arch": "arch", "seed": "seed"}
    loss_map = {"w_data": "data_weight", "w_phys": "physics_weight"}
    curriculum_map = {"n_collo_end": "n_collo_end"}
    refinement_map = {
        "refine_optimizer": "optimizer", "refine_steps": "steps",
        "refine_rtol": "rtol", "refine_atol": "atol",
        "n_collo_refine": "n_collo", "lbfgs_t_subsample": "lbfgs_t_subsample",
        "lbfgs_max_iter": "lbfgs_max_iter", "lbfgs_history": "lbfgs_history",
        "gn_lr": "gn_lr", "gn_cg_iters": "gn_cg_iters",
        "gn_damping": "gn_damping", "gn_log_every": "gn_log_every",
    }
    merged = {**eff_kw, **args_kw}
    config = replace(
        _BASE_CONFIG,
        run=replace(_BASE_CONFIG.run, **{
            target: merged[source] for source, target in run_map.items() if source in merged
        }),
        loss=replace(_BASE_CONFIG.loss, **{
            target: merged[source] for source, target in loss_map.items() if source in merged
        }),
        curriculum=replace(_BASE_CONFIG.curriculum, **{
            target: merged[source]
            for source, target in curriculum_map.items() if source in merged
        }),
        refinement=replace(_BASE_CONFIG.refinement, **{
            target: merged[source]
            for source, target in refinement_map.items() if source in merged
        }),
    )
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config,
        model=model,
        ns_fn=ns_fn,
        datasets=[dataset],
        re_batches=[re_batch],
    )


def test_refine_none_consumes_no_rng():
    """spec §5.1 #6：refine 關閉時主 stream 一個 key 都不得消耗。

    `refine()` 在 `refine_optimizer == "none"` 時直接 `return state`，
    固定 collocation 的 `rng, sub = jax.random.split(rng)` 在 early return
    **之後**。這個位置是承重的：整個 Kolmogorov paper campaign 都跑
    refine=none，若那個 split 被搬到 early return 之前（例如有人為了「把
    collocation 準備集中在一處」而上移），所有既有 run 的 `finalize` 落盤
    rng 會整條位移，而沒有任何 fixture 錄過 refine，既有測試不會紅。

    斷言 `refine()` 回傳的 `state.rng` 與輸入 bit-identical。
    """
    rng = jax.random.PRNGKey(31337)
    advanced, _sub = jax.random.split(rng)  # split 被上移到 early return 之前的結果

    # 鑑別力自檢
    assert not _key_eq(rng, advanced), (
        "split 後的 key 竟等於原 key——本測試無法分辨主流有沒有被消耗"
    )

    ctx = _refine_ctx(
        args_kw={"refine_optimizer": "none"},
        eff_kw={},
        re_batch=_re_batch(T=2, K=2, t0=0.0, t1=1.0),
    )
    out = refine(ctx, _state(rng))

    assert _key_eq(out.rng, rng), (
        "refine_optimizer='none' 卻改變了主 rng——split #6 的條件性被破壞"
    )
    assert not _key_eq(out.rng, advanced), (
        "refine_optimizer='none' 的回傳 rng 等於 split(rng)[0]——主流被無條件推進了一次"
    )
    print("✓ refine_none_consumes_no_rng")


def test_refine_lm_splits_the_main_stream_exactly_once(monkeypatch):
    """spec §5.1 #6：refine 啟用時主 stream 恰好 split 一次，且固定 collocation
    取自那次 split 的 `sub`。

    走 LM 分支是因為它最輕：把 `make_pinn_residual_vector_fn` 與 `lm_refine`
    換成替身後，真正的 model / 物理殘差完全不被觸及，剩下的就只有這條
    RNG 路徑。若 refine 內部多 split 一次（或把 `cx/cy/ct` 改成直接從主流
    抽），落盤 rng 與 refine 用的 collocation 都會變，但 refine 路徑目前
    零 fixture 覆蓋，不會有任何測試出聲。
    """
    import pi_lnn_jax.refiners as refiners

    rng = jax.random.PRNGKey(555)
    n_collo_refine = 5
    t0, t1 = 0.25, 1.75

    # 規格形式：主流 split 一次，sub 再 split 成 3 把 collocation key
    once, sub = jax.random.split(rng)
    twice, _ = jax.random.split(once)
    keys = jax.random.split(sub, 3)
    exp_cx = jax.random.uniform(keys[0], (n_collo_refine,), minval=0.0, maxval=1.0)
    exp_cy = jax.random.uniform(keys[1], (n_collo_refine,), minval=0.0, maxval=1.0)
    exp_ct = jax.random.uniform(keys[2], (n_collo_refine,), minval=t0, maxval=t1)

    # 鑑別力自檢 1：split 0/1/2 次三者互異，「恰好一次」才有意義
    assert not _key_eq(rng, once) and not _key_eq(once, twice)
    # 鑑別力自檢 2：若 collocation 改由主流本身（而非 sub）派生，數值必不同
    wrong_keys = jax.random.split(rng, 3)
    wrong_cx = jax.random.uniform(wrong_keys[0], (n_collo_refine,), minval=0.0, maxval=1.0)
    assert not np.allclose(np.asarray(exp_cx), np.asarray(wrong_cx)), (
        "從 sub 與從主流派生的 cx 竟相同——本測試無法分辨 collocation 的來源"
    )

    captured: dict = {}

    def _fake_make_res(model, sensor_vals, sensor_pos, sensor_time, **kw):
        del model, sensor_vals, sensor_pos, sensor_time
        captured.update(kw)
        return object()

    def _fake_lm_refine(params, res_fn, **kw):
        del res_fn, kw
        captured["lm_called"] = captured.get("lm_called", 0) + 1
        return params, None

    monkeypatch.setattr(refiners, "make_pinn_residual_vector_fn", _fake_make_res)
    monkeypatch.setattr(refiners, "lm_refine", _fake_lm_refine)

    ctx = _refine_ctx(
        args_kw={
            "refine_optimizer": "lm", "arch": "liquid",
            "n_collo_refine": n_collo_refine, "refine_steps": 3,
            "refine_rtol": 1e-6, "refine_atol": 1e-6,
        },
        eff_kw={"n_collo_end": 9, "w_data": 1.0, "w_phys": 1.0},
        re_batch=_re_batch(T=4, K=3, t0=t0, t1=t1),
    )
    out = refine(ctx, _state(rng))

    assert captured.get("lm_called") == 1, "LM 分支沒跑到，本測試沒有覆蓋 split #6"
    assert np.allclose(np.asarray(captured["cx"]), np.asarray(exp_cx))
    assert np.allclose(np.asarray(captured["cy"]), np.asarray(exp_cy))
    assert np.allclose(np.asarray(captured["ct"]), np.asarray(exp_ct)), (
        "refine 固定 collocation 與 split #6 的 sub 派生值不符——RNG 來源被改了"
    )
    assert _key_eq(out.rng, once), (
        "refine 回傳的主 rng 不等於 split 一次的結果——主流的 split 次數被改了"
    )
    assert not _key_eq(out.rng, rng) and not _key_eq(out.rng, twice)
    print("✓ refine_lm_splits_the_main_stream_exactly_once")


class _StubLiquidOperator:
    """假 LiquidOperator：只滿足 `mb_lbfgs_loss` 的 `model.apply(..., method=)` 分派。

    `refine()` 的 LBFGS 分支在每個 outer 迴圈末尾會探測一次 loss 並印出來，
    因此 `mb_lbfgs_loss` 一定會被實際執行；但它的**數值**不影響任何 RNG
    決策，故回傳固定形狀的零張量即可。
    """

    def encode(self):  # noqa: D102 - 僅作為 apply 的 method dispatch tag
        raise NotImplementedError

    def decode_query(self):  # noqa: D102
        raise NotImplementedError

    def get_forcing(self):  # noqa: D102
        raise NotImplementedError

    def apply(self, params, *rest, method=None):
        del params
        cls = type(self)
        if method is cls.get_forcing:
            return jnp.float32(1.0), jnp.float32(4.0)
        if method is cls.encode:
            return jnp.zeros((1, 1), dtype=jnp.float32)
        return jnp.zeros((rest[0].shape[0], 2), dtype=jnp.float32)


def test_refine_lbfgs_samples_from_the_independent_seed_plus_7919_stream(monkeypatch):
    """spec §5.3：LBFGS refine 的 outer-loop 取樣走獨立 stream
    `PRNGKey(seed + 7919)`，與主 stream 無關。

    這條獨立 stream 有兩種被「順手清理」的方式，兩種都會靜靜改變 refine
    的取樣：把 `+ 7919` 這個 magic number 拿掉（變成與 `initialize` 同源），
    或者乾脆重用主 stream 的 `rng` / `sub`。兩者在功能上都「跑得起來」，
    差別只在數值——而 refine 路徑零 fixture 覆蓋。

    同時順帶守住：即使 LBFGS 分支自己有一條獨立 stream，主 stream 仍然只
    因 split #6 推進一次。
    """
    import pi_lnn_jax.refiners as refiners

    seed = 20260730
    rng = jax.random.PRNGKey(seed)  # 主流；與 refine stream 同 seed，必須互不相干
    n_collo_refine = 5
    T, K, T_sub = 4, 3, 2
    t0, t1 = 0.25, 1.75

    # 規格形式：獨立 stream 由 seed + 7919 起手
    rng_refine = jax.random.PRNGKey(seed + 7919)
    rng_refine, sub_o = jax.random.split(rng_refine)
    kk = jax.random.split(sub_o, 4)
    exp_t_idx = jax.random.choice(kk[0], T, (T_sub,), replace=False)
    exp_cx = jax.random.uniform(kk[1], (n_collo_refine,), minval=0.0, maxval=1.0)

    # 壞形式 A：忘了 +7919，直接用 seed（與主流同源）
    bad_a_sub = jax.random.split(jax.random.PRNGKey(seed))[1]
    bad_a_cx = jax.random.uniform(
        jax.random.split(bad_a_sub, 4)[1], (n_collo_refine,), minval=0.0, maxval=1.0
    )
    # 壞形式 B：重用主流 split #6 之後的 rng
    bad_b_sub = jax.random.split(jax.random.split(rng)[0])[1]
    bad_b_cx = jax.random.uniform(
        jax.random.split(bad_b_sub, 4)[1], (n_collo_refine,), minval=0.0, maxval=1.0
    )

    # 鑑別力自檢：規格形式必須與兩種壞形式都不同
    assert not np.allclose(np.asarray(exp_cx), np.asarray(bad_a_cx)), (
        "seed 與 seed+7919 派生的 cx 竟相同——本測試分辨不出 7919 有沒有被拿掉"
    )
    assert not np.allclose(np.asarray(exp_cx), np.asarray(bad_b_cx)), (
        "獨立 stream 與重用主流派生的 cx 竟相同——本測試分辨不出 stream 有沒有被合併"
    )

    captured: dict = {}

    def _fake_lbfgs_refine(params, loss_fn, *, args, **kw):
        del loss_fn, kw
        captured.setdefault("args", []).append(args)
        return params, None

    monkeypatch.setattr(refiners, "lbfgs_refine", _fake_lbfgs_refine)

    ctx = _refine_ctx(
        args_kw={
            "refine_optimizer": "lbfgs", "refine_steps": 1,
            "lbfgs_t_subsample": T_sub, "lbfgs_max_iter": 3, "lbfgs_history": 5,
            "n_collo_refine": n_collo_refine,
            "refine_rtol": 1e-6, "refine_atol": 1e-6,
        },
        eff_kw={"seed": seed, "n_collo_end": 9, "w_data": 1.0, "w_phys": 1.0},
        re_batch=_re_batch(T=T, K=K, t0=t0, t1=t1),
        model=_StubLiquidOperator(),
        ns_fn=lambda *a, **kw: (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
    )
    out = refine(ctx, _state(rng))

    assert len(captured.get("args", [])) == 1, "LBFGS outer loop 沒跑到，本測試等於空轉"
    got_t_idx, got_cx, _got_cy, _got_ct = captured["args"][0]

    assert np.array_equal(np.asarray(got_t_idx), np.asarray(exp_t_idx)), (
        "LBFGS outer-loop 的 sensor mini-batch index 與 PRNGKey(seed + 7919) 派生值不符"
    )
    assert np.allclose(np.asarray(got_cx), np.asarray(exp_cx)), (
        "LBFGS outer-loop collocation 與 PRNGKey(seed + 7919) 派生值不符——獨立 stream 被改了"
    )
    assert not np.allclose(np.asarray(got_cx), np.asarray(bad_a_cx))
    assert not np.allclose(np.asarray(got_cx), np.asarray(bad_b_cx))
    # 獨立 stream 不得影響主流：主流仍只因 split #6 推進一次
    assert _key_eq(out.rng, jax.random.split(rng)[0])
    print("✓ refine_lbfgs_samples_from_the_independent_seed_plus_7919_stream")


# ─────────────────────────────────────────────────────────────────────────────
# `trig_weighting` 的無條件求值所依賴的 schema 保證（spec §8.3 第 7 項）
# ─────────────────────────────────────────────────────────────────────────────

def test_gradnorm_freq_is_guaranteed_positive_by_schema():
    """`_plan_step` 無條件求 `s % eff["gradnorm_freq"]`，除以零的防線只剩 schema。

    重構前（`28b007a:1325,1335`）這個模數藏在短路後面：
    `eff["use_gradnorm"] and compute_grad_norms is not None and s % eff["gradnorm_freq"] == 0`。
    現在四個觸發條件提前一次算完（讓 ledger 與控制流共用同一個值，避免兩份條件
    漂移），代價是模數變成無條件求值——`gradnorm_freq = 0` 會從「安靜略過」
    變成 ZeroDivisionError。

    不恢復短路：那會改變 ledger 記錄的語意並作廢六份 fixture。改為把這條依賴
    釘住——schema 若哪天放寬成 `_non_negative`，這裡會紅，提醒同時處理 run.py。

    `eff["gradnorm_freq"] = tk.get("gradnorm_freq", 1000)`，而 `tk` 是
    `cfg["train_kwargs"]`（已通過 schema），故兩條路徑都不可能給出 0：
    TOML 有寫 → 被驗證；沒寫 → 走這裡一併釘住的正值 fallback。
    """
    import pytest

    from pi_lnn_jax.config import TRAIN_SCHEMA, _validate_one

    assert _validate_one("gradnorm_freq", 1, TRAIN_SCHEMA) == 1
    with pytest.raises(ValueError):
        _validate_one("gradnorm_freq", 0, TRAIN_SCHEMA)
    with pytest.raises(ValueError):
        _validate_one("gradnorm_freq", -1, TRAIN_SCHEMA)

    # schema 預設與 case policy 的相容 fallback 是**兩個**獨立的正值來源，都要 > 0。
    assert TRAIN_SCHEMA["gradnorm_freq"][1] > 0
    from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy
    assert KolmogorovPolicy._defaults["loss.gradnorm_freq"] > 0
    print("✓ gradnorm_freq_is_guaranteed_positive_by_schema")


def test_al_update_freq_is_guaranteed_positive_by_schema():
    """`trig_al` 的模數被 `eff["use_al"]` 短路擋著，但只擋一半：
    `use_al` 為真而 `al_update_freq` 為 0 仍會除以零。同樣只剩 schema 這道防線。"""
    import pytest

    from pi_lnn_jax.config import TRAIN_SCHEMA, _validate_one

    with pytest.raises(ValueError):
        _validate_one("al_update_freq", 0, TRAIN_SCHEMA)
    assert TRAIN_SCHEMA["al_update_freq"][1] > 0
    print("✓ al_update_freq_is_guaranteed_positive_by_schema")


# ─────────────────────────────────────────────────────────────────────────────
# Resume continuity guard 真的接上了嗎
# ─────────────────────────────────────────────────────────────────────────────

def test_previous_sensor_loss_reads_the_last_completed_run():
    """守衛的輸入來源。回不出值，整個比較就退化成永遠跳過。"""
    with tempfile.TemporaryDirectory() as td:
        art = Path(td)
        (art / "summary.json").write_text(json.dumps(
            {"last_train_metrics": {"step": 900, "sensor": 1.25e-3, "total": 7.0}}))
        assert _previous_sensor_loss(art) == pytest.approx(1.25e-3)


@pytest.mark.parametrize("payload,why", [
    (None, "首次訓練：沒有前一輪"),
    ("{ not json", "壞掉的 summary 不該擋下整輪 resume"),
    (json.dumps({}), "跑到一半被砍：summary 在但沒有 last_train_metrics"),
    (json.dumps({"last_train_metrics": {"sensor": 0.0}}), "0 不是有效的相對誤差基準"),
])
def test_previous_sensor_loss_returns_none_without_raising(payload, why):
    with tempfile.TemporaryDirectory() as td:
        art = Path(td)
        if payload is not None:
            (art / "summary.json").write_text(payload)
        assert _previous_sensor_loss(art) is None, why


_UNVERIFIED_LINE = "continuity 未驗證"


def _run_restore(artifacts_dir) -> str:
    """跑一次 restore()，回傳它印出的東西。"""
    buf = io.StringIO()
    ctx = _restore_ctx(ckpt_rng=jax.random.PRNGKey(3), ckpt_step=7,
                       loss_fn=_RecordingLossFn(), artifacts_dir=artifacts_dir)
    with contextlib.redirect_stdout(buf):
        restore(ctx, _state(jax.random.PRNGKey(11)))
    return buf.getvalue()


def test_restore_actually_feeds_the_guard_not_a_hardcoded_none():
    """承重的一條：這個守衛先前整段沒在執行，因為呼叫端寫死 `last_logged_loss=None`。

    只測 helper 不夠——helper 正確而呼叫端沒接上，正是原本的狀態。這裡走完整條
    `restore()`，用它印出的東西判定守衛到底有沒有拿到值。
    """
    with tempfile.TemporaryDirectory() as td:
        art = Path(td)
        (art / "summary.json").write_text(json.dumps(
            {"last_train_metrics": {"sensor": 4.2e-3}}))
        out_with = _run_restore(art)

    # 鑑別力自檢：沒有前一輪時必須印出「未驗證」，否則下面那條斷言分辨不出任何事。
    with tempfile.TemporaryDirectory() as td2:
        out_without = _run_restore(Path(td2))
    assert _UNVERIFIED_LINE in out_without, (
        "拿不到前一輪時未印出「未驗證」——本測試無法分辨守衛有沒有被餵值")

    assert _UNVERIFIED_LINE not in out_with, (
        "summary.json 明明有 last_train_metrics.sensor，守衛卻報未驗證"
        "——呼叫端疑似又寫死了 last_logged_loss=None")
