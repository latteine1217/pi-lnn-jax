"""Tests for pi_lnn_jax.ckpt — orbax-backed train-state save/restore.

What: 驗證 CheckpointManager 的 round-trip 完整性、max_to_keep 政策、
       latest_step 行為，以及 GradNormState / ALState 的 bit-equal 保存。

Why: pi-lnn EXP-082 silent state corruption 屬於「ckpt 看似 load 成功但
     數值漂移」類型 bug；本測試套件強制每個欄位（含 optimizer state 內
     NamedTuple 子樹、靜態 metadata）逐 leaf 比對，回歸時能立即 catch。
"""
from __future__ import annotations

import sys
import tempfile
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from _paths import REPO_ROOT

# 確保可以從 repo root 匯入 pi_lnn_jax（PYTHONPATH=. 已在 CLI 設定，但保險起見）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_lnn_jax.ckpt import (  # noqa: E402
    CheckpointManager,
    TrainState,
    verify_params_tree,
)
from pi_lnn_jax.losses import ALState, GradNormState, al_init, gradnorm_init  # noqa: E402


# 抑制 orbax 對非 sharded array 的 UserWarning：POC 單機 case 預期會出現
warnings.filterwarnings("ignore", category=UserWarning, module="orbax.*")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (純函式風格，不依賴 pytest 框架避免新增 dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_train_state(
    step: int = 42,
    seed: int = 7,
    with_gradnorm: bool = False,
    with_al: bool = False,
) -> TrainState:
    """構造一個小型但結構完整的 TrainState（params + opt_state + step + rng）。

    params:    dict 模擬 Flax {'params': {...}} 結構
    opt_state: 模擬 optax NamedTuple-of-arrays 結構（用 dict 簡化）
    """
    params = {
        "params": {
            "encoder": {
                "kernel": jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
                "bias": jnp.zeros((4,), dtype=jnp.float32),
            },
            "decoder": {
                "kernel": jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32).reshape(4, 2),
            },
        }
    }
    opt_state = {
        "mu": jax.tree_util.tree_map(jnp.zeros_like, params),
        "nu": jax.tree_util.tree_map(jnp.ones_like, params),
        "count": jnp.int32(step),  # 模擬 ScheduleFree internal _global_step
    }
    gradnorm_state = (
        gradnorm_init([1.0, 0.01, 0.01]) if with_gradnorm else None
    )
    al_state = al_init(init_lambda=0.5, rho=1.0, lambda_clip=10.0) if with_al else None

    return TrainState(
        params=params,
        opt_state=opt_state,
        step=jnp.int32(step),
        rng_key=jax.random.PRNGKey(seed),
        gradnorm_state=gradnorm_state,
        al_state=al_state,
    )


def _assert_pytree_equal(name: str, tree_a, tree_b) -> None:
    """逐 leaf 比對兩個 pytree；array 用 bit-equal，scalar/str 用 ==。

    Raises AssertionError 帶上 path 資訊，方便 debug。
    """
    paths_a = jax.tree_util.tree_leaves_with_path(tree_a)
    paths_b = jax.tree_util.tree_leaves_with_path(tree_b)
    assert len(paths_a) == len(paths_b), (
        f"{name}: leaf 數量不同 ({len(paths_a)} vs {len(paths_b)})"
    )
    for (path_a, val_a), (path_b, val_b) in zip(paths_a, paths_b):
        assert path_a == path_b, f"{name}: path 不同 {path_a} vs {path_b}"
        if isinstance(val_a, (jnp.ndarray, np.ndarray)) or hasattr(val_a, "shape"):
            arr_a = np.asarray(val_a)
            arr_b = np.asarray(val_b)
            assert arr_a.shape == arr_b.shape, (
                f"{name}{path_a}: shape {arr_a.shape} != {arr_b.shape}"
            )
            assert arr_a.dtype == arr_b.dtype, (
                f"{name}{path_a}: dtype {arr_a.dtype} != {arr_b.dtype}"
            )
            assert np.array_equal(arr_a, arr_b), (
                f"{name}{path_a}: 值不相等\n  a={arr_a}\n  b={arr_b}"
            )
        else:
            assert val_a == val_b, (
                f"{name}{path_a}: scalar {val_a!r} != {val_b!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: save → load round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_save_load_round_trip() -> None:
    """完整保存 TrainState 並還原，所有欄位逐 leaf bit-equal。"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(directory=tmp, max_to_keep=3, save_interval_steps=1)
        state = _make_dummy_train_state(step=42, seed=7)

        wrote = mgr.save(step=42, train_state=state)
        assert wrote, "首次 save 應落盤（save_interval_steps=1）"

        # 用 reference 還原（structure 完整還原成 TrainState NamedTuple）
        reference = _make_dummy_train_state(step=0, seed=0)
        restored = mgr.restore(step=42, reference_state=reference)

        assert isinstance(restored, TrainState), (
            f"restore 必須回傳 TrainState，得到 {type(restored).__name__}"
        )
        _assert_pytree_equal("params", state.params, restored.params)
        _assert_pytree_equal("opt_state", state.opt_state, restored.opt_state)
        assert int(restored.step) == 42, f"step={int(restored.step)} != 42"
        assert np.array_equal(np.asarray(state.rng_key), np.asarray(restored.rng_key)), (
            "rng_key 不一致"
        )
        assert restored.gradnorm_state is None, "未傳 gradnorm 應為 None"
        assert restored.al_state is None, "未傳 al 應為 None"
        mgr.close()
    print("[PASS] test_save_load_round_trip")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: max_to_keep=3 → save 5 個 step → 只剩最後 3 個
# ─────────────────────────────────────────────────────────────────────────────

def test_keep_last_n() -> None:
    """max_to_keep=3 政策確實保留最新 3 個 step；最舊的被淘汰。"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(directory=tmp, max_to_keep=3, save_interval_steps=1)
        steps_to_save = [100, 200, 300, 400, 500]
        for s in steps_to_save:
            state = _make_dummy_train_state(step=s, seed=s)
            wrote = mgr.save(step=s, train_state=state, force=True)
            assert wrote, f"step={s} 應落盤 (force=True)"

        kept = mgr.all_steps()
        assert kept == [300, 400, 500], (
            f"max_to_keep=3 應保留最新 3 個 [300,400,500]，實得 {kept}"
        )
        # 確認舊 step 已不可還原
        reference = _make_dummy_train_state()
        for old_step in (100, 200):
            try:
                mgr.restore(step=old_step, reference_state=reference)
                raise AssertionError(f"step={old_step} 已被淘汰但仍可還原")
            except Exception:
                pass  # 預期失敗
        mgr.close()
    print("[PASS] test_keep_last_n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: latest_step 行為
# ─────────────────────────────────────────────────────────────────────────────

def test_latest_step() -> None:
    """save 多 step 後 latest_step() 回傳最大 step；空目錄回 None。"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(directory=tmp, max_to_keep=5, save_interval_steps=1)

        assert mgr.latest_step() is None, "空目錄 latest_step 應為 None"
        assert mgr.all_steps() == [], "空目錄 all_steps 應為 []"

        for s in [50, 150, 100, 250, 200]:  # 故意亂序，驗證內部按 step value 排序
            state = _make_dummy_train_state(step=s, seed=s)
            mgr.save(step=s, train_state=state, force=True)

        latest = mgr.latest_step()
        assert latest == 250, f"latest_step 應為 250，實得 {latest}"

        # 不傳 step 給 restore → 用 latest
        reference = _make_dummy_train_state()
        restored = mgr.restore(step=None, reference_state=reference)
        assert int(restored.step) == 250, (
            f"restore(step=None) 應載入最新 step=250，實得 {int(restored.step)}"
        )
        mgr.close()
    print("[PASS] test_latest_step")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: GradNormState + ALState 完整保存
# ─────────────────────────────────────────────────────────────────────────────

def test_gradnorm_al_state_preserved() -> None:
    """帶 GradNorm + AL state 一起 save/load，所有 array bit-equal 且
       靜態 metadata（task_names tuple、rho/lambda_clip/ema_momentum）保留。

    EXP-082 root cause 之一就是 GradNorm log_weights 未被一併還原；本測試
    是該 regression 的直接防線。
    """
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(directory=tmp, max_to_keep=2, save_interval_steps=1)

        # 構造一個「已經 update 過幾次」的 state，確保 log_weights 不是 init 值
        state = _make_dummy_train_state(
            step=88, seed=11, with_gradnorm=True, with_al=True,
        )
        # 手動把 GradNorm 推離 init 值，模擬訓練中途
        new_log_w = jnp.array([0.0, -2.30, -1.61, -3.91], dtype=jnp.float32)
        gn_perturbed = state.gradnorm_state._replace(log_weights=new_log_w)
        al_perturbed = state.al_state._replace(
            lambda_=jnp.float32(2.345),
            ema_C=jnp.float32(0.0789),
            initialized=jnp.bool_(True),
        )
        state = state._replace(gradnorm_state=gn_perturbed, al_state=al_perturbed)

        mgr.save(step=88, train_state=state, force=True)

        reference = _make_dummy_train_state(
            step=0, seed=0, with_gradnorm=True, with_al=True,
        )
        restored = mgr.restore(step=88, reference_state=reference)

        # 型別必須完整還原
        assert isinstance(restored.gradnorm_state, GradNormState), (
            f"gradnorm_state 應為 GradNormState，得到 {type(restored.gradnorm_state)}"
        )
        assert isinstance(restored.al_state, ALState), (
            f"al_state 應為 ALState，得到 {type(restored.al_state)}"
        )

        # log_weights bit-equal
        assert np.array_equal(
            np.asarray(restored.gradnorm_state.log_weights),
            np.asarray(new_log_w),
        ), (
            f"log_weights mismatch\n  expected={new_log_w}\n"
            f"  got={restored.gradnorm_state.log_weights}"
        )
        # task_names static metadata 保留
        assert restored.gradnorm_state.task_names == ("data", "ns_u", "ns_v"), (
            f"task_names 不一致：{restored.gradnorm_state.task_names}"
        )

        # AL lambda / ema_C bit-equal
        assert np.array_equal(
            np.asarray(restored.al_state.lambda_),
            np.asarray(al_perturbed.lambda_),
        ), f"AL lambda mismatch: {restored.al_state.lambda_}"
        assert np.array_equal(
            np.asarray(restored.al_state.ema_C),
            np.asarray(al_perturbed.ema_C),
        ), f"AL ema_C mismatch: {restored.al_state.ema_C}"
        assert bool(restored.al_state.initialized), "AL initialized flag 遺失"

        # AL 靜態 hyperparameters 也應保留
        assert restored.al_state.rho == 1.0, f"rho={restored.al_state.rho}"
        assert restored.al_state.lambda_clip == 10.0, (
            f"lambda_clip={restored.al_state.lambda_clip}"
        )
        assert restored.al_state.ema_momentum == 0.5, (
            f"ema_momentum={restored.al_state.ema_momentum}"
        )
        mgr.close()
    print("[PASS] test_gradnorm_al_state_preserved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def test_lra_state_is_a_ckpt_compatibility_tombstone(tmp_path) -> None:
    """`lra_state` 永遠是 None，但**欄位必須留著**。

    LRA controller 已於 2026-08-03 移除（實測與 GradNorm 數值等價、production
    零使用）。欄位保留純為 ckpt 格式相容：實測移除 TrainState 的欄位會讓既有
    ckpt 全部 restore 失敗——orbax 連 `None` 值也寫 metadata entry
    （`value_type='None', skip_deserialize=True`），少一個 key 就樹結構不符。

    這條測試存在的理由是擋下「它總是 None，刪掉吧」那個念頭。
    """
    assert "lra_state" in TrainState._fields, (
        "TrainState 少了 lra_state——既有 ckpt 會全部 restore 失敗。"
        "它是相容性墓碑，不是死欄位。")

    state = TrainState(
        params={"w": jnp.ones((2, 2))},
        opt_state=(jnp.zeros((2,)),),
        step=jnp.int32(11),
        rng_key=jnp.zeros((2,), dtype=jnp.uint32),
        gradnorm_state=None,
        al_state=None,
        lra_state=None,
    )
    mgr = CheckpointManager(directory=str(tmp_path), max_to_keep=1, save_interval_steps=1)
    mgr.save(step=11, train_state=state, force=True)
    restored = mgr.restore(step=11, reference_state=state)
    assert restored.lra_state is None
    mgr.close()


def test_verify_params_tree() -> None:
    """verify_params_tree：eval 端 ckpt↔config 錯配的 fail-fast 閘門。

    Why: lenient restore 不驗結構；Flax apply 會忽略 ckpt 多出的參數
    silent 跑殘缺 forward（例：tau-off config 評 tau-on ckpt）。
    """
    import pytest

    base = {"params": {"enc": {"w": jnp.zeros((3, 4)), "b": jnp.zeros((4,))},
                       "dec": {"w": jnp.zeros((4, 2))}}}
    same = jax.tree_util.tree_map(lambda x: x + 1.0, base)  # 值不同、結構相同 → OK
    n = verify_params_tree(base, same)
    assert n == 3

    # ckpt 多一個 leaf（config 缺）→ raise
    extra = {"params": {**base["params"], "tau": {"g": jnp.zeros((4,))}}}
    with pytest.raises(ValueError, match="config 缺"):
        verify_params_tree(base, extra)

    # ckpt 少一個 leaf（config 有、ckpt 缺）→ raise
    missing = {"params": {"enc": base["params"]["enc"]}}
    with pytest.raises(ValueError, match="ckpt 缺"):
        verify_params_tree(base, missing)

    # 同 path 不同 shape → raise
    reshaped = {"params": {"enc": {"w": jnp.zeros((3, 4)), "b": jnp.zeros((4,))},
                           "dec": {"w": jnp.zeros((4, 8))}}}
    with pytest.raises(ValueError, match="shape 不符"):
        verify_params_tree(base, reshaped)
    print("[PASS] test_verify_params_tree")
