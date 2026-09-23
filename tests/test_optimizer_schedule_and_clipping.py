"""LR 排程的形狀與梯度裁剪的存在 —— `chapter02.tex:516` 的實作對應。

Why this file exists:
    稿子 §2 對 optimiser 的敘述有三個可檢查的宣稱：linear warmup 2000 步到
    1e-3、**連續**（非階梯）指數衰減每 2000 步 ×0.9、全域 L2 裁剪 1.0。
    實測這三個宣稱在既有測試下**沒有任何保護**：

        把 `_wrap_chain` 改成直接回 inner（拿掉裁剪）  -> 1154 passed
        把 `_build_lr_schedule` 一律回 constant         -> 1154 passed
        schedule_free 底下 inner b1 從 0.0 改回 0.9     -> 全綠

    `tests/test_optimizers.py` 只斷言 `info["lr_schedule"] == "constant"`
    （warmup=decay=0 的退化情形），且它自己的 docstring 就寫明 toy 梯度
    小到不會觸發 clip。所以「排程長什麼樣」「有沒有裁剪」是零覆蓋。

    本檔全部是純數值斷言，不需要模型，跑完 < 1s。
"""
from __future__ import annotations

import jax.numpy as jnp
import optax
import pytest

from pi_lnn_jax.optimizers import _build_lr_schedule, build_optimizer

# 主線 config（exp_245_b3_les_T50.toml）解析出來的實際值。
PEAK, WARMUP, DECAY_STEPS, GAMMA, FLOOR = 1e-3, 2000, 2000, 0.9, 1e-6


@pytest.fixture
def schedule():
    fn, desc = _build_lr_schedule(PEAK, WARMUP, DECAY_STEPS, GAMMA, FLOOR)
    assert desc == "warmup_2000_exponential_decay_2000", desc
    return fn


def test_warmup_reaches_the_peak_exactly_at_the_declared_step(schedule):
    """稿子：linear warmup over 2000 steps to a peak of 1e-3。"""
    assert float(schedule(WARMUP)) == pytest.approx(PEAK, rel=1e-6)  # fp32
    # 起點不是 0 而是 floor——optax 的 schedule_free 在 lr=0 時 (lr/max_lr)^p 會 0/0
    assert 0.0 < float(schedule(0)) <= FLOOR * 1.01
    # 線性：半程應約為峰值的一半
    assert float(schedule(WARMUP // 2)) == pytest.approx(PEAK / 2, rel=1e-2)


def test_decay_is_geometric_at_the_declared_rate(schedule):
    """稿子：exponential decay at 0.9 per 2000 steps，衰減時鐘從 warmup 結束起算。"""
    assert float(schedule(WARMUP + DECAY_STEPS)) == pytest.approx(PEAK * GAMMA, rel=1e-6)
    assert float(schedule(WARMUP + 2 * DECAY_STEPS)) == pytest.approx(PEAK * GAMMA ** 2, rel=1e-6)


def test_decay_is_continuous_not_staircase(schedule):
    """稿子明寫 continuous；階梯版在 2000..3999 之間會是常數。

    optax 的 `staircase` 預設 False，但那是預設值不是斷言——改成 True
    不會有任何錯誤訊息，只會讓 LR 曲線變成另一種東西。
    """
    lo, mid, hi = (float(schedule(WARMUP + k)) for k in (1, DECAY_STEPS // 2, DECAY_STEPS - 1))
    assert lo > mid > hi, f"衰減不是連續下降：{lo:.6e} / {mid:.6e} / {hi:.6e}"
    assert lo != pytest.approx(hi, rel=1e-6), "整段相等＝階梯，與稿子的 continuous 不符"


def test_floor_is_never_reached_within_the_training_budget(schedule):
    """稿子寫「floored at 1e-6」，但 20000 步的預算下那個下限從不生效。

    釘住它是為了讓「那句規格永不生效」變成可稽核的事實，而不是讀者的假設。
    """
    assert float(schedule(20_000)) == pytest.approx(3.874204e-4, rel=1e-5)
    assert float(schedule(20_000)) > FLOOR * 100


def test_gradient_clipping_is_wired_and_bounds_the_global_norm():
    """稿子：‖∇L‖₂ ≤ 1.0，單一全域 L2（非 per-layer、非 L∞）。"""
    params = {"a": jnp.zeros((4,)), "b": jnp.zeros((3,))}
    huge = {"a": jnp.array([1e6, 0.0, 0.0, 0.0]), "b": jnp.array([0.0, 1e6, 0.0])}

    clip = optax.clip_by_global_norm(1.0)
    out, _ = clip.update(huge, clip.init(params), params)
    flat = jnp.concatenate([out["a"], out["b"]])
    assert float(jnp.linalg.norm(flat)) == pytest.approx(1.0, rel=1e-6)

    # 接線：**Adam 驗不了這件事**。它的首步是 -lr·g/(√(g²)+ε) = -lr·sign(g)，
    # 對 g 的尺度不變，裁不裁剪都一樣——本測試先前在這裡放了一個
    # `|upd| <= lr*1.01` 的斷言並宣稱「裁剪沒接上就會爆掉」，那是恆真的空洞斷言
    # （2026-09-10 獨立複查抓到：拿掉三個呼叫點的 `_wrap_chain` 後它照樣綠）。
    #
    # 判別器要跨兩步：巨大梯度會把 Adam 的二階動量 v̂ 撐大，未裁剪時第二步的
    # update 因此被壓到極小；裁剪過則 v̂ 有界，第二步照常。單步分不出來。
    modest = {"a": jnp.array([1.0, 0.0, 0.0, 0.0]), "b": jnp.array([0.0, 1.0, 0.0])}

    def second_step(max_grad_norm):
        tx, _ = build_optimizer(name="adam", learning_rate=PEAK,
                                max_grad_norm=max_grad_norm)
        st = tx.init(params)
        _, st = tx.update(huge, st, params)          # 第一步：巨大梯度
        upd, _ = tx.update(modest, st, params)       # 第二步：正常梯度
        return float(jnp.linalg.norm(jnp.concatenate([upd["a"], upd["b"]])))

    on, off = second_step(1.0), second_step(0.0)
    # 實測分離度 1.49×（開 1.414e-3 / 關 9.476e-4）。分離度不大是 Adam 的性質——
    # β2=0.999 讓 v̂ 很快回復——但方向確定：接線斷掉時兩者恆等，比值退到 1.00。
    assert on / off > 1.25, (
        f"裁剪開／關的第二步 update 範數是 {on:.3e} / {off:.3e}（比值 {on / off:.3f}，"
        "預期約 1.49）。退到 1.00 就代表 max_grad_norm 沒有走進 build_optimizer 的接線。")
    # 每條 optimizer 路徑都經過 `_wrap_chain` 這件事另由
    # `tests/test_production_wiring.py::test_every_optimizer_path_wraps_the_clip` 釘住。


def test_zero_max_grad_norm_disables_clipping_deliberately():
    """`max_grad_norm=0` 是「關閉」而非「裁到 0」——把這個分支釘成刻意的。"""
    from pi_lnn_jax.optimizers import _wrap_chain

    inner = optax.sgd(1.0)
    params = {"a": jnp.zeros((2,))}
    huge = {"a": jnp.array([1e6, 0.0])}
    off, _ = _wrap_chain(inner, 0.0).update(huge, _wrap_chain(inner, 0.0).init(params), params)
    assert float(jnp.abs(off["a"]).max()) > 1e5, "0 應該是關閉裁剪"
    on, _ = _wrap_chain(inner, 1.0).update(huge, _wrap_chain(inner, 1.0).init(params), params)
    assert float(jnp.linalg.norm(on["a"])) == pytest.approx(1.0, rel=1e-6)


def test_schedule_free_zeroes_the_inner_momentum():
    """`optimizers.py:290-299` 特地把 inner b1 壓成 0（避免 double momentum）。

    生效值只出現在 `info["name"]` 裡；改回 0.9 不會有任何錯誤訊息。
    """
    pytest.importorskip("optax.contrib")
    from pi_lnn_jax.optimizers import is_soap_available

    if not is_soap_available():
        pytest.skip("soap 不可用；b1 的字串觀測點只存在於 soap 的 inner_name")
    # production 走的就是 schedule_free(soap)；生效值編在 inner_name 裡
    _, info = build_optimizer(name="schedule_free", learning_rate=PEAK,
                              base_optimizer="soap", max_grad_norm=1.0)
    assert "b0.0/" in info["name"], (
        f"schedule_free 下 inner SOAP 的 b1 應為 0，info['name']={info['name']!r}")
    # 對照：純 soap（無 schedule_free）不該被壓成 0
    _, plain = build_optimizer(name="soap", learning_rate=PEAK, max_grad_norm=1.0)
    assert "b0.0/" not in plain["name"], (
        f"只有 schedule_free 才壓 b1；純 soap 不該受影響，得到 {plain['name']!r}")
