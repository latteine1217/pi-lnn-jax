"""`pi_lnn_jax.rollout` —— branch 自迴歸的延伸序列。

三個承重面必須在本機驗得動：
1. 延伸是**自迴歸**的（後段 pseudo 幀基於前段，不是全部共用凍結狀態）；
2. 原本的幀原封不動（data loss 與 t≤資料末端的行為不得被動到）；
3. pseudo 觀測不回傳梯度（它們是輸入，不是被優化的對象）。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.rollout import (
    check_extension_inputs,
    extend_sensor_sequence,
    pseudo_frame_times,
)


class _FakeModel:
    """只記錄呼叫、行為可預測的假 model：encode 回序列長度，decode 回該長度的常數。

    這樣「第 r 輪解出的值」就直接暴露了它看到的序列長度 —— 自迴歸與否一眼可辨。
    """

    def __init__(self):
        self.encode_lengths = []

    def apply(self, params, *args, method=None, **kw):
        if method == "encode":
            vals = args[0]
            self.encode_lengths.append(int(vals.shape[0]))
            return jnp.asarray(float(vals.shape[0]))
        xy_q, t_q, h_states = args[0], args[1], args[2]
        return jnp.broadcast_to(h_states, (xy_q.shape[0], 3))


def _inputs(T=6, K=3, C=2):
    return (
        jnp.zeros((T, K, C)),
        jnp.asarray(np.random.default_rng(0).uniform(size=(K, 2))),
        jnp.asarray(1.0),
        jnp.linspace(0.0, 5.0, T),
    )


def test_pseudo_frame_times_fail_fast():
    assert np.allclose(pseudo_frame_times(5.0, 6.0, 0.5), [5.5, 6.0])
    with pytest.raises(ValueError, match="無法整除"):
        pseudo_frame_times(5.0, 10.0, 0.3)
    with pytest.raises(ValueError, match="未超過資料末端"):
        pseudo_frame_times(5.0, 5.0, 0.5)
    print("✓ pseudo_frame_times_fail_fast")


def test_extension_is_autoregressive_across_rounds():
    """後一輪必須看到前一輪補上的幀 —— 否則就不是自迴歸。"""
    vals, pos, re_norm, times = _inputs()
    m = _FakeModel()
    pt = pseudo_frame_times(5.0, 7.0, 0.5)          # 4 幀
    out_vals, out_times = extend_sensor_sequence(
        m, {}, vals, pos, re_norm, times, pt, rounds=2,
        encode_method="encode", decode_method="decode")

    assert m.encode_lengths == [6, 8], (
        f"第二輪應該在 8 幀的序列上重新 encode，實得 {m.encode_lengths}")
    # 假 model 讓 pseudo 值 = 當輪序列長度 → 兩輪的值必須不同
    assert float(out_vals[6, 0, 0]) == 6.0 and float(out_vals[8, 0, 0]) == 8.0
    assert out_vals.shape[0] == 10 and out_times.shape[0] == 10
    assert np.allclose(np.asarray(out_times[6:]), pt)
    print("✓ extension_is_autoregressive_across_rounds")


def test_rounds_one_reuses_the_frozen_state():
    """鑑別力自檢：rounds=1 時全部 pseudo 幀共用同一次 encode（最弱版本）。"""
    vals, pos, re_norm, times = _inputs()
    m = _FakeModel()
    extend_sensor_sequence(m, {}, vals, pos, re_norm, times,
                           pseudo_frame_times(5.0, 7.0, 0.5), rounds=1,
                           encode_method="encode", decode_method="decode")
    assert m.encode_lengths == [6]
    print("✓ rounds_one_reuses_the_frozen_state")


def test_original_frames_are_untouched():
    """原本的幀與時間軸不得被動到（t≤資料末端的行為必須不變）。"""
    vals, pos, re_norm, times = _inputs()
    vals = vals + 0.37
    out_vals, out_times = extend_sensor_sequence(
        _FakeModel(), {}, vals, pos, re_norm, times,
        pseudo_frame_times(5.0, 6.0, 0.5), rounds=2,
        encode_method="encode", decode_method="decode")

    assert np.allclose(np.asarray(out_vals[:vals.shape[0]]), np.asarray(vals))
    assert np.allclose(np.asarray(out_times[:times.shape[0]]), np.asarray(times))
    print("✓ original_frames_are_untouched")


def test_pseudo_observations_carry_no_gradient():
    """pseudo 觀測走 stop_gradient：它們是輸入，梯度不得沿 rollout 回穿。"""
    vals, pos, re_norm, times = _inputs()
    pt = pseudo_frame_times(5.0, 6.0, 0.5)

    class _ParamModel(_FakeModel):
        def apply(self, params, *args, method=None, **kw):
            if method == "encode":
                return params["w"] * 1.0
            xy_q, _, h = args[0], args[1], args[2]
            return jnp.broadcast_to(h, (xy_q.shape[0], 3))

    def total(p):
        v, _ = extend_sensor_sequence(
            _ParamModel(), p, vals, pos, re_norm, times, pt, rounds=1,
            encode_method="encode", decode_method="decode")
        return jnp.sum(v)

    g = jax.grad(total)({"w": jnp.asarray(2.0)})
    assert float(g["w"]) == 0.0, "pseudo 觀測仍在傳梯度 —— stop_gradient 沒生效"
    print("✓ pseudo_observations_carry_no_gradient")


def test_exp533_config_differs_from_530_only_in_autoreg():
    """EXP-533 的單一變因：除了自迴歸與落點，其餘解析值必須與 EXP-530 相同。"""
    from pathlib import Path

    from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs

    repo = Path(__file__).resolve().parent.parent
    base = resolve_inputs(["--config", str(repo / "configs/exp_530_b3_extrap_T10.toml")]).config
    arm = resolve_inputs(
        ["--config", str(repo / "configs/exp_533_b3_extrap_T10_autoreg.toml")]).config

    assert arm.curriculum.autoreg_pseudo_dt > 0.0 and arm.curriculum.autoreg_rounds > 1
    assert base.model == arm.model and base.loss == arm.loss and base.data == arm.data
    diff = {
        f for f in base.curriculum.__dataclass_fields__
        if getattr(base.curriculum, f) != getattr(arm.curriculum, f)
    }
    assert diff == {"autoreg_pseudo_dt", "autoreg_rounds"}, f"多出非預期差異：{diff}"
    # pseudo 幀必須整除延伸區間（否則 build_context 會 fail-fast）
    n = pseudo_frame_times(5.0, arm.curriculum.physics_t_max,
                           arm.curriculum.autoreg_pseudo_dt)
    assert len(n) == 10 and n[-1] == pytest.approx(10.0)
    print("✓ exp533_config_differs_from_530_only_in_autoreg")


def test_autoreg_and_sensor_cut_are_mutually_exclusive():
    """兩者都改寫 decode 端時間軸，同時開會無法歸因 —— 必須 fail-fast。"""
    import inspect

    from pi_lnn_jax.pipeline.kolmogorov import assembly

    src = inspect.getsource(assembly.build_context)
    assert "sensor_cut_min_frac > 0.0" in src and "無法歸因" in src, (
        "build_context 少了互斥檢查 —— 兩個旗標同時開會靜靜訓出無法解釋的東西")
    print("✓ autoreg_and_sensor_cut_are_mutually_exclusive")


def test_runs_under_jit_with_traced_sensor_time():
    """**實跑一次 jit**：延伸函式在 traced sensor_time 之下不得取任何值。

    這條是實際踩過的坑：值層面的前提檢查（`float(sensor_time[-1])`）在本機用具體
    陣列測全綠，進到 jit 的 step_fn 就 TracerArrayConversionError，job 50 秒掛掉。
    具體陣列測不到 traced 路徑——要驗就得真的 jit 一次。
    """
    vals, pos, re_norm, times = _inputs()
    pt = pseudo_frame_times(5.0, 6.0, 0.5)

    @jax.jit
    def run(v, p_, rn, ti):
        out_v, out_t = extend_sensor_sequence(
            _FakeModel(), {}, v, p_, rn, ti, pt, rounds=2,
            encode_method="encode", decode_method="decode")
        return jnp.sum(out_v), out_t[-1]

    total, last_t = run(vals, pos, re_norm, times)
    assert np.isfinite(float(total))
    assert float(last_t) == pytest.approx(6.0)
    print("✓ runs_under_jit_with_traced_sensor_time")


def test_value_level_checks_live_at_construction_time():
    """值層面的前提改由 check_extension_inputs 擋（建構期，有具體值）。"""
    with pytest.raises(ValueError, match="晚於資料末端"):
        check_extension_inputs(np.array([4.5, 5.5]), t_last=5.0, rounds=1)
    with pytest.raises(ValueError, match="rounds"):
        check_extension_inputs(np.array([5.5, 6.0]), t_last=5.0, rounds=3)
    assert np.allclose(check_extension_inputs(np.array([5.5, 6.0]), 5.0, 2), [5.5, 6.0])
    print("✓ value_level_checks_live_at_construction_time")
