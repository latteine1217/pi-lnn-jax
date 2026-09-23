"""`physics_t_max` —— 把 physics collocation 時窗從資料時窗解耦。

這個鍵存在的唯一理由：讓 collocation 落到**沒有 sensor 資料**的時間段
（該段只有 PDE residual 約束），即時間外推實驗。因此每個測試都同時檢查
「延伸有沒有真的發生」與「關閉時是否逐字回到舊行為」——後者是 bit-identical
契約的承重面，`_plan_step` 的 RNG 消費不得因為多了這個鍵而改變。

不跑 training：`_plan_step` 是純 host-side 決策，可在本機 CPU 秒級驗完。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.pipeline.kolmogorov.assembly import (
    ReBatch,
    TrainingContext,
    check_physics_t_max,
)
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.run import _plan_step

_CONFIG = str(Path(__file__).resolve().parent.parent / "configs/_ledger_single_re.toml")

DATA_T_MIN, DATA_T_MAX = 0.0, 5.0


def _ctx(config):
    """單-Re、無 CRP/dropout/RAR 的最小 ctx：`_plan_step` 只讀得到這些欄位。"""
    fields = dict.fromkeys(ReBatch._fields)
    fields["sensor_vals"] = jnp.zeros((20, 10, 2))
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config,
        datasets=[{"re_norm": 1.0}],
        re_batches=[ReBatch(**fields)],
        re_t_min_host=[DATA_T_MIN],
        re_t_max_host=[DATA_T_MAX],
        crp_interp={},
        crp_re_norm_scale=10000.0,
        n_sensor_query=0,
    )


def _plan(config, step=1):
    key = jax.random.PRNGKey(0)
    return _plan_step(
        _ctx(config), step,
        params={}, rng_collo=key, rng_re=key, rng_crp=key, rar_state=None,
        n_datasets=1, sensor_dropout_rate=0.0, rar_residual_fn=None,
    )


def _with(config, **curriculum_kwargs):
    return replace(config, curriculum=replace(config.curriculum, **curriculum_kwargs))


def test_default_keeps_data_span_and_bitwise_identical_sampling():
    """預設 0.0 → t_upper 與 collocation 取樣逐位元等同「沒有這個鍵」的舊行為。"""
    config = resolve_inputs(["--config", _CONFIG]).config
    assert config.curriculum.physics_t_max == 0.0, "schema 預設必須是 0.0（= 沿用資料時窗）"

    plan = _plan(config)
    assert plan.t_upper == DATA_T_MAX
    assert float(jnp.max(plan.ct)) <= DATA_T_MAX

    # 鑑別力自檢：顯式寫成資料時窗上界，取樣必須逐位元相同（差異只可能來自時窗本身）
    same = _plan(_with(config, physics_t_max=DATA_T_MAX))
    assert np.array_equal(np.asarray(plan.ct), np.asarray(same.ct))
    assert np.array_equal(np.asarray(plan.cx), np.asarray(same.cx))
    print("✓ default_keeps_data_span_and_bitwise_identical_sampling")


def test_extends_collocation_beyond_data_window():
    """設 10.0（資料只到 5.0）→ collocation 必須真的取到 5.0 之外。"""
    config = resolve_inputs(["--config", _CONFIG]).config
    plan = _plan(_with(config, physics_t_max=10.0))

    assert plan.t_upper == 10.0
    ct = np.asarray(plan.ct)
    assert ct.max() > DATA_T_MAX, "時窗延伸了卻沒有任何 collocation 落在資料之外"
    assert ct.min() >= DATA_T_MIN
    # 外推段應佔約一半（U(0,10) 中 t>5 的比例）；只驗「有實質比例」，不釘死隨機值
    assert 0.2 < float((ct > DATA_T_MAX).mean()) < 0.8
    print("✓ extends_collocation_beyond_data_window")


def test_time_marching_ramps_to_extended_window():
    """time marching 的 ramp 天花板必須是延伸後的時窗，否則外推段永遠取不到。"""
    config = resolve_inputs(["--config", _CONFIG]).config
    tm = _with(config, use_time_marching=True, tm_start_frac=0.5,
               tm_warmup_steps=10, tm_ramp_steps=100, physics_t_max=10.0)

    assert _plan(tm, step=1).t_upper == pytest.approx(5.0)      # warmup 內 = 0.5 × 10.0
    assert _plan(tm, step=1000).t_upper == pytest.approx(10.0)  # ramp 完 → 全時窗
    # 鑑別力自檢：若天花板誤用資料時窗，ramp 終點會是 5.0 而非 10.0
    off = _with(config, use_time_marching=True, tm_start_frac=0.5,
                tm_warmup_steps=10, tm_ramp_steps=100)
    assert _plan(off, step=1000).t_upper == pytest.approx(DATA_T_MAX)
    print("✓ time_marching_ramps_to_extended_window")


def test_check_physics_t_max_fail_fast():
    """兩個前提不成立時必須炸；0.0 是 no-op。"""
    check_physics_t_max(0.0, 5.0, 5.0)          # 關閉 → 不檢查、不炸
    check_physics_t_max(10.0, 5.0, 10.0)        # 合法：延伸且 T_total 跟上

    with pytest.raises(ValueError, match="小於資料時窗上界"):
        check_physics_t_max(3.0, 5.0, 10.0)
    with pytest.raises(ValueError, match="超過 model.T_total"):
        check_physics_t_max(10.0, 5.0, 5.0)     # 相位錨週期 5 → 外推段與訓練段混疊
    print("✓ check_physics_t_max_fail_fast")


@pytest.mark.parametrize("name,n_collo", [
    ("exp_530_b3_extrap_T10", 1024),
    ("exp_531_b3_extrap_T10_collo2x", 2048),
])
def test_extrapolation_config_parses(name, n_collo):
    """外推臂 config：時窗、T_total、早期錨門檻三者必須一致，且只差 collocation 密度。"""
    cfg_path = Path(__file__).resolve().parent.parent / f"configs/{name}.toml"
    config = resolve_inputs(["--config", str(cfg_path)]).config

    assert config.curriculum.physics_t_max == 10.0
    assert config.model.T_total == 10.0, "T_total 必須跟上外推時窗，否則相位錨混疊"
    # t_early 視窗是 t_early_threshold × T_total；T_total 加倍後門檻要減半才維持 0.25 s
    assert config.loss.t_early_threshold * config.model.T_total == pytest.approx(0.25)
    assert config.curriculum.n_collo_start == n_collo
    assert config.curriculum.n_collo_end == n_collo
    print(f"✓ extrapolation_config_parses[{name}]")


def test_collo2x_arm_differs_from_530_only_in_density():
    """密度對照臂的單一變因：除了 collocation 點數與落點，其餘解析值必須相同。"""
    root = Path(__file__).resolve().parent.parent
    base = resolve_inputs(["--config", str(root / "configs/exp_530_b3_extrap_T10.toml")]).config
    arm = resolve_inputs(
        ["--config", str(root / "configs/exp_531_b3_extrap_T10_collo2x.toml")]).config

    assert base.model == arm.model, "模型建構值不得有差異"
    assert base.loss == arm.loss
    assert base.schedule == arm.schedule
    assert base.data == arm.data, "sensor / DNS / Re 必須是同一份"
    assert replace(base.run, artifacts_dir=None, config_path=None) == replace(
        arm.run, artifacts_dir=None, config_path=None)
    diff = {
        f for f in base.curriculum.__dataclass_fields__
        if getattr(base.curriculum, f) != getattr(arm.curriculum, f)
    }
    assert diff == {"n_collo_start", "n_collo_end"}, f"curriculum 多出非預期差異：{diff}"
    print("✓ collo2x_arm_differs_from_530_only_in_density")
