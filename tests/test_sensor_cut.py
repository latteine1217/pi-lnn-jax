"""`sensor_cut_min_frac` —— 造出「dt>0 且有真值」的監督訊號。

存在理由：既有訓練裡被監督的 query 其 `dt_to_query` **恆為 0**（query 時刻就是
sensor 幀），大 dt 只出現在 collocation 上而 PDE residual 選不出軌跡。截斷讓截點
之後的幀變成 dt>0 的監督點——那是 EXP-532 要測的唯一變因。

關閉（預設 0.0）時必須逐位元回到舊行為：不抽 RNG、jit 圖不變。
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
    cut_sensor_time,
)
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.run import _plan_step

REPO = Path(__file__).resolve().parent.parent
_CONFIG = str(REPO / "configs/_ledger_single_re.toml")

T_FRAMES, K_SENSORS = 21, 10
SENSOR_TIME = jnp.linspace(0.0, 5.0, T_FRAMES)


def _ctx(config):
    fields = dict.fromkeys(ReBatch._fields)
    fields["sensor_vals"] = jnp.zeros((T_FRAMES, K_SENSORS, 2))
    fields["sensor_time"] = SENSOR_TIME
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config, datasets=[{"re_norm": 1.0}], re_batches=[ReBatch(**fields)],
        re_t_min_host=[0.0], re_t_max_host=[5.0], crp_interp={},
        crp_re_norm_scale=10000.0, n_sensor_query=0,
    )


def _plan(config, step=1, key=0):
    k = jax.random.PRNGKey(key)
    return _plan_step(
        _ctx(config), step, params={}, rng_collo=k, rng_re=k, rng_crp=k, rar_state=None,
        n_datasets=1, sensor_dropout_rate=0.0, rar_residual_fn=None,
    )


def _with(config, **kw):
    return replace(config, curriculum=replace(config.curriculum, **kw))


def _decoder_idx_and_dt(sensor_time, t_q):
    """複製 `models.py` decoder 的取幀式（見下方 sentinel 測試釘住它沒漂移）。"""
    idx = int(jnp.sum(sensor_time <= t_q) - 1)
    idx = int(np.clip(idx, 0, sensor_time.shape[0] - 1))
    return idx, float(t_q - sensor_time[idx])


def test_cut_hides_trailing_frames_from_the_decoder():
    """截點之後的幀不再被選中，且截點之後的 query 拿到 dt>0。"""
    cut = 10                                   # t = 2.5
    st_cut = cut_sensor_time(SENSOR_TIME, jnp.asarray(cut))

    assert float(st_cut[cut]) == pytest.approx(2.5)
    assert float(st_cut[cut + 1]) > 1e29, "截點之後必須是哨兵值"

    # 截點之前：行為與未截斷完全相同（dt 仍為 0）
    for t in (0.0, 1.0, 2.5):
        assert _decoder_idx_and_dt(st_cut, t) == _decoder_idx_and_dt(SENSOR_TIME, t)

    # 截點之後：idx 卡在截點、dt 變正 —— 這正是要造出來的監督訊號
    idx, dt = _decoder_idx_and_dt(st_cut, 4.0)
    assert idx == cut
    assert dt == pytest.approx(1.5)
    assert _decoder_idx_and_dt(SENSOR_TIME, 4.0)[1] == pytest.approx(0.0), (
        "未截斷時同一個 query 的 dt 是 0——這就是既有訓練從未監督 dt>0 的原因")
    print("✓ cut_hides_trailing_frames_from_the_decoder")


def test_decoder_index_formula_has_not_drifted():
    """反漂移哨兵：上面的複製式必須與 decoder 實際用的一致。"""
    src = (REPO / "pi_lnn_jax" / "models.py").read_text()
    assert "jnp.sum(sensor_time[None, :] <= t_q[:, None], axis=1).astype(jnp.int32) - 1" in src, (
        "decoder 的取幀式變了 —— `cut_sensor_time` 的哨兵手法與本檔的複製式都要重新檢查")
    print("✓ decoder_index_formula_has_not_drifted")


def test_disabled_by_default_and_consumes_no_rng():
    """關閉時不抽 RNG：collocation 取樣必須與「沒有這個功能」逐位元相同。"""
    config = resolve_inputs(["--config", _CONFIG]).config
    assert config.curriculum.sensor_cut_min_frac == 0.0

    plan = _plan(config)
    assert plan.sensor_cut_idx is None
    # 鑑別力自檢：開啟後 RNG 被多消耗一次 → collocation 取樣必然改變
    on = _plan(_with(config, sensor_cut_min_frac=0.5))
    assert on.sensor_cut_idx is not None
    assert not np.array_equal(np.asarray(plan.ct), np.asarray(on.ct))
    print("✓ disabled_by_default_and_consumes_no_rng")


def test_cut_index_stays_in_the_declared_range():
    """抽出的截點必須落在 [min_frac·(T−1), T−1]，且是 traced int（不觸發 retrace）。"""
    config = _with(resolve_inputs(["--config", _CONFIG]).config, sensor_cut_min_frac=0.5)
    lo = int(round(0.5 * (T_FRAMES - 1)))

    seen = set()
    for key in range(30):
        cut = _plan(config, key=key).sensor_cut_idx
        assert isinstance(cut, jax.Array) and cut.dtype == jnp.int32
        seen.add(int(cut))
    assert min(seen) >= lo and max(seen) <= T_FRAMES - 1
    assert len(seen) > 1, "每步都抽到同一個截點 → 取樣沒有作用"
    print(f"✓ cut_index_stays_in_the_declared_range (seen {min(seen)}..{max(seen)})")


def test_exp532_config_differs_from_530_only_in_sensor_cut():
    """EXP-532 的單一變因：除了截斷與落點，其餘解析值必須與 EXP-530 相同。"""
    base = resolve_inputs(["--config", str(REPO / "configs/exp_530_b3_extrap_T10.toml")]).config
    arm = resolve_inputs(
        ["--config", str(REPO / "configs/exp_532_b3_extrap_T10_sensorcut.toml")]).config

    assert arm.curriculum.sensor_cut_min_frac > 0.0
    assert base.model == arm.model and base.loss == arm.loss and base.data == arm.data
    diff = {
        f for f in base.curriculum.__dataclass_fields__
        if getattr(base.curriculum, f) != getattr(arm.curriculum, f)
    }
    assert diff == {"sensor_cut_min_frac"}, f"curriculum 多出非預期差異：{diff}"
    print("✓ exp532_config_differs_from_530_only_in_sensor_cut")
