"""評估協定的解析規則。

Why 這個 module 存在：eval 的時間軸取樣與訓練 cadence 的關係，先前散在
`submit_crossre.sh` 的一個 sed、`eval_exp301_enstrophy.sbatch` 的一句註解、
以及 `eval_dropout_sweep.sbatch.tmpl` 的一句「與訓練 time_strides 無關」裡。
三種說法互不相容，而且**三種都是對的**——它們描述的是三種不同的協定。

實測（2026-08-06）：27 份 T20 系 final_eval 中有 12 份的評估 stride 與訓練宣告
不符（多數是訓練 8 / 評估 2）。`evaluate_exp245` 內建的 allclose 攔不到——sensor
與 DNS 一起用錯的 stride 時兩者仍互相一致。這個 module 讓那件事在 follow_training
模式下變成 raise。
"""
from __future__ import annotations

import pytest

from pi_lnn_jax.evaluation_protocol import (
    TRAINING_DEFAULT_TIME_STRIDE,
    EvaluationProtocol,
    ProtocolMode,
    resolve_protocol,
)


# ── follow_training：單一真實來源是訓練 config ────────────────────────────

def test_follow_training_takes_the_stride_from_the_training_config():
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[8])

    assert p.sensor_time_stride == 8
    assert p.training_time_stride == 8
    # 對齊模式必須看到完整 DNS 候選池，否則該匹配的幀可能被 stride 掉。
    assert p.dns_time_stride == 1
    assert "time_strides" in p.basis


def test_follow_training_uses_the_training_fallback_when_unset():
    """訓練端在 time_strides 為空時吃 2（assembly.py）。協定必須複製同一條規則，
    不是自己另訂一個預設——那會讓兩邊在「沒寫」的情況下悄悄分岔。"""
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[])

    assert p.sensor_time_stride == TRAINING_DEFAULT_TIME_STRIDE == 2
    assert "fallback" in p.basis


def test_follow_training_picks_the_stride_for_the_requested_re():
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                         training_time_strides=[2, 4, 8], re_index=2)

    assert p.sensor_time_stride == 8


def test_follow_training_refuses_a_cli_stride_that_disagrees():
    """TD-4 的修法：CLI 與訓練不一致就失敗，不靜默採用其中一個。"""
    with pytest.raises(ValueError, match="與訓練"):
        resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                         training_time_strides=[8], cli_time_stride=2)


def test_follow_training_accepts_a_cli_stride_that_agrees():
    """明寫一致的值不該被當成錯——呼叫端把假設寫出來是好事。"""
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                         training_time_strides=[8], cli_time_stride=8)

    assert p.sensor_time_stride == 8


# ── fixed_grid：刻意偏離，必須說出理由 ────────────────────────────────────

def test_fixed_grid_requires_an_explicit_stride():
    with pytest.raises(ValueError, match="fixed_grid"):
        resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[8],
                         reason="snapshot-density：訓練 cadence 是自變數")


def test_fixed_grid_requires_a_reason():
    """刻意與訓練不同是合法的（snapshot-density 全族靠它），但沉默的偏離
    與「忘了帶旗標」在產物上長得一模一樣。理由進 provenance 才分得出來。"""
    with pytest.raises(ValueError, match="reason"):
        resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[8],
                         cli_time_stride=2)


def test_fixed_grid_records_the_training_stride_it_departs_from():
    p = resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[20],
                         cli_time_stride=2,
                         reason="snapshot-density：訓練 cadence 是自變數，eval 格點須固定才可比")

    assert p.sensor_time_stride == 2
    assert p.training_time_stride == 20      # 偏離的對象要記下來
    assert "snapshot-density" in p.basis
    assert p.dns_time_stride == 1


# ── sensor_time_independent：不對齊，是協定不是放寬 ──────────────────────

def test_sensor_time_independent_does_not_align_and_strides_the_query_grid():
    """thesis §7.2 的間歇評估：query 走完整 DNS 格點、sensor context 走自己的
    不等距序列。DNS 的 stride 在此是 query 解析度，不是對齊用的。"""
    p = resolve_protocol(mode=ProtocolMode.SENSOR_TIME_INDEPENDENT,
                         training_time_strides=[1], cli_time_stride=2)

    assert p.aligns_sensor_to_dns is False
    assert p.dns_time_stride == 2
    assert p.sensor_time_stride == 2


def test_matched_modes_align():
    for mode in (ProtocolMode.FOLLOW_TRAINING, ProtocolMode.FIXED_GRID):
        kwargs = {"cli_time_stride": 2, "reason": "x"} if mode is ProtocolMode.FIXED_GRID else {}
        p = resolve_protocol(mode=mode, training_time_strides=[2], **kwargs)
        assert p.aligns_sensor_to_dns is True, mode


# ── 模式必填、無預設 ──────────────────────────────────────────────────────

def test_mode_has_no_default():
    """有預設就會有人吃到錯的那一個——那正是 TD-4 的失效形狀。"""
    with pytest.raises(TypeError):
        resolve_protocol(training_time_strides=[8])  # type: ignore[call-arg]


def test_mode_string_is_accepted_and_validated():
    assert resolve_protocol(mode="follow_training",
                            training_time_strides=[8]).mode is ProtocolMode.FOLLOW_TRAINING
    with pytest.raises(ValueError):
        resolve_protocol(mode="lenient", training_time_strides=[8])


def test_protocol_is_immutable():
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[8])
    with pytest.raises(Exception):
        p.sensor_time_stride = 1  # type: ignore[misc]


def test_protocol_serializes_for_provenance():
    """協定要跟著數字走進產物；stdout 的說明會被滾走。"""
    p = resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[20],
                         cli_time_stride=2, reason="snapshot-density")

    d = p.to_provenance()
    assert d["mode"] == "fixed_grid"
    assert d["sensor_time_stride"] == 2
    assert d["training_time_stride"] == 20
    assert "snapshot-density" in d["basis"]


def test_sensor_T_is_carried_and_optional():
    assert resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                            training_time_strides=[8]).sensor_T is None
    assert resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                            training_time_strides=[8], sensor_T=50).sensor_T == 50


def test_negative_or_zero_stride_is_refused():
    with pytest.raises(ValueError):
        resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[8],
                         cli_time_stride=0, reason="x")
    with pytest.raises(ValueError):
        resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[-1])


def test_re_index_out_of_range_is_refused():
    """靜默取 [0] 會讓多-Re 的每個 Re 都用第一個 stride。"""
    with pytest.raises(IndexError):
        resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                         training_time_strides=[2, 4], re_index=5)


def test_exported_type_is_the_dataclass():
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[8])
    assert isinstance(p, EvaluationProtocol)


# ── load_for_evaluation：一份對齊實作 ────────────────────────────────────
#
# 真實資料只在 lab-server 上，故把兩個 loader 換成合成資料，驗對齊行為本身。

import numpy as np  # noqa: E402

from pi_lnn_jax.evaluation_protocol import load_for_evaluation  # noqa: E402


def _patch(monkeypatch, *, n_dns=16, sensor_take=None, N=8, K=3):
    dns_t = (np.arange(n_dns) * 0.25).astype(np.float64)
    dns_u = np.arange(n_dns * N * N, dtype=np.float64).reshape(n_dns, N, N)
    dns_v = dns_u + 0.5
    take = sensor_take if sensor_take is not None else list(range(0, n_dns, 4))
    s_time = dns_t[take]

    def _sensors(_p, time_stride=1):
        # 模仿真實 loader：sensor 自身也被 stride
        return {"sensor_vals": np.zeros((len(s_time[::time_stride]), K, 2)),
                "sensor_pos": np.zeros((K, 2)),
                "sensor_time": s_time[::time_stride],
                "norm_stats": {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0}}

    def _dns(_p, time_stride=1, return_p=False):
        return dns_u[::time_stride], dns_v[::time_stride], dns_t[::time_stride]

    monkeypatch.setattr("pi_lnn_jax.data.load_sensors_from_path", _sensors)
    monkeypatch.setattr("pi_lnn_jax.data.load_dns_from_path", _dns)
    return dns_t


def test_matched_mode_selects_the_frames_whose_times_equal_the_sensor_times(monkeypatch):
    dns_t = _patch(monkeypatch)
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[1])

    out = load_for_evaluation("s.json", "d.npy", protocol=p, viscosity=1e-4)

    np.testing.assert_array_equal(out.dns_t_eval, out.sensor_time)
    np.testing.assert_array_equal(out.dns_t_eval, dns_t[out.eval_idx])
    assert out.context.times == tuple(float(t) for t in out.dns_t_eval)
    assert out.context.viscosity == 1e-4
    assert out.context.grid_shape == (out.Nprime, out.Nprime)


def test_independent_mode_keeps_the_whole_dns_grid_as_query(monkeypatch):
    """query 走完整 DNS 格點、sensor 走自己的序列，兩者長度本就不同。"""
    _patch(monkeypatch, n_dns=16)
    p = resolve_protocol(mode=ProtocolMode.SENSOR_TIME_INDEPENDENT,
                         training_time_strides=[1], cli_time_stride=2)

    out = load_for_evaluation("s.json", "d.npy", protocol=p)

    assert out.eval_idx is None
    assert len(out.dns_t_eval) == 8           # 16 幀 / stride 2
    assert len(out.sensor_time) != len(out.dns_t_eval)


def test_matched_mode_fails_loudly_when_sensor_times_have_no_dns_counterpart(monkeypatch):
    """對齊模式不放寬：對不上就是對不上，不靜默取最近的。"""
    _patch(monkeypatch, sensor_take=[0, 1], n_dns=16)
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[1])
    # 把 DNS 時間軸整體平移，使 sensor 時間落在兩幀正中間
    import pi_lnn_jax.data as _d
    orig = _d.load_dns_from_path
    monkeypatch.setattr("pi_lnn_jax.data.load_dns_from_path",
                        lambda p_, time_stride=1, return_p=False: tuple(
                            (a if i < 2 else a + 0.125) for i, a in enumerate(orig(p_, time_stride))))

    with pytest.raises(ValueError):
        load_for_evaluation("s.json", "d.npy", protocol=p)


def test_sensor_T_truncates(monkeypatch):
    _patch(monkeypatch, n_dns=16)
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING,
                         training_time_strides=[1], sensor_T=2)

    out = load_for_evaluation("s.json", "d.npy", protocol=p)

    assert out.T == 2 and len(out.dns_t_eval) == 2


def test_returned_fields_cover_the_previous_load_aligned_re_surface(monkeypatch):
    """三支 caller 逐鍵索引舊回傳面；欄位少一個就是一個 AttributeError。"""
    _patch(monkeypatch)
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[1])

    out = load_for_evaluation("s.json", "d.npy", protocol=p)

    for f in ("sensor_phys", "sensor_pos", "T", "K", "N", "s", "Nprime", "eval_idx",
              "dns_t_eval", "dns_u_eval", "dns_v_eval", "dns_u_full", "dns_v_full"):
        assert hasattr(out, f), f


# ── 真實資料上的等價（遷移背書；隨 load_aligned_re 一同退場）──────────────

_REAL_SENSOR = "data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json"
_REAL_DNS = ("data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_"
             "dt1p95e4_si128_seed42.npy")


def _real_data_present() -> bool:
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    return (root / _REAL_SENSOR).exists() and (root / _REAL_DNS).exists()


@pytest.mark.skipif(not _real_data_present(),
                    reason=f"需要 {_REAL_SENSOR} 與 {_REAL_DNS}（僅資料機上有）")
@pytest.mark.parametrize("stride,sensor_T,max_grid", [(8, None, 0), (2, 50, 0), (4, 50, 128)])
def test_new_aligner_matches_load_aligned_re_on_real_data(stride, sensor_T, max_grid):
    """`load_for_evaluation` 與既有 `load_aligned_re` 在真實 sensor/DNS 上逐欄相同。

    上面的對齊測試全走合成資料；這條把「等價」從推理變成量測。實測（2026-08-06，
    Re=10000 T20 組）三種設定下所有欄位 max|Δ| = 0，含 101×256×256 的場陣列。

    **本測試隨 `load_aligned_re` 一同退場**——它存在的目的是背書那次替換。
    """
    import os

    import numpy as np

    from pi_lnn_jax.baseline_eval import load_aligned_re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sj, dp = os.path.join(root, _REAL_SENSOR), os.path.join(root, _REAL_DNS)

    old = load_aligned_re(sj, dp, sensor_time_stride=stride,
                          sensor_T=sensor_T if sensor_T else 10 ** 9,
                          grid_stride=1, max_grid=max_grid)
    p = resolve_protocol(mode=ProtocolMode.FIXED_GRID, training_time_strides=[8],
                         cli_time_stride=stride, sensor_T=sensor_T,
                         reason="等價探針")
    new = load_for_evaluation(sj, dp, protocol=p, viscosity=1e-4, max_grid=max_grid)

    for k in ("T", "K", "N", "s", "Nprime"):
        assert old[k] == getattr(new, k), k
    assert list(old["eval_idx"]) == list(new.eval_idx)
    for k in ("sensor_phys", "sensor_pos", "dns_t_eval", "dns_u_eval", "dns_v_eval"):
        np.testing.assert_array_equal(np.asarray(old[k]), np.asarray(getattr(new, k)), err_msg=k)
    np.testing.assert_array_equal(np.asarray(new.context.times), np.asarray(new.dns_t_eval))


def test_training_time_strides_are_read_from_the_toml(tmp_path):
    from pi_lnn_jax.evaluation_protocol import training_time_strides_from_config

    c = tmp_path / "c.toml"
    c.write_text('[data_kwargs]\nre_values = [1000.0]\n'
                 'sensor_jsons = ["a.json"]\ndns_paths = ["a.npy"]\n'
                 'time_strides = [4, 8]\n')

    assert training_time_strides_from_config(c) == [4, 8]


def test_missing_time_strides_reads_as_empty_not_as_a_guess(tmp_path):
    """空 list 讓 resolve_protocol 走與訓練相同的 fallback；此處不代它猜。"""
    from pi_lnn_jax.evaluation_protocol import training_time_strides_from_config

    c = tmp_path / "c.toml"
    c.write_text('[data_kwargs]\nre_values = [1000.0]\n'
                 'sensor_jsons = ["a.json"]\ndns_paths = ["a.npy"]\n')

    assert training_time_strides_from_config(c) == []


def test_pressure_is_opt_in_and_flows_into_the_context(monkeypatch):
    """`pressure_evaluated` 是 EvaluationContext 的欄位；它必須反映實際載了什麼，
    否則 build_metric_artifact 會把 p_rel_err 的缺席判成 UNAVAILABLE 而非 NOT_APPLICABLE。"""
    n_dns, N, K = 16, 8, 3
    dns_t = (np.arange(n_dns) * 0.25).astype(np.float64)
    dns_u = np.arange(n_dns * N * N, dtype=np.float64).reshape(n_dns, N, N)

    def _sensors(_p, time_stride=1):
        return {"sensor_vals": np.zeros((len(dns_t[::4]), K, 2)),
                "sensor_pos": np.zeros((K, 2)), "sensor_time": dns_t[::4],
                "norm_stats": {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0}}

    def _dns(_p, time_stride=1, return_p=False):
        if return_p:
            return dns_u, dns_u + 0.5, dns_u + 1.5, dns_t
        return dns_u, dns_u + 0.5, dns_t

    monkeypatch.setattr("pi_lnn_jax.data.load_sensors_from_path", _sensors)
    monkeypatch.setattr("pi_lnn_jax.data.load_dns_from_path", _dns)
    p = resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[1])

    off = load_for_evaluation("s", "d", protocol=p)
    on = load_for_evaluation("s", "d", protocol=p, with_pressure=True)

    assert off.dns_p_eval is None and off.context.pressure_evaluated is False
    assert on.dns_p_eval is not None and on.context.pressure_evaluated is True
    assert on.dns_p_eval.shape == on.dns_u_eval.shape
