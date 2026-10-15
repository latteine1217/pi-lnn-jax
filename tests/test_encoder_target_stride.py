"""encoder 輸入與監督目標的時間 stride 拆開——K–Δt 平面 2×2 消融的前提。

Why 這組測試存在：
    `spg` 重跑批量到「比部署節奏更密一倍，K=400 買到 −3.00%」，而七項隱藏機制都
    排除之後，剩下兩個不是 bug 的解釋分不開：
      (a) 監督量翻倍——0.025 臂的 sensor loss 母體 T×K 大一倍；
      (b) encoder 看到的幀翻倍——CfC 多積分一倍的步數、也多看一倍的輸入。
    原本 encoder 輸入與監督目標是**同一個** `sensor_vals`，兩者只能一起變。這裡釘住
    把它們拆開的契約：

      * `ReBatch.sensor_vals/sensor_time` 永遠是**監督目標**——母體 `T*K` 在
        run.py／assembly.py 四處各自從它推，所以它們自動正確，不必逐處改。
      * encoder 的輸入走 `encoder_context(rb)`：有覆寫用覆寫，否則回 `sensor_vals`。
      * 兩個 stride 預設都是 1 → 不產生覆寫欄位 → 圖與舊版逐字相同。

    「預設路徑逐位元不變」的最終背書是 lab-server 的 CPU A/B 對拍（CLAUDE.md §7.1），
    這裡只能證明路由正確、以及覆寫欄位在預設下不存在。
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.pipeline.kolmogorov import assembly

K, T, N = 5, 9, 8


def _dataset() -> dict:
    rng = np.random.default_rng(0)
    return {
        "sensor_vals": rng.normal(size=(T, K, 3)).astype(np.float32),
        "sensor_pos": rng.uniform(size=(K, 2)).astype(np.float32),
        "sensor_time": np.linspace(0.0, 1.0, T).astype(np.float32),
        "re_norm": 0.1,
        "re_value": 1.0e4,
        "norm_stats": {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0,
                       "p_mean": 0.0, "p_std": 1.0},
    }


# ── ReBatch 與 encoder_context ────────────────────────────────────────────────

def test_default_rebatch_has_no_encoder_override():
    rb = assembly._make_re_batch(_dataset())
    assert rb.enc_vals is None
    assert rb.enc_time is None


def test_explicit_unit_strides_equal_default():
    """顯式給 1 必須與不給完全相同——不得因此產生覆寫欄位。"""
    a = assembly._make_re_batch(_dataset())
    b = assembly._make_re_batch(_dataset(), encoder_time_stride=1, target_time_stride=1)
    assert b.enc_vals is None and b.enc_time is None
    np.testing.assert_array_equal(np.asarray(a.sensor_vals), np.asarray(b.sensor_vals))
    np.testing.assert_array_equal(np.asarray(a.sensor_time), np.asarray(b.sensor_time))


def test_encoder_context_falls_back_to_supervision_arrays():
    """沒有覆寫時回傳的必須是**同一個物件**，不是拷貝——這是預設圖不變的前提。"""
    rb = assembly._make_re_batch(_dataset())
    v, t = assembly.encoder_context(rb)
    assert v is rb.sensor_vals
    assert t is rb.sensor_time


@pytest.mark.parametrize("enc,tgt", [(2, 1), (1, 2)])
def test_strides_split_encoder_input_from_supervision(enc, tgt):
    d = _dataset()
    rb = assembly._make_re_batch(d, encoder_time_stride=enc, target_time_stride=tgt)
    # 監督目標 = sensor_vals（母體從這裡推）
    np.testing.assert_array_equal(np.asarray(rb.sensor_vals), d["sensor_vals"][::tgt])
    np.testing.assert_array_equal(np.asarray(rb.sensor_time), d["sensor_time"][::tgt])
    # encoder 輸入 = 覆寫欄位
    v, t = assembly.encoder_context(rb)
    np.testing.assert_array_equal(np.asarray(v), d["sensor_vals"][::enc])
    np.testing.assert_array_equal(np.asarray(t), d["sensor_time"][::enc])


def test_non_nested_strides_are_rejected():
    """兩個 stride 都 ≠1 時兩條時間軸不互為子集，decoder 對齊的語意就說不清——拒收。"""
    with pytest.raises(ValueError, match="其中一個必須是 1"):
        assembly._make_re_batch(_dataset(), encoder_time_stride=2, target_time_stride=3)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_stride_rejected(bad):
    with pytest.raises(ValueError):
        assembly._make_re_batch(_dataset(), encoder_time_stride=bad)


# ── loss_fn 與 GradNorm probe 都必須走 encoder_context ────────────────────────

def _loss_and_probe():
    model = LiquidOperator(**minimal_kwargs(3))
    d = _dataset()
    params = model.init(jax.random.PRNGKey(3), jnp.asarray(d["sensor_vals"]),
                        jnp.asarray(d["sensor_pos"]), 0.1, jnp.asarray(d["sensor_time"]),
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    ns_fn, poisson_fn = make_ns_residual_fn(model)
    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn, use_poisson=False, use_al=False, al_rho=0.0,
        w_poisson=0.0, T_total=1.0)
    probe = assembly._build_grad_norm_fn(model, ns_fn, T_total=1.0)
    colloc = (jax.random.uniform(jax.random.PRNGKey(4), (N,)),
              jax.random.uniform(jax.random.PRNGKey(5), (N,)),
              jax.random.uniform(jax.random.PRNGKey(6), (N,)))
    return model, params, loss_fn, probe, colloc, d


def _loss(loss_fn, params, colloc, rb):
    cx, cy, ct = colloc
    total, _ = loss_fn(params, cx, cy, ct, jnp.array([1.0, 0.1, 0.1]),
                       jnp.asarray(0.0), jnp.asarray(1.0), rb, 0.0)
    return float(total)


def test_explicit_override_equal_to_supervision_reproduces_default_loss():
    """把覆寫欄位填成與監督目標**同一組陣列**，loss 必須與預設逐位元相同——
    證明覆寫路徑只換了 encoder 的來源，沒改到別的東西。"""
    _, params, loss_fn, _, colloc, d = _loss_and_probe()
    rb = assembly._make_re_batch(d)
    same = rb._replace(enc_vals=rb.sensor_vals, enc_time=rb.sensor_time)
    assert _loss(loss_fn, params, colloc, rb) == _loss(loss_fn, params, colloc, same)


@pytest.mark.parametrize("enc,tgt", [(2, 1), (1, 2)])
def test_loss_changes_when_encoder_input_differs(enc, tgt):
    """覆寫真的有被用到：encoder 輸入或監督目標一變，loss 就變。"""
    _, params, loss_fn, _, colloc, d = _loss_and_probe()
    base = _loss(loss_fn, params, colloc, assembly._make_re_batch(d))
    split = _loss(loss_fn, params, colloc, assembly._make_re_batch(
        d, encoder_time_stride=enc, target_time_stride=tgt))
    assert np.isfinite(split)
    assert split != base


def test_loss_and_probe_both_route_through_encoder_context(monkeypatch):
    """loss 與 probe 量的必須是同一個東西（B-10 的教訓）：兩者都要從
    encoder_context 取 encoder 輸入，不得有一條還直接讀 sensor_vals。"""
    _, params, loss_fn, probe, colloc, d = _loss_and_probe()
    rb = assembly._make_re_batch(d, encoder_time_stride=2, target_time_stride=1)
    real = assembly.encoder_context
    seen = []

    def spy(re_batch):
        v, t = real(re_batch)
        seen.append(int(v.shape[0]))
        return v, t

    monkeypatch.setattr(assembly, "encoder_context", spy)
    _loss(loss_fn, params, colloc, rb)
    n_loss = len(seen)
    cx, cy, ct = colloc
    probe(params, cx, cy, ct, rb, 0.0)
    n_probe = len(seen) - n_loss
    assert n_loss >= 1, "loss_fn 沒有走 encoder_context"
    assert n_probe >= 1, "GradNorm probe 沒有走 encoder_context"
    assert set(seen) == {(T + 1) // 2}, f"encoder 看到的幀數不對：{seen}"


# ── 建構期守門 ────────────────────────────────────────────────────────────────

def _guard_kwargs(**over):
    kw = {"encoder_time_stride": 2, "target_time_stride": 1, "arch": "liquid",
          "n_datasets": 1, "sensor_cut_min_frac": 0.0, "autoreg_pseudo_dt": 0.0,
          "sensor_dropout_rate": 0.0, "use_continuous_re_physics": False,
          "refine_optimizer": "none"}
    kw.update(over)
    return kw


def test_guard_passes_for_supported_combination():
    assembly.check_encoder_target_strides(**_guard_kwargs())


def test_guard_is_silent_at_default_strides():
    """預設 (1,1) 時守門不得對任何其他設定有意見——否則等於改了既有 run 的行為。"""
    assembly.check_encoder_target_strides(**_guard_kwargs(
        encoder_time_stride=1, target_time_stride=1, arch="vanilla", n_datasets=3,
        sensor_cut_min_frac=0.5, autoreg_pseudo_dt=0.1, sensor_dropout_rate=0.2,
        use_continuous_re_physics=True, refine_optimizer="lbfgs"))


@pytest.mark.parametrize("over,msg", [
    ({"arch": "vanilla"}, "arch"),
    ({"n_datasets": 2}, "dataset"),
    ({"sensor_cut_min_frac": 0.3}, "sensor_cut_min_frac"),
    ({"autoreg_pseudo_dt": 0.1}, "autoreg_pseudo_dt"),
    ({"sensor_dropout_rate": 0.2}, "sensor_dropout_rate"),
    ({"use_continuous_re_physics": True}, "continuous_re_physics"),
    ({"refine_optimizer": "lbfgs"}, "refine"),
])
def test_guard_rejects_untested_combinations(over, msg):
    with pytest.raises(ValueError, match=msg):
        assembly.check_encoder_target_strides(**_guard_kwargs(**over))


# ── 設定鍵 ────────────────────────────────────────────────────────────────────

def test_config_defaults_are_unit_strides():
    from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy

    assert KolmogorovPolicy._defaults["curriculum.encoder_time_stride"] == 1
    assert KolmogorovPolicy._defaults["curriculum.target_time_stride"] == 1


def test_config_keys_resolve_from_toml(tmp_path):
    from pi_lnn_jax.pipeline.kolmogorov import resolve_inputs

    src = Path("configs/exp_spg_k400_st1_s42.toml").read_text()
    assert "[train]" in src
    patched = src.replace("[train]", "[train]\nencoder_time_stride = 2\ntarget_time_stride = 1", 1)
    p = tmp_path / "c.toml"
    p.write_text(patched)
    cur = resolve_inputs(["--config", str(p)]).config.curriculum
    assert cur.encoder_time_stride == 2
    assert cur.target_time_stride == 1
    base = resolve_inputs(["--config", "configs/exp_spg_k400_st1_s42.toml"]).config.curriculum
    assert base.encoder_time_stride == 1 and base.target_time_stride == 1


def test_eval_reads_encoder_stride_from_config(tmp_path):
    """evaluator 必須餵模型訓練時 encoder 看到的那個情境，而不是監督目標的情境。"""
    from pi_lnn_jax.evaluation_protocol import encoder_time_stride_from_config

    src = Path("configs/exp_spg_k400_st1_s42.toml").read_text()
    p = tmp_path / "c.toml"
    p.write_text(src.replace("[train]", "[train]\nencoder_time_stride = 2", 1))
    assert encoder_time_stride_from_config(p) == 2
    assert encoder_time_stride_from_config("configs/exp_spg_k400_st1_s42.toml") == 1
