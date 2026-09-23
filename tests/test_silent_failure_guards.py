"""把六條「靜默失效」變成大聲失敗 —— model-audit §5 的可修部分。

共同形狀：條件成立時什麼都不會發生（沒有錯誤、沒有警告），只是跑出來的東西
與 config 說的不是同一件事。六條的觸發條件目前全都不成立，所以對既有數字
零影響——它們是陷阱，不是活的 bug。

不在此檔的兩條，理由寫在各自的測試裡：
  * `dts[0]` 吃絕對時刻      -> 改了會動 §7.2 的數字，見 test_cfc_irregular_clock
  * vector 模式忽略 heads    -> 硬 raise 會打死 tab:cyl_main 的重現路徑（見下）
"""
from __future__ import annotations

import pytest

from pi_lnn_jax.pipeline.kolmogorov.assembly import (
    _assert_declared_npz_matches,
    assert_soap_betas_are_used,
)
from pi_lnn_jax.pipeline.kolmogorov.config import KolmogorovPolicy
from pi_lnn_jax.pipeline.kolmogorov.run import assert_al_statics_match

_DEFAULT_BETAS = (KolmogorovPolicy._defaults["run.soap_b1"],
                  KolmogorovPolicy._defaults["run.soap_b2"])


# ── 1. AL 的靜態超參被 ckpt 蓋掉 config ───────────────────────────────────

class _AL:
    def __init__(self, rho, clip):
        self.rho, self.lambda_clip = rho, clip


class _Loss:
    def __init__(self, rho, clip):
        self.al_rho, self.al_lambda_clip = rho, clip


def test_al_statics_matching_the_config_pass_through():
    assert_al_statics_match(_AL(0.1, 10.0), _Loss(0.1, 10.0))
    assert_al_statics_match(None, _Loss(0.1, 10.0))          # 沒有 ckpt state


@pytest.mark.parametrize("ckpt,cfg,needle", [
    ((0.1, 10.0), (1.0, 10.0), "al_rho"),
    ((0.1, 10.0), (0.1, 5.0), "al_lambda_clip"),
])
def test_al_statics_drifting_from_the_config_fail_loudly(ckpt, cfg, needle):
    """改了 al_rho 再 resume，跑的是 ckpt 的舊值而 summary.json 記的是 config 的新值。

    ρ 與 Λmax 是 `ALState` 的 NamedTuple 欄位 = pytree leaf，orbax 會序列化並
    還原它們。這條路沒有任何地方會發現不一致（appendix07 的 μ sweep 若有任何
    一臂是從別的 μ 的 ckpt resume 出來的，標示的 μ 就與實跑不符）。
    """
    with pytest.raises(ValueError, match=needle):
        assert_al_statics_match(_AL(*ckpt), _Loss(*cfg))


# ── 2. 宣告了 SOAP betas 卻拿到純 Adam ────────────────────────────────────

def test_declared_soap_betas_without_soap_fail_loudly():
    """主線 config 的 soap_betas=[0.9, 0.999] 在沒帶 CLI 旗標時被完全忽略。

    SOAP + Schedule-Free 只存在於 sbatch 的旗標，TOML 沒有任何鍵能開它。照
    `CLAUDE.md` §5 的指令重現的人會靜默拿到純 Adam。
    """
    with pytest.raises(ValueError, match="都不會生效"):
        assert_soap_betas_are_used(0.9, 0.999, "adam", "adam")


def test_soap_betas_pass_when_soap_is_actually_used():
    assert_soap_betas_are_used(0.9, 0.999, "schedule_free", "soap")
    assert_soap_betas_are_used(0.9, 0.999, "soap", "adam")


def test_base_soap_alone_does_not_excuse_an_adam_outer():
    """`--base_optimizer soap` 配 `--optimizer adam` 不算開了 SOAP。

    `build_optimizer` 的 adam 路徑根本不讀 `base_optimizer`，betas 一樣被丟——
    那正是這道守衛要擋的那一類。放行條件先前寫成 `"soap" in (optimizer, base)`，
    會把這個組合放過去（2026-09-10 獨立複查）。
    """
    with pytest.raises(ValueError, match="都不會生效"):
        assert_soap_betas_are_used(0.9, 0.999, "adam", "soap")


def test_untouched_betas_are_not_flagged():
    """沒有任何 TOML/CLI 宣告過 betas 時不該擋——那是「沒設」不是「設了被忽略」。"""
    assert_soap_betas_are_used(*_DEFAULT_BETAS, "adam", "adam")


# ── 6. sensor_npzs 是純裝飾的設定鍵 ───────────────────────────────────────

def test_declared_npz_mismatch_fails_loudly():
    """把 `sensor_npzs` 指到別的檔完全沒效果——實際檔案由 JSON 的 meta 決定。

    危險在於 provenance 會忠實記錄那個從未被開啟的路徑。
    """
    with pytest.raises(ValueError, match="不參與解析"):
        _assert_declared_npz_matches(
            ["data/sensors/noisy.npz"], {"npz_path": "/pool/clean.npz"}, "s.json")


def test_declared_npz_matching_or_absent_passes():
    _assert_declared_npz_matches([], {"npz_path": "/pool/clean.npz"}, "s.json")
    _assert_declared_npz_matches(
        ["data/sensors/clean.npz"], {"npz_path": "/pool/clean.npz"}, "s.json")


# ── 4. vector 模式忽略 decoder_attention_heads（tripwire，不 raise）────────

def test_vector_attention_ignores_the_head_count_today():
    """`attention_kind="vector"` 下 `decoder_attention_heads` 完全沒有作用。

    **釘住而非 raise**：主線 cylinder pipeline 走 scalar 不受影響，但 legacy 的
    `train_cylinder_v1.py`（產生 `tab:cyl_main` 的**唯一**路徑）是
    `--mode geo --attention_kind vector` 且 CFG 寫 `decoder_attention_heads=4`。
    硬擋會打死那個已發表數字的重現路徑，代價高於缺陷本身。

    修好的那天本測試會變紅，提醒一併檢查 cylinder 的重現路徑與 model fingerprint。
    """
    import jax
    import jax.numpy as jnp
    from _minimal_model import minimal_kwargs

    from pi_lnn_jax.models import LiquidOperator

    K, T, N = 5, 4, 6
    outs = []
    for heads in (1, 2):
        kw = dict(minimal_kwargs(3))
        kw["attention_kind"] = "vector"
        kw["decoder_attention_heads"] = heads
        model = LiquidOperator(**kw)
        sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
        st = jnp.linspace(0.0, 1.0, T)
        sv = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
        xy = jax.random.uniform(jax.random.PRNGKey(9), (N, 2))
        tq = jnp.linspace(0.1, 0.9, N)
        p = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st, xy, tq)
        outs.append(model.apply(p, sv, sp, 0.1, st, xy, tq))

    assert jnp.array_equal(outs[0], outs[1]), (
        "vector 模式現在會用 heads 了——若這是刻意修正，請檢查 legacy cylinder 的"
        "重現路徑（--attention_kind vector + heads=4）與 model fingerprint，"
        "並更新 model-audit 的該條")


# ── 5. sensor 三件組的形狀契約 ────────────────────────────────────────────

def _write_sensor_pair(tmp_path, *, k_decl=4, k_data=4, n_time=6, n_frames=6):
    """造一組 JSON + NPZ。四個參數分開給，才能製造 K 或 T 不一致。"""
    import json

    import numpy as np

    npz = tmp_path / "vals.npz"
    rng = np.random.default_rng(0)
    np.savez(npz,
             time=np.linspace(0.0, 1.0, n_time).astype(np.float32),
             u=rng.normal(size=(k_data, n_frames)).astype(np.float32),
             v=rng.normal(size=(k_data, n_frames)).astype(np.float32))
    js = tmp_path / "sensors.json"
    js.write_text(json.dumps({
        "K": k_decl,
        "selected_coordinates": rng.uniform(size=(k_decl, 2)).tolist(),
        "dns_values_npz": npz.name,
        "channels": ["u", "v"],
    }))
    return js


def test_wellformed_sensor_pair_loads(tmp_path):
    from pi_lnn_jax.data import load_sensors_from_path

    d = load_sensors_from_path(_write_sensor_pair(tmp_path), time_stride=1)
    assert d["sensor_vals"].shape == (6, 4, 2)


def test_sensor_count_mismatch_names_the_layer_it_broke_in(tmp_path):
    """JSON 宣告 K=4 而 NPZ 只有 3 個 sensor：要在資料層擋，不是等下游 broadcast 炸。

    原本只驗了 coords 是 (K,2)，channel 陣列的 K 完全沒驗。手工組裝或半途中斷的
    NPZ 會被載進來，直到訓練端 `broadcast_to((T,K,2))` 才失敗——而那個 JAX 錯誤
    指向的是模型不是資料。
    """
    from pi_lnn_jax.data import load_sensors_from_path

    js = _write_sensor_pair(tmp_path, k_decl=4, k_data=3)
    with pytest.raises(ValueError, match="個 sensor"):
        load_sensors_from_path(js, time_stride=1)


def test_frame_count_mismatch_is_caught(tmp_path):
    """同一個 NPZ 裡 channel 的幀數與 'time' 不一致——靜默切齊會餵錯時間軸。"""
    from pi_lnn_jax.data import load_sensors_from_path

    js = _write_sensor_pair(tmp_path, n_time=6, n_frames=5)
    with pytest.raises(ValueError, match="個時刻"):
        load_sensors_from_path(js, time_stride=1)


# ── RAR pool 太小 ────────────────────────────────────────────────────────────

def test_rar_pool_guard_catches_the_incompatible_defaults():
    """schema 預設 rar_pool_size=512 配主線 n_collo=1024 不相容——要當場擋下。

    不擋的話 `lax.top_k(pool=512, k=819)` 會在 rar_warmup 之後的第一個觸發步才炸，
    那時已經跑了幾百步。這正是 RAR 從未能在主線 config 上跑起來的原因。
    """
    from pi_lnn_jax.pipeline.kolmogorov.assembly import assert_rar_pool_is_large_enough

    with pytest.raises(ValueError, match=r"rar_pool_size=512 < top-k 需要的 819"):
        assert_rar_pool_is_large_enough(rar_freq=1, rar_pool_size=512, n_collo_end=1024)

    # RAR 關閉時不管 pool 多小都放行
    assert_rar_pool_is_large_enough(rar_freq=0, rar_pool_size=1, n_collo_end=1024)
    # pool 夠大就放行（8× 是 docstring 建議的下限）
    assert_rar_pool_is_large_enough(rar_freq=1, rar_pool_size=8192, n_collo_end=1024)
    # 邊界：恰好等於 n_top 要放行
    assert_rar_pool_is_large_enough(rar_freq=1, rar_pool_size=819, n_collo_end=1024)


def test_rar_pool_guard_tracks_the_exploration_ratio():
    """exploration_ratio 越高，需要的 top-k 越少 —— guard 必須跟著鬆。

    釘住這個鍵**真的被 guard 讀進去**，而不是收了參數就擱著。
    """
    from pi_lnn_jax.pipeline.kolmogorov.assembly import assert_rar_pool_is_large_enough

    # expl=0.2 → n_top=819；expl=0.5 → n_top=512。pool=600 只夠後者。
    with pytest.raises(ValueError, match="819"):
        assert_rar_pool_is_large_enough(1, 600, 1024, 0.2)
    assert_rar_pool_is_large_enough(1, 600, 1024, 0.5)


# ── 第三道閘門：反正規化常數 ─────────────────────────────────────────────────

def _write_summary(tmp_path, **extra):
    import json
    (tmp_path / "summary.json").write_text(json.dumps({"ckpt_dir": "x", **extra}))
    return tmp_path


def test_norm_stats_gate_accepts_matching_constants(tmp_path):
    from pi_lnn_jax.ckpt import verify_norm_stats
    ns = {"u_mean": -0.0155, "u_std": 0.4167, "v_mean": -0.0029, "v_std": 0.3059}
    _write_summary(tmp_path, norm_stats=[ns])
    out = verify_norm_stats(tmp_path, dict(ns))
    assert out["norm_stats_verified"] is True


def test_norm_stats_gate_raises_on_stride_style_mismatch(tmp_path):
    """實測的 stride-20 偏移（v_std 0.3059 → 0.2992，2.17%）必須擋下。

    參數樹閘門與建構指紋對它完全隱形——norm_stats 既不是參數也不是建構值。
    """
    from pi_lnn_jax.ckpt import verify_norm_stats
    trained = {"u_mean": -0.015539, "u_std": 0.416693,
               "v_mean": -0.002942, "v_std": 0.305857}
    at_stride_20 = {**trained, "u_std": 0.418413, "v_std": 0.299243}
    _write_summary(tmp_path, norm_stats=[trained])
    with pytest.raises(ValueError, match="time_stride"):
        verify_norm_stats(tmp_path, at_stride_20)


def test_norm_stats_gate_flags_old_ckpts_and_can_be_made_strict(tmp_path, monkeypatch):
    """2026-09-14 之前的 ckpt 沒有這個欄位，且無法回溯補上 —— 警告 + 蓋印記。"""
    from pi_lnn_jax.ckpt import NORM_STATS_STRICT_ENV, verify_norm_stats
    ns = {"u_mean": 0.0, "u_std": 1.0}
    _write_summary(tmp_path)                       # 沒有 norm_stats 欄位
    out = verify_norm_stats(tmp_path, ns)
    assert out["norm_stats_verified"] is False and out["eval_norm_stats"] == ns

    monkeypatch.setenv(NORM_STATS_STRICT_ENV, "1")
    with pytest.raises(ValueError, match=NORM_STATS_STRICT_ENV):
        verify_norm_stats(tmp_path, ns)


def test_eval_script_actually_calls_the_norm_stats_gate():
    """守衛被測得再紮實，沒有人呼叫它就是零效果（M18）。"""
    import ast
    from _paths import REPO_ROOT
    tree = ast.parse((REPO_ROOT / "scripts" / "evaluate_exp245.py").read_text())
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "verify_norm_stats" in called, (
        "evaluate_exp245.py 不再呼叫 verify_norm_stats——第三道閘門被摘掉了")
