"""Tests for pi_lnn_jax.ckpt.restore_eval_params — eval 端 restore 的唯一入口。

What: 驗證 fail-fast 契約——缺目錄、空目錄、指定不存在的 step、參數樹與 config
      重建樹錯配，四種情況都必須 raise 而非回退。**外加建構指紋閘門**：
      參數樹比對看不見「改 forward 卻不改樹」的旗標，那一維由指紋負責。

Why: 這段 restore 三步舞原本在 evaluate_exp245 / evaluate_multi_re / dump_cp_fields
     各寫一次，其中 dump_cp_fields 漏了 `verify_params_tree`、也沒檢查
     latest_step() 回 None。漏掉的後果不是 crash 而是「安靜用錯的 params 產圖」。
     收斂成單一 helper 後，fail-fast 就不再是各檔自律的問題。

指紋閘門的三條路徑各自要有測試，因為它們的失敗模式完全不同：
  相符 → 通過並標記 verified；不符 → 無條件 raise 且點名欄位；
  無指紋（舊 ckpt，資訊真的不存在）→ 警告 + 蓋印記，strict 下才失敗。
第三條最容易退化成裝飾性——所以 `test_strict_env_turns_missing_into_failure`
與「印記真的出現在回傳值裡」兩條是本檔的承重面。
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_lnn_jax.ckpt import (  # noqa: E402
    FINGERPRINT_STRICT_ENV,
    CheckpointManager,
    TrainState,
    reference_params_for,
    restore_eval_params,
)
from pi_lnn_jax.model_factory import build_model  # noqa: E402
from pi_lnn_jax.model_factory import model_fingerprint  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="orbax.*")


@dataclasses.dataclass(frozen=True)
class _FakeModel:
    """指紋只讀 dataclass 欄位，所以不必付真 LiquidOperator 的建構成本。"""
    width: int = 4
    disable_cross_attention: bool = False


_MODEL = _FakeModel()


def _params(scale: float = 1.0):
    return {"dense": {"kernel": jnp.ones((3, 4)) * scale, "bias": jnp.zeros((4,))}}


def _save(ckpt_dir: Path, step: int, params) -> None:
    mgr = CheckpointManager(directory=ckpt_dir, max_to_keep=3, save_interval_steps=1)
    state = TrainState(
        params=params,
        opt_state=(),
        step=jnp.asarray(step, jnp.int32),
        rng_key=jax.random.PRNGKey(0),
    )
    mgr.save(step, state, force=True)


def _write_summary(ckpt_dir: Path, model=None, *, ckpt_dir_override=None) -> Path:
    """在 artifacts_dir（= ckpt_dir.parent）落一份 summary.json。"""
    summary = {"ckpt_dir": str(ckpt_dir_override or ckpt_dir)}
    if model is not None:
        summary["model_construction"] = model_fingerprint(model)
    p = ckpt_dir.parent / "summary.json"
    p.write_text(json.dumps(summary))
    return p


# ─── fail-fast 契約（原有） ────────────────────────────────────────────────

def test_restores_latest_and_reports_step():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _save(d, 300, _params())
        _write_summary(d, _MODEL)
        params, step, prov = restore_eval_params(
            d, "latest", reference_params=_params(), model=_MODEL)
        assert step == 300
        assert params["dense"]["kernel"].shape == (3, 4)
        assert prov["fingerprint_verified"] is True


def test_restores_explicit_step():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _save(d, 300, _params())
        _write_summary(d, _MODEL)
        _, step, _ = restore_eval_params(
            d, 100, reference_params=_params(), model=_MODEL)
        assert step == 100


def test_missing_directory_raises():
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "nope" / "checkpoints"
        with pytest.raises(FileNotFoundError, match="checkpoint 目錄不存在"):
            restore_eval_params(missing, "latest", reference_params=_params(), model=_MODEL)


def test_empty_directory_raises_instead_of_returning_none():
    """空目錄時 latest_step() 回 None——不得帶著 None 往下走。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        d.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="無 ckpt 可 restore"):
            restore_eval_params(d, "latest", reference_params=_params(), model=_MODEL)


def test_nonexistent_explicit_step_raises():
    """指定的 step 不在盤上時要點名它，不要默默換成最近的一個。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        with pytest.raises(FileNotFoundError, match="step=999"):
            restore_eval_params(d, 999, reference_params=_params(), model=_MODEL)


def test_params_tree_mismatch_raises():
    """config/ckpt 錯配是本專案踩過的坑：必須擋在算數字之前。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        wrong = {"dense": {"kernel": jnp.ones((3, 8)), "bias": jnp.zeros((8,))}}
        with pytest.raises(ValueError, match="ckpt 參數樹與 config 重建的模型不一致"):
            restore_eval_params(d, "latest", reference_params=wrong, model=_MODEL)


# ─── 建構指紋閘門 ──────────────────────────────────────────────────────────

def test_fingerprint_mismatch_raises_and_names_the_field():
    """參數樹相同、建構不同——這正是 verify_params_tree 放行的那一類。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _write_summary(d, _FakeModel(disable_cross_attention=False))

        with pytest.raises(ValueError, match="disable_cross_attention") as exc:
            restore_eval_params(d, "latest", reference_params=_params(),
                                model=_FakeModel(disable_cross_attention=True))
        assert "安靜給出錯誤數字" in str(exc.value)


def test_missing_summary_is_flagged_not_silently_passed():
    """舊 ckpt：資訊真的不存在，只能標記——但必須標記，不得靜默放行。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _, _, prov = restore_eval_params(
            d, "latest", reference_params=_params(), model=_MODEL)
        assert prov["fingerprint_verified"] is False
        assert "summary.json" in prov["reason"]
        # 印記必須帶著 eval 端的建構值，否則事後無從判斷當時用了什麼。
        assert prov["eval_construction"]["disable_cross_attention"] == "False"


def test_summary_without_model_construction_is_flagged():
    """指紋機制之前的 run：summary 在，欄位不在。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _write_summary(d, None)
        _, _, prov = restore_eval_params(
            d, "latest", reference_params=_params(), model=_MODEL)
        assert prov["fingerprint_verified"] is False
        assert "model_construction" in prov["reason"]


def test_strict_env_turns_missing_into_failure(monkeypatch):
    """弱路徑的防線。沒有這一條，驗收 job 會在無指紋下報綠。"""
    monkeypatch.setenv(FINGERPRINT_STRICT_ENV, "1")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        with pytest.raises(ValueError, match=FINGERPRINT_STRICT_ENV):
            restore_eval_params(d, "latest", reference_params=_params(), model=_MODEL)


def test_summary_describing_another_ckpt_raises():
    """artifacts_dir 未對齊是本專案記錄有案的失敗模式。

    讀到**別人的** summary 比讀不到更糟：它會讓指紋比對通過，給出假綠。
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        _write_summary(d, _MODEL, ckpt_dir_override=Path(td) / "somewhere_else")
        with pytest.raises(ValueError, match="artifacts_dir 未對齊"):
            restore_eval_params(d, "latest", reference_params=_params(), model=_MODEL)


def test_corrupt_summary_raises_rather_than_degrading_to_unverified():
    """壞掉的 summary 不得被偽裝成「舊 ckpt」——那會把解析失敗變成弱路徑。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        (d.parent / "summary.json").write_text("{ not json")
        with pytest.raises(ValueError, match="summary.json 無法解析"):
            restore_eval_params(d, "latest", reference_params=_params(), model=_MODEL)


def test_model_parameter_is_required():
    """可選的閘門＝裝飾性閘門。少傳 model 必須是 TypeError，不是靜默跳過。"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "checkpoints"
        _save(d, 100, _params())
        with pytest.raises(TypeError):
            restore_eval_params(d, "latest", reference_params=_params())


# ─────────────────────────────────────────────────────────────────────────────
# reference_params_for —— 收斂四份 dummy-init 的 adapter
#
# 這一節是 adapter 的承重面。它宣稱「四個腳本歷史上各自的 dummy-init 差異都不
# 影響參數樹，只有 sensor 通道數影響」——那不是推論，是可測的，所以在這裡測。
# 沒有 test_channel_count_is_the_load_bearing_dimension 這條，前三條就只是
# 「兩段等價的程式碼互相同意」，證不出閘門還看得見東西。
# ─────────────────────────────────────────────────────────────────────────────

_RP_KWARGS = {
    "sensor_value_dim": 2,
    "d_time": 8,
    "domain_length": 6.283185307179586,
    "use_temporal_anchor": True,
    "T_total": 20.0,
    "temporal_anchor_harmonics": 2,
    "output_head_gain": 1.0,
    "fourier_embed_dim": 8,
    "query_mlp_hidden_dim": 8,
    "num_query_mlp_layers": 2,
    "operator_rank": 4,
    "d_model": 8,
    "num_spatial_encoder_layers": 1,
    "num_temporal_cfc_layers": 1,
    "token_attention_heads": 2,
}
_RP_T, _RP_K, _RP_C = 5, 6, 2


def _rp_sensors(channels: int = _RP_C):
    rs = np.random.RandomState(0)
    return (
        jnp.asarray(rs.uniform(size=(_RP_T, _RP_K, channels)).astype(np.float32)),
        jnp.asarray(rs.uniform(size=(_RP_K, 2)).astype(np.float32)),
        jnp.asarray(np.linspace(0.0, 1.0, _RP_T).astype(np.float32)),
    )


def _tree_index(tree):
    return {
        jax.tree_util.keystr(kp): tuple(getattr(leaf, "shape", ()))
        for kp, leaf in jax.tree_util.tree_leaves_with_path(tree)
    }


def _historical_dance(variant, model, sensor_vals, sensor_pos, sensor_time):
    """逐字重演四個腳本各自的 dummy-init。

    差異刻意保留原樣（split / 不 split、re_norm 0.5 vs 讀 config、T_total 各自
    fallback），因為那正是要證明「不影響參數樹」的東西。
    """
    seed = 42
    T_total = float(_RP_KWARGS["T_total"])
    init_xy = jnp.asarray(
        np.random.RandomState(seed).uniform(0, 1, (8, 2)).astype(np.float32))
    init_t = jnp.asarray(
        np.random.RandomState(seed).uniform(0, T_total, (8,)).astype(np.float32))

    if variant == "evaluate_exp245":          # split 一次，用 rng_init
        _, rng = jax.random.split(jax.random.PRNGKey(seed))
        re_norm = 1.0
    elif variant == "evaluate_multi_re":      # 不 split，re_norm 寫死 0.5
        rng = jax.random.PRNGKey(seed)
        re_norm = 0.5
    elif variant == "diag_trainable_fourier":  # split[1]
        rng = jax.random.split(jax.random.PRNGKey(seed))[1]
        re_norm = 1.0
    elif variant == "dump_cp_fields":          # 不 split，re_norm 由寫死的 scale 算
        rng = jax.random.PRNGKey(seed)
        re_norm = float(np.log(10000.0) / np.log(10000.0))
    else:
        raise AssertionError(variant)

    return model.init(
        rng, sensor_vals, sensor_pos, re_norm, sensor_time, init_xy, init_t)


@pytest.mark.parametrize("variant", [
    "evaluate_exp245", "evaluate_multi_re", "diag_trainable_fourier", "dump_cp_fields",
])
def test_reference_params_reproduces_each_historical_dance(variant):
    """四個呼叫端原本各自的 dummy-init 與 adapter 產出同一棵樹（path + shape）。

    這是遷移的驗收條件：樹不同 → restore 端的 verify_params_tree 會炸。
    """
    sv, sp, st = _rp_sensors()
    model, _ = build_model("liquid", dict(_RP_KWARGS), K_sensors=_RP_K)
    assert _tree_index(reference_params_for(model, sv, sp, st)) == \
        _tree_index(_historical_dance(variant, model, sv, sp, st))


@pytest.mark.parametrize("arch", ["liquid", "vanilla", "pinn"])
def test_reference_params_works_for_every_arch(arch):
    """三個 operator 的 __call__ 同形，adapter 不該只對 B3 成立。"""
    sv, sp, st = _rp_sensors()
    model, _ = build_model(arch, dict(_RP_KWARGS), K_sensors=_RP_K)
    assert _tree_index(reference_params_for(model, sv, sp, st))


def test_channel_count_is_the_load_bearing_dimension():
    """鑑別力：唯一該改變參數樹的維度，真的會改變它。

    沒有這條，上面的等價性測試證不出閘門還看得見任何東西。
    """
    model2, _ = build_model(
        "liquid", dict(_RP_KWARGS, sensor_value_dim=2), K_sensors=_RP_K)
    model3, _ = build_model(
        "liquid", dict(_RP_KWARGS, sensor_value_dim=3), K_sensors=_RP_K)
    t2 = _tree_index(reference_params_for(model2, *_rp_sensors(channels=2)))
    t3 = _tree_index(reference_params_for(model3, *_rp_sensors(channels=3)))
    assert t2 != t3


def test_query_count_does_not_change_the_tree():
    """docstring 宣稱查詢點不影響參數樹——那是 adapter 敢固定 xy/t_q 的理由。

    這條若紅，代表某個 arch 讓查詢維度進了 param 形狀，adapter 的固定值就不再
    安全（而 verify_params_tree 看不見這件事，所以只能在這裡擋）。
    """
    sv, sp, st = _rp_sensors()
    model, _ = build_model("liquid", dict(_RP_KWARGS), K_sensors=_RP_K)
    ref = _tree_index(reference_params_for(model, sv, sp, st))
    for n_query in (1, 3, 64):
        xy = jnp.zeros((n_query, 2), jnp.float32)
        t_q = jnp.broadcast_to(st[:1], (n_query,)).astype(jnp.float32)
        other = model.init(jax.random.PRNGKey(0), sv, sp, 0.0, st, xy, t_q)
        assert _tree_index(other) == ref, f"n_query={n_query} 改變了參數樹"
