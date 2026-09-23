"""建構指紋：補上 `verify_params_tree` 看不見的那一維。

問題（架構深化候選 A）：`ckpt.verify_params_tree` 是 eval 端唯一真正接上的閘門，
但它只比 path + shape。**改 forward 卻不改參數樹**的旗標因此完全隱形——
訓練時開、eval 時關（或反過來），restore 乾乾淨淨通過，數字是錯的。

這不是假想：`configs/exp_245_b1_les_T50.toml`（B1 ablation 臂）就設了
`disable_cross_attention = true`。

本檔最承重的一條是 `test_fingerprint_catches_what_the_params_tree_cannot`——
它先證明 `verify_params_tree` 放行，再證明指紋擋下。少了前半段，這個測試
就只是在說「兩個不同的 dict 不相等」，那不構成任何背書。
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.ckpt import verify_params_tree
from pi_lnn_jax.model_factory import (
    build_model,
    fingerprint_diff,
    model_fingerprint,
)
from pi_lnn_jax.models import LiquidOperator

_BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=0,
    query_mlp_hidden_dim=16, operator_rank=8, decoder_attention_heads=1,
    use_temporal_anchor=True, T_total=1.0, temporal_anchor_harmonics=2,
    domain_length=1.0,
)
_T, _K, _N = 5, 6, 4


def _inputs():
    rng = jax.random.PRNGKey(0)
    return (jnp.zeros((_T, _K, 2)), jax.random.uniform(rng, (_K, 2)), 0.5,
            jnp.linspace(0, 1, _T), jax.random.uniform(rng, (_N, 2)), jnp.zeros((_N,)))


def _built(**over):
    model = LiquidOperator(**{**_BASE, **over})
    params = model.init(jax.random.PRNGKey(42), *_inputs())
    return model, params


def test_fingerprint_catches_what_the_params_tree_cannot():
    """本檔的存在理由。兩段缺一不可。"""
    m_off, p_off = _built(disable_cross_attention=False)
    m_on, p_on = _built(disable_cross_attention=True)

    # (1) 先證明現有閘門放行——否則下面那段不代表任何事。
    n = verify_params_tree(p_off, p_on)
    assert n > 0, "verify_params_tree 未比對任何 leaf，前提不成立"

    # 而 forward 確實不同：這才是「放行」之所以危險。
    out_off = m_off.apply(p_off, *_inputs())
    out_on = m_on.apply(p_off, *_inputs())      # 同一份 params，兩個 flag
    drift = float(jnp.abs(out_off - out_on).max())
    assert drift > 1e-3, f"兩個 flag 的 forward 沒有實質差異（drift={drift:.2e}），前提不成立"

    # (2) 指紋擋下，且點名了是哪個欄位。
    problems = fingerprint_diff(model_fingerprint(m_off), model_fingerprint(m_on))
    assert problems, "指紋未偵測到差異——它沒有補上盲點"
    assert any("disable_cross_attention" in p for p in problems), (
        f"指紋偵測到差異但沒點名欄位，診斷價值為零：{problems}")


@pytest.mark.parametrize("flag,value", [
    ("domain_length", 2.0),
    ("T_total", 9.0),
    ("cfc_tau_mod_scale", 1.5),
    ("use_sdf_features", True),
    ("body_radius", 0.5),
])
def test_other_tree_invisible_flags_are_covered(flag, value):
    """盲點不只一個。這些旗標同樣改 forward 而不改樹。"""
    a, b = LiquidOperator(**_BASE), LiquidOperator(**{**_BASE, flag: value})
    assert any(flag in p for p in
               fingerprint_diff(model_fingerprint(a), model_fingerprint(b)))


def test_identical_construction_has_no_diff():
    """相同建構不得有差異——否則閘門會擋下每一次正當的 eval。"""
    assert fingerprint_diff(model_fingerprint(LiquidOperator(**_BASE)),
                            model_fingerprint(LiquidOperator(**_BASE))) == []


@pytest.mark.parametrize("arch", ["liquid", "vanilla", "pinn"])
def test_fingerprint_covers_every_arch(arch):
    """三個 arch 都要有指紋，且反映**實際建構**而非傳入的 config。

    `build_model` 對 vanilla/pinn 會用 COMMON_KEYS 濾掉一半 kwargs；
    指紋取自實例，所以濾掉的東西不該出現在裡面。
    """
    kwargs = {**_BASE, "disable_cross_attention": True}
    model, _ = build_model(arch, dict(kwargs), K_sensors=_K)
    fp = model_fingerprint(model)

    assert fp, f"{arch} 的指紋是空的"
    assert all(isinstance(v, str) for v in fp.values()), "指紋值必須全為字串（JSON-safe）"

    expected = {f.name for f in dataclasses.fields(model)} - {"parent", "name"}
    # 方向1 opt-in：mid-band 分支關閉（wavenumbers=()）時，model_fingerprint 刻意省略這兩欄，
    # 讓 flag-off 指紋與「從未有此功能」的模型相同（保 §7.1 A/B 與既有 ckpt 契約）。
    # 完整性守衛照樣抓其他意外漏欄位；此處只登錄這個刻意例外。
    if getattr(model, "mid_band_wavenumbers", ()) == ():
        expected -= {"mid_band_wavenumbers", "mid_band_embed_dim", "mid_band_init_sigma"}
    if not getattr(model, "use_trainable_fourier", False):
        expected -= {"use_trainable_fourier", "trainable_fourier_dim",
                     "trainable_fourier_init_scale", "trainable_fourier_lowpass_kc"}
    if not getattr(model, "drag_alpha", 0.0):
        expected -= {"drag_alpha"}
    if not (getattr(model, "mmlp_branch_u", False)
            or getattr(model, "mmlp_gate_branch", False)):
        expected -= {"mmlp_branch_u", "mmlp_gate_branch"}
    if getattr(model, "relpos_bias_norm", "layernorm") == "layernorm":
        expected -= {"relpos_bias_norm"}
    if not getattr(model, "relpos_bias_zero_init", False):
        expected -= {"relpos_bias_zero_init"}
    assert set(fp) == expected, f"{arch} 的指紋欄位與 dataclass 不符"

    if arch != "liquid":
        assert "disable_cross_attention" not in fp, (
            f"{arch} 的建構子不吃這個 flag，指紋卻記了它——那不是實際建構")


def test_fingerprint_is_json_round_trippable():
    """指紋要進 summary.json；round-trip 後比對出假差異會讓閘門變成噪音源。"""
    import json
    fp = model_fingerprint(LiquidOperator(**_BASE))
    assert fingerprint_diff(fp, json.loads(json.dumps(fp))) == []


def test_diff_reports_missing_and_extra_fields():
    """欄位增減也要點名——換 arch 評估舊 ckpt 是真實會發生的錯誤。"""
    problems = fingerprint_diff({"a": "1", "b": "2"}, {"a": "1", "c": "3"})
    assert any("b" in p and "eval 無此欄位" in p for p in problems)
    assert any("c" in p and "訓練時無此欄位" in p for p in problems)
