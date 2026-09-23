"""Tests for pi_lnn_jax.model_factory — arch → Flax Module 的唯一 factory。

What: 驗證三個 arch（liquid / vanilla / pinn）的 kwargs 過濾規則、model_name
      標籤，以及「訓練端與 eval 端共用同一個 factory 物件」這件事本身。

Why: 此 factory 原本在 train_kolmogorov.py 與 scripts/evaluate_exp245.py 各存
     一份逐行複製（後者註解自承「對齊 train_kolmogorov.py _build_model」）。
     兩份漂移不會 crash，只會讓 eval 拿「另一個架構」去評 checkpoint，安靜產出
     錯誤的 headline 數字。test_single_factory_shared_by_train_and_eval 就是防
     這條回歸：任何一端改回自己的 local factory，此測試立刻紅。
"""
from __future__ import annotations

import sys

import pytest
from _paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_lnn_jax.model_factory import build_model  # noqa: E402
from pi_lnn_jax.models import (  # noqa: E402
    LiquidOperator,
    StandardPINNOperator,
    VanillaDeepONetOperator,
)


# LiquidOperator 專用 key（d_model / token_attention_heads / 層數）刻意混入，
# 用來驗證 vanilla / pinn 路徑會把它們濾掉——不濾就 TypeError。
MODEL_KWARGS = {
    "sensor_value_dim": 2,
    "d_time": 16,
    "domain_length": 6.283185307179586,
    "use_temporal_anchor": True,
    "T_total": 20.0,
    "temporal_anchor_harmonics": 3,
    "output_head_gain": 1.0,
    "fourier_embed_dim": 32,
    "query_mlp_hidden_dim": 128,
    "num_query_mlp_layers": 4,
    "operator_rank": 96,
    # 以下為 liquid 專用，vanilla / pinn constructor 不吃
    "d_model": 64,
    "num_spatial_encoder_layers": 2,
    "num_temporal_cfc_layers": 2,
    "token_attention_heads": 4,
}


def test_liquid_passes_kwargs_through():
    model, name = build_model("liquid", dict(MODEL_KWARGS), K_sensors=100)
    assert isinstance(model, LiquidOperator)
    assert model.d_model == 64
    assert "B3" in name


def test_liquid_mid_band_wavenumbers_list_becomes_tuple():
    """config 傳 list（TOML array）→ build_model 轉成 tuple（Flax module field 需 hashable）。"""
    kwargs = {**MODEL_KWARGS, "mid_band_wavenumbers": [6.0, 8.0, 10.0], "mid_band_embed_dim": 16}
    model, _ = build_model("liquid", dict(kwargs), K_sensors=100)
    assert isinstance(model.mid_band_wavenumbers, tuple), "list 未轉 tuple → Flax 會在 jit 時炸"
    assert model.mid_band_wavenumbers == (6.0, 8.0, 10.0)


def test_liquid_mid_band_init_sigma_plumbs_through():
    """mid_band_init_sigma 可從 config 設（高頻段需調小 σ 避免二階導爆炸）。"""
    kwargs = {**MODEL_KWARGS, "mid_band_wavenumbers": [6.0, 8.0], "mid_band_init_sigma": 0.5}
    model, _ = build_model("liquid", dict(kwargs), K_sensors=100)
    assert model.mid_band_init_sigma == 0.5


def test_vanilla_filters_liquid_only_kwargs():
    model, name = build_model("vanilla", dict(MODEL_KWARGS), K_sensors=100)
    assert isinstance(model, VanillaDeepONetOperator)
    assert model.K_sensors == 100
    # query_mlp_* 映射到 vanilla 自己的欄位名
    assert model.hidden_dim == MODEL_KWARGS["query_mlp_hidden_dim"]
    assert model.operator_rank == MODEL_KWARGS["operator_rank"]
    assert model.num_branch_layers == MODEL_KWARGS["num_query_mlp_layers"]
    assert model.num_trunk_layers == MODEL_KWARGS["num_query_mlp_layers"]
    assert "B0" in name and "K=100" in name


def test_pinn_drops_sensor_value_dim():
    model, name = build_model("pinn", dict(MODEL_KWARGS), K_sensors=100)
    assert isinstance(model, StandardPINNOperator)
    assert not hasattr(model, "sensor_value_dim")
    assert model.hidden_dim == MODEL_KWARGS["query_mlp_hidden_dim"]
    assert model.num_layers == MODEL_KWARGS["num_query_mlp_layers"]
    assert "B2" in name


@pytest.mark.parametrize(
    "given, expected",
    [
        (4, 4),        # 正常值原樣沿用
        (0, 6),        # falsy → 退回 pinn 預設 6 層（`or 6`）
        (None, 6),     # 同上
        (-3, 1),       # 負值被 max(1, ...) 夾到 1
    ],
)
def test_pinn_num_layers_resolution(given, expected):
    """釘住 `max(1, get(...) or 6)` 的實際語意（此為既有行為，非新設計）。"""
    kwargs = dict(MODEL_KWARGS, num_query_mlp_layers=given)
    model, _ = build_model("pinn", kwargs, K_sensors=100)
    assert model.num_layers == expected


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("does_not_exist", dict(MODEL_KWARGS), K_sensors=100)


def test_build_model_does_not_mutate_caller_kwargs():
    """factory 內部對 pinn 做 pop("sensor_value_dim")，不得污染呼叫端 dict。"""
    kwargs = dict(MODEL_KWARGS)
    build_model("pinn", kwargs, K_sensors=100)
    assert kwargs == MODEL_KWARGS


def test_single_factory_shared_by_train_and_eval():
    """反漂移哨兵：訓練端與 eval 端必須指向同一個 factory 物件，且不得自建分身。

    這是本檔存在的主要理由——兩份複製品曾經共存，且漂移不會 crash。
    訓練端的 build_model 參照已隨建構期搬進 pi_lnn_jax.pipeline.kolmogorov.assembly。
    """
    from pi_lnn_jax.pipeline.kolmogorov import assembly as train_assembly

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import evaluate_exp245

    for mod in (train_assembly, evaluate_exp245):
        assert mod.build_model is build_model, (
            f"{mod.__name__} 的 build_model 不是 pi_lnn_jax.model_factory 那一個"
        )
        # 曾經的重複實作叫 _build_model；任一端重新長回 local factory 就紅
        assert not hasattr(mod, "_build_model"), (
            f"{mod.__name__} 又出現 local _build_model — factory 必須唯一"
        )
