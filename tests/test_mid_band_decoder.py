"""方向1 Increment A cycle 3：decoder opt-in band-targeted 分支的參數樹契約。

flag 關（mid_band_wavenumbers=()，預設）→ decoder 完全不實例化 band 分支 → 參數樹
與 baseline 無異（bit-identical seam）。
flag 開 → decoder 出現 band_spatial_emb 參數分支（trunk 取得 mid-band 容量）。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from pi_lnn_jax.model_factory import model_fingerprint
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


def _param_paths(**over) -> set[str]:
    model = LiquidOperator(**{**_BASE, **over})
    params = model.init(jax.random.PRNGKey(42), *_inputs())
    return {"/".join(map(str, k)) for k in flatten_dict(params).keys()}


def test_band_branch_only_when_flag_on():
    off = _param_paths()  # mid_band_wavenumbers=() 預設
    on = _param_paths(mid_band_wavenumbers=(6.0, 12.0), mid_band_embed_dim=16)
    assert not any("band_spatial_emb" in p for p in off), "flag-off 不應有 band 分支"
    assert any("band_spatial_emb" in p for p in on), "flag-on 應有 band_spatial_emb 分支"


def test_fingerprint_hides_mid_band_when_off():
    """opt-in 關閉時，指紋不含 mid_band 欄位 → 與加功能前的 baseline 指紋逐鍵相同
    （保 §7.1 A/B 與既有 ckpt 契約）。開啟時才出現，正確可辨為不同 arch。"""
    off = LiquidOperator(**_BASE)  # mid_band_wavenumbers=() 預設
    on = LiquidOperator(**{**_BASE, "mid_band_wavenumbers": (6.0, 12.0), "mid_band_embed_dim": 16})
    fp_off = model_fingerprint(off)
    fp_on = model_fingerprint(on)
    assert "mid_band_wavenumbers" not in fp_off
    assert "mid_band_embed_dim" not in fp_off
    assert fp_on["mid_band_wavenumbers"] == "(6.0, 12.0)"
    assert "mid_band_embed_dim" in fp_on
