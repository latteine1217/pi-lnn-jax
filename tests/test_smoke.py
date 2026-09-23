"""Smoke test: 確認 LiquidOperator init + forward 在 EXP-030 shape 設定下可跑通。

What: shape/dtype/no-NaN 三條檢查；不對比 PyTorch 數值（留 test_parity.py）。
Why: POC 第一步先確保 Flax module 可正確 setup + 跑 forward；
     init + lax.scan + cross-attention 任一處 shape mismatch 都會在這裡爆。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from pi_lnn_jax.models import LiquidOperator


def _build_exp030_model() -> LiquidOperator:
    """EXP-030 shape：d_model=64, T=51, K=100, N=128"""
    return LiquidOperator(
        sensor_value_dim=2,
        d_model=64,
        d_time=8,
        num_spatial_encoder_layers=1,
        num_temporal_cfc_layers=1,
        domain_length=1.0,
        use_temporal_anchor=True,
        T_total=5.0,
        temporal_anchor_harmonics=2,
        num_token_attention_layers=1,
        token_attention_heads=4,
        num_query_mlp_layers=1,
        query_mlp_hidden_dim=64,
        operator_rank=64,
        decoder_attention_heads=1,
    )


def test_liquid_operator_init_forward_jit():
    rng = jax.random.PRNGKey(42)
    rng_init, rng_data = jax.random.split(rng)

    # EXP-030 規模
    T, K, N = 51, 100, 32
    sensor_vals = jax.random.normal(rng_data, (T, K, 2))
    sensor_pos = jax.random.uniform(rng_data, (K, 2), minval=0.0, maxval=1.0)
    sensor_time = jnp.linspace(0.0, 5.0, T)
    re_norm = 0.1  # placeholder (Re normalized)
    xy = jax.random.uniform(rng_init, (N, 2), minval=0.0, maxval=1.0)
    t_q = jax.random.uniform(rng_init, (N,), minval=0.0, maxval=5.0)

    model = _build_exp030_model()
    print("=== Init ===")
    params = model.init(rng_init, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  total params: {n_params:,}")
    # 主要參數塊
    for path, leaf in jax.tree_util.tree_leaves_with_path(params['params']):
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if leaf.size >= 1000:  # 只 print 較大的
            print(f"  {path_str}: shape={leaf.shape}, size={leaf.size}")

    print("\n=== Forward ===")
    out = model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q)
    print(f"  output shape: {out.shape}  (expect ({N}, 3))")
    print(f"  output dtype: {out.dtype}")
    print(f"  has NaN: {bool(jnp.any(jnp.isnan(out)))}")
    print(f"  has Inf: {bool(jnp.any(jnp.isinf(out)))}")
    print(f"  mean / std / min / max: "
          f"{float(out.mean()):.4f} / {float(out.std()):.4f} / "
          f"{float(out.min()):.4f} / {float(out.max()):.4f}")

    assert out.shape == (N, 3), f"shape mismatch: {out.shape}"
    assert not bool(jnp.any(jnp.isnan(out))), "output has NaN"
    assert not bool(jnp.any(jnp.isinf(out))), "output has Inf"

    print("\n=== Forward (jit) ===")
    apply_jit = jax.jit(model.apply)
    out2 = apply_jit(params, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q)
    print(f"  output equal to non-jit: {bool(jnp.allclose(out, out2, atol=1e-5))}")

    print("\n=== Smoke PASS ===")
