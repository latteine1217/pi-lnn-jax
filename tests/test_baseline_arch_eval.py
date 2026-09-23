"""非 liquid arch（B0 vanilla / B2 pinn）也能走 evaluate_time_series。

鎖定 arch-agnostic 契約：evaluate_time_series 只透過 reconstruct_field 的標準
__call__(sensor_vals, sensor_pos, re_norm, sensor_time, xy, t) 介面，三種 arch
（LiquidOperator / VanillaDeepONetOperator / StandardPINNOperator）共用同一簽名與
[N, 3] 輸出，故 eval 不該對 arch 設限。對應移除 evaluate_exp245.py 的 liquid-only guard。
"""
from __future__ import annotations

import jax
import numpy as np

from pi_lnn_jax.evaluate import evaluate_time_series
from pi_lnn_jax.models import StandardPINNOperator, VanillaDeepONetOperator


def _tiny_dns(T: int = 3, N: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    dns_u = rng.standard_normal((T, N, N)).astype(np.float32)
    dns_v = rng.standard_normal((T, N, N)).astype(np.float32)
    dns_t = (np.arange(T) * 0.5).astype(np.float32)
    return dns_u, dns_v, dns_t


def _run_eval(model, K: int = 4, T_sensor: int = 3):
    dns_u, dns_v, dns_t = _tiny_dns()
    sensor_pos = (
        np.linspace(0.0, 1.0, K, endpoint=False)[:, None].repeat(2, axis=1).astype(np.float32)
    )
    sensor_time = dns_t.copy()
    sensor_vals = np.zeros((T_sensor, K, 2), dtype=np.float32)  # C=sensor_value_dim=2
    norm_stats = {"u_mean": 0.0, "u_std": 1.0, "v_mean": 0.0, "v_std": 1.0}
    init_xy = np.zeros((1, 2), np.float32)
    init_t = np.zeros((1,), np.float32)
    params = model.init(
        jax.random.PRNGKey(0), sensor_vals, sensor_pos, 0.5, sensor_time, init_xy, init_t
    )
    return evaluate_time_series(
        model, params, sensor_vals, sensor_pos, 0.5, sensor_time,
        norm_stats, dns_u, dns_v, dns_t, verbose=False,
    )


def test_pinn_arch_runs_time_series_eval():
    model = StandardPINNOperator(num_layers=2, hidden_dim=16, fourier_embed_dim=16, d_time=4)
    out = _run_eval(model)
    mm = out["metrics_mean"]
    for k in ("u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err"):
        assert np.isfinite(mm[k]), f"{k} non-finite"
    assert len(out["ke_t_pred"]) == 3
    assert np.isfinite(out["ke_t_errors"]["ke_t_rel_l2"])


def test_vanilla_arch_runs_time_series_eval():
    model = VanillaDeepONetOperator(
        K_sensors=4, hidden_dim=16, operator_rank=8,
        num_branch_layers=2, num_trunk_layers=2, fourier_embed_dim=16, d_time=4,
    )
    out = _run_eval(model)
    assert np.isfinite(out["metrics_mean"]["ke_rel_err"])
    assert len(out["ke_t_pred"]) == 3
