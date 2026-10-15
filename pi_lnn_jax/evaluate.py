"""DNS 全場 offline evaluation: KE rel-err / divergence L2 / per-channel rel-err.

對齊 pi-lnn 的 evaluator 概念（src/pi_con/evaluate_deeponet_cfc.py）：
  - KE = 0.5 * (u² + v²)，逐 grid point；rel_err = ||KE_pred - KE_dns||_2 / ||KE_dns||_2
  - per-channel rel-err: ||u_pred - u_dns||_L2 / ||u_dns||_L2

Why offline benchmark: ENGINEERING_VISION 明示「sensor 訓練、DNS 對照」是合法工程做法。
                       eval 只在 inference 路徑用 DNS，不污染 training loss。

POC 簡化：
  - 預設只算單一 time slice (t=2.5 中段)，避免 M3 跑全 T 的 ~T×N² queries。
  - 若指定 t_indices=None 則跑所有 sensor_time 的 slice (T=11)。
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.metric_artifact import (
    aggregate_metric_rows,
    band_spectral_rel_error,
    compute_divergence_l2,
    compute_energy_spectrum,
    compute_grad_frobenius,
    compute_metrics,
    compute_vorticity,
    dissipation_wavenumber,
    forcing_mode_coeff_u,
    energy_timeseries_errors,
    per_t_to_arrays,
    sparsity_yardsticks,
    spectral_coherence,
    spectrum_rel_error,
)


# ── jit 的 forward 入口 ──
# model 為 static（Flax Module 依欄位值 hash/eq），故同一 model 同 shape 在 process 內只編譯
# 一次；params 走引數而非 closure，避免把整份權重當常數烘進 executable。
@functools.partial(jax.jit, static_argnums=0)
def _encode(model, params, sensor_vals, sensor_pos, re_norm, sensor_time):
    return model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                       method=LiquidOperator.encode)


@functools.partial(jax.jit, static_argnums=0)
def _decode(model, params, xy, t_q, h_states, sensor_time, sensor_pos):
    return model.apply(params, xy, t_q, h_states, sensor_time, sensor_pos,
                       method=LiquidOperator.decode_query)


@functools.partial(jax.jit, static_argnums=0)
def _apply(model, params, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q):
    return model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q)


def _make_query_fn(model, params, sensor_vals, sensor_pos, re_norm, sensor_time):
    """回傳 `f(xy [n,2], t_q [n]) -> [n, 3]` normalized 的查詢函式。

    Why encode 在此預算: sensor 編碼（spatial encoder + CfC scan over T）與查詢點、查詢時刻
    都無關；經 `__call__` 逐 chunk 前向時它在每個 chunk × 每個時刻重跑一次（K=200/T=101/256²
    即 ~6,700 次）。LiquidOperator 有 encode/decode_query 拆分 → 預算一次；B0/PINN 無此拆分
    （見 physics.make_ns_residual_fn_baseline），走 jit 的完整 forward。
    """
    if isinstance(model, LiquidOperator):
        h_states = _encode(model, params, sensor_vals, sensor_pos, re_norm, sensor_time)
        return lambda xy, t_q: _decode(model, params, xy, t_q, h_states, sensor_time, sensor_pos)
    return lambda xy, t_q: _apply(
        model, params, sensor_vals, sensor_pos, re_norm, sensor_time, xy, t_q)


def reconstruct_field(
    model: LiquidOperator,
    params,
    sensor_vals: jnp.ndarray,
    sensor_pos: jnp.ndarray,
    re_norm: float,
    sensor_time: jnp.ndarray,
    norm_stats: dict,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_slice: float,
    chunk_size: int = 4096,
    return_p: bool = False,
) -> tuple:
    """在 (x_grid × y_grid) 對指定 t_slice 跑 forward，返回 (u_pred, v_pred) physical。
    return_p=True 時額外回 p（physical）：回 (u, v, p)；否則 (u, v)。

    axis convention（極關鍵，對齊 pi-lnn DNS）:
      x_grid 對應 axis_1 (row index)，y_grid 對應 axis_2 (col index)
      → output shape (Nx, Ny)，u[i, j] 對應 (x_grid[i], y_grid[j])

    多時刻重建請走 evaluate_time_series / evaluate_against_dns：它們整個評估只 encode 一次，
    本函式每次呼叫 encode 一次。
    """
    query_fn = _make_query_fn(model, params, sensor_vals, sensor_pos, re_norm, sensor_time)
    return _reconstruct(query_fn, int(sensor_pos.shape[0]), norm_stats,
                        x_grid, y_grid, t_slice, chunk_size, return_p)


def _reconstruct(
    query_fn,
    K_sensors: int,
    norm_stats: dict,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_slice: float,
    chunk_size: int = 4096,
    return_p: bool = False,
) -> tuple:
    """reconstruct_field 的本體，查詢函式由 caller 預建（見 _make_query_fn）。"""
    if return_p and ("p_mean" not in norm_stats or "p_std" not in norm_stats):
        raise KeyError(
            "norm_stats 無 p_mean/p_std；此 checkpoint/dataset 未含 p 通道，無法 return_p"
        )
    Nx, Ny = len(x_grid), len(y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')  # [Nx, Ny]
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)  # [Nx*Ny, 2]

    # 分塊 forward 避免 memory peak。大 K：decode cross-attention 中介 ∝ chunk×K×d_model，
    # chunk=4096 在 K>128 會於 XLA autotuning OOM。forward-only；每個 query 點的輸出只依賴
    # 該點 + sensors（不跨 query batch 交互）→ 分塊在**演算法上逐點獨立**。K≤128 維持原 chunk。
    #
    # PILNN_EVAL_CHUNK_BUDGET：K>128 時 chunk×K 的上限，預設 200000 = 既有行為（不設即不變）。
    # ⚠️ 分塊逐點獨立，但**非 bitwise 相同**：XLA 對不同 chunk shape 編出不同 matmul kernel，
    # 浮點累加順序不同 → 結果在 ~1e-5 相對量級有差（實測 chunk=1000 vs 2000 於 Re=1e6 d384：
    # 全部 headline metrics abs 差 ≤ 3e-4，論文 2 位%精度下完全相同）。故加大 chunk 的驗收
    # 條件是「headline metrics 於報告精度相同」，**不是**逐位相同——後者因 XLA 本就達不到。
    # 註：本預算只鎖 chunk×K，**未計入 d_model**（d384 比 d256 多 1.5× 記憶體，OOM 邊界更早）；
    # 實測 d384/K200 安全上限 chunk=2000（budget 400000），chunk=4000 OOM。
    u_phys_list, v_phys_list = [], []
    p_phys_list = []
    n_total = xy_flat.shape[0]
    _chunk_budget = int(os.environ.get("PILNN_EVAL_CHUNK_BUDGET", "200000"))
    eff_chunk = chunk_size if K_sensors <= 128 else max(256, _chunk_budget // K_sensors)
    eff_chunk = min(eff_chunk, n_total)
    # 每塊固定 shape [eff_chunk]：最後一塊不足時補零點再截掉，jit 只編譯一次。
    # 補的點不影響真實點——同上，查詢點逐點獨立。
    t_chunk = jnp.full((eff_chunk,), float(t_slice), dtype=jnp.float32)
    for i in range(0, n_total, eff_chunk):
        chunk_xy = xy_flat[i:i + eff_chunk]
        n_valid = chunk_xy.shape[0]
        if n_valid < eff_chunk:
            chunk_xy = np.pad(chunk_xy, ((0, eff_chunk - n_valid), (0, 0)))
        pred = np.asarray(query_fn(jnp.asarray(chunk_xy), t_chunk))[:n_valid]  # [n, 3] normalized
        u_phys = pred[:, 0] * norm_stats["u_std"] + norm_stats["u_mean"]
        v_phys = pred[:, 1] * norm_stats["v_std"] + norm_stats["v_mean"]
        u_phys_list.append(u_phys)
        v_phys_list.append(v_phys)
        if return_p:
            p_phys = pred[:, 2] * norm_stats["p_std"] + norm_stats["p_mean"]
            p_phys_list.append(p_phys)
    u = np.concatenate(u_phys_list).reshape(Nx, Ny)
    v = np.concatenate(v_phys_list).reshape(Nx, Ny)
    if return_p:
        p = np.concatenate(p_phys_list).reshape(Nx, Ny)
        return u, v, p
    return u, v


def evaluate_against_dns(
    model: LiquidOperator,
    params,
    sensor_vals: jnp.ndarray,
    sensor_pos: jnp.ndarray,
    re_norm: float,
    sensor_time: jnp.ndarray,
    norm_stats: dict,
    dns_u: np.ndarray,       # [T_dns, Nx, Ny]
    dns_v: np.ndarray,       # [T_dns, Nx, Ny]
    dns_t: np.ndarray,       # [T_dns]
    eval_t_indices: Optional[list[int]] = None,
    x_grid: Optional[np.ndarray] = None,
    y_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> list[dict]:
    """對 DNS 多個 time slice 算 metrics。Default: 只 t=mid (省時)."""
    if eval_t_indices is None:
        # 預設只測中段一個 slice，最快
        mid = dns_u.shape[0] // 2
        eval_t_indices = [mid]
    if x_grid is None:
        Nx = dns_u.shape[1]
        x_grid = np.linspace(0.0, 1.0, Nx, endpoint=False).astype(np.float32)
    if y_grid is None:
        Ny = dns_u.shape[2]
        y_grid = np.linspace(0.0, 1.0, Ny, endpoint=False).astype(np.float32)

    query_fn = _make_query_fn(model, params, sensor_vals, sensor_pos, re_norm, sensor_time)
    K_sensors = int(sensor_pos.shape[0])
    results = []
    for tidx in eval_t_indices:
        t_slice = float(dns_t[tidx])
        u_pred, v_pred = _reconstruct(
            query_fn, K_sensors, norm_stats, x_grid, y_grid, t_slice,
        )
        metrics = compute_metrics(u_pred, v_pred, dns_u[tidx], dns_v[tidx])
        metrics["t_slice"] = t_slice
        metrics["t_idx"] = tidx
        results.append(metrics)
        if verbose:
            print(f"  t={t_slice:.3f} (idx={tidx}): "
                  f"u_err={metrics['u_rel_err']:.4f}  v_err={metrics['v_rel_err']:.4f}  "
                  f"KE_err={metrics['ke_rel_err']:.4f}  "
                  f"ω_err={metrics['omega_rel_err']:.4f}  "
                  f"low_band_err={metrics['low_band_rel_err']:.4f}  "
                  f"div_pred/dns={metrics['div_pred_l2']:.2e}/{metrics['div_dns_l2']:.2e}")
    return results


def evaluate_time_series(
    model: LiquidOperator,
    params,
    sensor_vals: jnp.ndarray,
    sensor_pos: jnp.ndarray,
    re_norm: float,
    sensor_time: jnp.ndarray,
    norm_stats: dict,
    dns_u: np.ndarray,
    dns_v: np.ndarray,
    dns_t: np.ndarray,
    verbose: bool = True,
    dns_p: Optional[np.ndarray] = None,
    eval_p: bool = False,
    nu: Optional[float] = None,
    collect_fields: bool = False,
) -> dict:
    """對全 dns_t 跑 reconstruction → KE(t) time series + 整體 metrics 平均。

    Returns: dict with 'ke_t_pred' [T], 'ke_t_dns' [T], 'metrics_per_t' [T dict], 'metrics_mean' dict

    collect_fields=True 時額外回傳 'u_fields'/'v_fields' [T,Nx,Ny]（重建場本身，
    供場圖與空間統計使用）。預設關閉：T×N² 的場在 T=101/N=256 約 50 MB，只有需要
    畫場的呼叫端該付這個記憶體。
    """
    if eval_p and dns_p is None:
        raise ValueError("eval_p=True 需要提供 dns_p")
    T = dns_u.shape[0]
    Nx, Ny = dns_u.shape[1], dns_u.shape[2]
    x_grid = np.linspace(0.0, 1.0, Nx, endpoint=False).astype(np.float32)
    y_grid = np.linspace(0.0, 1.0, Ny, endpoint=False).astype(np.float32)
    ke_t_pred = np.zeros(T)
    ke_t_dns = np.zeros(T)
    metrics_per_t = []
    u_fields = np.zeros((T, Nx, Ny), dtype=np.float32) if collect_fields else None
    v_fields = np.zeros((T, Nx, Ny), dtype=np.float32) if collect_fields else None
    # sensor encode 整個評估只做一次（見 _make_query_fn）
    query_fn = _make_query_fn(model, params, sensor_vals, sensor_pos, re_norm, sensor_time)
    K_sensors = int(sensor_pos.shape[0])
    for tidx in range(T):
        t_slice = float(dns_t[tidx])
        if eval_p:
            u_pred, v_pred, p_pred = _reconstruct(
                query_fn, K_sensors, norm_stats, x_grid, y_grid, t_slice, return_p=True,
            )
            m = compute_metrics(u_pred, v_pred, dns_u[tidx], dns_v[tidx],
                                p_pred=p_pred, p_dns=dns_p[tidx], nu=nu)
        else:
            u_pred, v_pred = _reconstruct(
                query_fn, K_sensors, norm_stats, x_grid, y_grid, t_slice,
            )
            m = compute_metrics(u_pred, v_pred, dns_u[tidx], dns_v[tidx], nu=nu)
        m["t"] = t_slice
        metrics_per_t.append(m)
        if collect_fields:
            u_fields[tidx] = np.asarray(u_pred, dtype=np.float32)
            v_fields[tidx] = np.asarray(v_pred, dtype=np.float32)
        ke_t_pred[tidx] = 0.5 * (u_pred ** 2 + v_pred ** 2).mean()
        ke_t_dns[tidx] = 0.5 * (dns_u[tidx] ** 2 + dns_v[tidx] ** 2).mean()
        if verbose:
            print(f"  t={t_slice:.2f}: KE rel-err={m['ke_rel_err']:.3f}  "
                  f"low_band={m['low_band_rel_err']:.3f}  ω={m['omega_rel_err']:.3f}")
    result = aggregate_metric_rows(
        metrics_per_t, ke_t_pred, ke_t_dns, pressure_evaluated=eval_p,
    )
    metrics_mean = result["metrics_mean"]
    ke_t_errors = result["ke_t_errors"]
    if verbose:
        print(f"\n  Mean over T={T}: "
              f"u={metrics_mean['u_rel_err']:.3f}  v={metrics_mean['v_rel_err']:.3f}  "
              f"KE={metrics_mean['ke_rel_err']:.3f}  ω={metrics_mean['omega_rel_err']:.3f}  "
              f"low_band={metrics_mean['low_band_rel_err']:.3f}  "
              f"div_pred={metrics_mean['div_pred_l2']:.2e}")
        print(f"  KE MAPE (pointwise)={ke_t_errors['ke_t_mape']:.4f}  "
              f"nMAE={ke_t_errors['ke_t_nmae']:.4f}  "
              f"[spatial-mean (old)={ke_t_errors['ke_t_mape_spatialmean']:.4f}]")
        print(f"  E(t) series: "
              f"rel-L2={ke_t_errors['ke_t_rel_l2']:.4f}  "
              f"rel-L∞={ke_t_errors['ke_t_rel_linf']:.4f}  "
              f"final={ke_t_errors['ke_t_final_rel']:.4f}")
    if collect_fields:
        result["u_fields"] = u_fields
        result["v_fields"] = v_fields
    return result
