"""Baseline eval harness 的純函式（高風險邏輯集中於此，便於單測）。

職責邊界：
  - 本檔只做「對齊 / 切分 / 反正規化 / 選 stride」的純計算，皆 fail-fast，不碰檔案 IO。
  - 重建方法在 baselines.py；CLI 串接在 scripts/evaluate_baselines.py。

對齊 CLAUDE.md evalStrictness：時間軸不可鬆散對齊、POD basis 不可洩漏 eval 快照、
grid stride 不可導致格點錯位、缺資料一律 fail-fast。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def match_sensor_dns_times(sensor_time, dns_t, rtol: float = 0.25) -> list[int]:
    """把每個 sensor 取樣時間對到「同一物理時間」的 DNS index（值匹配，非 index-stride）。

    fail-fast:
      - 任一 sensor 時間找不到 gap < rtol*median(dt_dns) 的 DNS 時間 → ValueError。
      - 兩個 sensor 時間對到同一 DNS index（時間軸不一致）→ ValueError。
    """
    sensor_time = np.asarray(sensor_time, dtype=np.float64)
    dns_t = np.asarray(dns_t, dtype=np.float64)
    if dns_t.size < 2:
        raise ValueError(f"DNS 時間點過少（{dns_t.size}），無法對齊")
    dt = float(np.median(np.diff(np.sort(dns_t))))
    tol = rtol * dt
    matched: list[int] = []
    for ts in sensor_time:
        j = int(np.argmin(np.abs(dns_t - ts)))
        gap = abs(float(dns_t[j]) - float(ts))
        if gap > tol:
            raise ValueError(
                f"sensor time {ts:.6g} 無對應 DNS time（最近 {dns_t[j]:.6g}, gap={gap:.3g} > tol={tol:.3g}）"
            )
        matched.append(j)
    if len(set(matched)) != len(matched):
        raise ValueError("sensor→DNS 時間映射有重複，時間軸不一致（sensor 取樣率與 DNS 不相容）")
    return matched


def basis_indices_excluding(n_total: int, eval_indices, min_basis: int = 8) -> np.ndarray:
    """回傳與 eval_indices 不相交的 DNS time index（leakage-free POD basis 來源）。

    fail-fast：complement 數量 < min_basis → ValueError（DNS 軌跡太短，無法建乾淨 basis）。
    """
    evalset = {int(i) for i in eval_indices}
    basis = np.array([i for i in range(int(n_total)) if i not in evalset], dtype=int)
    if basis.size < min_basis:
        raise ValueError(
            f"leakage-free POD basis 不足：complement={basis.size} < min_basis={min_basis}；"
            f"DNS 軌跡需比 eval 視窗更長，或改用其他 basis 來源"
        )
    return basis


def choose_grid_stride(n: int, grid_stride: int = 1, max_grid: int = 0) -> int:
    """選空間 stride s（必須整除 n，確保 strided 格點仍落在 j/N' 的均勻格上）。

    - max_grid>0：選「能整除 n 且 n//s ≤ max_grid」的最小 s（最細不超預算的格）。
    - 否則用 grid_stride（>1 時須整除 n，否則 fail-fast）。
    """
    n = int(n)
    if max_grid and max_grid > 0:
        for s in range(1, n + 1):
            if n % s == 0 and (n // s) <= max_grid:
                return s
        return n
    if grid_stride and grid_stride > 1:
        if n % grid_stride != 0:
            raise ValueError(f"grid_stride={grid_stride} 未整除 DNS 格點 N={n}（會造成格點錯位）")
        return int(grid_stride)
    return 1


def denormalize_sensors(sensor_vals, norm_stats: dict) -> np.ndarray:
    """把 normalized sensor_vals [T,K,2] 反正規化回物理單位（對齊 raw DNS 比對）。

    channel 0=u、1=v；norm_stats 需含 u_mean/u_std/v_mean/v_std。
    """
    for key in ("u_mean", "u_std", "v_mean", "v_std"):
        if key not in norm_stats:
            raise KeyError(f"norm_stats 缺 '{key}'，無法反正規化 sensor")
    out = np.array(sensor_vals, dtype=np.float64)
    out[..., 0] = out[..., 0] * norm_stats["u_std"] + norm_stats["u_mean"]
    out[..., 1] = out[..., 1] * norm_stats["v_std"] + norm_stats["v_mean"]
    return out


def select_pod_modes_by_validation(basis_u, basis_v, sensor_pos, mode_grid,
                                   *, val_frac: float = 0.3, seed: int = 0) -> tuple[int, dict]:
    """以「從 basis 隨機切出的 validation 子集」選 gappy-POD 模態數（leakage-free）。

    Why:
      在 eval metric 上挑模態 = test-set tuning（不公平）。改在 basis 內切 fit/val：POD basis 用
      fit、模態數用 val 誤差挑；val 與 eval 快照不相交，故對 eval 無洩漏。自然避開模態≈2K 的
      過擬合點（fit_size < 2K 時掃描格被 cap，且過擬合點 val 誤差高會被淘汰）。

    回傳 (best_modes, {modes: val_combined_rel_err})。
    """
    from pi_lnn_jax.baselines import GappyPOD, _sensor_grid_indices

    basis_u = np.asarray(basis_u, dtype=np.float64)
    basis_v = np.asarray(basis_v, dtype=np.float64)
    B, N, _ = basis_u.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    n_val = max(1, int(round(val_frac * B)))
    val_i, fit_i = perm[:n_val], perm[n_val:]
    if fit_i.size < 1:
        raise ValueError(f"basis 太小（B={B}），無法切出 fit/val")

    fit_u, fit_v = basis_u[fit_i], basis_v[fit_i]
    val_u, val_v = basis_u[val_i], basis_v[val_i]
    ix, iy = _sensor_grid_indices(sensor_pos, N)
    val_sv = np.stack([val_u[:, ix, iy], val_v[:, ix, iy]], axis=2)  # [V,K,2] val 感測量測

    cap = min(int(fit_i.size), 2 * int(sensor_pos.shape[0]))  # 模態 ≤ fit 快照數且 ≤ 量測數
    grid = sorted({min(int(m), cap) for m in mode_grid if int(m) >= 1})
    if not grid:
        raise ValueError("mode_grid cap 後為空")

    base = GappyPOD(n_modes=max(grid)).fit(fit_u, fit_v)
    full_modes = base.modes
    curve: dict = {}
    for m in grid:
        base.modes = full_modes[:, :m]
        u_pred, v_pred = base.reconstruct(val_sv, sensor_pos)
        eu = np.linalg.norm(u_pred - val_u) / (np.linalg.norm(val_u) + 1e-12)
        ev = np.linalg.norm(v_pred - val_v) / (np.linalg.norm(val_v) + 1e-12)
        curve[m] = float((eu + ev) / 2)
    best = min(curve, key=curve.get)
    return best, curve


def load_aligned_re(sensor_json, dns_path, *, sensor_time_stride: int, sensor_T: int,
                    grid_stride: int, max_grid: int) -> dict:
    """載入單一 Re 的 sensors+DNS、denorm、選 stride、時間值匹配對齊（IO 編排，供各 eval 腳本共用）。

    回傳 dict 含 sensor_phys/sensor_pos/T/K/N/s/Nprime/eval_idx/dns_t_eval/
    dns_u_eval/dns_v_eval/dns_u_full/dns_v_full。
    所有 fail-fast 條件由內部純函式負責（時間對齊 / stride 整除）。

    `dns_t_eval` 是被選中那些幀的**物理時間值**（非索引），供呼叫端填
    `metric_artifact.EvaluationContext.times`。沒有它，呼叫端只能拿 eval_idx
    充數，artifact 的時間軸就與物理時間脫鉤。
    """
    # lazy import：保持本模組 top-level 輕量（純函式單測不需 data 依賴）
    from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path

    d = load_sensors_from_path(sensor_json, time_stride=sensor_time_stride)
    T = min(sensor_T, d["sensor_vals"].shape[0])
    sensor_pos = np.asarray(d["sensor_pos"])
    sensor_time = np.asarray(d["sensor_time"][:T])
    sensor_phys = denormalize_sensors(d["sensor_vals"][:T], d["norm_stats"])  # [T,K,2] physical

    dns_u_full, dns_v_full, dns_t_full = load_dns_from_path(Path(dns_path), time_stride=1)
    N = int(dns_u_full.shape[1])
    s = choose_grid_stride(N, grid_stride=grid_stride, max_grid=max_grid)
    eval_idx = match_sensor_dns_times(sensor_time, dns_t_full)  # fail-fast 值匹配
    dns_u_eval = np.asarray(dns_u_full[eval_idx][:, ::s, ::s])
    dns_v_eval = np.asarray(dns_v_full[eval_idx][:, ::s, ::s])
    return {
        "sensor_phys": sensor_phys, "sensor_pos": sensor_pos,
        "T": int(T), "K": int(sensor_pos.shape[0]), "N": N, "s": int(s),
        "Nprime": int(dns_u_eval.shape[1]), "eval_idx": eval_idx,
        "dns_t_eval": np.asarray(dns_t_full)[eval_idx],
        "dns_u_eval": dns_u_eval, "dns_v_eval": dns_v_eval,
        "dns_u_full": dns_u_full, "dns_v_full": dns_v_full,
    }
