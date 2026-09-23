#!/usr/bin/env python3
"""diag_offgrid_query.py — 在 sensor 取樣點**之間**的時刻查詢模型，量「連續時間」宣稱的代價。

What:
    sensor 輸入維持訓練時的 stride（101 幀），但把 query 時刻換成 DNS 的**全部**幀
    （201 個，stride 1）。DNS 的偶數幀與 sensor 時戳重合（staleness = 0），奇數幀落在
    兩個 sensor 之間（staleness = Δt/2）。兩組的誤差差值就是 off-grid 查詢的代價。

Why:
    `chapter01.tex` 宣稱 "query-anywhere continuous-time evaluation"，但三個評估協定
    **沒有一條能查 off-grid**：`follow_training` 與 `fixed_grid` 都走
    `match_sensor_dns_times` 把 sensor 對齊到 DNS，中點幀在對齊時就被丟掉；
    `sensor_time_independent` 的 `--time-stride` 同時控制 sensor 與 query，設 1 會讓
    sensor 吃 201 幀——那是換了模型輸入，不是「在樣本之間查詢」。

    也就是說那句宣稱從未被測過，不是因為沒人想到，是因為工具做不到。本診斷補上那條路。
    branch 的值路徑在兩個時戳之間是階梯常數（decoder 取 `h_states[idx]` 持平），所以
    預期 off-grid 較差；量的是差多少。

    這是**診斷**不是 eval producer：不落 metric artifact、不寫進 artifacts_dir 的
    評估產物命名空間，避免污染 `scripts/CLAUDE.md` §2 那條唯一落盤路徑。

Usage:
    PYTHONPATH=. uv run python scripts/diag_offgrid_query.py \\
        --config configs/exp_245_b3_les_T50.toml --ckpt latest

    需要 GPU 寫出的 ckpt → 走 Slurm（head node 的 CPU JAX 推不出 orbax sharding）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax import ckpt as ckpt_mod
from pi_lnn_jax.config import load_config
from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path
from pi_lnn_jax.evaluate import evaluate_against_dns
from pi_lnn_jax.model_factory import build_model

# sensor 時戳與 DNS 幀時間視為「同一時刻」的容差。DNS dt = 0.025 s，sensor dt = 0.05 s，
# 所以真正的中點距離最近時戳 0.025 s；容差取其 1/100，不可能把中點誤判成 on-grid。
T_MATCH_TOL = 2.5e-4


# 訓練端的 fallback（單一來源：pipeline/kolmogorov/assembly.py:194 與
# pi_lnn_jax/config.py:306）。eval 紅線禁的是「**與訓練不同**的靜默預設」，
# 不是「config 省略了訓練自己也省略的鍵」——主線 exp_main5_* 兩個鍵都沒設，
# 硬失敗會讓診斷在它唯一該用的 run 上跑不起來。所以照抄訓練端的解析，
# 但把來源印出來、也寫進輸出，讓「用了 fallback」永遠可稽核。
TRAIN_TIME_STRIDE_FALLBACK = 2       # assembly.py:194  `if time_strides else 2`
TRAIN_RE_NORM_SCALE_FALLBACK = 1e4   # config.py:306    schema default


def resolve_training_axes(data_kwargs: dict) -> tuple[int, float, dict]:
    """回 (sensor_stride, re_norm_scale, provenance)；provenance 記每個值的來源。"""
    strides = data_kwargs.get("time_strides") or []
    if strides:
        stride, stride_src = int(strides[0]), "config.time_strides[0]"
    else:
        stride, stride_src = TRAIN_TIME_STRIDE_FALLBACK, "training fallback (assembly.py:194)"
    if "re_norm_scale" in data_kwargs:
        scale, scale_src = float(data_kwargs["re_norm_scale"]), "config.re_norm_scale"
    else:
        scale, scale_src = TRAIN_RE_NORM_SCALE_FALLBACK, "schema default (config.py:306)"
    return stride, scale, {"sensor_stride_source": stride_src, "re_norm_scale_source": scale_src}


def split_on_off_grid(dns_t, sensor_time, tol: float = T_MATCH_TOL):
    """把 DNS 幀分成「落在 sensor 時戳上」與「落在兩者之間」。

    回 (on_grid, off_grid, gap)：兩個 bool 遮罩與每幀到最近 sensor 時戳的距離。
    兩邊都不得為空——空的那邊代表這組資料沒有可比較的對象，由呼叫端 fail-fast。
    """
    dns_t = np.asarray(dns_t, dtype=float)
    sensor_time = np.asarray(sensor_time, dtype=float)
    if dns_t.ndim != 1 or sensor_time.ndim != 1:
        raise ValueError(f"時間軸須為 1D，收到 {dns_t.shape} 與 {sensor_time.shape}")
    gap = np.abs(dns_t[:, None] - sensor_time[None, :]).min(axis=1)
    on_grid = gap <= tol
    return on_grid, ~on_grid, gap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="訓練用的 TOML（決定 sensor stride 與模型建構值）")
    p.add_argument("--ckpt", default="latest", help="'latest' 或 step 整數")
    p.add_argument("--arch", default="liquid", help="build_model 的 arch；預設 liquid")
    p.add_argument("--artifacts-dir", default=None, help="覆寫 config 的 artifacts_dir")
    p.add_argument("--out", default=None, help="輸出 json 路徑；預設 <artifacts>/diag_offgrid_query.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_kwargs, data_kwargs, model_kwargs = (
        cfg["train_kwargs"], cfg["data_kwargs"], cfg["model_kwargs"])

    # ── 輸入解析：全部 fail-fast，不套預設、不靜默截短 ──────────────────
    sensor_jsons = data_kwargs.get("sensor_jsons") or []
    dns_paths = data_kwargs.get("dns_paths") or []
    re_values = data_kwargs.get("re_values") or []
    if not sensor_jsons:
        raise SystemExit(f"{args.config}: data_kwargs.sensor_jsons 為空")
    if not dns_paths:
        raise SystemExit(f"{args.config}: data_kwargs.dns_paths 為空——本診斷需要 DNS 全幀")
    if not re_values:
        raise SystemExit(f"{args.config}: data_kwargs.re_values 為空")
    sensor_stride, re_norm_scale, axes_prov = resolve_training_axes(data_kwargs)
    print(f"[axes] sensor stride = {sensor_stride}  ({axes_prov['sensor_stride_source']})")
    print(f"[axes] re_norm_scale = {re_norm_scale:g}  ({axes_prov['re_norm_scale_source']})")

    artifacts_dir = Path(args.artifacts_dir or train_kwargs.get("artifacts_dir") or "").resolve()
    if not artifacts_dir.name:
        raise SystemExit("無 artifacts_dir：config 未設且未給 --artifacts-dir")
    ckpt_dir = artifacts_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint 目錄不存在：{ckpt_dir}")

    # ── 載入：sensor 照訓練 stride，DNS 全幀 ────────────────────────────
    d = load_sensors_from_path(sensor_jsons[0], time_stride=sensor_stride)
    sensor_vals, sensor_pos = d["sensor_vals"], d["sensor_pos"]
    sensor_time = np.asarray(d["sensor_time"])
    norm_stats = d["norm_stats"]
    re_norm = float(np.log(float(re_values[0])) / np.log(re_norm_scale))

    dns_u, dns_v, dns_t = load_dns_from_path(dns_paths[0], time_stride=1)
    dns_t = np.asarray(dns_t)

    # ── 分組：DNS 每一幀是落在 sensor 時戳上，還是落在兩者之間 ──────────
    on_grid, off_grid, gap = split_on_off_grid(dns_t, sensor_time)
    if not on_grid.any():
        raise SystemExit(
            f"沒有任何 DNS 幀對得上 sensor 時戳（最小 gap {gap.min():.3e} > 容差 {T_MATCH_TOL}）"
            "——時間軸不相容，比較無意義。")
    if not off_grid.any():
        raise SystemExit(
            "所有 DNS 幀都落在 sensor 時戳上——此資料組合沒有 off-grid 時刻可查，"
            "本診斷無對象（需要 DNS 的時間解析度高於 sensor）。")

    print(f"[setup] sensor: {len(sensor_time)} 幀 @ stride {sensor_stride}, "
          f"dt={np.diff(sensor_time).mean():.4f}s")
    print(f"[setup] DNS   : {len(dns_t)} 幀, dt={np.diff(dns_t).mean():.4f}s")
    print(f"[setup] on-grid {int(on_grid.sum())} 幀（staleness=0）／"
          f"off-grid {int(off_grid.sum())} 幀（最大 staleness {gap[off_grid].max():.4f}s）")

    # ── 模型與參數：兩道閘門都帶（scripts/CLAUDE.md §5）─────────────────
    model, model_name = build_model(args.arch, model_kwargs, K_sensors=int(sensor_pos.shape[0]))
    reference_params = ckpt_mod.reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    params, step, provenance = ckpt_mod.restore_eval_params(
        ckpt_dir, args.ckpt, reference_params, model)
    print(f"[ckpt] {model_name} step={step}  fingerprint_verified="
          f"{provenance.get('fingerprint_verified')}")

    # ── 評估：同一份 sensor 輸入，query 走 DNS 全部時刻 ─────────────────
    rows = evaluate_against_dns(
        model, params, sensor_vals, sensor_pos, re_norm, sensor_time, norm_stats,
        dns_u, dns_v, dns_t, eval_t_indices=list(range(len(dns_t))), verbose=False)

    uv = np.array([r["uv_rel_err"] for r in rows], dtype=float)
    if len(uv) != len(dns_t):
        raise SystemExit(f"評估回傳 {len(uv)} 列，DNS 有 {len(dns_t)} 幀——不可對齊，中止")

    summary = {
        "config": str(args.config),
        "ckpt_step": int(step),
        "sensor_stride": sensor_stride,
        "axes_provenance": axes_prov,
        "n_on_grid": int(on_grid.sum()),
        "n_off_grid": int(off_grid.sum()),
        "max_staleness_s": float(gap[off_grid].max()),
        "uv_rel_err_on_grid_mean": float(uv[on_grid].mean()),
        "uv_rel_err_off_grid_mean": float(uv[off_grid].mean()),
        "uv_rel_err_gap_pp": float((uv[off_grid].mean() - uv[on_grid].mean()) * 100.0),
        "uv_rel_err_all_mean": float(uv.mean()),
        "provenance": provenance,
    }
    print("\n=== off-grid 查詢的代價 ===")
    print(f"  on-grid  (staleness=0)      uv_rel_err = {uv[on_grid].mean()*100:.4f}%  "
          f"(n={int(on_grid.sum())})")
    print(f"  off-grid (staleness>0)      uv_rel_err = {uv[off_grid].mean()*100:.4f}%  "
          f"(n={int(off_grid.sum())})")
    print(f"  差值                        {summary['uv_rel_err_gap_pp']:+.4f} pp")

    out = Path(args.out) if args.out else artifacts_dir / "diag_offgrid_query.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {**summary, "per_frame": [
            {"t": float(t), "on_grid": bool(g), "uv_rel_err": float(e)}
            for t, g, e in zip(dns_t, on_grid, uv)]}, indent=2, default=str))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
