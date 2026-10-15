"""diag_pressure_poisson_jump.py — 兩個審稿追問，同一次重建量完。

What:
    對一個 Kolmogorov 主線 run，在評估協定決定的每一幀重建 (u, v, p)，並量：

    (A) 網路壓力 vs「由網路自己的速度解 Poisson」的壓力（p_pois）。
        審稿人問：壓力是 u 的泛函；網路的 p 有沒有比直接拿重建的 u 解 Poisson 更好？
        三組比較，全部經 `eval_pressure_limit.pressure_frame_metrics`（同定義）：
          net_vs_ref    網路 p      vs 參考壓力（= eval_pressure_limit 的原數字，當閘門用）
          pois_vs_ref   p_pois      vs 參考壓力
          net_vs_pois   網路 p      vs p_pois
    (B) 狀態切換的跳躍幅度。decoder 取「≤ t 的最新 sensor 狀態」（models.py:985，
        idx = Σ[st ≤ t] − 1），所以在 t_i − ε 查詢得到前一個狀態、t_i 得到新狀態，
        兩者之差就是 t_i 處的跳躍（trunk 對 t 連續，ε 小時 trunk 的貢獻可忽略——
        由對照組 t_i − 2ε vs t_i − ε 實測背書，兩者同一狀態）。
          jump_rel      ‖u(t_i) − u(t_i−ε)‖ / ‖u(t_i)‖        （速度向量 L2）
          dns_step_rel  ‖u_dns(t_i) − u_dns(t_{i−1})‖ / ‖u_dns(t_i)‖ （一個 sensor 步的真實變化，量級參考）
          uv_err_left   左極限場 u(t_i−ε) 對 DNS(t_i) 的 uv_rel_err（新狀態有沒有讓誤差變小）
          ctrl_rel      ‖u(t_i−ε) − u(t_i−2ε)‖ / ‖u(t_i)‖     （每 10 幀一次的連續性對照）

Why 用 eval_pressure_limit 的重建而不是另寫:
    encode-once 迴圈、uv 指紋閘門、norm_stats 與建構指紋三道閘門都已在那支腳本實作並被
    已發表的壓力表背書。本診斷 import 它的函式，只多做 (A)(B) 兩件事；
    uv 閘門照舊——重建若與該 run 既有評估對不上，整個 run 不寫輸出。

這是**診斷**不是 eval producer：不落 metric artifact（`tests/test_no_metric_write_bypass.py`
的 METRIC_PRODUCERS 不含本檔），輸出只寫到 `--out`。

Usage（與 eval_pressure_limit 相同的參數）:
    PYTHONPATH=. uv run python scripts/diag_pressure_poisson_jump.py \
        --arch liquid --protocol follow_training \
        --artifacts-dir <run> --config <toml> --expect-metrics <run>/final_eval/metrics.json \
        --arm K200_al --out <job>/out/<run>.json
    需要 GPU 寫出的 ckpt → 走 Slurm。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pressure_limit as epl
from _common.periodic_fields import reference_pressure

from pi_lnn_jax.ckpt import (
    reference_params_for,
    restore_eval_params,
    verify_norm_stats,
)
from pi_lnn_jax.config import DATA_SCHEMA, load_config
from pi_lnn_jax.data import _resolve_data_path
from pi_lnn_jax.evaluation_protocol import (
    encoder_time_stride_from_config,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.metric_artifact import compute_metrics
from pi_lnn_jax.model_factory import build_model

PRODUCER = "scripts/diag_pressure_poisson_jump.py"
SCHEMA = "pressure_poisson_jump.v1"

#: 左極限的時間偏移。sensor 間距 0.05、float32 在 t≤5 的間距 ~4.8e-7，
#: 1e-4 遠大於 float 解析度、遠小於 sensor 間距（trunk 在 ε 內的變化由對照組量）。
EPS_T = 1e-4
#: 連續性對照每隔幾個切換點做一次（多一次 decode 的成本）。
CTRL_EVERY = 10

COMPARISONS = ("net_vs_ref", "pois_vs_ref", "net_vs_pois")


def state_index(sensor_time: np.ndarray, t: float) -> int:
    """與 decoder 相同的查詢規則（models.py:985）：Σ[st ≤ t] − 1，clip 到 [0, T−1]。"""
    st = np.asarray(sensor_time, dtype=np.float32)
    idx = int(np.sum(st <= np.float32(t))) - 1
    return min(max(idx, 0), len(st) - 1)


def left_limit_times(sensor_time: np.ndarray, eps: float) -> np.ndarray:
    """每個切換點 t_i（i ≥ 1）的左極限查詢時刻 t_i − eps，並斷言它確實落在前一個狀態。

    不成立代表 eps 相對 float32 或 sensor 間距選錯——量到的就不是跳躍，fail-fast。
    """
    st = np.asarray(sensor_time, dtype=np.float32)
    if st.ndim != 1 or len(st) < 2 or np.any(np.diff(st) <= 0):
        raise ValueError("sensor_time 須為嚴格遞增的 1D 陣列且至少兩幀")
    if eps >= float(np.diff(st).min()):
        raise ValueError(f"eps={eps} 不小於最小 sensor 間距 {np.diff(st).min()}")
    t_minus = (st[1:] - np.float32(eps)).astype(np.float32)
    for i, tm in enumerate(t_minus, start=1):
        got = state_index(st, float(tm))
        if got != i - 1:
            raise ValueError(f"t_{i}−eps={tm!r} 查到狀態 {got}，預期 {i - 1}")
        if state_index(st, float(st[i])) != i:
            raise ValueError(f"t_{i}={st[i]!r} 查到的不是狀態 {i}")
    return t_minus


def rel_diff_uv(u_a, v_a, u_b, v_b, u_n, v_n) -> float:
    """‖(u_a − u_b, v_a − v_b)‖ / ‖(u_n, v_n)‖（速度向量 L2，float64 累加）。"""
    u_a, v_a, u_b, v_b, u_n, v_n = (np.asarray(x, dtype=np.float64)
                                    for x in (u_a, v_a, u_b, v_b, u_n, v_n))
    num = np.sum((u_a - u_b) ** 2 + (v_a - v_b) ** 2)
    den = np.sum(u_n ** 2 + v_n ** 2)
    return float(np.sqrt(num / den))


def decode_uv_at(model, params, h_states, sensor_time, sensor_pos, xy, n_total, chunk,
                 norm_stats, N, t: float):
    """單一時刻重建 (u, v)，與 epl.iter_uvp_frames 同一個 decode 與反正規化。"""
    t_q = jnp.full((chunk,), float(t), dtype=jnp.float32)
    pred = np.concatenate([
        np.asarray(epl._decode(model, params, xy[i:i + chunk], t_q, h_states,
                               sensor_time, sensor_pos))
        for i in range(0, xy.shape[0], chunk)
    ])[:n_total]
    u = pred[:, 0] * norm_stats["u_std"] + norm_stats["u_mean"]
    v = pred[:, 1] * norm_stats["v_std"] + norm_stats["v_mean"]
    return u.reshape(N, N), v.reshape(N, N)


def prepare(args) -> SimpleNamespace:
    """輸入解析、協定、restore 與兩道閘門——與 eval_pressure_limit.run 逐步相同。

    本檔與 diag_spectral_floor 共用；只做到「拿到可 decode 的 params」為止，
    uv 指紋閘門由呼叫端依自己重建的幀執行。
    """
    config = args.config.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    for path in (config, ckpt_dir, artifacts_dir / "summary.json", args.expect_metrics):
        if not path.exists():
            raise FileNotFoundError(f"不存在: {path}")
    summary = epl.read_training_summary(artifacts_dir, arch=args.arch, config=config)
    expect = epl.read_expectation(args.expect_metrics.resolve(), ckpt_dir=ckpt_dir, config=config)

    cfg = load_config(config)
    data_kwargs, model_kwargs = cfg["data_kwargs"], cfg["model_kwargs"]
    sensor_jsons, dns_paths = data_kwargs.get("sensor_jsons", []), data_kwargs.get("dns_paths", [])
    re_values = data_kwargs.get("re_values", [])
    if not (len(sensor_jsons) == len(dns_paths) == len(re_values) == 1):
        raise ValueError(f"只支援單-Re config：{sensor_jsons} {dns_paths} {re_values}")
    if encoder_time_stride_from_config(config) != 1:
        raise NotImplementedError("encoder_time_stride≠1 未支援（主線為 1）")
    re_value = float(re_values[0])
    re_norm_scale = float(data_kwargs.get("re_norm_scale", DATA_SCHEMA["re_norm_scale"][1]))
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))
    protocol = resolve_protocol(
        mode=args.protocol, training_time_strides=training_time_strides_from_config(config),
        cli_time_stride=args.time_stride, reason=args.protocol_reason)
    dns_path = _resolve_data_path(dns_paths[0])
    dns_L = float(np.load(dns_path, allow_pickle=True).item()["config"]["L"])
    if dns_L != 1.0:
        raise NotImplementedError(f"DNS 的 L={dns_L}：只支援 L=1")
    aligned = load_for_evaluation(sensor_jsons[0], dns_path, protocol=protocol,
                                  viscosity=1.0 / re_value, with_pressure=False)
    if aligned.s != 1:
        raise ValueError(f"評估網格必須是 DNS 網格本身，grid stride={aligned.s}")
    times = np.asarray(aligned.dns_t_eval, dtype=np.float64)
    if len(times) != len(expect.times) or not np.allclose(times, expect.times, atol=1e-6, rtol=0):
        raise ValueError(f"評估時刻與 {expect.path} 不一致——協定錯位")

    sensor_vals = jnp.asarray(aligned.sensor_vals_normalized)
    sensor_pos = jnp.asarray(aligned.sensor_pos)
    sensor_time = jnp.asarray(aligned.sensor_time)
    model, model_name = build_model(args.arch, model_kwargs, K_sensors=aligned.K)
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    params, restored_step, restore_prov = restore_eval_params(
        ckpt_dir, expect.ckpt_step, reference_params=params, model=model)
    norm_prov = verify_norm_stats(artifacts_dir, aligned.norm_stats)
    stale = sorted(set(epl.P_NORM_FALLBACK) & set(aligned.norm_stats))
    if stale:
        raise ValueError(f"sensor 的 norm_stats 含 {stale}：p 的反正規化前提不成立")
    norm_stats = {**{k: float(v) for k, v in aligned.norm_stats.items()}, **epl.P_NORM_FALLBACK}
    print(f"[setup] {model_name} step={restored_step} K={aligned.K} frames={len(times)} "
          f"N={aligned.N} fingerprint_verified={restore_prov['fingerprint_verified']} "
          f"norm_stats_verified={norm_prov['norm_stats_verified']}")
    repo_root = Path(epl.pi_lnn_jax.__file__).resolve().parent.parent
    return SimpleNamespace(
        config=config, artifacts_dir=artifacts_dir, summary=summary, expect=expect, cfg=cfg,
        re_value=re_value, re_norm=re_norm, protocol=protocol, dns_L=dns_L, aligned=aligned,
        times=times, N=int(aligned.N), model=model, model_name=model_name, params=params,
        restored_step=int(restored_step), restore_prov=restore_prov, norm_prov=norm_prov,
        norm_stats=norm_stats, sensor_vals=sensor_vals, sensor_pos=sensor_pos,
        sensor_time=sensor_time, repo_root=repo_root)


def provenance(ctx: SimpleNamespace, args, script: Path) -> dict:
    """兩支診斷共用的輸出 provenance。"""
    return {
        "arm": args.arm,
        "run": {"artifacts_dir": str(ctx.artifacts_dir), "config": str(ctx.config),
                "arch": args.arch, "model_name": ctx.model_name, "K": int(ctx.aligned.K),
                "seed": int(ctx.cfg["train_kwargs"]["seed"]),
                "use_al": bool(ctx.summary["use_al"]), "ckpt_step": ctx.restored_step,
                "re_value": ctx.re_value},
        "script": {"path": str(script.resolve()), "sha256": epl.file_sha256(script),
                   "eval_pressure_limit_sha256": epl.file_sha256(Path(epl.__file__))},
        "repo": {"root": str(ctx.repo_root), "git_sha": epl.git_sha(ctx.repo_root)},
        "protocol": ctx.protocol.to_provenance(),
        "restore_provenance": ctx.restore_prov,
        "norm_stats_provenance": ctx.norm_prov,
    }


def run(args) -> dict:
    t_start = time.time()
    ctx = prepare(args)
    times, aligned, N, dns_L = ctx.times, ctx.aligned, ctx.N, ctx.dns_L
    model, params, norm_stats = ctx.model, ctx.params, ctx.norm_stats
    sensor_vals, sensor_pos, sensor_time = ctx.sensor_vals, ctx.sensor_pos, ctx.sensor_time
    expect = ctx.expect
    sensor_time_np = np.asarray(aligned.sensor_time, dtype=np.float32)
    # (B) 需要評估時刻與 sensor 時刻逐一重合：第 i 幀就是第 i 個切換點。
    if len(times) != len(sensor_time_np) or not np.allclose(times, sensor_time_np, atol=1e-6):
        raise ValueError(
            f"評估時刻（{len(times)}）與 sensor 時刻（{len(sensor_time_np)}）不重合——"
            "跳躍量測假設每個評估幀就是一個切換點")
    t_minus = left_limit_times(sensor_time_np, EPS_T)

    # decode 用的查詢格點與分塊：與 epl.iter_uvp_frames 相同
    h_states = epl._encode(model, params, sensor_vals, sensor_pos, ctx.re_norm, sensor_time)
    grid = np.linspace(0.0, 1.0, N, endpoint=False).astype(np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    xy_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    n_total = xy_np.shape[0]
    chunk = epl._query_chunk(int(aligned.K), n_total)
    xy = jnp.asarray(np.pad(xy_np, ((0, (-n_total) % chunk), (0, 0))))
    dec = {"model": model, "params": params, "h_states": h_states, "sensor_time": sensor_time,
               "sensor_pos": sensor_pos, "xy": xy, "n_total": n_total, "chunk": chunk,
               "norm_stats": norm_stats, "N": N}

    per_frame = {c: {k: [] for k in ("grad_p_rel_l2", "p_gr_rel_l2")} for c in COMPARISONS}
    uv_series, jumps, dns_steps, uv_left, ctrl = [], [], [], [], []
    frames = epl.iter_uvp_frames(model, params, sensor_vals, sensor_pos, ctx.re_norm, sensor_time,
                                 norm_stats, N, times)
    for i, (u_pred, v_pred, p_pred) in enumerate(frames):
        u_ref, v_ref = aligned.dns_u_eval[i], aligned.dns_v_eval[i]
        p_ref = reference_pressure(u_ref, v_ref, dns_L)
        p_pois = reference_pressure(u_pred, v_pred, dns_L)
        rows = {
            "net_vs_ref": epl.pressure_frame_metrics(p_pred, p_ref, dns_L),
            "pois_vs_ref": epl.pressure_frame_metrics(p_pois, p_ref, dns_L),
            "net_vs_pois": epl.pressure_frame_metrics(p_pred, p_pois, dns_L),
        }
        for c, row in rows.items():
            for k in per_frame[c]:
                if not np.isfinite(row[k]):
                    raise FloatingPointError(f"t={times[i]:.4f} {c}.{k} 非有限值")
                per_frame[c][k].append(row[k])
        uv_series.append(compute_metrics(u_pred, v_pred, u_ref, v_ref)["uv_rel_err"])

        if i >= 1:   # 第 i 幀是第 i 個切換點
            tm = float(t_minus[i - 1])
            u_l, v_l = decode_uv_at(**dec, t=tm)
            jumps.append(rel_diff_uv(u_pred, v_pred, u_l, v_l, u_pred, v_pred))
            du, dv = aligned.dns_u_eval, aligned.dns_v_eval
            dns_steps.append(rel_diff_uv(du[i], dv[i], du[i - 1], dv[i - 1], du[i], dv[i]))
            uv_left.append(compute_metrics(u_l, v_l, u_ref, v_ref)["uv_rel_err"])
            if i % CTRL_EVERY == 0:
                u_l2, v_l2 = decode_uv_at(**dec, t=float(np.float32(tm) - np.float32(EPS_T)))
                if state_index(sensor_time_np, float(np.float32(tm) - np.float32(EPS_T))) != i - 1:
                    raise ValueError(f"對照時刻 t_{i}−2ε 跨了狀態")
                ctrl.append({"i": i, "rel": rel_diff_uv(u_l, v_l, u_l2, v_l2, u_pred, v_pred)})
        if i % 10 == 0 or i == len(times) - 1:
            print(f"  [{i + 1:3d}/{len(times)}] t={times[i]:.3f} uv={uv_series[-1]:.4f} "
                  f"pois_vs_ref.grad={rows['pois_vs_ref']['grad_p_rel_l2']:.4f} "
                  f"net_vs_ref.grad={rows['net_vs_ref']['grad_p_rel_l2']:.4f}"
                  + (f" jump={jumps[-1]:.4g}" if jumps else ""))

    uv_mean = float(np.mean(uv_series))
    diff = abs(uv_mean - expect.uv_rel_err)
    gate = {"expect_metrics": str(expect.path), "expected": expect.uv_rel_err,
            "measured": uv_mean, "abs_diff": diff, "atol": epl.UV_GATE_ATOL,
            "passed": bool(diff <= epl.UV_GATE_ATOL)}
    print(f"[gate] uv {uv_mean:.6f} vs 既有 {expect.uv_rel_err:.6f} |Δ|={diff:.2e} → "
          f"{'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        raise SystemExit(f"[FAIL] uv 指紋不符（|Δ|={diff:.3e}）——重建的不是既有評估的那個模型，未寫輸出")

    return {
        "producer": PRODUCER, "schema": SCHEMA,
        **provenance(ctx, args, Path(__file__)),
        "uv_gate": gate,
        "eps_t": EPS_T,
        "pressure": {c: {k: {"time_mean": float(np.mean(v)), "per_frame": v}
                         for k, v in per_frame[c].items()} for c in COMPARISONS},
        "jump": {
            "definition": "‖u(t_i) − u(t_i−ε)‖ / ‖u(t_i)‖, velocity vector L2, i = 1..T−1",
            "jump_rel": {"mean": float(np.mean(jumps)), "max": float(np.max(jumps)),
                         "per_switch": jumps},
            "dns_step_rel": {"mean": float(np.mean(dns_steps)), "per_switch": dns_steps},
            "uv_err_left_limit": {"mean": float(np.mean(uv_left)), "per_switch": uv_left},
            "uv_err_at_switch": {"mean": float(np.mean(uv_series[1:])),
                                 "per_switch": uv_series[1:]},
            "continuity_control": {"mean": float(np.mean([c["rel"] for c in ctrl])),
                                   "points": ctrl},
        },
        "wall_seconds": time.time() - t_start,
    }


def main(argv=None) -> int:
    args = epl.parse_args(argv)
    out = run(args)
    text = json.dumps(out, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    p, j = out["pressure"], out["jump"]
    print("=== pressure (time mean) ===")
    for c in COMPARISONS:
        print(f"  {c:<12s} grad_p_rel_l2={p[c]['grad_p_rel_l2']['time_mean']:.5f} "
              f"p_gr_rel_l2={p[c]['p_gr_rel_l2']['time_mean']:.5f}")
    print("=== state-switch jump ===")
    print(f"  jump_rel mean={j['jump_rel']['mean']:.5f} max={j['jump_rel']['max']:.5f}  "
          f"dns_step_rel mean={j['dns_step_rel']['mean']:.5f}  "
          f"control={j['continuity_control']['mean']:.2e}")
    print(f"  uv_err left-limit={j['uv_err_left_limit']['mean']:.5f} "
          f"at-switch={j['uv_err_at_switch']['mean']:.5f}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
