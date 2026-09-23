#!/usr/bin/env python
"""tab:inference_cost 的 per-trajectory inference 量測。

Why this exists:
    該表的四個 inference 數字（encode / field query / throughput / full-sequence）
    全庫查無生成端，而 `current-status.md` 反而記「Tier 2.4 wall-clock 對照未成表」。
    硬體欄標的 "M3 MPS" 是 PyTorch 的 Metal backend——本庫是 JAX，沒有那條路徑，
    所以那組數字必然產於 PyTorch 世代。本腳本補上量測端。

What it measures:
    encode：整段 sensor 序列 → branch hidden states，每條軌跡一次。
    query ：已有 hidden states 後，解碼一批 (x, y, t) 查詢點。
    兩者分開量才有意義——論文的賣點是「encode 一次、之後任意點查詢近乎免費」，
    那需要兩個獨立的數字才能支持。

Timing discipline:
    JAX 是非同步派發，不 block 就會量到派發時間而非執行時間；且第一次呼叫含
    JIT 編譯。因此每個量測都 (1) 先 warm-up 到編譯完成，(2) 每輪 block_until_ready，
    (3) 報 n 次的 median 與 sd——median 對偶發的系統雜訊比 mean 穩健。

Usage（lab-server GPU node；ckpt 由 GPU 寫出，CPU 上 orbax 推不出 sharding）:
    uv run python scripts/bench_inference.py \
        --config configs/exp_main5_s42.toml --ckpt latest \
        --output artifacts/bench/inference_cost.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def _timeit(fn, n: int, warmup: int = 3) -> dict:
    """量 fn() 的 wall-clock。回傳 median / sd / min，單位秒。"""
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return {"median_s": float(np.median(a)), "sd_s": float(a.std(ddof=1)),
            "min_s": float(a.min()), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--arch", default="liquid")
    ap.add_argument("--protocol", default="follow_training")
    ap.add_argument("--grid", type=int, default=128,
                    help="全場查詢的邊長；論文表用 128（16384 點/批）")
    ap.add_argument("--n-repeat", type=int, default=20)
    ap.add_argument("--compute-device", default=None, choices=["cpu", "gpu"],
                    help="在此裝置上量測。ckpt 必須以寫出它的 backend 載入（GPU 寫的 "
                         "ckpt 在純 CPU JAX 上 orbax 推不出 sharding），所以載入後才搬。"
                         "不給則用預設裝置")
    ap.add_argument("--artifacts-dir", default=None,
                    help="覆寫 config 的 train_kwargs.artifacts_dir")
    ap.add_argument("--output", default="artifacts/bench/inference_cost.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params
    from pi_lnn_jax.config import load_config
    from pi_lnn_jax.data import load_sensors_from_path
    from pi_lnn_jax.evaluation_protocol import (
        load_for_evaluation, resolve_protocol, training_time_strides_from_config)
    from pi_lnn_jax.model_factory import build_model

    cfg = load_config(a.config)
    data_kwargs = cfg["data_kwargs"]
    model_kwargs = cfg["model_kwargs"]
    artifacts_dir = Path(
        a.artifacts_dir if a.artifacts_dir is not None
        else cfg["train_kwargs"].get("artifacts_dir", "artifacts/run")
    ).resolve()

    for key in ("sensor_jsons", "dns_paths", "re_values"):
        if not data_kwargs.get(key):
            raise SystemExit(f"[fail-fast] {a.config} 的 data_kwargs.{key} 為空——"
                             "benchmark 需要與訓練同一組輸入才量得到有意義的 encode 成本")
    ckpt_dir = artifacts_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[fail-fast] 找不到 checkpoints：{ckpt_dir}")

    re_value = float(data_kwargs["re_values"][0])
    if "re_norm_scale" not in data_kwargs:
        print("[warn] config 缺 re_norm_scale，fallback 10000.0"
              "（multi-Re ckpt 用其他 scale 會餵錯 re_norm 條件）")
    re_norm_scale = float(data_kwargs.get("re_norm_scale", 10000.0))
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))

    protocol = resolve_protocol(
        mode=a.protocol,
        training_time_strides=training_time_strides_from_config(a.config),
        cli_time_stride=None)
    probe = load_sensors_from_path(data_kwargs["sensor_jsons"][0],
                                   time_stride=protocol.sensor_time_stride)
    aligned = load_for_evaluation(
        data_kwargs["sensor_jsons"][0], Path(data_kwargs["dns_paths"][0]),
        protocol=protocol, viscosity=1.0 / re_value,
        with_pressure="p_mean" in probe["norm_stats"])

    sv = jnp.asarray(aligned.sensor_vals_normalized)
    sp = jnp.asarray(aligned.sensor_pos)
    st = jnp.asarray(aligned.sensor_time)
    T, K = int(sv.shape[0]), int(sv.shape[1])

    model, model_name = build_model(a.arch, model_kwargs, K_sensors=K)
    params = reference_params_for(model, sv, sp, st)
    params, step, prov = restore_eval_params(
        ckpt_dir, a.ckpt, reference_params=params, model=model)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    # ── 搬到量測裝置 ──
    # ckpt 以寫出它的 backend 載入完成後才搬；反過來（先切 backend 再載）會讓
    # orbax 拿不到 sharding 而失敗（job 5421 即為此）。
    if a.compute_device is not None:
        try:
            devs = jax.devices(a.compute_device)
        except RuntimeError as e:
            raise SystemExit(
                f"[fail-fast] backend '{a.compute_device}' 不在 JAX 裝置列表：{e}\n"
                f"  JAX_PLATFORMS 需同時包含載入 ckpt 的 backend 與量測 backend，"
                f"例如 JAX_PLATFORMS=cuda,cpu") from e
        if not devs:
            raise SystemExit(f"[fail-fast] 無可用的 {a.compute_device} 裝置")
        tgt = devs[0]
        params = jax.device_put(params, tgt)
        sv, sp, st = (jax.device_put(x, tgt) for x in (sv, sp, st))
        print(f"[bench] 量測裝置搬至 {tgt.device_kind} ({tgt.platform})")

    # ── encode：整段 sensor 序列 → hidden states，每條軌跡一次 ──
    enc = jax.jit(lambda p_: model.apply(p_, sv, sp, re_norm, st,
                                         method=model.__class__.encode))
    r_enc = _timeit(lambda: enc(params), a.n_repeat)
    h = jax.block_until_ready(enc(params))

    # ── query：已有 hidden states，解碼一批 (x,y,t) ──
    g = a.grid
    xs = jnp.linspace(0.0, 1.0, g, endpoint=False)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    t_q = jnp.full((xy.shape[0],), float(np.asarray(st)[-1]))
    dec = jax.jit(lambda p_, q, tq: model.apply(p_, q, tq, h, st, sp,
                                                method=model.__class__.decode_query))
    r_field = _timeit(lambda: dec(params, xy, t_q), a.n_repeat)

    per_pt_us = 1e6 * r_field["median_s"] / xy.shape[0]
    thr = xy.shape[0] / r_field["median_s"]
    full_seq_s = r_enc["median_s"] + T * r_field["median_s"]

    dev = list(jax.tree_util.tree_leaves(params))[0].devices().pop()
    out = {
        "provenance": {
            "script": "scripts/bench_inference.py",
            "config": str(Path(a.config).resolve()),
            "ckpt_dir": str(ckpt_dir),
            "ckpt_step": int(step),
            "arch": a.arch, "model": model_name, "n_params": int(n_params),
            "ckpt_provenance": {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                                    else str(v)) for k, v in dict(prov).items()},
            "jax_version": jax.__version__,
            "device_kind": dev.device_kind, "platform": dev.platform,
            "host": platform.platform(),
            "T": T, "K": K, "query_grid": g, "query_points": int(xy.shape[0]),
            "n_repeat": a.n_repeat,
            "note": ("median over n_repeat after warm-up; every call "
                     "block_until_ready'd so JAX async dispatch is not timed"),
        },
        "encode": r_enc,
        "field_query": r_field,
        "derived": {
            "per_query_point_us": per_pt_us,
            "throughput_q_per_s": thr,
            "full_sequence_s": full_seq_s,
            "full_sequence_note": f"encode once + {T} field queries",
        },
    }
    op = Path(a.output); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n[bench] {model_name}  params={n_params:,}  step={step}")
    print(f"[bench] device={dev.device_kind} ({dev.platform})  jax={jax.__version__}")
    print(f"[bench] T={T}  K={K}  query grid={g}² = {xy.shape[0]} pts  n={a.n_repeat}\n")
    print(f"  encode (T={T}, K={K})      {1e3*r_enc['median_s']:8.1f} ± {1e3*r_enc['sd_s']:.1f} ms")
    print(f"  field query ({xy.shape[0]} pts)  {1e3*r_field['median_s']:8.1f} ± {1e3*r_field['sd_s']:.1f} ms")
    print(f"  per query point           {per_pt_us:8.2f} us")
    print(f"  throughput                {thr:8.0f} q/s")
    print(f"  full sequence ({T} fields) {full_seq_s/60:8.2f} min")
    print(f"\n[out] {op}")


if __name__ == "__main__":
    main()
