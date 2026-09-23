#!/usr/bin/env python3
"""diag_gradnorm_target.py — GradNorm 沒有 floor 的話想把權重放在哪。

What:
    在既有 ckpt 的參數上直接算 GradNorm 的**未受約束目標**
    `w_computed_i = (G_0 + 1e-5·Ḡ) / (G_i + 1e-5·Ḡ)`（`losses.gradnorm_step:98-99`
    的逐字形式），對 `trunk_out` 與 `temporal_encoder` 兩條 reference path 各算一次。
    不重訓、不更新任何 state。

Why:
    job 5633 實測 K=100 主線的兩個 physics 權重在**全部 5 個 seed 上都精確等於**
    `gradnorm_min_weight=0.05` —— clamp 全程在咬，所以看得到的權重是 floor 給的，
    不是平衡規則給的。`04_method.tex` 已如此描述（"floor-limited regulariser"）。

    缺的是「floor 底下還有多深」。那個數字決定 floor 是溫和護欄（目標 0.04，
    clamp 幾乎無作用）還是整個 physics 權重都是它（目標 1e-4，clamp 撐起全部）。
    **這不需要新的訓練 run**：目標值是當前參數的函數。

    兩條 path 都算，因為 `_build_grad_norm_fn` 的 docstring 說 `trunk_out` 會因
    Fourier 特徵二階空間微分的 (2π·k)² 放大讓 `G_phys >> G_data`，使 GradNorm
    **反向運作**；它的預設因此改成 `temporal_encoder`。但 `assembly.py:962` 為了
    bit-identical 契約把它覆蓋回 `trunk_out`——所有歷史 production run 都是這樣跑的。
    兩者的比值就是那個覆蓋的代價，而它從沒被量過（技術債 TD-2）。

    這是**診斷**不是 eval producer：不落 metric artifact。

Usage:
    PYTHONPATH=. uv run python scripts/diag_gradnorm_target.py \\
        --config configs/exp_245_b3_les_T50.toml --resume latest

    需要 GPU（ckpt restore 要推 orbax sharding）→ 走 Slurm。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from pi_lnn_jax.pipeline.kolmogorov import build_context, resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.assembly import _build_grad_norm_fn
from pi_lnn_jax.pipeline.kolmogorov.run import RunJournal, initialize

from diag_grad_accum_al_bias import _restore_without_sanity_check

REF_PATHS = {
    "trunk_out (production；assembly.py:962 的覆蓋值)": ("query_decoder", "trunk_out"),
    "temporal_encoder (_build_grad_norm_fn 的預設，v6 修正)": ("temporal_encoder",),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="latest")
    p.add_argument("--draws", type=int, default=5,
                   help="獨立 collocation 抽樣次數；G 依賴抽樣，單次不足以下判斷。")
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--optimizer", default="schedule_free")
    p.add_argument("--base_optimizer", default="soap")
    p.add_argument("--soap_precondition_frequency", default="2")
    p.add_argument("--grad_accum", type=int, default=None,
                   help="覆寫 cfg.curriculum.grad_accum_chunks（只影響 probe，不重訓）。"
                        "probe 的 data 項在 M>1 時截成前 n0=n_collo/M 個點，M=1 則全量 T*K。")
    p.add_argument("--n_collo_end", type=int, default=None,
                   help="覆寫 collocation 數。data 項不讀 cx/cy/ct，所以這個旗標對 G_data "
                        "的**唯一**影響管道是 n0=n_collo/M —— 是 n0 的單因子旋鈕。")
    p.add_argument("--artifacts_dir", default=None,
                   help="覆寫 ckpt 位置（ckpt 可能落在別的 worktree）。")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    argv = ["--config", args.config, "--resume_step", str(args.resume),
            "--optimizer", args.optimizer, "--base_optimizer", args.base_optimizer,
            "--soap_precondition_frequency", str(args.soap_precondition_frequency)]
    if args.n_collo_end is not None:
        argv += ["--n_collo_end", str(args.n_collo_end)]
    if args.artifacts_dir is not None:
        argv += ["--artifacts_dir", args.artifacts_dir]
    resolved = resolve_inputs(argv)
    cfg = resolved.config
    ctx = build_context(cfg)

    summary_path = Path(ctx.artifacts_dir) / "summary.json"
    if summary_path.exists():
        rec = json.loads(summary_path.read_text()).get("optimizer")
        if rec and rec != ctx.opt_info["name"]:
            raise SystemExit(
                f"optimizer 不符：解析為 {ctx.opt_info['name']!r}，summary.json 記 {rec!r}")

    state = _restore_without_sanity_check(
        ctx, cfg, initialize(ctx, RunJournal()), args.resume)
    if int(state.step) == 0:
        raise SystemExit("沒有還原到 ckpt——目標值是**訓練後**參數的函數。")

    floor = float(cfg.loss.gradnorm_min)
    d0 = ctx.re_batches[0]
    st = np.asarray(d0.sensor_time)
    n_collo = int(cfg.curriculum.n_collo_end)
    grad_accum = (args.grad_accum if args.grad_accum is not None
                  else int(cfg.curriculum.grad_accum_chunks))
    # probe 的 data 項實際看到幾個點、橫跨幾個時刻 —— 判讀 G_data 必須配著這兩個數看。
    T_sensors, K_sensors = int(d0.sensor_vals.shape[0]), int(d0.sensor_vals.shape[1])
    n_probe = (T_sensors * K_sensors if grad_accum <= 1
               else min(n_collo // grad_accum, T_sensors * K_sensors))
    t_cover = n_probe / K_sensors        # 攤平是 time-major，故涵蓋的時刻數 = n_probe/K

    fns = {}
    for label, path in REF_PATHS.items():
        try:
            _sub = state.params["params"]
            for k in path:
                _sub = _sub[k]
        except (KeyError, TypeError):
            print(f"[skip] {label}：ref path {path} 不在此 arch 的參數樹內")
            continue
        fns[label] = _build_grad_norm_fn(
            ctx.model, ctx.ns_fn, ns_fn_baseline=ctx.ns_fn_baseline,
            ref_param_path=path,
            sensor_channel_weights=tuple(cfg.loss.sensor_channel_weights),
            grad_accum=grad_accum,
            cont_gradnorm=cfg.loss.cont_gradnorm,
            t_early_weight=cfg.loss.t_early_weight,
            t_early_threshold=cfg.loss.t_early_threshold,
            T_total=cfg.model.T_total,
            use_causal=cfg.loss.use_causal)

    print(f"\n=== {args.config}  step {int(state.step)} ===")
    print(f"  K={K_sensors}  T={T_sensors}  n_collo={n_collo}  M={grad_accum}")
    print(f"  probe 的 data 項：{n_probe} 點 / {T_sensors * K_sensors}，"
          f"涵蓋 {t_cover:.2f} 個時刻（共 {T_sensors}）")
    print(f"  floor = {floor}   實際權重 = {np.array2string(np.asarray(state.task_weights), precision=6)}")

    report = {"config": args.config, "ckpt_step": int(state.step), "floor": floor,
              "K": K_sensors, "T": T_sensors, "n_collo": n_collo, "grad_accum": grad_accum,
              "n_probe_points": n_probe, "probe_time_coverage": t_cover,
              "actual_weights": np.asarray(state.task_weights).tolist(), "ref_paths": {}}
    key = jax.random.PRNGKey(args.seed)
    for label, fn in fns.items():
        rows = []
        for _ in range(args.draws):
            key, sub = jax.random.split(key)
            k = jax.random.split(sub, 3)
            cx = jax.random.uniform(k[0], (n_collo,))
            cy = jax.random.uniform(k[1], (n_collo,))
            ct = jax.random.uniform(k[2], (n_collo,),
                                    minval=float(st[0]), maxval=float(st[-1]))
            G = np.asarray(fn(state.params, cx, cy, ct, d0,
                             cfg.loss.causal_eps if cfg.loss.use_causal else 0.0))
            # losses.gradnorm_step:98-99 逐字；不重寫一份公式
            mean_G = G.mean()
            w_raw = mean_G / (G + 1e-5 * mean_G)
            rows.append({"G": G.tolist(),
                         "w_computed": (w_raw / max(w_raw[0], 1e-8)).tolist()})
        W = np.array([r["w_computed"] for r in rows])
        Gs = np.array([r["G"] for r in rows])
        med = np.median(W, axis=0)
        print(f"\n── ref path: {label}")
        print(f"   G (median)        {np.array2string(np.median(Gs, axis=0), precision=4)}")
        print(f"   G_phys / G_data   {np.median(Gs, axis=0)[1:] / max(np.median(Gs, axis=0)[0], 1e-30)}")
        print(f"   w_computed 中位數  {np.array2string(med, precision=6)}")
        print(f"   floor / 目標      {[f'{floor / max(m, 1e-30):.1f}x' for m in med[1:]]}"
              f"   ← floor 比目標高幾倍（>>1 = clamp 撐起整個 physics 權重）")
        report["ref_paths"][label] = {
            "G_median": np.median(Gs, axis=0).tolist(),
            "w_computed_median": med.tolist(),
            "w_computed_all": W.tolist(),
            "floor_over_target": [float(floor / max(m, 1e-30)) for m in med[1:]]}

    out = Path(args.out or (Path(ctx.artifacts_dir) / "diag_gradnorm_target.json"))
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
