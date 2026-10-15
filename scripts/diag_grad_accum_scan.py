#!/usr/bin/env python3
"""diag_grad_accum_scan.py — grad accumulation 的迴圈寫法：時間、記憶體、數值。

What:
    以真實的一步輸入（`diag_step_cost.kolmogorov_setup`）比較三種累積寫法，全部套在
    現行結構上（518b71d 起 M>1 時 encode 只算一次＋checkpoint，逐塊只 decode）：

      scan    `ctx.step_fn` 本身（`optimizers.accumulate_grads`，lax.scan）——現行
      unroll  同一 scan 加 `unroll=True`
      pyloop  Python 迴圈展開，`jnp.stack` 後 `mean(axis=0)`（與 scan 的平均同一算式）

    unroll／pyloop 在 trace 當下把 `assembly.accumulate_grads` 換成對應寫法、trace 完即還原，
    其餘程式碼與 step_fn 共用同一個 `assembly.accumulate_value_and_grad`。
    各量整步時間，並逐葉比對輸出（更新後 params、opt_state、total 與 aux），附同碼兩跑的
    確定性對照（GPU 上同碼兩跑本就不逐位相同，job 6026）。

    `--memory-only`：只編譯不執行，印 XLA memory_analysis；多列 `old`（TD-34 之前每塊
    各自 encode 的結構）作基準。展開放棄了 scan「峰值只佔一塊」的保證，大 K 先確認。

Why:
    TD-34 開銷 2：整步比可歸屬的分項和多出一段時間。job 6026 在舊結構上量到 unroll 對
    K=400 省 17.7%、K=200 只省 2.9%；結構改過之後要重量才知道還剩多少。

Usage:
    PYTHONPATH=. uv run python scripts/diag_grad_accum_scan.py \\
        --config configs/exp_ksweep_k200_s42.toml --out <json> \\
        --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_step_cost import _median_ms, kolmogorov_setup

from pi_lnn_jax.optimizers import accumulate_grads
from pi_lnn_jax.pipeline.kolmogorov import assembly


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--optimizer", required=True)
    p.add_argument("--base_optimizer", required=True)
    p.add_argument("--soap_precondition_frequency", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--memory-only", action="store_true",
                   help="只編譯 old／scan／unroll，印 memory_analysis，不執行")
    return p.parse_args()


def _accum_unroll(vg, params, chunks):
    def _body(_, chunk):
        (loss, aux), grads = vg(params, *chunk)
        return None, (grads, loss, aux)
    _, (g, l, a) = jax.lax.scan(_body, None, chunks, unroll=True)
    m = lambda x: jnp.mean(x, axis=0)
    return jax.tree_util.tree_map(m, g), m(l), jax.tree_util.tree_map(m, a)


def _accum_pyloop(vg, params, chunks):
    M = chunks[0].shape[0]
    outs = [vg(params, *(c[i] for c in chunks)) for i in range(M)]
    stack = lambda *xs: jnp.mean(jnp.stack(xs), axis=0)
    grads = jax.tree_util.tree_map(stack, *[o[1] for o in outs])
    loss = stack(*[o[0][0] for o in outs])
    aux = jax.tree_util.tree_map(stack, *[o[0][1] for o in outs])
    return grads, loss, aux


def _make_step(ctx, M, encode_fn, accum):
    """重建 assembly.step_fn 的 grad_accum>1 分支。

    encode_fn=None → TD-34 之前的結構（每塊呼叫完整 loss_fn）；否則走
    `assembly.accumulate_value_and_grad`，並在 trace 期間把其內部的 accumulate_grads
    換成 `accum`。
    """
    @jax.jit
    def step(params_, opt_state_, cx, cy, ct, task_w, al_lambda, w_phys_now, re_batch,
             causal_eps, sensor_idx, re_batch_crp, sensor_input_mask, sensor_cut_idx):
        chunks = (cx.reshape(M, -1), cy.reshape(M, -1), ct.reshape(M, -1),
                  sensor_idx.reshape(M, -1))
        if encode_fn is None:
            def _vg(p, cx_c, cy_c, ct_c, sidx_c):
                return jax.value_and_grad(ctx.loss_fn, has_aux=True)(
                    p, cx_c, cy_c, ct_c, task_w, al_lambda, w_phys_now, re_batch, causal_eps,
                    sidx_c, re_batch_crp, sensor_input_mask, sensor_cut_idx)
            grads, total, aux = accum(_vg, params_, chunks)
        else:
            orig = assembly.accumulate_grads
            assembly.accumulate_grads = accum     # 只在這次 trace 期間生效
            try:
                grads, total, aux = assembly.accumulate_value_and_grad(
                    ctx.loss_fn, encode_fn, params_, chunks, task_w, al_lambda, w_phys_now,
                    re_batch, causal_eps, re_batch_crp, sensor_input_mask, sensor_cut_idx)
            finally:
                assembly.accumulate_grads = orig
        updates, opt_state_ = ctx.tx.update(grads, opt_state_, params_)
        params_ = jax.tree_util.tree_map(lambda p, u: p + u, params_, updates)
        return (params_, opt_state_, total, *aux)
    return step


def _compare(x, y):
    """逐葉比對：回 (是否全部逐位相同, 最大絕對差, 不同的葉數/總葉數)。"""
    lx, ly = jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)
    if len(lx) != len(ly):
        raise AssertionError(f"pytree 葉數不同：{len(lx)} vs {len(ly)}")
    n_diff, dmax = 0, 0.0
    for a, b in zip(lx, ly):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise AssertionError(f"葉形狀不同：{a.shape} vs {b.shape}")
        if not np.array_equal(a, b, equal_nan=True):
            n_diff += 1
            if np.issubdtype(a.dtype, np.floating):
                dmax = max(dmax, float(np.nanmax(np.abs(a.astype(np.float64) - b))))
    return n_diff == 0, dmax, f"{n_diff}/{len(lx)}"


def _write(a, rep):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n寫入 {out}")


def main() -> int:
    a = parse_args()
    dev = jax.devices()[0]
    print(f"JAX {jax.__version__}  device={dev}")
    cfg, ctx, state, plan, causal_eps = kolmogorov_setup(a)
    M = int(cfg.curriculum.grad_accum_chunks)
    if M <= 1 or plan.sensor_idx is None:
        raise SystemExit(f"本診斷需要 grad_accum_chunks>1 且 sensor_idx 非 None（M={M}）")
    if float(cfg.curriculum.autoreg_pseudo_dt) > 0.0:
        raise SystemExit("本診斷未涵蓋 autoreg 延伸（autoreg_pseudo_dt>0）")
    encode_fn = assembly._build_encode_fn(
        ctx.model, use_continuous_re_physics=cfg.loss.use_continuous_re_physics)
    if encode_fn is None:
        raise SystemExit("本診斷需要有 encode/decode_query 拆分的 arch")
    p, o = state.params, state.opt_state
    args = (plan.cx, plan.cy, plan.ct, state.task_weights, state.al_state.lambda_,
            plan.w_phys_now, plan.re_batch, causal_eps, plan.sensor_idx, plan.re_batch_crp,
            plan.sensor_input_mask, plan.sensor_cut_idx)

    if a.memory_only:
        rep = {"config": a.config, "device": str(dev), "M": M, "memory_bytes": {}}
        print(f"\n=== {a.config}  M={M}  device={dev}  （只編譯）===")
        for k, f in (("old", _make_step(ctx, M, None, accumulate_grads)),
                     ("scan", ctx.step_fn),
                     ("unroll", _make_step(ctx, M, encode_fn, _accum_unroll))):
            ma = f.lower(p, o, *args).compile().memory_analysis()
            m = {n: int(getattr(ma, n)) for n in ("temp_size_in_bytes", "argument_size_in_bytes",
                                                  "output_size_in_bytes", "alias_size_in_bytes")}
            rep["memory_bytes"][k] = m
            print(f"  {k:<8s} temp {m['temp_size_in_bytes'] / 2**30:7.2f} GiB   "
                  f"args {m['argument_size_in_bytes'] / 2**30:6.2f} GiB   "
                  f"out {m['output_size_in_bytes'] / 2**30:6.2f} GiB")
        _write(a, rep)
        return 0

    variants = {
        "scan": ctx.step_fn,
        "unroll": _make_step(ctx, M, encode_fn, _accum_unroll),
        "pyloop": _make_step(ctx, M, encode_fn, _accum_pyloop),
    }
    outs = {k: jax.block_until_ready(f(p, o, *args)) for k, f in variants.items()}
    # 確定性對照：GPU 上同碼兩跑未必逐位相同，不先量這個雜訊，「不逐位相同」無從判讀。
    det_same, det_d, det_fr = _compare(outs["scan"],
                                       jax.block_until_ready(variants["scan"](p, o, *args)))
    rep = {"config": a.config, "device": str(dev), "M": M,
           "determinism_control": {"bitwise_equal": det_same, "diff_leaves": det_fr,
                                   "max_abs_diff": det_d},
           "variants": {}}
    print(f"\n=== {a.config}  M={M}  device={dev} ===")
    print(f"  確定性對照（scan 跑兩次）：逐位相同={det_same}  差異葉 {det_fr}  max|Δ|={det_d:.3e}")
    if not det_same:
        print("  ⚠️ 本裝置非確定性：下表的 max|Δ| 須與確定性對照的量級比較，不能只看逐位")
    print(f"\n  {'variant':<8s} {'median ms':>10s} {'min':>9s} {'max':>9s}   "
          f"{'bitwise=scan':>12s} {'diff leaves':>12s} {'max|Δ|':>10s}")
    for k in ("scan", "unroll", "pyloop"):
        t = _median_ms(lambda f=variants[k]: f(p, o, *args)[2], a.reps)
        s, d, fr = _compare(outs["scan"], outs[k])
        print(f"  {k:<8s} {t[0]:10.2f} {t[1]:9.2f} {t[2]:9.2f}   {s!s:>12s} {fr:>12s} {d:10.3e}")
        rep["variants"][k] = {"ms_median_min_max": t, "bitwise_equal_scan": s,
                              "diff_leaves": fr, "max_abs_diff": d}
    _write(a, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
