#!/usr/bin/env python3
"""diag_step_cost.py — 一個訓練步的時間花在哪？

What:
    在 GPU 上以**真實的一步輸入**（各案自己的 `_plan_step`）拆解訓練步成本：

    Kolmogorov
      - `step_fn` 整步（含 M 塊 grad accumulation 與 optimizer 更新）
      - encode（checkpoint：前向＋重算＋反向）與單一塊的 value_and_grad（只 decode）
        ——518b71d 起 M>1 的結構；分項和與整步的差額即 TD-34 開銷 2
      - optimizer 更新（`tx.update` + apply）
      - GradNorm probe（`grad_norm_fn`），並依 `gradnorm_freq` 攤成每步平均
    cylinder
      - `step_fn` 整步
      - GradNorm probe，依 `gradnorm_freq` 攤成每步平均

Why:
    TD-34：分項和與整步之間若有差額，那段時間不屬於任何可歸屬的計算（迴圈結構、排程或
    融合的代價），是下一個加速槓桿的候選。job 6025 用本腳本量到 M>1 時每塊重跑 encode
    （已由 73cf69b／518b71d 修掉），之後拆解改為對應現行結構。cylinder 無分塊，但
    GradNorm 每 25 步對三個 task 各做一次完整 grad，另列攤到每步的比例。

    計時與參數數值無關 → 用 `initialize` 的初始參數，不還原 ckpt（也不需要 orbax）。

Usage:
    PYTHONPATH=. uv run python scripts/diag_step_cost.py --case kolmogorov \\
        --config configs/exp_ksweep_k200_s42.toml \\
        --optimizer schedule_free --base_optimizer soap --soap_precondition_frequency 2
    PYTHONPATH=. uv run python scripts/diag_step_cost.py --case cylinder \\
        --config configs/exp_cyl_cexp002.toml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=["kolmogorov", "cylinder"])
    p.add_argument("--config", required=True)
    p.add_argument("--reps", type=int, default=20, help="計時重複次數（取中位數）")
    # Kolmogorov 的 optimizer 旗標不在 config 內（sbatch 的 TRAIN_FLAGS），SOAP 本身是每步
    # 成本大宗，故必填、不給預設——給錯等於量另一個 optimizer。
    p.add_argument("--optimizer")
    p.add_argument("--base_optimizer")
    p.add_argument("--soap_precondition_frequency")
    p.add_argument("--out", required=True, help="JSON 輸出路徑")
    return p.parse_args()


def _median_ms(fn, reps):
    jax.block_until_ready(fn())                   # warmup + compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


def kolmogorov_setup(a):
    """建 ctx、初始 state 與最後一步的真實 plan。供本檔與其他訓練步診斷共用。

    回傳 (cfg, ctx, state, plan, causal_eps)。
    """
    from pi_lnn_jax.pipeline.kolmogorov import build_context, resolve_inputs
    from pi_lnn_jax.pipeline.kolmogorov.run import (
        RunJournal,
        _open_loop_streams,
        _plan_step,
        initialize,
    )

    missing = [f for f in ("optimizer", "base_optimizer", "soap_precondition_frequency")
               if getattr(a, f) is None]
    if missing:
        raise SystemExit(f"kolmogorov 必須明示 optimizer 旗標（與訓練 sbatch 一致）：缺 {missing}")
    cfg = resolve_inputs([
        "--config", a.config, "--optimizer", a.optimizer,
        "--base_optimizer", a.base_optimizer,
        "--soap_precondition_frequency", str(a.soap_precondition_frequency)]).config
    ctx = build_context(cfg)
    state = initialize(ctx, RunJournal())
    cur, loss = cfg.curriculum, cfg.loss
    if cur.rar_freq > 0:
        raise SystemExit("本診斷不涵蓋 RAR（rar_freq>0）：RAR 取樣本身另有成本，會混進分母")

    _, rar_state, rng_collo, rng_re, rng_crp = _open_loop_streams(
        loss.use_continuous_re_physics, state.rng)

    def _no_rar(*_a, **_k):
        raise AssertionError("rar_freq=0 下不應呼叫 RAR")

    # 取最後一步：collocation 數與 time-marching 上界都已到終值，是訓練大部分時間的形狀
    s = int(cfg.run.steps)
    plan = _plan_step(
        ctx, s, params=state.params, rng_collo=rng_collo, rng_re=rng_re, rng_crp=rng_crp,
        rar_state=rar_state, n_datasets=len(ctx.re_batches),
        sensor_dropout_rate=float(cur.sensor_dropout_rate), rar_residual_fn=_no_rar)
    causal_eps = loss.causal_eps if loss.use_causal else 0.0
    return cfg, ctx, state, plan, causal_eps


def _kolmogorov(a):
    """拆解現行 step_fn（518b71d 起 M>1 時 encode 只算一次＋checkpoint）。

    M>1：整步 ≈ encode（checkpoint：前向＋重算＋反向）＋ M×單塊（只 decode，對 (p, h) 求梯度）
    ＋ optimizer；差額即未解釋的開銷（TD-34 開銷 2）。M=1：整步 ≈ 全量 value_and_grad ＋ optimizer。
    """
    from pi_lnn_jax.pipeline.kolmogorov import assembly

    cfg, ctx, state, plan, causal_eps = kolmogorov_setup(a)
    cur, loss = cfg.curriculum, cfg.loss
    if float(cur.autoreg_pseudo_dt) > 0.0:
        raise SystemExit("本診斷未涵蓋 autoreg 延伸（autoreg_pseudo_dt>0）")
    p, o, tw, lam = state.params, state.opt_state, state.task_weights, state.al_state.lambda_
    rb, crp, mask = plan.re_batch, plan.re_batch_crp, plan.sensor_input_mask

    step_args = (plan.cx, plan.cy, plan.ct, tw, lam, plan.w_phys_now, rb, causal_eps,
                 plan.sensor_idx, crp, mask, plan.sensor_cut_idx)
    t_step = _median_ms(lambda: ctx.step_fn(p, o, *step_args)[2], a.reps)

    M = int(cur.grad_accum_chunks)
    n_c = plan.cx.shape[0] // M
    sidx = plan.sensor_idx
    sidx_c = sidx[: sidx.shape[0] // M] if (M > 1 and sidx is not None) else sidx
    chunk_args = (plan.cx[:n_c], plan.cy[:n_c], plan.ct[:n_c], tw, lam, plan.w_phys_now,
                  rb, causal_eps, sidx_c, crp, mask, plan.sensor_cut_idx)
    encode_fn = assembly._build_encode_fn(
        ctx.model, use_continuous_re_physics=loss.use_continuous_re_physics)
    split = M > 1 and encode_fn is not None

    t_enc = None
    if split:
        enc = lambda p_: encode_fn(p_, rb, crp, mask)
        h_pre = jax.jit(enc)(p)
        cot = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(0), x.shape, x.dtype), h_pre)
        enc_fb = jax.jit(lambda p_: jax.vjp(jax.checkpoint(enc), p_)[1](cot)[0])
        t_enc = _median_ms(lambda: enc_fb(p), a.reps)
        vg = jax.jit(jax.value_and_grad(
            lambda p_, h_: ctx.loss_fn(p_, *chunk_args, h_pre=h_), argnums=(0, 1), has_aux=True))
        t_chunk = _median_ms(lambda: vg(p, h_pre)[0][0], a.reps)
        g = vg(p, h_pre)[1][0]
        h_shape = tuple(h_pre[0].shape)
    else:
        vg = jax.jit(jax.value_and_grad(ctx.loss_fn, has_aux=True))
        t_chunk = _median_ms(lambda: vg(p, *chunk_args)[0][0], a.reps)
        g = vg(p, *chunk_args)[1]
        h_shape = None

    # 新的 opt_state 必須一併回傳：只取 updates 的話，XLA 會把整段 SOAP 狀態更新
    # （二階統計量累加、預條件矩陣更新）當死碼刪掉，量到的只剩套用預條件那一小步（job 6044）。
    def _update(g_, o_, p_):
        updates, o_new = ctx.tx.update(g_, o_, p_)
        return jax.tree_util.tree_map(lambda x, u: x + u, p_, updates), o_new
    upd = jax.jit(_update)
    t_opt = _median_ms(lambda: upd(g, o, p), a.reps)
    # 走過一步的 opt_state：SOAP 的預條件更新依步數條件觸發（precondition_frequency），
    # 只用初始 opt_state 量可能永遠落在同一個分支。
    o_next = ctx.step_fn(p, o, *step_args)[1]
    t_opt_next = _median_ms(lambda: upd(g, o_next, p), a.reps)

    # ── 二分：差額落在「累積」還是「optimizer」──
    if M > 1 and not split:
        raise SystemExit("本診斷未涵蓋無 encode 拆分的 arch 在 M>1 的累積")

    def _grads(p_):
        if not split:  # M=1：與 step_fn 同一條 value_and_grad
            return jax.grad(lambda q: ctx.loss_fn(q, *step_args)[0])(p_)
        chunks = (plan.cx.reshape(M, -1), plan.cy.reshape(M, -1), plan.ct.reshape(M, -1),
                  sidx.reshape(M, -1))
        return assembly.accumulate_value_and_grad(
            ctx.loss_fn, encode_fn, p_, chunks, tw, lam, plan.w_phys_now, rb, causal_eps,
            crp, mask, plan.sensor_cut_idx)[0]
    acc = jax.jit(_grads)
    acc_opt = jax.jit(lambda p_, o_: _update(_grads(p_), o_, p_))
    t_acc = _median_ms(lambda: acc(p), a.reps)
    t_acc_opt = _median_ms(lambda: acc_opt(p, o), a.reps)
    t_step_next = _median_ms(lambda: ctx.step_fn(p, o_next, *step_args)[2], a.reps)

    t_gn = None
    if loss.use_gradnorm and ctx.grad_norm_fn is not None:
        t_gn = _median_ms(lambda: ctx.grad_norm_fn(p, plan.cx, plan.cy, plan.ct, rb, causal_eps),
                          a.reps)

    ts, tc, to = t_step[0], t_chunk[0], t_opt[0]
    parts = (t_enc[0] if t_enc else 0.0) + M * tc + to
    gn_per_step = (t_gn[0] / loss.gradnorm_freq) if t_gn and loss.gradnorm_freq > 0 else 0.0
    print(f"\n=== kolmogorov  {a.config} ===")
    print(f"  M={M}  n_collo={plan.cx.shape[0]} (每塊 {n_c})  "
          f"n_sensor_query={None if sidx is None else sidx.shape[0]}  "
          f"h_states={h_shape}  gradnorm_freq={loss.gradnorm_freq}  encode 提出分塊={split}")
    print(f"\n--- {a.reps} 次中位數（min/max）---")
    rows = [("step_fn 整步", t_step)]
    if t_enc:
        rows.append(("encode（checkpoint）", t_enc))
    rows += [("單塊 value_and_grad", t_chunk), ("optimizer 更新", t_opt),
             ("optimizer（走過一步）", t_opt_next), ("累積梯度（無 opt）", t_acc),
             ("累積＋tx.update", t_acc_opt), ("step_fn（走過一步）", t_step_next)]
    if t_gn:
        rows.append(("GradNorm probe", t_gn))
    for name, t in rows:
        print(f"  {name:<22s} {t[0]:9.2f} ms   ({t[1]:.2f} / {t[2]:.2f})")
    print("\n--- 推論 ---")
    print(f"  分項和 = {parts:9.2f} ms   vs 整步 {ts:.2f} ms   差 {ts - parts:+.2f} ms"
          f"（整步的 {100 * (ts - parts) / ts:+.1f}%）")
    acc_parts = parts - to
    print(f"  累積段：實測 {t_acc[0]:.2f} ms  vs encode＋M×單塊 {acc_parts:.2f} ms"
          f"   差 {t_acc[0] - acc_parts:+.2f} ms")
    print(f"  optimizer 段：累積＋update − 累積 = {t_acc_opt[0] - t_acc[0]:.2f} ms"
          f"   vs 單獨量 {to:.2f}（走過一步 {t_opt_next[0]:.2f}）")
    print(f"  整步 − 累積＋update = {ts - t_acc_opt[0]:+.2f} ms")
    print(f"  GradNorm 攤到每步 = {gn_per_step:.3f} ms = 整步的 {100 * gn_per_step / ts:.2f}%")
    return {"case": "kolmogorov", "config": a.config, "M": M, "n_collo": int(plan.cx.shape[0]),
            "n_sensor_query": None if sidx is None else int(sidx.shape[0]),
            "h_states_shape": list(h_shape) if h_shape else None, "encode_split": split,
            "gradnorm_freq": int(loss.gradnorm_freq),
            "optimizer": [a.optimizer, a.base_optimizer, a.soap_precondition_frequency],
            "ms_median_min_max": {"step": t_step, "encode_ckpt_fb": t_enc, "chunk_vg": t_chunk,
                                  "opt_update": t_opt, "opt_update_after_step": t_opt_next,
                                  "accumulate_only": t_acc, "accumulate_plus_update": t_acc_opt,
                                  "step_after_step": t_step_next,
                                  "gradnorm_probe": t_gn},
            "sum_of_parts_ms": parts, "step_minus_parts_ms": ts - parts,
            "gradnorm_frac_per_step": gn_per_step / ts}


def _cylinder(a):
    from pi_lnn_jax.pipeline.cylinder import build_context, resolve_inputs
    from pi_lnn_jax.pipeline.cylinder.run import (
        _open_loop_streams,
        _plan_step,
        initialize,
    )

    cfg = resolve_inputs(["--config", a.config]).config
    ctx = build_context(cfg)
    state = initialize(ctx)
    loss = cfg.loss
    rk, np_rng = _open_loop_streams(cfg.run.seed)
    # 取一個會觸發 GradNorm 的步：各輸入形狀與一般步相同
    s = int(loss.gradnorm_freq) if loss.gradnorm_freq > 0 else 1
    plan = _plan_step(ctx, s, rk=rk, np_rng=np_rng)
    p, o = state.params, state.opt_state
    args = (plan.cx, plan.cy, plan.ct, state.task_weights, state.al_state.lambda_, plan.sq_idx,
            jnp.asarray(plan.t_max, jnp.float32), plan.bc_in, plan.bc_body, plan.bc_slip,
            plan.bc_bvel)
    t_step = _median_ms(lambda: ctx.step_fn(p, o, *args)[2], a.reps)
    t_gn = None
    if ctx.grad_norm_fn is not None:
        t_gn = _median_ms(lambda: ctx.grad_norm_fn(p, plan.cx, plan.cy, plan.ct, plan.sq_idx),
                          a.reps)
    ts = t_step[0]
    gn_per_step = (t_gn[0] / loss.gradnorm_freq) if t_gn and loss.gradnorm_freq > 0 else 0.0
    print(f"\n=== cylinder  {a.config} ===")
    print(f"  n_collo={plan.cx.shape[0]}  gradnorm_freq={loss.gradnorm_freq}")
    print(f"  step_fn 整步       {ts:9.2f} ms   ({t_step[1]:.2f} / {t_step[2]:.2f})")
    if t_gn:
        print(f"  GradNorm probe     {t_gn[0]:9.2f} ms   ({t_gn[1]:.2f} / {t_gn[2]:.2f})"
              f"   = {t_gn[0] / ts:.2f} 個整步")
    print(f"  GradNorm 攤到每步 = {gn_per_step:.2f} ms = 整步的 {100 * gn_per_step / ts:.1f}%")
    return {"case": "cylinder", "config": a.config, "n_collo": int(plan.cx.shape[0]),
            "gradnorm_freq": int(loss.gradnorm_freq),
            "ms_median_min_max": {"step": t_step, "gradnorm_probe": t_gn},
            "gradnorm_frac_per_step": gn_per_step / ts}


def main() -> int:
    a = parse_args()
    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    rep = _kolmogorov(a) if a.case == "kolmogorov" else _cylinder(a)
    rep["device"] = str(jax.devices()[0])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
