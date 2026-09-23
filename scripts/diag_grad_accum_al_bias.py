#!/usr/bin/env python3
"""diag_grad_accum_al_bias.py — 生產設定下，梯度累積讓 AL 的 C² 項偏多少。

What:
    在**既有 ckpt 的參數上**、用**生產 config 的分塊數 M**，量 per-chunk 的 AL 約束值
    `al_c` 的塊間變異，並由此給出 AL 二次項的偏差絕對量級與它佔 total loss 的比例。

Why (model-audit B-11):
    `accumulate_grads` 對 M 塊取平均。data / momentum / `t_early` / CRP / λ·C 都是
    mean 型，逐位元等價；唯一破壞者是 `0.5·ρ·C²`——分塊算的是 `0.5·ρ·E[C²]`，
    全量算的是 `0.5·ρ·(E[C])²`，差值**恰好**是

        bias = 0.5·ρ·Var_chunks[C]                                     (1)

    這是恆等式不是近似，所以生產量級不需要跑一次全量 backward 就能得到——而在
    K=800/M=16 上全量 backward 本來就會 OOM（M=16 存在的理由就是這個）。

    稽核與複驗都量過 rel-diff，但倍率隨 M 變動、不是可轉述的常數（M=4 → 2.5×，
    M=32 → 1.9×），且都在小規模隔離設定上。`chapter03` 的 `para:grad_accum` 已改寫成
    只講機制與兩個方向性後果，**不引用任何倍率**。缺的是：在真正跑出論文數字的那組
    設定下，這個偏差到底佔多少。若它佔 total 的 1e-6，B-11 就是純學術；若佔 1e-2，
    `tab:ksweep_series` 沿 K 的 Δ 就有一部分是 M 變了造成的。

    **內部一致性檢查**：`--full-ref` 開啟時另跑一次 M=1 全量，驗證實測的 loss 差是否
    等於式 (1)。K=200(M=4) 記憶體放得下，K=800(M=16) 放不下——所以判準在 K=200 上建立，
    再套用到 K=800。探針算錯時這一步會先崩（式 (1) 對不上實測），這是本腳本唯一的
    自我檢查點。

    這是**診斷**不是 eval producer：不落 metric artifact，不寫進既有 run 的產物目錄。

Usage:
    PYTHONPATH=. uv run python scripts/diag_grad_accum_al_bias.py \\
        --config configs/exp_ksweep_k200_s42.toml --resume latest --full-ref
    PYTHONPATH=. uv run python scripts/diag_grad_accum_al_bias.py \\
        --config configs/exp_ksweep_k800_s42.toml --resume latest

    需要 GPU（ckpt restore 要推 orbax sharding；head node 的 CPU JAX 推不出來）→ 走 Slurm。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.optimizers import accumulate_grads
from pi_lnn_jax.pipeline.kolmogorov import build_context, resolve_inputs
from pi_lnn_jax.losses import gradnorm_weights
from pi_lnn_jax.pipeline.kolmogorov.run import (
    RunJournal, _to_ckpt_state, assert_al_statics_match, initialize,
    physics_weight_at_step)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="latest", help="'latest' 或 step 整數")
    # optimizer 三旗標不影響本診斷要量的東西（loss_fn 與 accumulate_grads 都不碰 tx），
    # 但 `initialize` 會用 `tx.init(params)` 造 restore 的參考 state，而 orbax 對
    # opt_state 結構嚴格檢查——給錯就 restore 不了。預設是生產組合，且由下方的
    # summary.json 對帳把「預設」升級成「已驗證」。
    p.add_argument("--cont_gradnorm", action="store_true",
                   help="把 continuity 當第四個 task 交給 GradNorm（該 run 訓練時用了就要帶，"
                        "否則權重長度不符、restore 的守衛會擋下）")
    p.add_argument("--optimizer", default="schedule_free")
    p.add_argument("--base_optimizer", default="soap")
    p.add_argument("--soap_precondition_frequency", default="2")
    p.add_argument("--draws", type=int, default=8,
                   help="獨立抽樣次數。單次抽樣的 Var 本身有抽樣噪聲，用多次給出散佈。")
    p.add_argument("--seed", type=int, default=20260909, help="抽樣用，與訓練 RNG 無關")
    p.add_argument("--weights-only", action="store_true",
                   help="只還原並印出 GradNorm 權重（全精度）後結束。"
                        "用來判別權重是被 min_weight clamp 住還是停在平衡點——"
                        "訓練 log 只印兩位小數，分不出 0.050 與 0.054。")
    p.add_argument("--grad-controls", action="store_true",
                   help="梯度層的兩個對照（需 --full-ref）："
                        "(a) 重建一份 al_rho=0 的 context——線性項 λ·C 在分塊下精確等價，"
                        "移掉二次項後分塊與全量在數學上完全相同，量到的差就是**純 float32 噪聲**；"
                        "(b) 相鄰兩次獨立抽樣的全量梯度差——那是 optimizer 每步本來就在承受的"
                        "取樣噪聲，用來判斷累積偏差重不重要。")
    p.add_argument("--full-ref", action="store_true",
                   help="另跑 M=1 全量作對照，驗證式 (1)。大 K 會 OOM。")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _global_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def _draw_batch(key, n_collo: int, n_sensor: int, t_lo: float, t_hi: float, n_pool: int):
    """與訓練端同分佈的一次抽樣：collocation 在 [0,1]²×[t_lo,t_hi] 均勻，sensor_idx 均勻整數。

    刻意**不**重放某個訓練步：本量測是同一批資料上「分塊 vs 全量」的配對比較，
    抽樣只影響散佈，不影響配對。重放特定步反而會讓結果綁在那一步的 collocation 上。
    """
    k = jax.random.split(key, 4)
    return (jax.random.uniform(k[0], (n_collo,)),
            jax.random.uniform(k[1], (n_collo,)),
            jax.random.uniform(k[2], (n_collo,), minval=t_lo, maxval=t_hi),
            jax.random.choice(k[3], n_pool, shape=(n_sensor,), replace=False))


def _restore_without_sanity_check(ctx, cfg, state, resume):
    """還原 params / al_state / task_weights，但**不跑** `run.restore` 的 resume sanity check。

    Why 繞開：那個 check 把 `sensor_idx` 傳 None（`run.py:292`），於是拿**整個**
    T×K sensor query 網格過一次 decoder，而訓練走的是 `num_sensor_query_points`
    的 mini-batch。K=200 時是 20200 點對 2000 點，decoder cross-attention 的中介
    張量隨之放大，在 r740 上直接 RESOURCE_EXHAUSTED（job 5627）。

    這不是本診斷的問題也不是可以順手修的東西——`restore()` 在 §7.1 的 bit-identical
    契約覆蓋範圍內，改它要走對拍。這裡只是不觸發它；還原本身仍走同一個
    `ckpt_mgr.restore` 與同一組守衛（AL 靜態值、GradNorm 權重長度），一個都沒少。
    """
    ref = _to_ckpt_state(
        params=state.params, opt_state=state.opt_state, step=0, rng_key=state.rng,
        gn_state=state.gn_state, al_state=state.al_state, lra_state=state.lra_state)
    target = (ctx.ckpt_mgr.latest_step() if resume == "latest" else int(resume))
    if target is None:
        raise SystemExit(f"無 ckpt 可還原：{ctx.ckpt_dir}")
    r = ctx.ckpt_mgr.restore(target, reference_state=ref)

    assert_al_statics_match(r.al_state, cfg.loss)
    task_weights = state.task_weights
    if cfg.loss.use_gradnorm and r.gradnorm_state is not None:
        task_weights = gradnorm_weights(r.gradnorm_state)
        n_expected = 4 if cfg.loss.cont_gradnorm else 3
        if task_weights.shape[0] != n_expected:
            raise SystemExit(
                f"ckpt 的 GradNorm 權重長度 {task_weights.shape[0]} 與 "
                f"cont_gradnorm={cfg.loss.cont_gradnorm} 所需的 {n_expected} 不符。")
    print(f"[restore] step={int(r.step)}  lambda={float(r.al_state.lambda_):.4f}  "
          f"task_weights={np.asarray(task_weights)}")
    return state._replace(params=r.params, opt_state=r.opt_state, step=int(r.step),
                          rng=r.rng_key, gn_state=r.gradnorm_state,
                          al_state=r.al_state, task_weights=task_weights)


def main() -> int:
    args = parse_args()
    resolved = resolve_inputs([
        "--config", args.config, "--resume_step", str(args.resume),
        "--optimizer", args.optimizer,
        "--base_optimizer", args.base_optimizer,
        "--soap_precondition_frequency", str(args.soap_precondition_frequency),
    ] + (["--cont_gradnorm"] if args.cont_gradnorm else []))
    cfg = resolved.config
    M = int(cfg.curriculum.grad_accum_chunks)
    rho = float(cfg.loss.al_rho)
    if M <= 1 and not args.weights_only:
        raise SystemExit(f"{args.config} 的 grad_accum_chunks={M}——沒有分塊就沒有這個偏差。")
    if cfg.loss.al_constraint_mode != "mse" and not args.weights_only:
        raise SystemExit(
            f"al_constraint_mode={cfg.loss.al_constraint_mode}：式 (1) 的推導只對 mse 成立"
            "（signed_mean 的 al_penalty 用的是 cont 而非 al_c²，是另一條路）。")

    ctx = build_context(cfg)

    # SOAP/ScheduleFree 只能從 CLI 開，TOML 沒有任何鍵能開它（assembly 的
    # `assert_soap_betas_are_used` 就是為此而在）。旗標給錯時 opt_state 結構不符，
    # 而那要到 orbax restore 才炸、訊息也不會指向旗標。這裡先跟該 run **自己寫下的**
    # 紀錄對帳：`summary.json["optimizer"]` 逐字就是 `opt_info["name"]`（run.py:1238）。
    summary_path = Path(ctx.artifacts_dir) / "summary.json"
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text()).get("optimizer")
        if recorded and recorded != ctx.opt_info["name"]:
            raise SystemExit(
                f"optimizer 不符：本次解析為 {ctx.opt_info['name']!r}，但 "
                f"{summary_path} 記的是 {recorded!r}。ckpt 的 opt_state 是後者寫的，"
                "restore 會失敗或對不上。請調整 --optimizer / --base_optimizer / "
                "--soap_precondition_frequency 使其一致。")
        print(f"[opt] {ctx.opt_info['name']}  == summary.json 記錄  ✓")
    else:
        print(f"[opt] {ctx.opt_info['name']}  （{summary_path} 不存在，無法對帳）")

    state = initialize(ctx, RunJournal())
    state = _restore_without_sanity_check(ctx, cfg, state, args.resume)
    if int(state.step) == 0:
        raise SystemExit(
            "restore 後 step 仍為 0——沒有還原到任何 ckpt。本診斷量的是**訓練後**參數上的"
            "量級；在 init 參數上量出來的 Var[C] 與生產無關。")

    if args.weights_only:
        w = np.asarray(state.task_weights)
        floor = float(cfg.loss.gradnorm_min)
        # 判別 clamp 與平衡點：clamp 住的權重會**精確**等於 floor（gradnorm_step 的
        # jnp.maximum 直接寫回），停在平衡點的不會。相對距離 < 1e-6 視為被咬住。
        at_floor = [bool(abs(float(x) - floor) <= 1e-6 * floor) for x in w[1:]]
        print(f"\n=== GradNorm 權重 @ step {int(state.step)} ===")
        print(f"  config     {args.config}")
        print(f"  floor      {floor}   init {list(cfg.loss.gradnorm_init_weights)}")
        print(f"  weights    {np.array2string(w, precision=8)}")
        print(f"  被 floor 咬住 {at_floor}   （True = 精確等於 floor，權重由 floor 而非平衡規則決定）")
        return 0

    params = state.params
    lam = state.al_state.lambda_
    task_w = state.task_weights
    n_collo = int(cfg.curriculum.n_collo_end)
    n_sensor = int(ctx.n_sensor_query)
    if n_collo % M or n_sensor % M:
        raise SystemExit(f"n_collo={n_collo} 或 n_sensor={n_sensor} 不能被 M={M} 整除")
    d0 = ctx.re_batches[0]
    st = np.asarray(d0.sensor_time)
    n_pool = int(d0.sensor_vals.shape[0] * d0.sensor_vals.shape[1])

    # physics 權重取**收斂後**的值（warmup/ramp 早已走完），與 ckpt 的參數同一個 regime。
    w_phys = physics_weight_at_step(
        int(state.step), cfg.loss.physics_weight,
        cfg.loss.physics_warmup_steps, cfg.loss.physics_ramp_steps)
    eps = 0.0 if not cfg.loss.use_causal else float(cfg.loss.causal_eps)

    vg = jax.jit(jax.value_and_grad(ctx.loss_fn, has_aux=True))

    def call(cx, cy, ct, sidx):
        (total, aux), g = vg(params, cx, cy, ct, task_w, lam, w_phys,
                             d0, eps, sidx, None, None, None)
        return float(total), aux, g

    print(f"[cfg] {args.config}  M={M}  rho={rho}  lambda={float(lam):.4f}  "
          f"w_phys={w_phys:.4g}  step={int(state.step)}")
    print(f"[batch] n_collo={n_collo} n_sensor={n_sensor} → 每塊 "
          f"{n_collo // M}/{n_sensor // M}")

    rows, batches, prev_g_full = [], [], None
    key = jax.random.PRNGKey(args.seed)
    for i in range(args.draws):
        key, sub = jax.random.split(key)
        cx, cy, ct, sidx = _draw_batch(
            sub, n_collo, n_sensor, float(st[0]), float(st[-1]), n_pool)
        chunks = (cx.reshape(M, -1), cy.reshape(M, -1),
                  ct.reshape(M, -1), sidx.reshape(M, -1))

        # 逐塊：拿到 accumulate_grads 平均掉的 per-chunk al_c
        per = [call(*[c[m] for c in chunks]) for m in range(M)]
        al_c = np.array([float(a[5]) for _, a, _ in per])
        tot_chunks = np.array([t for t, _, _ in per])

        # 同一批、走真正的 accumulate_grads（不是我自己平均）——兩者不合就是探針壞了
        def _vg(p, a, b, c, d):
            return jax.value_and_grad(ctx.loss_fn, has_aux=True)(
                p, a, b, c, task_w, lam, w_phys, d0, eps, d, None, None, None)
        g_acc, tot_acc, aux_acc = accumulate_grads(_vg, params, chunks)
        # 這個差**不是** 0：lax.scan 的累加順序與 numpy 的 mean 不同，float32 下
        # 相對差落在 1e-5 量級（job 5629 實測 2.4e-5）。它有兩個用途：
        #   1. 結構檢查——差到 1e-3 以上代表我切錯塊或呼叫錯函式，不是捨入。
        #   2. **它就是本次量測的噪聲地板。** 底下的 bias_share 若比它還小，
        #      誠實的結論是「低於 float32 解析度」，而不是報那個小數字。
        accum_vs_loop = abs(float(tot_acc) - tot_chunks.mean()) / abs(tot_chunks.mean())
        assert accum_vs_loop <= 1e-3, (
            f"逐塊平均 {tot_chunks.mean():.8e} 與 accumulate_grads {float(tot_acc):.8e} "
            f"相對差 {accum_vs_loop:.2e} —— 遠大於 float32 捨入，是切塊或呼叫錯了")

        bias = 0.5 * rho * float(al_c.var())            # 式 (1)
        row = {
            "draw": i,
            "al_c_mean": float(al_c.mean()),
            "al_c_std": float(al_c.std()),
            "al_c_cv": float(al_c.std() / max(abs(al_c.mean()), 1e-30)),
            "total_chunked": float(tot_acc),
            "al_quadratic_chunked": 0.5 * rho * float((al_c ** 2).mean()),
            "al_quadratic_full_implied": 0.5 * rho * float(al_c.mean() ** 2),
            "bias_abs": bias,
            "bias_share_of_total": bias / abs(float(tot_acc)),
            "float32_noise_floor": accum_vs_loop,
        }

        if args.full_ref:
            tot_full, aux_full, g_full = call(cx, cy, ct, sidx)
            batches.append((cx, cy, ct, sidx))
            # 相鄰兩次獨立抽樣的全量梯度差＝訓練每步本來就有的取樣噪聲尺度。
            # 累積偏差要與**這個**比，而不是與 0 比。
            if prev_g_full is not None:
                row["grad_draw_to_draw"] = (
                    _global_norm(jax.tree_util.tree_map(lambda a, b: a - b, g_full, prev_g_full))
                    / max(_global_norm(g_full), 1e-30))
            prev_g_full = g_full
            row["total_full"] = tot_full
            row["loss_gap_measured"] = float(tot_acc) - tot_full
            row["loss_gap_predicted"] = bias
            row["identity_rel_err"] = abs(
                row["loss_gap_measured"] - bias) / max(abs(bias), 1e-30)
            row["grad_rel_diff"] = (
                _global_norm(jax.tree_util.tree_map(lambda a, b: a - b, g_acc, g_full))
                / max(_global_norm(g_full), 1e-30))
        rows.append(row)
        print(f"  draw {i}: al_c={row['al_c_mean']:.4e} cv={row['al_c_cv']:.3f}  "
              f"bias={bias:.4e}  share={row['bias_share_of_total']:.3e}"
              + (f"  identity_rel_err={row['identity_rel_err']:.2e}"
                 f"  grad_rel={row['grad_rel_diff']:.3e}" if args.full_ref else ""))

    # ── (a) ρ=0 對照：梯度層的純噪聲地板 ───────────────────────────────────
    # AL 項是 λ·C + 0.5ρ·C²。C 是平均，所以線性項在分塊下**精確等價**；唯一破壞
    # 等價的是二次項。ρ=0 時分塊與全量在數學上完全相同，量到的任何差都是捨入。
    # loss 層有對照（逐塊平均 vs accumulate_grads），梯度層先前**沒有**——
    # 所以 grad_rel_diff 一直無法歸因。這一段就是補那個對照。
    if args.grad_controls:
        if not args.full_ref:
            raise SystemExit("--grad-controls 需要 --full-ref（要有全量梯度才比得了）")
        import dataclasses
        print("\n[rho=0] 重建 context（只改 al_rho；模型與資料相同，params 沿用）…")
        cfg0 = dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, al_rho=0.0))
        ctx0 = build_context(cfg0)
        assert float(cfg0.loss.al_rho) == 0.0
        vg0 = jax.jit(jax.value_and_grad(ctx0.loss_fn, has_aux=True))

        def _vg0(p, a, b, c, d):
            return jax.value_and_grad(ctx0.loss_fn, has_aux=True)(
                p, a, b, c, task_w, lam, w_phys, d0, eps, d, None, None, None)

        for r, (cx, cy, ct, sidx) in zip([x for x in rows if "grad_rel_diff" in x], batches):
            chunks = (cx.reshape(M, -1), cy.reshape(M, -1),
                      ct.reshape(M, -1), sidx.reshape(M, -1))
            g_acc0, _, _ = accumulate_grads(_vg0, params, chunks)
            (_, _), g_full0 = vg0(params, cx, cy, ct, task_w, lam, w_phys,
                                  d0, eps, sidx, None, None, None)
            r["grad_rel_diff_rho0"] = (
                _global_norm(jax.tree_util.tree_map(lambda a, b: a - b, g_acc0, g_full0))
                / max(_global_norm(g_full0), 1e-30))
            print(f"  draw {r['draw']}: grad_rel(ρ={rho}) = {r['grad_rel_diff']:.3e}   "
                  f"grad_rel(ρ=0) = {r['grad_rel_diff_rho0']:.3e}")

    def agg(k):
        v = np.array([r[k] for r in rows if k in r])
        return {"mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}

    summary = {k: agg(k) for k in
               ("al_c_cv", "bias_abs", "bias_share_of_total", "float32_noise_floor")
               + (("identity_rel_err", "grad_rel_diff") if args.full_ref else ())
               + (("grad_draw_to_draw",) if args.full_ref and args.draws > 1 else ())
               + (("grad_rel_diff_rho0",) if args.grad_controls else ())}
    # 判讀的分水嶺：bias 佔比與噪聲地板誰大。小於就是量不到，不是「很小」。
    share, floor = summary["bias_share_of_total"]["mean"], summary["float32_noise_floor"]["mean"]
    if args.grad_controls:
        gr, g0 = summary["grad_rel_diff"]["mean"], summary["grad_rel_diff_rho0"]["mean"]
        dd = summary.get("grad_draw_to_draw", {}).get("mean", float("nan"))
        summary["grad_verdict"] = (
            f"grad_rel(ρ={rho}) {gr:.3e} vs ρ=0 對照 {g0:.3e} = {gr / max(g0, 1e-30):.2f}× —— "
            + ("**與純噪聲同量級，AL 項在梯度層不可歸因**" if gr < 2 * g0
               else "**高於純噪聲，AL 項在梯度層可量測**")
            + f"；而取樣噪聲（相鄰抽樣的全量梯度差）是 {dd:.3e}"
            + (f"，累積偏差只有它的 {gr / dd:.1e}" if dd == dd and dd > 0 else ""))
    summary["verdict"] = (
        f"bias_share {share:.3e} / noise_floor {floor:.3e} = {share / max(floor, 1e-30):.2f}× —— "
        + ("**低於 float32 解析度，本設定下量不到**" if share < floor
           else "高於噪聲地板，是可量測的真實偏差"))
    print("\n=== 匯總 ===")
    for k, v in summary.items():
        if isinstance(v, str):
            print(f"  {k:24s} {v}")
        else:
            print(f"  {k:24s} mean={v['mean']:.4e}  [{v['min']:.3e}, {v['max']:.3e}]")

    report = {"config": args.config, "M": M, "rho": rho,
              "lambda": float(lam), "ckpt_step": int(state.step),
              "n_collo": n_collo, "n_sensor": n_sensor,
              "full_ref": bool(args.full_ref), "draws": rows, "summary": summary}
    out = Path(args.out or (Path(ctx.artifacts_dir) / "diag_grad_accum_al_bias.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
