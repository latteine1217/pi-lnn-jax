#!/usr/bin/env python3
"""diag_hoist_kv_projections.py — 把 k/v/token 投影移到 gather 之前，值一樣嗎？省多少？

What:
    `DeepONetCfCDecoder` 先 `h_branch_tokens = h_states[idx]`（[T,K,d] → [N,K,d]），
    再對它跑三個 Dense（`branch_token_proj` → `branch_key_proj` / `branch_value_proj`，
    `models.py:987, 1012-1014`）。Dense 是逐列仿射、gather 只選列，兩者可交換，
    所以「先投影再 gather」算的是同一件事，但列數從 N·K 降到 T·K。

    本腳本用**真 ckpt 的權重、真 encode 出來的 h_states、真 query 時刻**，
    在 GPU 上量兩件事：

      1. **輸出是否逐位元相同。** CPU 上可交換是代數事實，但兩序的矩陣形狀不同
         （[N·K,d] vs [T·K,d]），XLA 很可能選不同 kernel／tiling，
         **點積的累加順序會變** → float32 下未必位元相同。這一題只有 GPU 答得了。
      2. **省多少時間。** 三個投影自己的倍率是 N/T（主線約 30×），但那不是整步的比例。
         同時量一次完整 `decode_query` 與一次 `step_fn` 當分母。

Why:
    model-audit §5 記「純浪費 FLOP，主線尺度 5.945e10 vs 1.986e9（29.9×）」，
    但沒有量過**省下來的時間佔一步多少**，也沒有驗過 GPU 上是否位元相同——
    後者決定這個修改能不能通過 §7.1 的對拍，或會不會報一個良性的紅。

Usage:
    PYTHONPATH=. uv run python scripts/diag_hoist_kv_projections.py \\
        --config configs/exp_pv_les_s42.toml --resume latest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax.pipeline.kolmogorov import build_context, resolve_inputs
from pi_lnn_jax.pipeline.kolmogorov.run import RunJournal, initialize

from diag_grad_accum_al_bias import _restore_without_sanity_check


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="latest")
    p.add_argument("--reps", type=int, default=30, help="計時重複次數（取中位數）")
    p.add_argument("--optimizer", default="schedule_free")
    p.add_argument("--base_optimizer", default="soap")
    p.add_argument("--soap_precondition_frequency", default="2")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _median_ms(fn, reps):
    fn().block_until_ready()                      # warmup + compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main() -> int:
    a = parse_args()
    resolved = resolve_inputs([
        "--config", a.config, "--resume_step", str(a.resume),
        "--optimizer", a.optimizer, "--base_optimizer", a.base_optimizer,
        "--soap_precondition_frequency", str(a.soap_precondition_frequency)])
    cfg = resolved.config
    ctx = build_context(cfg)
    state = _restore_without_sanity_check(ctx, cfg, initialize(ctx, RunJournal()), a.resume)
    params = state.params
    d0 = ctx.re_batches[0]
    st = jnp.asarray(d0.sensor_time)

    # ── 真的 h_states ──
    h_states = ctx.model.apply(params, d0.sensor_vals, d0.sensor_pos, d0.re_norm, st,
                               method=ctx.model.encode)
    T, K, dm = h_states.shape

    # ── 真的 query 時刻 → idx（與 models.py:985-986 逐字相同）──
    n_collo = int(cfg.curriculum.n_collo_end)
    key = jax.random.PRNGKey(0)
    t_q = jax.random.uniform(key, (n_collo,), minval=float(st[0]), maxval=float(st[-1]))
    idx = jnp.clip(jnp.sum(st[None, :] <= t_q[:, None], axis=1).astype(jnp.int32) - 1,
                   0, T - 1)
    N = int(t_q.shape[0])

    # ── 三個投影：用 ckpt 的權重，以模型自己的 nn.Dense 施加 ──
    qd = params["params"]["query_decoder"]
    hid = qd["branch_token_proj"]["kernel"].shape[-1]
    dt, dk, dv = nn.Dense(hid), nn.Dense(hid), nn.Dense(hid)

    def stage(x):
        tok = dt.apply({"params": qd["branch_token_proj"]}, x)
        return (dk.apply({"params": qd["branch_key_proj"]}, tok),
                dv.apply({"params": qd["branch_value_proj"]}, tok))

    now = jax.jit(lambda hs, i: stage(hs[i]))            # 現行：gather 再投影
    hoisted = jax.jit(lambda hs, i: tuple(y[i] for y in stage(hs)))  # 提升：投影再 gather

    k_a, v_a = now(h_states, idx)
    k_b, v_b = hoisted(h_states, idx)
    same_k = bool(jnp.array_equal(k_a, k_b)); same_v = bool(jnp.array_equal(v_a, v_b))
    dk_max = float(jnp.max(jnp.abs(k_a - k_b))); dv_max = float(jnp.max(jnp.abs(v_a - v_b)))
    rel = float(jnp.max(jnp.abs(k_a - k_b)) / (jnp.max(jnp.abs(k_a)) + 1e-30))

    t_now = _median_ms(lambda: now(h_states, idx)[0], a.reps)
    t_hoist = _median_ms(lambda: hoisted(h_states, idx)[0], a.reps)

    # ── 分母：一次完整 decode_query ──
    xy = jax.random.uniform(jax.random.PRNGKey(1), (n_collo, 2))
    dec = jax.jit(lambda p_, x_, t_: ctx.model.apply(
        p_, x_, t_, h_states, st, d0.sensor_pos, method=ctx.model.decode_query))
    t_dec = _median_ms(lambda: dec(params, xy, t_q), a.reps)

    print(f"\n=== {a.config}  step {int(state.step)} ===")
    print(f"  形狀：h_states [T={T}, K={K}, d={dm}]   N={N}   hidden={hid}")
    print(f"  列數：現行 N·K = {N*K:,}   提升後 T·K = {T*K:,}   倍率 {N/T:.1f}×")
    print(f"\n--- 1. 值是否相同 ---")
    print(f"  逐位元相同：k={same_k}  v={same_v}")
    print(f"  max|Δ|：k={dk_max:.3e}  v={dv_max:.3e}   相對 {rel:.3e}")
    print(f"\n--- 2. 時間（{a.reps} 次中位數）---")
    print(f"  三個投影 現行     {t_now:8.3f} ms")
    print(f"  三個投影 提升後   {t_hoist:8.3f} ms   省 {t_now - t_hoist:.3f} ms（{100*(1-t_hoist/t_now):.1f}%）")
    print(f"  一次 decode_query {t_dec:8.3f} ms   ← 分母")
    print(f"  → 提升可省整個 decode 的 {100*(t_now - t_hoist)/t_dec:.1f}%")

    rep = {"config": a.config, "step": int(state.step), "T": T, "K": K, "N": N,
           "hidden": hid, "bitwise_same": {"k": same_k, "v": same_v},
           "max_abs_diff": {"k": dk_max, "v": dv_max}, "max_rel_diff": rel,
           "ms": {"now": t_now, "hoisted": t_hoist, "decode_query": t_dec},
           "saving_frac_of_decode": (t_now - t_hoist) / t_dec}
    out = Path(a.out or (Path(ctx.artifacts_dir) / "diag_hoist_kv_projections.json"))
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
