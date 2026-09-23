#!/usr/bin/env python
"""train_baseline_shred — SHRED 深度 baseline 的訓練 + 多-Re 評估（Phase B；lab-server GPU）。

Why:
  SHRED（LSTM over sensor 窗 → shallow decoder → 全場，full-field 監督）是 PI-CON 的外部對照：
  其離散時間 LSTM 對照 PI-CON 的 continuous-time CfC（Tier 0.1 baseline + 1.2b 對照證據）。
  eval 與其他 baseline / PI-CON 共用 metric_artifact.evaluate_field_series 這道 seam（公平比較）。

設計選擇（請於 lab-server 跑前確認；皆影響結果）:
  - 共同輸出格 g（--grid，預設 64）：各 Re 的 DNS 下採樣到 g×g（需 N % g == 0，否則 fail-fast），
    以便跨 Re pool 訓練且 decoder 大小固定。
  - field 正規化：用「pooled train-Re 場」的全域逐通道 stats；預測後反正規化比對 raw DNS。
  - 時間對齊：用 match_sensor_dns_times 把 sensor 時間對到 DNS 時間（fail-fast），SHRED 預測
    窗末端時間的場（causal）；故每個 Re 只評估 t≥L-1 的快照。
  - leakage：train 只用 non-held-out Re；held-out Re 的場「絕不」進訓練（與 PI-CON 同條件）。

※ 本機（Mac）不跑此腳本（no-local-training）；只 syntax 檢查。實跑見 scripts/slurm/train_shred.sbatch.tmpl。

用法（lab-server）:
  uv run python scripts/train_baseline_shred.py --config configs/eval_multi_re_train5_crp.toml \
    --held-out 3000,8000,30000 --grid 64 --window 10 --epochs 200 \
    --output artifacts/shred/metrics.json --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from pi_lnn_jax.baselines_deep import SHRED, build_windows  # noqa: E402
from pi_lnn_jax.data import (  # noqa: E402
    parse_re_set,
    resolve_re_inputs,
)
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity  # noqa: E402
from pi_lnn_jax.metric_artifact import write_compatibility_projection  # noqa: E402

_METRIC_KEYS = ["u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err", "low_band_rel_err", "div_pred_l2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHRED baseline 訓練 + 多-Re 評估")
    p.add_argument("--config", default=None, help="manifest TOML（讀 data_kwargs）")
    p.add_argument("--re-values", default="")
    p.add_argument("--sensor-jsons", default="")
    p.add_argument("--dns-paths", default="")
    p.add_argument("--held-out", default="", help="comma-sep Re（不進訓練）")
    p.add_argument("--re-subset", default="")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）")
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--sensor-time-stride", type=int, default=None,
                   help="明示 stride；follow_training 下與 config 不一致即失敗")
    p.add_argument("--sensor-T", type=int, default=50)
    p.add_argument("--grid", type=int, default=64, help="共同輸出格 g（各 Re 下採樣到 g×g；需整除 N）")
    p.add_argument("--window", type=int, default=10, help="LSTM 時間窗長 L")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--decoder-hidden", default="350,400")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="僅記錄；JAX_PLATFORMS 由 env 控")
    p.add_argument("--output", default="artifacts/shred/metrics.json")
    p.add_argument("--save-params", default="", help="非空則存 flax msgpack 參數")
    return p.parse_args()




def _load_re(sj, dp, args, *, protocol, re_value):
    """依協定載入並對齊；DNS 場下採樣到共同格 g（需整除 N，否則 fail-fast）。

    SHRED 的 sensor 走 normalized 值（decoder 的輸入慣例），故取
    `sensor_vals_normalized` 而非 denormalized 的 `sensor_phys`。
    """
    aligned = load_for_evaluation(sj, dp, protocol=protocol, viscosity=1.0 / re_value,
                                  max_grid=args.grid)
    if aligned.Nprime != args.grid:
        raise ValueError(
            f"DNS N={aligned.N} 無法以整除 stride 降到 g={args.grid}"
            f"（得到 {aligned.Nprime}）；格點錯位會讓跨 Re pool 失效")
    sensor_vals = aligned.sensor_vals_normalized                     # [T,K,2] normalized
    field = np.stack([aligned.dns_u_eval, aligned.dns_v_eval], axis=-1)  # [T,g,g,2] raw
    return {"sensor_vals": sensor_vals, "field": field,
            "t": aligned.dns_t_eval, "K": sensor_vals.shape[1],
            "context": aligned.context, "protocol": aligned.protocol}


def main() -> int:
    args = parse_args()
    inputs = resolve_re_inputs(
        config=args.config or None,
        re_values=args.re_values, sensor_jsons=args.sensor_jsons, dns_paths=args.dns_paths,
    )
    held = parse_re_set(args.held_out)
    subset = parse_re_set(args.re_subset)
    g = args.grid
    L = args.window
    decoder_hidden = tuple(int(x) for x in args.decoder_hidden.split(",") if x.strip())
    out_dim = g * g * 2

    print("=" * 80)
    print(f"train_baseline_shred — grid={g} window={L} hidden={args.hidden} epochs={args.epochs}")
    print(f"  held-out: {sorted(held) or '(none)'}  device={args.device}")
    print("=" * 80)

    # ── 載入全部 Re（一次）──
    training_strides = training_time_strides_from_config(args.config) if args.config else []

    data = {}
    Ks = set()
    for re_index, (re_value, sj, dp) in enumerate(inputs):
        if subset and re_value not in subset:
            continue
        protocol = resolve_protocol(
            mode=args.protocol, training_time_strides=training_strides,
            re_index=re_index, cli_time_stride=args.sensor_time_stride,
            sensor_T=args.sensor_T, reason=args.protocol_reason)
        data[re_value] = _load_re(sj, dp, args, protocol=protocol, re_value=re_value)
        Ks.add(data[re_value]["K"])
    if len(Ks) != 1:
        raise ValueError(f"跨 Re 的 K 不一致：{Ks}（SHRED pool 需相同感測點數）")

    train_res = [r for r in data if r not in held]
    if not train_res:
        raise ValueError("無 train Re（全部被 held-out？）")

    # ── 全域 field stats（只用 train Re，避免 held-out 洩漏）──
    train_fields = np.concatenate([data[r]["field"] for r in train_res], axis=0)  # [ΣT,g,g,2]
    fmean = train_fields.reshape(-1, 2).mean(axis=0)   # [2]
    fstd = train_fields.reshape(-1, 2).std(axis=0) + 1e-8
    print(f"[field stats] mean={fmean}  std={fstd}  (train Re={sorted(train_res)})")

    def _norm(field):   # [.,g,g,2] → flat normalized
        return ((field - fmean) / fstd).reshape(field.shape[0], -1)

    def _denorm(flat):  # [.,out_dim] → [.,g,g,2] physical
        return flat.reshape(-1, g, g, 2) * fstd + fmean

    # ── 組訓練窗（只 train Re）──
    Xs, Ys = [], []
    for r in train_res:
        X, Y = build_windows(data[r]["sensor_vals"], _norm(data[r]["field"]), L)
        Xs.append(X)
        Ys.append(Y)
    X = jnp.asarray(np.concatenate(Xs, axis=0))
    Y = jnp.asarray(np.concatenate(Ys, axis=0))
    print(f"[train] windows={X.shape[0]}  x={X.shape}  y={Y.shape}")

    # ── 建模 + 訓練 ──
    model = SHRED(out_dim=out_dim, hidden=args.hidden, decoder_hidden=decoder_hidden)
    rng = jax.random.PRNGKey(args.seed)
    params = model.init(rng, X[: min(args.batch, X.shape[0])])
    opt = optax.adam(args.lr)
    state = opt.init(params)

    @jax.jit
    def train_step(params, state, xb, yb):
        def loss_fn(p):
            return jnp.mean((model.apply(p, xb) - yb) ** 2)
        loss, g_ = jax.value_and_grad(loss_fn)(params)
        upd, state2 = opt.update(g_, state)
        return optax.apply_updates(params, upd), state2, loss

    n = X.shape[0]
    t0 = time.time()
    for ep in range(args.epochs):
        perm = np.asarray(jax.random.permutation(jax.random.fold_in(rng, ep), n))
        last = None
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            params, state, last = train_step(params, state, X[idx], Y[idx])
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:>4}  mse={float(last):.5e}  ({time.time() - t0:.1f}s)", flush=True)

    # ── 評估（所有 Re；SHRED 只能評 t≥L-1 的窗末端快照）──
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 一個 evaluation run 建一次 recorder：它自建構時抓一次 code revision（與遷移前
    # 的 repository_revision(_REPO_ROOT) 同一個 repo root），並擁有 stamp/命名/寫盤。
    # out_path 已 .resolve()——傳解析後的它，讓寫出的檔名與舊版逐字相同。
    recorder = EvaluationRunRecorder("scripts/train_baseline_shred.py", out_path)

    rows = []
    for re_value in sorted(data):
        is_held = re_value in held
        X_e, _ = build_windows(data[re_value]["sensor_vals"], _norm(data[re_value]["field"]), L)
        pred = _denorm(np.asarray(model.apply(params, jnp.asarray(X_e))))  # [Nwin,g,g,2] physical
        target = data[re_value]["field"][L - 1:]                          # 對齊窗末端 raw

        # recorder 依 identity 重現凍結檔名 {stem}_re{Re}（re-only，無 method/modes），
        # 並自行蓋上 per-re_value 的 evaluation_protocol（caller 不得再供）。
        projection = recorder.record(
            pred[:, :, :, 0], pred[:, :, :, 1], target[:, :, :, 0], target[:, :, :, 1],
            # SHRED 只評窗末端，故 context 的時間軸要切掉前 L-1 個（caller-side 編修）
            context=dataclasses.replace(
                data[re_value]["context"],
                times=tuple(float(t) for t in data[re_value]["t"][L - 1:])),
            protocol=data[re_value]["protocol"],
            identity=RunArtifactIdentity(re=re_value),
            inputs=(("config", str(Path(args.config).resolve())
                     if args.config else "direct"),),
            # SHRED 的預測來自本次訓練——seed 與訓練超參是這筆記錄不可省的身分。
            # 其他 baseline 是確定性重建，這一支不是（見 knowledge 台帳）。
            details_extras={
                "method": "shred", "re_value": re_value, "held_out": is_held,
                "seed": args.seed, "epochs": args.epochs, "window": L,
                "hidden": args.hidden, "batch": args.batch, "lr": args.lr,
                "sensor_time_stride": args.sensor_time_stride,
                "sensor_T": args.sensor_T, "grid": g,
                "K": data[re_value]["K"], "eval_grid": [g, g],
                "n_windows": int(pred.shape[0]),
            },
        )

        mean = projection["metrics_mean"]
        row = {"Re": re_value, "method": "shred", "held_out": is_held,
               "eval_grid": [g, g], "n_windows": int(pred.shape[0]),
               **{k: (float(mean[k]) if k in mean and np.isfinite(mean[k]) else None)
                  for k in _METRIC_KEYS}}
        rows.append(row)
        print(f"[Re={re_value:>9.0f} {'HELD' if is_held else 'train':>5}] "
              f"u_err={row['u_rel_err']:.4f} ke_err={row['ke_rel_err']:.4f} ω_err={row['omega_rel_err']:.4f}",
              flush=True)

    summary = {"model": "shred", "grid": g, "window": L, "hidden": args.hidden,
               "epochs": args.epochs, "held_out": sorted(held),
               "field_mean": fmean.tolist(), "field_std": fstd.tolist(), "rows": rows}
    write_compatibility_projection(out_path, summary)
    print(f"\n[out] {out_path}")

    if args.save_params:
        from flax import serialization
        pp = Path(args.save_params).resolve()
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_bytes(serialization.to_bytes(params))
        print(f"[params] {pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
