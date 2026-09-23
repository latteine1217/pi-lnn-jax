#!/usr/bin/env python
"""eval_gappy_cross_re — cross-Re gappy-POD（pooled train-Re basis，共同格 g）。

Why:
  公平對照。evaluate_baselines.py 的 gappy 用 per-Re basis（吃 held-out Re 自己的軌跡 → 樂觀）。
  本腳本只用 **train-Re** 的 POD basis（不給 held-out Re 任何自己的資料），檢驗 gappy 是否同樣
  在 held-out Re 崩潰 → 強化「held-out 不轉移是學習式法的通病、非 PI-CON 特有」（§6.4 limitations）。

設定:
  - 所有 Re 場下採樣到共同格 g（需 N % g == 0，否則 fail-fast），以便 pool 成單一 POD basis。
  - basis = pooled train-Re 場（held-out Re 不在 basis → 乾淨 cross-Re；train Re 在 basis → in-dist 樂觀，僅供參考）。
  - 每個 Re 用「該 Re 自己的 sensors」重建（basis 共用），compute_metrics vs 該 Re DNS（共同格）。

※ 本機只有單一 Re 資料 → 此腳本須在 lab-server 跑（全 8-Re）。CPU。
用法:
  uv run python scripts/eval_gappy_cross_re.py --config configs/eval_multi_re_train5_crp.toml \
    --held-out 3000,8000,30000 --grid 64 --output artifacts/baseline_eval/gappy_cross_re.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.baseline_eval import select_pod_modes_by_validation  # noqa: E402
from pi_lnn_jax.baselines import GappyPOD  # noqa: E402
from pi_lnn_jax.data import parse_re_set, resolve_re_inputs  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity  # noqa: E402
from pi_lnn_jax.metric_artifact import write_compatibility_projection  # noqa: E402

_METRIC_KEYS = ["u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err", "low_band_rel_err", "div_pred_l2"]


def parse_args():
    p = argparse.ArgumentParser(description="cross-Re gappy-POD（train-Re basis）")
    p.add_argument("--config", required=True)
    p.add_argument("--held-out", default="")
    p.add_argument("--re-subset", default="")
    p.add_argument("--grid", type=int, default=64, help="共同格 g（各 Re 下採樣到 g×g；需整除 N）")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）")
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--sensor-time-stride", type=int, default=None,
                   help="明示 stride；follow_training 下與 config 不一致即失敗")
    p.add_argument("--sensor-T", type=int, default=50)
    p.add_argument("--pod-modes", type=int, default=0, help="0=validation 選（pooled basis + 首個 train Re sensors）")
    p.add_argument("--output", default="artifacts/baseline_eval/gappy_cross_re.json")
    return p.parse_args()


def _load_re(sj, dp, g, args, *, protocol, re_value):
    """依協定載入並對齊；空間下採樣到共同格 g（需整除 N，否則 fail-fast）。"""
    aligned = load_for_evaluation(sj, dp, protocol=protocol, viscosity=1.0 / re_value,
                                  max_grid=g)
    if aligned.Nprime != g:
        raise ValueError(
            f"DNS N={aligned.N} 無法以整除 stride 降到共同格 g={g}"
            f"（得到 {aligned.Nprime}）；格點錯位會讓跨 Re 的比較失效")
    return {"sensor_phys": aligned.sensor_phys, "sensor_pos": aligned.sensor_pos,
            "u": aligned.dns_u_eval, "v": aligned.dns_v_eval,
            "t": aligned.dns_t_eval, "K": aligned.K,
            "context": aligned.context, "protocol": aligned.protocol}


def main():
    args = parse_args()
    inputs = resolve_re_inputs(config=args.config)
    held = parse_re_set(args.held_out)
    subset = parse_re_set(args.re_subset)
    g = args.grid

    training_strides = training_time_strides_from_config(args.config)

    data = {}
    for re_index, (re_value, sj, dp) in enumerate(inputs):
        if subset and re_value not in subset:
            continue
        protocol = resolve_protocol(
            mode=args.protocol, training_time_strides=training_strides,
            re_index=re_index, cli_time_stride=args.sensor_time_stride,
            sensor_T=args.sensor_T, reason=args.protocol_reason)
        data[re_value] = _load_re(sj, dp, g, args, protocol=protocol, re_value=re_value)

    train_res = [r for r in data if r not in held]
    if not train_res:
        raise ValueError("無 train Re")

    # pooled train-Re basis（held-out 不在內）
    pooled_u = np.concatenate([data[r]["u"] for r in train_res], axis=0)  # [ΣT,g,g]
    pooled_v = np.concatenate([data[r]["v"] for r in train_res], axis=0)
    K0 = data[train_res[0]]["K"]
    if args.pod_modes > 0:
        r_modes = min(args.pod_modes, 2 * K0, pooled_u.shape[0])
    else:
        grid = [max(1, K0 // 8), K0 // 4, K0 // 2, K0, 3 * K0 // 2, 2 * K0]
        r_modes, _ = select_pod_modes_by_validation(
            pooled_u, pooled_v, data[train_res[0]]["sensor_pos"], grid, val_frac=0.3, seed=0)
    print(f"[cross-Re gappy] g={g}  pooled snapshots={pooled_u.shape[0]}  modes={r_modes}  "
          f"train Re={sorted(train_res)}", flush=True)

    model = GappyPOD(n_modes=r_modes).fit(pooled_u, pooled_v)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # 一個 evaluation run 建一次 recorder：它自建構時抓一次 code revision（與遷移前
    # 的 repository_revision(_REPO_ROOT) 同一個 repo root），並擁有 stamp/命名/寫盤。
    # outp 刻意不 .resolve()——傳未解析的它，讓寫出的檔名與舊版逐字相同。
    recorder = EvaluationRunRecorder("scripts/eval_gappy_cross_re.py", outp)

    rows = []
    for re_value in sorted(data):
        is_held = re_value in held
        u_pred, v_pred = model.reconstruct(data[re_value]["sensor_phys"], data[re_value]["sensor_pos"])
        ue, ve = data[re_value]["u"], data[re_value]["v"]

        # recorder 依 identity 重現凍結檔名 {stem}_re{Re}（re-only，無 method/modes），
        # 並自行蓋上 per-re_value 的 evaluation_protocol（caller 不得再供）。
        projection = recorder.record(
            u_pred, v_pred, ue, ve,
            context=data[re_value]["context"],
            protocol=data[re_value]["protocol"],
            identity=RunArtifactIdentity(re=re_value),
            inputs=(("config", str(Path(args.config).resolve())),),
            details_extras={
                "method": "gappy_cross_re", "re_value": re_value,
                "held_out": is_held, "in_basis": re_value in train_res,
                "basis_train_re": sorted(train_res),
                "pod_modes": int(r_modes),
                "pooled_snapshots": int(pooled_u.shape[0]),
                "sensor_time_stride": args.sensor_time_stride,
                "sensor_T": args.sensor_T, "grid": g,
                "K": data[re_value]["K"], "eval_grid": [g, g],
            },
        )

        mean = projection["metrics_mean"]
        row = {"Re": re_value, "method": "gappy_cross_re", "held_out": is_held,
               "in_basis": (re_value in train_res), "eval_grid": [g, g],
               **{k: (float(mean[k]) if k in mean and np.isfinite(mean[k]) else None)
                  for k in _METRIC_KEYS}}
        rows.append(row)
        flag = "(in basis)" if re_value in train_res else "(NOT in basis)"
        print(f"[Re={re_value:>9.0f} {'HELD' if is_held else 'train':>5} {flag:>14}] "
              f"u_err={row['u_rel_err']:.4f} ke_err={row['ke_rel_err']:.4f} ω_err={row['omega_rel_err']:.4f}",
              flush=True)

    def _mean(h):
        vals = [r["u_rel_err"] for r in rows if r["held_out"] == h and isinstance(r.get("u_rel_err"), (int, float))]
        return sum(vals) / len(vals) if vals else None
    print(f"\nmean u_err — in-train(in basis)={_mean(False)}  held-out(NOT in basis)={_mean(True)}")

    write_compatibility_projection(
        outp, {"grid": g, "pod_modes": int(r_modes), "rows": rows})
    print(f"[out] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
