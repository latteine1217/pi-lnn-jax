#!/usr/bin/env python
"""evaluate_multi_re — 單一 multi-Re checkpoint 對多個 Re 各跑 DNS reconstruction，
輸出 train (in-distribution) vs held-out 泛用性表。

Why:
  multi-Re 訓練（configs/exp_multi_re_train5.toml）的核心產出是「對未見 Re 的泛用性」。
  訓練期 mid-eval 只看 d0=Re1000；本腳本載最終 ckpt，對 config 列的每個 Re 各跑全場
  reconstruction vs 該 Re 的 DNS，量 in-dist 與 held-out 的差距。

正確性要點（高風險 eval，對齊 CLAUDE.md evalStrictness）:
  - re_norm 用 config 的 re_norm_scale（multi-Re=1e6），與訓練一致；用錯 scale → 餵錯條件。
  - 每個 Re 的 DNS 必須與其 sensors 同一 realization（皆 sweep；sensors 由該 DNS QR 產生）。
  - sensor 輸入截到 T=sensor_T（預設 50）對齊訓練 cadence；feed 不同長度 = encoder OOD。
  - 缺 DNS/ckpt 一律 fail-fast（不靜默 skip）；分階段評估用 --re-subset 明示子集。

用法:
  # 全 8 Re（需所有 DNS 已同步）
  uv run python scripts/evaluate_multi_re.py --config configs/eval_multi_re_train5.toml \\
    --ckpt latest --held-out 3000,8000,30000
  # 先跑 held-out + Re1000（DNS 子集）
  uv run python scripts/evaluate_multi_re.py --config configs/eval_multi_re_train5.toml \\
    --re-subset 1000,3000,8000,30000 --held-out 3000,8000,30000
  # 高 Re 用 grid-stride 控成本（在 N//s 網格上比對）
  uv run python scripts/evaluate_multi_re.py --config configs/eval_multi_re_train5.toml \\
    --grid-stride 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params  # noqa: E402
from pi_lnn_jax.config import load_config  # noqa: E402
from pi_lnn_jax.data import load_sensors_from_path  # noqa: E402
from pi_lnn_jax.evaluate import evaluate_time_series  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.metric_artifact import (  # noqa: E402
    SourceProvenance,
    build_metric_artifact,
    freeze_provenance_details,
    repository_revision,
    sparsity_yardsticks,
    write_compatibility_projection,
    write_metric_artifact,
)
from pi_lnn_jax.model_factory import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-Re checkpoint 泛用性評估")
    p.add_argument("--config", required=True, help="eval config TOML（含 8-Re sensor/dns 列 + arch + re_norm_scale）")
    p.add_argument("--ckpt", default="latest", help="ckpt step（'latest' 或整數）")
    p.add_argument("--held-out", default="", help="comma-sep Re，標記為 held-out（其餘為 in-train）")
    p.add_argument("--re-subset", default="", help="comma-sep Re，只評估子集（分階段用）；空=全部")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）")
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--sensor-time-stride", type=int, default=None,
                   help="明示 stride；follow_training 下與 config 不一致即失敗")
    p.add_argument("--sensor-T", type=int, default=50, help="sensor 輸入截到此 T（對齊訓練 cadence）")
    p.add_argument("--grid-stride", type=int, default=1, help="全域 DNS 空間網格子採樣（在 N//s 網格比對）")
    p.add_argument("--max-grid", type=int, default=0, help="per-Re eval 網格上限 N'（>0 則自動 stride 到 ≤N'；低 Re 保 native、高 Re 才降，蓋過 --grid-stride）")
    p.add_argument("--artifacts_dir", default=None, help="覆寫 ckpt 所在 artifacts_dir")
    p.add_argument("--output", default=None, help="輸出 JSON 路徑（預設 artifacts_dir/multi_re_eval/metrics.json）")
    p.add_argument("--snap-held-out-re-norm", action="store_true",
                   help="診斷：held-out Re 改餵『最近 in-train Re 的 re_norm』（sensors/DNS 不變）。"
                        "隔離『未見 re_norm 值』是否為失效主因——若 u_err 大降則 FiLM 離散記憶說成立。")
    return p.parse_args()


def _parse_re_list(s: str) -> set[float]:
    return {float(x) for x in s.split(",") if x.strip()} if s.strip() else set()


def _f(x, fmt=".4f") -> str:
    """None-safe 數值格式（evaluate_time_series 的 metrics_mean 只聚合部分鍵）。"""
    return format(x, fmt) if isinstance(x, (int, float)) else "n/a"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_kwargs = cfg["model_kwargs"]
    data_kwargs = cfg["data_kwargs"]
    train_kwargs = cfg["train_kwargs"]

    re_values = [float(r) for r in data_kwargs.get("re_values", [])]
    sensor_jsons = data_kwargs.get("sensor_jsons", [])
    dns_paths = data_kwargs.get("dns_paths", [])
    if "re_norm_scale" not in data_kwargs:
        print("[warn] config data_kwargs 缺 re_norm_scale，fallback 10000.0；"
              "multi-Re ckpt 若用其他 scale（如 1e6）會餵錯 re_norm 條件。", flush=True)
    re_norm_scale = float(data_kwargs.get("re_norm_scale", 10000.0))
    if not (len(re_values) == len(sensor_jsons) == len(dns_paths)) or not re_values:
        raise ValueError(
            f"re_values({len(re_values)})/sensor_jsons({len(sensor_jsons)})/"
            f"dns_paths({len(dns_paths)}) 長度需相等且非空"
        )

    held_out = _parse_re_list(args.held_out)
    subset = _parse_re_list(args.re_subset)

    artifacts_dir = Path(
        args.artifacts_dir if args.artifacts_dir else train_kwargs.get("artifacts_dir", "artifacts/run")
    ).resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    out_path = Path(args.output) if args.output else artifacts_dir / "multi_re_eval" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"evaluate_multi_re — config={Path(args.config).name}")
    print(f"  artifacts:     {artifacts_dir}")
    print(f"  re_norm_scale: {re_norm_scale:.0f}  (re_norm = log(Re)/log(scale))")
    print(f"  held-out:      {sorted(held_out) or '(none)'}")
    print(f"  grid-stride:   {args.grid_stride}  max-grid: {args.max_grid or '(off)'}  sensor-T: {args.sensor_T}")
    print("=" * 80)

    # ── Model + restore params（一次；params 跨 Re 共用，模型靠 re_norm 條件化）──
    d0 = load_sensors_from_path(sensor_jsons[0], time_stride=args.sensor_time_stride)
    K = int(d0["sensor_pos"].shape[0])
    # 走 factory 而非直接建構：訓練端與 eval 端必須是同一個物件（見 scripts/CLAUDE.md §2）。
    # 本腳本沒有 --arch，只支援 B3——這裡把那個假設寫出來，而不是靠「剛好只建 B3」。
    # 拿 B0/B2 的 config 餵它時，restore 會因參數樹不符而擋下（訊息是樹不符，不是 arch 用錯）。
    model, _ = build_model("liquid", model_kwargs, K_sensors=K)
    init_params = reference_params_for(
        model,
        jnp.asarray(d0["sensor_vals"][: args.sensor_T]),
        jnp.asarray(d0["sensor_pos"]),
        jnp.asarray(d0["sensor_time"][: args.sensor_T]),
    )
    params, restored_step, ckpt_provenance = restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=init_params, model=model)

    # 稀疏度標尺（單一真實來源；背書 sec:count / tab:kscaling）：把 K 釘在 Foias–Temam
    # determining-node 下界與 2D Nyquist 解析度上界之間。不重算 DS critical（regime 不符文獻）。
    yard = sparsity_yardsticks(K)
    print(f"[yardstick] K={yard['K']}  Nyquist k_max=√(K/π)={yard['nyquist_kmax']:.2f}  "
          f"determining-node floor={yard['determining_nodes']} (Foias–Temam 1984)  "
          f"K/K_dn={yard['determining_ratio']:.1f}×\n")

    # 診斷用：in-train Re 的 re_norm 集合（--snap-held-out-re-norm 把 held-out snap 到最近者）
    in_train_re_norms = sorted(
        float(np.log(rv) / np.log(re_norm_scale)) for rv in re_values if rv not in held_out
    )

    training_strides = training_time_strides_from_config(args.config)

    rows = []
    metric_artifacts = []
    revision, code_dirty = repository_revision(_REPO_ROOT)
    for re_index, (re_value, sj, dp) in enumerate(zip(re_values, sensor_jsons, dns_paths)):
        if subset and re_value not in subset:
            continue
        is_held = re_value in held_out
        re_norm_true = float(np.log(re_value) / np.log(re_norm_scale))
        re_norm = re_norm_true
        if is_held and args.snap_held_out_re_norm and in_train_re_norms:
            re_norm = min(in_train_re_norms, key=lambda x: abs(x - re_norm_true))

        protocol = resolve_protocol(
            mode=args.protocol, training_time_strides=training_strides,
            re_index=re_index, cli_time_stride=args.sensor_time_stride,
            sensor_T=args.sensor_T, reason=args.protocol_reason)
        # 對齊、空間下採樣與 evaluation context 全由協定層負責；本檔只消費結果。
        # 先前這裡自行推導 t_stride 再以 allclose 驗，那是第三份對齊實作
        # （另兩份在 evaluate_exp245 與 baseline_eval）。
        aligned = load_for_evaluation(
            sj, dp, protocol=protocol, viscosity=1.0 / re_value,
            grid_stride=args.grid_stride, max_grid=args.max_grid)
        T = aligned.T
        sensor_vals = jnp.asarray(aligned.sensor_vals_normalized)
        sensor_pos = jnp.asarray(aligned.sensor_pos)
        sensor_time = jnp.asarray(aligned.sensor_time)
        norm_stats = aligned.norm_stats
        dns_u, dns_v, dns_t = aligned.dns_u_eval, aligned.dns_v_eval, aligned.dns_t_eval

        tag = "HELD-OUT" if is_held else "in-train"
        snap_note = f" (snap {re_norm_true:.4f}→{re_norm:.4f})" if re_norm != re_norm_true else ""
        print(f"[Re={re_value:>9.0f} {tag}] re_norm={re_norm:.4f}{snap_note}  sensor T={T},K={K}  "
              f"eval grid={dns_u.shape[1]}x{dns_u.shape[2]} × {dns_u.shape[0]}t …", flush=True)
        t0 = time.time()
        out = evaluate_time_series(
            model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
            norm_stats, dns_u, dns_v, dns_t, verbose=False,
            # ν=1/Re，per-Re 各自計算：band 邊界的 k_η 依 Re 而異，共用一個會錯
            nu=1.0 / re_value,
        )
        artifact_path = out_path.with_name(
            f"{out_path.stem}_re{re_value:g}.metric_artifact.json"
        )
        metric_artifacts.append((
            artifact_path,
            build_metric_artifact(
                out,
                context=aligned.context,
                provenance=SourceProvenance(
                    producer="scripts/evaluate_multi_re.py",
                    code_revision=revision,
                    code_dirty=code_dirty,
                    inputs=(
                        ("config", str(Path(args.config).resolve())),
                        ("checkpoint_dir", str(ckpt_dir)),
                        ("sensor", str(sj)),
                        ("dns", str(dp)),
                    ),
                    details=freeze_provenance_details({
                        "checkpoint": ckpt_provenance,
                        "checkpoint_step": restored_step,
                        "re_value": re_value,
                        "evaluation_protocol": protocol.to_provenance(),
                        "re_norm": re_norm,
                        "re_norm_true": re_norm_true,
                    }),
                ),
            ),
        ))
        mm = out["metrics_mean"]
        row = {
            "Re": re_value, "re_norm": re_norm, "re_norm_true": re_norm_true,
            "re_norm_snapped": bool(re_norm != re_norm_true), "held_out": is_held,
            "eval_grid": [int(dns_u.shape[1]), int(dns_u.shape[2])], "T_eval": int(T),
            "uv_rel_err": mm.get("uv_rel_err"),   # primary field-fidelity metric
            "u_rel_err": mm.get("u_rel_err"), "v_rel_err": mm.get("v_rel_err"),
            "ke_rel_err": mm.get("ke_rel_err"), "omega_rel_err": mm.get("omega_rel_err"),
            "low_band_rel_err": mm.get("low_band_rel_err"),
            "div_pred_l2": mm.get("div_pred_l2"),
            "metrics_mean": mm, "wall_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"    u_err={_f(row['u_rel_err'])} v_err={_f(row['v_rel_err'])} "
              f"KE_err={_f(row['ke_rel_err'])} ω_err={_f(row['omega_rel_err'])} "
              f"low_band={_f(row['low_band_rel_err'])} "
              f"div_pred={_f(row['div_pred_l2'], '.2e')}  ({row['wall_s']}s)", flush=True)

    # ── 泛用性表 ──
    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

    print("\n" + "=" * 88)
    print(f"{'Re':>10} {'split':>9} {'re_norm':>8} {'u_err':>8} {'v_err':>8} "
          f"{'KE_err':>8} {'ω_err':>8} {'low_band':>9}")
    print("-" * 88)
    for r in sorted(rows, key=lambda r: (r["held_out"], r["Re"])):
        print(f"{r['Re']:>10.0f} {'HELD' if r['held_out'] else 'train':>9} "
              f"{r['re_norm']:>8.4f} {_fmt(r['u_rel_err']):>8} {_fmt(r['v_rel_err']):>8} "
              f"{_fmt(r.get('ke_rel_err')):>8} {_fmt(r.get('omega_rel_err')):>8} "
              f"{_fmt(r.get('low_band_rel_err')):>9}")
    print("-" * 88)

    def _mean(key, held):
        vals = [r[key] for r in rows if r["held_out"] == held and isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    in_u, ho_u = _mean("u_rel_err", False), _mean("u_rel_err", True)
    if in_u is not None and ho_u is not None:
        print(f"  mean u_err — in-train={in_u:.4f}  held-out={ho_u:.4f}  "
              f"gap={ho_u - in_u:+.4f}")
    print("=" * 88)

    summary = {
        "config": str(Path(args.config).resolve()),
        "ckpt_step": restored_step, "re_norm_scale": re_norm_scale,
        "grid_stride": args.grid_stride, "max_grid": args.max_grid, "sensor_T": args.sensor_T,
        "held_out": sorted(held_out), "yardsticks": yard, "rows": rows,
        "ckpt_provenance": ckpt_provenance,
    }
    for artifact_path, artifact in metric_artifacts:
        write_metric_artifact(artifact_path, artifact)
        print(f"[out] canonical artifact → {artifact_path}")
    write_compatibility_projection(out_path, summary)
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
