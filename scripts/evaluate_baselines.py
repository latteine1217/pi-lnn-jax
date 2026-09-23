#!/usr/bin/env python
"""evaluate_baselines — classical sparse-reconstruction baselines 對多 Re 的對照表。

Why:
  補 McGreivy rule 2 的「外部強 baseline」（interp / gappy-POD）。與 PI-CON 共用
  metric / sensors / DNS（metric_artifact.evaluate_field_series 這道 seam），公平比較
  by construction。
  baselines 不需 model / ckpt / re_norm（各 Re 獨立，從自身 sensors+DNS 重建）。

正確性要點（高風險 eval，對齊 CLAUDE.md evalStrictness）:
  - sensor_vals 為 normalized → 先 denormalize 回物理單位再重建、再比對 raw DNS。
  - 時間軸用「值匹配」對齊（match_sensor_dns_times，fail-fast），非 index-stride。
  - gappy-POD 的 POD basis 只用「與 eval 快照不相交」的 DNS 時間（leakage-free, basis_indices_excluding）。
    注意：此為「per-Re basis」（每個 Re 用自身軌跡的不相交時段），對 held-out Re 屬樂觀對照——
    報告時須標明，勿與 PI-CON 的「未見 held-out DNS」混淆。
  - grid stride 必須整除 N（choose_grid_stride，fail-fast），避免格點錯位。
  - 缺 DNS/sensor/NPZ 一律 fail-fast。

用法:
  # A) 用既有 eval config（讀 data_kwargs 的 re_values/sensor_jsons/dns_paths）
  uv run python scripts/evaluate_baselines.py --config configs/eval_multi_re_train5_crp.toml \
    --held-out 3000,8000,30000 --baselines interp_linear,gappy_pod
  # B) direct（單/多 Re 快速檢查，免 config）
  uv run python scripts/evaluate_baselines.py --re-values 10000 \
    --sensor-jsons data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json \
    --dns-paths data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_dt1p95e4_si128_seed42.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.baseline_eval import (  # noqa: E402
    basis_indices_excluding,
    select_pod_modes_by_validation,
)
from pi_lnn_jax.baselines import GappyDMD, GappyPOD, InterpBaseline  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="classical baseline 多-Re 重建對照表")
    src = p.add_argument_group("輸入來源（--config 或 direct 三件組擇一）")
    src.add_argument("--config", default=None, help="eval config TOML（讀 data_kwargs 的 re/sensor/dns）")
    src.add_argument("--re-values", default="", help="direct: comma-sep Re")
    src.add_argument("--sensor-jsons", default="", help="direct: comma-sep sensor JSON 路徑")
    src.add_argument("--dns-paths", default="", help="direct: comma-sep DNS .npy 路徑")

    p.add_argument("--held-out", default="", help="comma-sep Re，標記為 held-out")
    p.add_argument("--re-subset", default="", help="comma-sep Re，只評估子集；空=全部")
    p.add_argument("--baselines", default="interp_linear,gappy_pod",
                   help="comma-sep：interp_linear / interp_cubic / gappy_pod / gappy_dmd")
    p.add_argument("--protocol", required=True,
                   choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）：follow_training 由 config 的 "
                        "time_strides 決定取樣；fixed_grid 固定格點（須另給 "
                        "--sensor-time-stride 與 --protocol-reason）；"
                        "sensor_time_independent 不對齊時間軸")
    p.add_argument("--protocol-reason", default=None,
                   help="fixed_grid 必填：為何刻意偏離訓練 cadence")
    p.add_argument("--sensor-time-stride", type=int, default=None,
                   help="明示 stride。follow_training 下若與 config 不一致即失敗；"
                        "不給則由 config 決定")
    p.add_argument("--sensor-T", type=int, default=50, help="sensor/eval 截到此 T")
    p.add_argument("--grid-stride", type=int, default=1, help="空間 stride（須整除 N）")
    p.add_argument("--max-grid", type=int, default=0, help=">0 則 per-Re 選最細整除 stride 使 N'≤此值")
    p.add_argument("--pod-modes", type=int, default=0, help="gappy-POD 模態數；0=auto=min(2K, basis_size)")
    p.add_argument("--min-basis", type=int, default=8, help="leakage-free POD basis 最少快照數")
    p.add_argument("--output", default="artifacts/baseline_eval/metrics.json", help="輸出 JSON 路徑")
    return p.parse_args()




def _make_baseline(name: str):
    if name == "interp_linear":
        return InterpBaseline(method="linear", periodic=True), False
    if name == "interp_cubic":
        return InterpBaseline(method="cubic", periodic=True), False
    if name in ("gappy_pod", "gappy_dmd"):
        return None, True  # 需 per-Re fit，延後建立
    raise ValueError(
        f"未知 baseline: {name}"
        "（支援 interp_linear/interp_cubic/gappy_pod/gappy_dmd）")


def _reconstruct_one(name, sensor_phys, sensor_pos, dns_u_eval, dns_v_eval,
                     dns_u_full, dns_v_full, eval_idx, s, args) -> tuple:
    """對單一 (Re, baseline) 重建全場序列。回傳 (u_pred, v_pred, extra_meta)。

    metric 的計算不在這裡——它走 `metric_artifact.evaluate_field_series`，
    與 PI-CON 端同一個 seam。本函式只負責「這個 baseline 怎麼重建」。
    """
    Nprime = dns_u_eval.shape[1]
    K = sensor_pos.shape[0]
    extra: dict = {}
    _, needs_basis = _make_baseline(name)

    if needs_basis:  # gappy-POD
        basis_idx = basis_indices_excluding(dns_u_full.shape[0], eval_idx, args.min_basis)
        train_u = np.asarray(dns_u_full[basis_idx][:, ::s, ::s])
        train_v = np.asarray(dns_v_full[basis_idx][:, ::s, ::s])
        if args.pod_modes > 0:  # 明示模態數（cap 在量測數/basis 大小）
            r = min(args.pod_modes, 2 * K, basis_idx.size)
            sel = "fixed"
        else:  # auto：validation 選模態（leakage-free，避過擬合點）
            grid = [max(1, K // 8), K // 4, K // 2, K, 3 * K // 2, 2 * K]
            r, _ = select_pod_modes_by_validation(train_u, train_v, sensor_pos, grid, val_frac=0.3, seed=0)
            sel = "validation"
        # gappy_dmd 沿用 POD 選出的同一個 r：本對照的變因只有基底（POD vs DMD），
        # 各自調秩會讓差異混入「哪一邊調得比較好」。
        cls = GappyDMD if name == "gappy_dmd" else GappyPOD
        model = cls(n_modes=r).fit(train_u, train_v)  # 最終 fit 用全 basis
        u_pred, v_pred = model.reconstruct(sensor_phys, sensor_pos)
        extra = {"pod_modes": int(r), "basis_size": int(basis_idx.size),
                 "mode_selection": sel, "basis_kind": "dmd" if name == "gappy_dmd" else "pod",
                 "actual_modes": int(model.modes.shape[1])}
    else:
        model, _ = _make_baseline(name)
        u_pred, v_pred = model.reconstruct(sensor_phys, sensor_pos, grid_shape=(Nprime, Nprime))

    return u_pred, v_pred, extra


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def main() -> int:
    args = parse_args()
    inputs = resolve_re_inputs(
        config=args.config or None,
        re_values=args.re_values, sensor_jsons=args.sensor_jsons, dns_paths=args.dns_paths,
    )
    held_out = parse_re_set(args.held_out)
    subset = parse_re_set(args.re_subset)
    baselines = [b for b in args.baselines.split(",") if b.strip()]
    for b in baselines:
        _make_baseline(b)  # 提早驗證名稱（fail-fast）

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 一個 evaluation run 建一次 recorder：它自建構時抓一次 code revision（與遷移前
    # 的 repository_revision(_REPO_ROOT) 同一個 repo root），並擁有 stamp/命名/寫盤。
    recorder = EvaluationRunRecorder("scripts/evaluate_baselines.py", out_path)

    print("=" * 80)
    print(f"evaluate_baselines — baselines={baselines}")
    print(f"  source:    {'config=' + Path(args.config).name if args.config else 'direct paths'}")
    print(f"  held-out:  {sorted(held_out) or '(none)'}")
    print(f"  grid:      stride={args.grid_stride} max-grid={args.max_grid or '(off)'}  sensor-T={args.sensor_T}")
    print("=" * 80)

    training_strides = training_time_strides_from_config(args.config) if args.config else []

    rows = []
    for re_index, (re_value, sj, dp) in enumerate(inputs):
        if subset and re_value not in subset:
            continue
        is_held = re_value in held_out

        # 協定先解析、再據以載入——「用什麼 stride」不再是載入函式的預設參數。
        protocol = resolve_protocol(
            mode=args.protocol, training_time_strides=training_strides,
            re_index=re_index, cli_time_stride=args.sensor_time_stride,
            sensor_T=args.sensor_T, reason=args.protocol_reason,
        )
        aligned = load_for_evaluation(
            sj, dp, protocol=protocol, viscosity=1.0 / re_value,
            grid_stride=args.grid_stride, max_grid=args.max_grid,
        )
        sensor_phys, sensor_pos = aligned.sensor_phys, aligned.sensor_pos
        T, K, N, s, Nprime = aligned.T, aligned.K, aligned.N, aligned.s, aligned.Nprime
        eval_idx = aligned.eval_idx
        dns_u_eval, dns_v_eval = aligned.dns_u_eval, aligned.dns_v_eval
        dns_u_full, dns_v_full = aligned.dns_u_full, aligned.dns_v_full
        context = aligned.context

        tag = "HELD-OUT" if is_held else "in-train"
        print(f"[Re={re_value:>9.0f} {tag}] T={T} K={K}  N={N}→{Nprime} (s={s})  "
              f"protocol={protocol.mode.value} stride={protocol.sensor_time_stride}  "
              f"eval_idx[0:3]={eval_idx[:3]} …", flush=True)

        for name in baselines:
            t0 = time.time()
            u_pred, v_pred, extra = _reconstruct_one(
                name, sensor_phys, sensor_pos, dns_u_eval, dns_v_eval,
                dns_u_full, dns_v_full, eval_idx, s, args)
            wall = round(time.time() - t0, 2)

            projection = recorder.record(
                u_pred, v_pred, dns_u_eval, dns_v_eval,
                context=context, protocol=protocol,
                identity=RunArtifactIdentity(re=re_value, method=name),
                inputs=(("sensor", str(sj)), ("dns", str(dp)),
                        ("config", str(Path(args.config).resolve())
                         if args.config else "direct")),
                details_extras={
                    "method": name, "re_value": re_value, "held_out": is_held,
                    "sensor_time_stride": args.sensor_time_stride,
                    "sensor_T": args.sensor_T, "grid_stride": args.grid_stride,
                    "max_grid": args.max_grid, "min_basis": args.min_basis,
                    "K": K, "T_eval": T, "eval_grid": [Nprime, Nprime],
                    **extra,
                },
                run_measurements={"wall_s": wall},
            )

            mean = projection["metrics_mean"]
            row = {
                "Re": re_value, "method": name, "held_out": is_held,
                "eval_grid": [Nprime, Nprime], "T_eval": T, "K": K,
                **{k: (float(mean[k]) if k in mean and np.isfinite(mean[k]) else None)
                   for k in _METRIC_KEYS},
                **extra, "wall_s": wall,
            }
            rows.append(row)
            print(f"    {name:>14}: u_err={_fmt(row['u_rel_err'])} v_err={_fmt(row['v_rel_err'])} "
                  f"KE_err={_fmt(row['ke_rel_err'])} ω_err={_fmt(row['omega_rel_err'])} "
                  f"low_band={_fmt(row['low_band_rel_err'])}  ({wall}s)", flush=True)

    # ── 對照表（依 method 分組）──
    print("\n" + "=" * 96)
    print(f"{'method':>14} {'Re':>9} {'split':>8} {'u_err':>8} {'v_err':>8} "
          f"{'KE_err':>8} {'ω_err':>8} {'low_band':>9}")
    print("-" * 96)
    for r in sorted(rows, key=lambda r: (r["method"], r["held_out"], r["Re"])):
        print(f"{r['method']:>14} {r['Re']:>9.0f} {'HELD' if r['held_out'] else 'train':>8} "
              f"{_fmt(r['u_rel_err']):>8} {_fmt(r['v_rel_err']):>8} {_fmt(r['ke_rel_err']):>8} "
              f"{_fmt(r['omega_rel_err']):>8} {_fmt(r['low_band_rel_err']):>9}")
    print("-" * 96)

    def _mean(method, key, held):
        vals = [r[key] for r in rows
                if r["method"] == method and r["held_out"] == held and isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    for name in baselines:
        iu, hu = _mean(name, "u_rel_err", False), _mean(name, "u_rel_err", True)
        seg = f"  [{name}] mean u_err — in-train={_fmt(iu)}"
        if hu is not None:
            seg += f"  held-out={_fmt(hu)}"
            if isinstance(iu, float):
                seg += f"  gap={hu - iu:+.4f}"
        print(seg)
    print("=" * 96)

    summary = {
        "source": str(Path(args.config).resolve()) if args.config else "direct",
        "baselines": baselines, "held_out": sorted(held_out),
        "sensor_T": args.sensor_T, "grid_stride": args.grid_stride, "max_grid": args.max_grid,
        "pod_modes": args.pod_modes, "rows": rows,
    }
    write_compatibility_projection(out_path, summary)
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
