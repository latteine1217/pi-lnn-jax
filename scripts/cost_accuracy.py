#!/usr/bin/env python
"""cost_accuracy — classical baseline 的 cost-vs-accuracy 報表（Tier 0.2，McGreivy fig-1a 式）。

Why:
  McGreivy & Hakim 2024 要求「在 equal-accuracy 或 equal-runtime 下比較」並畫 cost-accuracy 曲線。
  本腳本掃 gappy-POD 的模態數（accuracy↑、cost↑ 的 frontier）+ interp（單點），量
  「每場重建耗時 vs 相對誤差」，輸出表 + 圖。inference cost 指「重建一個場」的 wall（不含
  offline POD basis SVD，basis 屬一次性）。

與 evaluate_baselines.py 共用 evaluation_protocol.load_for_evaluation（同協定/denorm/
時間對齊/leakage 切分/stride 契約）。
gappy-POD 對單一 Re 只做一次 SVD（fit max modes），其餘模態數用切片重建，省時。

用法（本機可跑，CPU）:
  uv run python scripts/cost_accuracy.py --re-values 10000 \
    --sensor-jsons data/sensors/re10000/sensors_les_qr_K100_N256_t0-20_si128.json \
    --dns-paths data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_dt1p95e4_si128_seed42.npy \
    --pod-modes-list 10,25,50,100,200 \
    --output artifacts/baseline_eval/cost_accuracy_re10000.json \
    --fig artifacts/baseline_eval/cost_accuracy_re10000.png
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

from pi_lnn_jax.baseline_eval import basis_indices_excluding  # noqa: E402
from pi_lnn_jax.baselines import GappyPOD, InterpBaseline  # noqa: E402
from pi_lnn_jax.data import resolve_re_inputs  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity  # noqa: E402
from pi_lnn_jax.metric_artifact import write_compatibility_projection  # noqa: E402

_METRIC_KEYS = ["u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err", "low_band_rel_err"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="classical baseline cost-vs-accuracy 報表")
    p.add_argument("--config", default=None, help="eval config TOML（讀 data_kwargs）")
    p.add_argument("--re-values", default="")
    p.add_argument("--sensor-jsons", default="")
    p.add_argument("--dns-paths", default="")
    p.add_argument("--re-subset", default="", help="只跑子集（cost-accuracy 通常選 1 個 Re）")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）")
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--sensor-time-stride", type=int, default=None,
                   help="明示 stride；follow_training 下與 config 不一致即失敗")
    p.add_argument("--sensor-T", type=int, default=50)
    p.add_argument("--grid-stride", type=int, default=1)
    p.add_argument("--max-grid", type=int, default=0)
    p.add_argument("--pod-modes-list", default="10,25,50,100,200", help="comma-sep gappy-POD 模態數掃描")
    p.add_argument("--min-basis", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5, help="重建計時重複次數（取 min 去 scheduler 噪音）")
    p.add_argument("--metric", default="ke_rel_err", help="圖 y 軸 metric（預設 ke_rel_err）")
    p.add_argument("--output", default="artifacts/baseline_eval/cost_accuracy.json")
    p.add_argument("--fig", default="artifacts/baseline_eval/cost_accuracy.png")
    return p.parse_args()


def _score(recorder, u_pred, v_pred, re, *, identity, inputs, details_extras,
           recon_s_per_field) -> dict:
    """跨 recorder seam 評一組重建、蓋 provenance、命名、落盤，回傳舊 row 的 metric 欄位。

    recorder 依 identity 重現凍結檔名、並自行蓋上 per-Re 的 evaluation_protocol
    （caller 的 details_extras 不得再供）。重建成本走 run-level 的 cost measurement
    ——它是本腳本那張圖的 x 軸，需要定義與單位，不是 provenance 上的一則註記。
    """
    projection = recorder.record(
        u_pred, v_pred, re.dns_u_eval, re.dns_v_eval,
        context=re.context, protocol=re.protocol,
        identity=identity, inputs=inputs, details_extras=details_extras,
        run_measurements={"recon_s_per_field": recon_s_per_field},
    )
    mean = projection["metrics_mean"]
    return {k: (float(mean[k]) if k in mean and np.isfinite(mean[k]) else None)
            for k in _METRIC_KEYS}


def _timed(fn, repeats: int):
    """跑 fn() repeats 次取最快（min 去 scheduler 噪音），回傳 (最後輸出, best_seconds)。"""
    best = float("inf")
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _eval_one_re(re_value, re, modes_list, min_basis, repeats, *,
                 args, recorder) -> list[dict]:
    # `re` 是 AlignedEvaluation：協定已在呼叫端解析，這裡只消費結果。
    """對單一 Re 跑 interp + gappy-POD(各模態數)，回傳 rows（含 timing；重建計時取 min）。"""
    rows = []
    T, K, Nprime = re.T, re.K, re.Nprime
    sensor_phys, sensor_pos = re.sensor_phys, re.sensor_pos

    inputs = (("config", str(Path(args.config).resolve())
               if args.config else "direct"),)

    def _details(method, modes, extra):
        # 舊 _provenance 的 details，去掉 evaluation_protocol（recorder 保證注入）。
        return {
            "method": method, "modes": modes, "re_value": re_value,
            "repeats": repeats, "min_basis": min_basis,
            "sensor_time_stride": args.sensor_time_stride,
            "sensor_T": args.sensor_T, "grid_stride": args.grid_stride,
            "max_grid": args.max_grid, "K": K, "T_eval": T,
            "eval_grid": [Nprime, Nprime], **extra,
        }

    # ── interp（單點，無 fit）──
    (u_pred, v_pred), recon_s = _timed(
        lambda: InterpBaseline("linear", periodic=True).reconstruct(
            sensor_phys, sensor_pos, grid_shape=(Nprime, Nprime)), repeats)
    mm = _score(recorder, u_pred, v_pred, re,
                identity=RunArtifactIdentity(re=re_value, method="interp_linear"),
                inputs=inputs,
                details_extras=_details("interp_linear", None, {"fit_s": 0.0}),
                recon_s_per_field=recon_s / T)
    rows.append({"Re": re_value, "method": "interp_linear", "modes": None,
                 "fit_s": 0.0, "recon_s_per_field": recon_s / T, **mm})

    # ── gappy-POD：一次 SVD（fit max modes）後切片重建各模態數 ──
    basis_idx = basis_indices_excluding(re.dns_u_full.shape[0], re.eval_idx, min_basis)
    s = re.s
    train_u = np.asarray(re.dns_u_full[basis_idx][:, ::s, ::s])
    train_v = np.asarray(re.dns_v_full[basis_idx][:, ::s, ::s])
    cap = min(2 * K, basis_idx.size)
    modes = sorted({min(m, cap) for m in modes_list})  # 去重 + cap
    _, fit_s = _timed(lambda: GappyPOD(n_modes=max(modes)).fit(train_u, train_v), 1)
    base = GappyPOD(n_modes=max(modes)).fit(train_u, train_v)
    full_modes = base.modes
    for m in modes:
        base.modes = full_modes[:, :m]  # 切片重用 SVD（避免重算）
        (u_pred, v_pred), recon_s = _timed(lambda: base.reconstruct(sensor_phys, sensor_pos), repeats)
        mm = _score(recorder, u_pred, v_pred, re,
                    identity=RunArtifactIdentity(re=re_value, method="gappy_pod", modes=int(m)),
                    inputs=inputs,
                    details_extras=_details("gappy_pod", int(m), {
                        "fit_s": fit_s, "basis_size": int(basis_idx.size)}),
                    recon_s_per_field=recon_s / T)
        rows.append({"Re": re_value, "method": "gappy_pod", "modes": int(m),
                     "fit_s": fit_s, "recon_s_per_field": recon_s / T,
                     "basis_size": int(basis_idx.size), **mm})
    return rows


def _save_figure(rows, metric, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = sorted({r["Re"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, re_value in enumerate(res):
        color = cmap(0.15 + 0.7 * (i / max(1, len(res) - 1)))
        g = sorted([r for r in rows if r["Re"] == re_value and r["method"] == "gappy_pod"],
                   key=lambda r: r["modes"])  # 依模態數（受控變數）連線，非 cost
        if g:
            x = [r["recon_s_per_field"] * 1e3 for r in g]
            y = [r[metric] for r in g]
            ax.plot(x, y, "-o", color=color, label=f"gappy-POD (Re={re_value:.0f})", markersize=5)
            for r in g:  # 標模態數
                ax.annotate(str(r["modes"]), (r["recon_s_per_field"] * 1e3, r[metric]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points", color=color)
        ip = [r for r in rows if r["Re"] == re_value and r["method"] == "interp_linear"]
        if ip:
            ax.plot(ip[0]["recon_s_per_field"] * 1e3, ip[0][metric], "s", color=color,
                    markersize=8, markerfacecolor="white", label=f"interp-linear (Re={re_value:.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction time per field (ms)")
    ax.set_ylabel(f"Relative L2 error ({metric})")
    ax.set_title("Cost vs. accuracy (lower-left is better)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_path.with_suffix(".pdf"))
    return str(fig_path)


def main() -> int:
    args = parse_args()
    inputs = resolve_re_inputs(
        config=args.config or None,
        re_values=args.re_values, sensor_jsons=args.sensor_jsons, dns_paths=args.dns_paths,
    )
    subset = {float(x) for x in args.re_subset.split(",") if x.strip()}
    modes_list = [int(x) for x in args.pod_modes_list.split(",") if x.strip()]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 一個 evaluation run 建一次 recorder：它自建構時抓一次 code revision（與遷移前
    # 的 repository_revision(_REPO_ROOT) 同一個 repo root），並擁有 stamp/命名/寫盤。
    # out_path 已 .resolve()——原樣傳入，讓寫出的檔名與舊版逐字相同。
    recorder = EvaluationRunRecorder("scripts/cost_accuracy.py", out_path)

    print("=" * 80)
    print(f"cost_accuracy — pod-modes-list={modes_list}  metric={args.metric}")
    print("=" * 80)

    training_strides = training_time_strides_from_config(args.config) if args.config else []

    rows = []
    for re_index, (re_value, sj, dp) in enumerate(inputs):
        if subset and re_value not in subset:
            continue
        protocol = resolve_protocol(
            mode=args.protocol, training_time_strides=training_strides,
            re_index=re_index, cli_time_stride=args.sensor_time_stride,
            sensor_T=args.sensor_T, reason=args.protocol_reason)
        re = load_for_evaluation(sj, dp, protocol=protocol, viscosity=1.0 / re_value,
                                 grid_stride=args.grid_stride, max_grid=args.max_grid)
        print(f"[Re={re_value:.0f}] T={re.T} K={re.K} N={re.N}→{re.Nprime} (s={re.s})  "
              f"protocol={protocol.mode.value} stride={protocol.sensor_time_stride} …",
              flush=True)
        rows.extend(_eval_one_re(re_value, re, modes_list, args.min_basis, args.repeats,
                                 args=args, recorder=recorder))

    # ── 表 ──
    print(f"\n{'method':>14} {'Re':>9} {'modes':>6} {'recon_ms/field':>15} {'ke_err':>8} {'u_err':>8} {'ω_err':>8}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: (r["Re"], r["method"], r["modes"] or 0)):
        print(f"{r['method']:>14} {r['Re']:>9.0f} {str(r['modes'] or '-'):>6} "
              f"{r['recon_s_per_field'] * 1e3:>15.2f} {r['ke_rel_err']:>8.4f} "
              f"{r['u_rel_err']:>8.4f} {r['omega_rel_err']:>8.4f}")
    print("-" * 78)

    fig_out = _save_figure(rows, args.metric, args.fig)
    write_compatibility_projection(
        out_path, {"pod_modes_list": modes_list, "metric": args.metric, "rows": rows})
    print(f"\n[out] {out_path}")
    print(f"[fig] {fig_out} (+.pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
