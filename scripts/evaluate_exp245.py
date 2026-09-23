"""Standalone DNS evaluation for EXP-245 (Re=10000) baseline checkpoints.

What:
    從 train_kolmogorov.py 寫出的 orbax checkpoint resume，跑完整 time series DNS
    對照評估（KE rel-err / per-channel rel-err / vorticity / divergence /
    low-band E(k) rel-err），並輸出 metrics_mean.json、ke_timeseries.npy 與
    energy spectrum PNG。

Why:
    train_kolmogorov.py 內部的 _load_dns_for_eval 寫死呼叫 pi_lnn_jax.data.load_dns()
    → Re=1000 DNS path（pi-lnn/data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128*.npy）。
    對 Re=10000 baseline（EXP-245）做 in-training mid-eval 會用錯 DNS。
    本 script 走獨立路徑：直接從 TOML `dns_paths[0]` 載 Re=10000 DNS，
    跑 final ckpt full-T eval，避開該 bug 並產生可決策的 metric。

Usage:
    PYTHONPATH=. uv run python scripts/evaluate_exp245.py \\
        --config configs/exp_245_b3_les_T50.toml --ckpt latest

    PYTHONPATH=. uv run python scripts/evaluate_exp245.py \\
        --config configs/exp_245_b3_les_T50.toml --ckpt 20000 \\
        --output-subdir final_eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# 確保 repo root 在 sys.path（從 scripts/ 直接執行時）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pi_lnn_jax.ckpt import (  # noqa: E402
    reference_params_for, restore_eval_params, verify_norm_stats,
)
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
    EvaluationContext,
    SourceProvenance,
    build_metric_artifact,
    freeze_provenance_details,
    per_t_to_arrays,
    repository_revision,
    write_compatibility_projection,
    write_metric_artifact,
)
from pi_lnn_jax.model_factory import build_model  # noqa: E402
from pi_lnn_jax.sensor_dropout import apply_sensor_dropout, make_keep_mask  # noqa: E402


def _write_metric_json_artifacts(
    out_dir: Path,
    *,
    projection: dict,
    evaluation: dict,
    context: EvaluationContext,
    provenance: SourceProvenance,
) -> None:
    """Write the compatibility projection and canonical artifact together."""
    artifact = build_metric_artifact(
        evaluation, context=context, provenance=provenance,
    )
    write_metric_artifact(
        out_dir / "metric_artifact.json",
        artifact,
    )
    write_compatibility_projection(out_dir / "metrics.json", projection)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-245 final ckpt DNS eval")
    p.add_argument("--config", type=str, required=True, help="TOML config 路徑")
    p.add_argument("--ckpt", type=str, default="latest",
                   help="ckpt step ('latest' 或 整數)")
    p.add_argument("--arch", choices=["liquid", "vanilla", "pinn"], default="liquid")
    p.add_argument("--optimizer", choices=["adam", "soap", "schedule_free"],
                   default="schedule_free",
                   help="必須與訓練 --optimizer 一致；orbax restore 需要 opt_state 結構吻合")
    p.add_argument("--base_optimizer", choices=["adam", "soap"], default="soap",
                   help="schedule_free 模式下的 inner optimizer，需與訓練端一致")
    p.add_argument("--iterate", choices=["y", "x"], default="y",
                   help="TD-1 診斷：評估哪一個 ScheduleFree iterate。預設 y（訓練 "
                        "iterate，= 現行所有已發表數字的慣例）。x 為 Polyak 平均的 "
                        "評估 iterate（optax 文件建議在其上評估，PyTorch 對照亦用 "
                        "它）。此旗標只改『評哪一組參數』，不改任何 metric 定義。")
    p.add_argument("--soap_precondition_frequency", type=int, default=2,
                   help="SOAP precondition freq，需與訓練端一致")
    p.add_argument("--artifacts_dir", type=str, default=None,
                   help="覆蓋 TOML artifacts_dir（必要當 train_kolmogorov.py 用 --artifacts_dir flag）")
    p.add_argument("--output-subdir", type=str, default="final_eval",
                   help="輸出到 artifacts_dir/<output-subdir>/")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode],
                   help="評估協定（必填，無預設）：follow_training 由 config 的 "
                        "time_strides 決定；fixed_grid 固定格點（須 --protocol-reason）；"
                        "sensor_time_independent 為 thesis §7.2 的間歇評估")
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--time-stride", type=int, default=None,
                   help="sensor 與 DNS 載入時的 time stride（與 train_kolmogorov.py 一致）")
    p.add_argument("--no-plot", action="store_true",
                   help="跳過 matplotlib spectrum plot（無 X 環境 / headless 用）")
    # 論文繪圖用的匯出（預設關閉：兩者都只有畫圖時才需要，訓練評估流程不該付這個代價）
    # 間歇 sensor（thesis §7.2）：這是不同的評估協定，不是放寬對齊。它已收進
    # --protocol 的 sensor_time_independent；本旗標保留為既有 sbatch 的相容別名，
    # 且必須與 --protocol 一致——兩者說不同的話時拒絕執行，不自行選一個。
    p.add_argument("--sensor-time-independent", action="store_true",
                   help="[相容別名] 等同 --protocol sensor_time_independent；"
                        "與 --protocol 不一致時失敗")
    p.add_argument("--autoreg-pseudo-dt", type=float, default=0.0,
                   help="branch 自迴歸：用模型自己的預測把 sensor 序列延伸到 --autoreg-t-end，"
                        "pseudo 幀間距（0=關）。必須與訓練該 ckpt 時的設定一致——"
                        "自迴歸改的是推論程序，訓練有、eval 沒有就是評到另一個方法。")
    p.add_argument("--autoreg-rounds", type=int, default=1,
                   help="自迴歸輪數（每輪重新 encode；須與訓練一致）")
    p.add_argument("--autoreg-t-end", type=float, default=None,
                   help="延伸到哪個時刻（預設 = DNS 時間軸末端）")
    p.add_argument("--export-arrays", action="store_true",
                   help="另存 series.npz：逐時 metrics 攤平成陣列（KE/div_ratio/u,v rel-err/"
                        "E(k)/forcing-mode 振幅與相位），供 trajectory 與能譜圖直接讀取")
    p.add_argument("--export-fields", action="store_true",
                   help="另存 fields.npz：重建場 u,v [T,Nx,Ny]（float32，T=101/N=256 約 50 MB），"
                        "供場圖、渦度圖與空間統計使用")
    # 實驗2：eval-time sensor dropout robustness（zero-mask，非移除；B0/B3 公平對照）
    p.add_argument("--sensor_dropout_rate", type=float, default=0.0,
                   help="eval 時隨機 zero-mask 的 sensor 比例（0=不 dropout，行為 bit-identical）")
    p.add_argument("--dropout_realizations", type=int, default=1,
                   help="每個 dropout rate 的隨機 mask realization 數（跨 realization 報 mean±std）")
    p.add_argument("--dropout_seed", type=int, default=0,
                   help="dropout mask 的 base PRNGKey（realization r 用 seed+r，可重現）")
    return p.parse_args()


def _find_schedule_free_state(tree):
    """在 opt_state pytree 內找出 ScheduleFreeState（同時具備 b1 與 z 的節點）。

    Why 用結構特徵而非型別：opt_state 經 orbax lenient restore 後是巢狀
    dict/tuple，型別資訊已丟失，`isinstance` 認不出來。b1+z 這組欄位是
    `optax.contrib.schedule_free_eval_params` 唯一需要的，用它當判別特徵。
    """
    import types
    if tree is None:
        return None
    has = lambda o, k: (isinstance(o, dict) and k in o) or hasattr(o, k)
    get = lambda o, k: o[k] if isinstance(o, dict) else getattr(o, k)
    if has(tree, "b1") and has(tree, "z"):
        return types.SimpleNamespace(b1=get(tree, "b1"), z=get(tree, "z"))
    children = tree.values() if isinstance(tree, dict) else (
        tree if isinstance(tree, (list, tuple)) else ())
    for c in children:
        found = _find_schedule_free_state(c)
        if found is not None:
            return found
    return None


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_kwargs = cfg["train_kwargs"]
    data_kwargs = cfg["data_kwargs"]
    model_kwargs = cfg["model_kwargs"]

    artifacts_dir = Path(
        args.artifacts_dir if args.artifacts_dir is not None
        else train_kwargs.get("artifacts_dir", "artifacts/run")
    ).resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    out_dir = artifacts_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"evaluate_exp245 — config={Path(args.config).name}")
    print(f"  artifacts:   {artifacts_dir}")
    print(f"  ckpt_dir:    {ckpt_dir}")
    print(f"  output:      {out_dir}")
    print(f"  ckpt:        {args.ckpt}")
    print("=" * 80)

    # ── Sensor + DNS ─────────────────────────────────────────────────
    sensor_json = data_kwargs.get("sensor_jsons", [])
    re_values = data_kwargs.get("re_values", [10000.0])
    dns_paths = data_kwargs.get("dns_paths", [])
    if not sensor_json:
        raise ValueError("TOML data_kwargs.sensor_jsons 為空")
    if not dns_paths:
        raise ValueError("TOML data_kwargs.dns_paths 為空（EXP-245 需 Re=10000 DNS）")

    re_value = float(re_values[0])
    # #4: 讀 config re_norm_scale（不可 hardcode log(10000)；multi-Re 用 1e6 等）
    if "re_norm_scale" not in data_kwargs:
        print("[warn] config data_kwargs 缺 re_norm_scale，fallback 10000.0"
              "（multi-Re ckpt 用其他 scale 會餵錯 re_norm 條件）")
    re_norm_scale = float(data_kwargs.get("re_norm_scale", 10000.0))
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))
    print(f"\n[data] sensor_json = {sensor_json[0]}")
    print(f"[data] dns_path    = {dns_paths[0]}")
    print(f"[data] Re          = {re_value}  re_norm = {re_norm:.4f}")

    if args.sensor_time_independent and args.protocol != ProtocolMode.SENSOR_TIME_INDEPENDENT.value:
        raise SystemExit(
            f"--sensor-time-independent 與 --protocol {args.protocol} 矛盾。"
            "前者是後者的相容別名；請只給 --protocol sensor_time_independent。")
    protocol = resolve_protocol(
        mode=args.protocol,
        training_time_strides=training_time_strides_from_config(args.config),
        cli_time_stride=args.time_stride, reason=args.protocol_reason)
    print(f"[protocol] {protocol.mode.value}  sensor_stride={protocol.sensor_time_stride} "
          f"dns_stride={protocol.dns_time_stride}  {protocol.basis}")

    # eval_p 取決於 sensor 的 norm_stats 是否含壓力；先探一次再決定要不要載 p。
    _probe = load_sensors_from_path(sensor_json[0], time_stride=protocol.sensor_time_stride)
    eval_p = "p_mean" in _probe["norm_stats"]

    aligned = load_for_evaluation(
        sensor_json[0], Path(dns_paths[0]), protocol=protocol,
        viscosity=1.0 / re_value, with_pressure=eval_p)
    sensor_vals = jnp.asarray(aligned.sensor_vals_normalized)
    sensor_pos = jnp.asarray(aligned.sensor_pos)
    sensor_time = jnp.asarray(aligned.sensor_time)
    norm_stats = aligned.norm_stats
    T, K = sensor_vals.shape[0], sensor_vals.shape[1]
    print(f"[data] sensor_vals {sensor_vals.shape}  T={T}  K={K}")
    dns_u, dns_v, dns_t = aligned.dns_u_eval, aligned.dns_v_eval, aligned.dns_t_eval
    dns_p = aligned.dns_p_eval
    if protocol.mode is ProtocolMode.SENSOR_TIME_INDEPENDENT:
        print(f"[eval] sensor-time-independent：query 走 DNS {dns_t.shape[0]} 幀格點，"
              f"sensor context {T} 幀（不等距，Δt 由 sensor 自身時間軸決定）")

    print(f"[data] DNS u {dns_u.shape}  t [{float(dns_t[0]):.3f}, "
          f"{float(dns_t[-1]):.3f}]  eval_p={eval_p}")

    # ── Model + reference TrainState（給 orbax restore 對 abstract template）──
    # 與 train_kolmogorov.py 共用同一個 factory；model_name 印出來便於核對評的是哪個 baseline
    model, model_name = build_model(args.arch, model_kwargs, K_sensors=K)
    print(f"[model] {model_name}")
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\n[model] {args.arch}  params={n_params:,}")

    # ── Restore ──
    # Why lenient（restore_eval_params 內部走 reference_state=None）：SOAP_JAX 的
    # opt_state 含內部 transform 結構，第一次 tx.update 後 EmptyState placeholder
    # 才實化為 dict；eval 端 tx.init(params) 還沒 update 過 → 結構與 disk mismatch。
    # eval 只需 params，不需 opt_state。結構驗證改由 reference_params 承擔。
    params, restored_step, ckpt_provenance = restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=params, model=model)

    # 第三道閘門：反正規化常數。參數樹比 shape、建構指紋比建構值，兩者都看不見
    # norm_stats——它是從 sensor 檔推導的，eval 的 time_stride 若與訓練不同就會拿到
    # 另一組（實測 stride 20 下 v_std 差 2.17%）。印記與另外兩道一樣跟著數字走。
    norm_stats_provenance = verify_norm_stats(artifacts_dir, norm_stats)

    # ── TD-1：可選的 x-iterate 評估 ──
    # 訓練端存的與 final eval 用的都是 y（訓練 iterate）；optax 文件建議在 x 上評，
    # 而 PyTorch 對照用的正是 x。差多少從未量化過，本旗標補上量測端。
    # x = (y − (1−b1)·z)/b1 —— 不自己抄公式，直接用 optax 的正典實作。
    iterate_note = None
    if args.iterate == "x":
        from optax.contrib import schedule_free_eval_params
        from pi_lnn_jax.ckpt import CheckpointManager
        if args.optimizer != "schedule_free":
            raise SystemExit("[fail-fast] --iterate x 只在 --optimizer schedule_free 下有意義")
        mgr = CheckpointManager(directory=ckpt_dir, max_to_keep=3, save_interval_steps=1)
        raw = mgr.restore(restored_step, reference_state=None)
        sf = _find_schedule_free_state(raw.get("opt_state"))
        if sf is None:
            raise SystemExit(
                "[fail-fast] opt_state 內找不到 ScheduleFreeState（需同時有 b1 與 z）；"
                f"無法重建 x-iterate。ckpt_dir={ckpt_dir}")
        params_y = params
        params = schedule_free_eval_params(sf, params_y)
        num = sum(float(jnp.sum((a - b) ** 2))
                  for a, b in zip(jax.tree_util.tree_leaves(params),
                                  jax.tree_util.tree_leaves(params_y)))
        den = sum(float(jnp.sum(b ** 2)) for b in jax.tree_util.tree_leaves(params_y))
        rel = (num / den) ** 0.5 if den > 0 else float("nan")
        iterate_note = {"iterate": "x", "b1": float(sf.b1),
                        "rel_param_distance_x_vs_y": rel}
        print(f"[TD-1] 評估 x-iterate（Polyak 平均）：b1={float(sf.b1):.4f}  "
              f"‖x−y‖/‖y‖ = {rel:.3e}")
    else:
        iterate_note = {"iterate": "y", "b1": None, "rel_param_distance_x_vs_y": None}

    # ── Eval (full time series) ──
    # arch-agnostic：evaluate_time_series 只用 reconstruct_field 的標準 forward
    # __call__(sensor_vals, sensor_pos, re_norm, sensor_time, xy, t) → [N, 3]，
    # liquid / vanilla(B0) / pinn(B2) 三者共用此簽名（liquid 的 encode/decode 僅訓練
    # 優化路徑）。契約由 tests/test_baseline_arch_eval.py 鎖定。
    # sensor dropout robustness（rate=0 → n_real=1、sv_in=sensor_vals → 與既有 eval bit-identical）
    drop_rate = float(args.sensor_dropout_rate)
    n_real = int(args.dropout_realizations) if drop_rate > 0.0 else 1
    if drop_rate > 0.0:
        print(f"\n[dropout] eval-time sensor dropout rate={drop_rate} "
              f"realizations={n_real} base_seed={args.dropout_seed} "
              f"(zero-mask, K={K}, keep={K - int(round(drop_rate * K))})")
    print(f"\n[eval] running full-T time series eval (T={T}) …")
    t0 = time.time()
    # branch 自迴歸：延伸 sensor 序列（與訓練端共用 pi_lnn_jax.rollout，不得各寫一份）。
    # 延伸在評估迴圈之外做一次：下游 evaluate_time_series 拿到的就是延伸後的序列，
    # 其餘路徑完全不變。
    if args.autoreg_pseudo_dt > 0.0:
        from pi_lnn_jax.rollout import extend_sensor_sequence, pseudo_frame_times
        t_last = float(np.asarray(sensor_time)[-1])
        t_end_ar = float(args.autoreg_t_end) if args.autoreg_t_end is not None \
            else float(np.asarray(dns_t)[-1])
        pseudo_t = pseudo_frame_times(t_last, t_end_ar, float(args.autoreg_pseudo_dt))
        sensor_vals, sensor_time = extend_sensor_sequence(
            model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
            pseudo_t, int(args.autoreg_rounds),
            encode_method=model.__class__.encode,
            decode_method=model.__class__.decode_query,
        )
        print(f"[autoreg] sensor 序列延伸 {len(pseudo_t)} 幀 → t ∈ "
              f"[{t_last:.3f}, {t_end_ar:.3f}]，{args.autoreg_rounds} 輪；"
              f"sensor_vals={tuple(sensor_vals.shape)}")

    out = None
    real_metrics = []
    for r in range(n_real):
        if drop_rate > 0.0:
            keep = make_keep_mask(jax.random.PRNGKey(int(args.dropout_seed) + r), K, drop_rate)
            sv_in = apply_sensor_dropout(sensor_vals, keep)
        else:
            sv_in = sensor_vals
        out_r = evaluate_time_series(
            model, params, sv_in, sensor_pos, re_norm, sensor_time,
            norm_stats, dns_u, dns_v, dns_t, verbose=(r == 0),
            dns_p=dns_p, eval_p=eval_p,
            # ν=1/Re（physics.py 的慣例）；band/γ 的 k_η 邊界需要它，未傳則那些欄位缺席
            nu=1.0 / re_value,
            # 場只從第一個 realization 收（out 亦取第一個，兩者必須是同一次重建）
            collect_fields=(args.export_fields and r == 0),
        )
        real_metrics.append(out_r["metrics_mean"])
        if out is None:
            out = out_r  # 主 out（plot / ke_timeseries）用第一個 realization
    print(f"[eval] wall = {time.time() - t0:.1f}s")

    # 跨 realization 聚合（rate=0 時 n_real=1，agg == 單值）
    _mk = list(real_metrics[0].keys())
    dropout_summary = None
    if drop_rate > 0.0:
        dropout_summary = {
            "rate": drop_rate,
            "realizations": n_real,
            "base_seed": int(args.dropout_seed),
            "keep_sensors": K - int(round(drop_rate * K)),
            "metrics_mean_over_realizations": {
                k: float(np.mean([m[k] for m in real_metrics])) for k in _mk
            },
            "metrics_std_over_realizations": {
                k: float(np.std([m[k] for m in real_metrics])) for k in _mk
            },
            "per_realization_metrics_mean": real_metrics,
        }
        _agg = dropout_summary["metrics_mean_over_realizations"]
        _std = dropout_summary["metrics_std_over_realizations"]
        print(f"[dropout] KE_err {_agg.get('ke_rel_err'):.4f}±{_std.get('ke_rel_err'):.4f}  "
              f"ω_err {_agg.get('omega_rel_err'):.4f}±{_std.get('omega_rel_err'):.4f}  "
              f"(over {n_real} realizations)")

    # ── 輸出 ──
    metrics_mean = out["metrics_mean"]
    ke_t_pred = np.asarray(out["ke_t_pred"])
    ke_t_dns = np.asarray(out["ke_t_dns"])
    metrics_per_t = out["metrics_per_t"]

    # JSON dump（per_t 內含 E_pred_k / E_dns_k list；保留方便後分析）
    summary = {
        "config": str(Path(args.config).resolve()),
        "ckpt_step": restored_step,
        "ckpt_dir": str(ckpt_dir),
        "re_value": re_value,
        "T_eval": int(T),
        "metrics_mean": metrics_mean,
        "ke_t_errors": out.get("ke_t_errors"),
        "metrics_per_t": metrics_per_t,
        "sensor_dropout": dropout_summary,
        # 印記跟著數字走：stdout 的警告會被滾走，而這個欄位會跟到它進論文表格那天。
        "ckpt_provenance": ckpt_provenance,
        "norm_stats_provenance": norm_stats_provenance,
        # 印記跟著數字走：沒有這欄，x 版與 y 版的產物長得一模一樣。
        "schedule_free_iterate": iterate_note,
    }
    revision, code_dirty = repository_revision(_REPO_ROOT)
    _write_metric_json_artifacts(
        out_dir,
        projection=summary,
        evaluation=out,
        context=aligned.context,
        provenance=SourceProvenance(
            producer="scripts/evaluate_exp245.py",
            code_revision=revision,
            code_dirty=code_dirty,
            inputs=(
                ("config", str(Path(args.config).resolve())),
                ("checkpoint_dir", str(ckpt_dir)),
                ("sensor", str(sensor_json[0])),
                ("dns", str(dns_paths[0])),
            ),
            details=freeze_provenance_details({
                "checkpoint": ckpt_provenance,
                "checkpoint_step": restored_step,
                "re_value": re_value,
            "evaluation_protocol": protocol.to_provenance(),
                "sensor_dropout": dropout_summary,
            }),
        ),
    )
    summary_path = out_dir / "metrics.json"
    print(f"\n[out] metrics.json     → {summary_path}")
    print(f"[out] metric_artifact.json → {out_dir / 'metric_artifact.json'}")

    np.savez(
        out_dir / "ke_timeseries.npz",
        t=np.asarray(dns_t), ke_pred=ke_t_pred, ke_dns=ke_t_dns,
    )
    print(f"[out] ke_timeseries.npz → {out_dir / 'ke_timeseries.npz'}")

    if args.export_arrays:
        series = per_t_to_arrays(metrics_per_t)
        series["ke_pred"] = ke_t_pred
        series["ke_dns"] = ke_t_dns
        np.savez(out_dir / "series.npz", **series)
        print(f"[out] series.npz        → {out_dir / 'series.npz'}  "
              f"fields={sorted(series)}")

    if args.export_fields:
        # 只存 pred 場；DNS 場已在 data/ 且體積相同，重存一份沒有意義。
        # dns_path 記進檔案，繪圖端據此取回配對的參考場，避免對錯 DNS。
        np.savez(
            out_dir / "fields.npz",
            u_pred=out["u_fields"], v_pred=out["v_fields"],
            t=np.asarray(dns_t),
            x=np.linspace(0.0, 1.0, dns_u.shape[1], endpoint=False),
            y=np.linspace(0.0, 1.0, dns_u.shape[2], endpoint=False),
            dns_path=np.array(str(dns_paths[0])),
        )
        print(f"[out] fields.npz        → {out_dir / 'fields.npz'}  "
              f"shape={out['u_fields'].shape}")

    # Spectrum plot（取中段 t slice 的 E(k)）
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mid = len(metrics_per_t) // 2
            E_pred = np.asarray(metrics_per_t[mid]["E_pred_k"])
            E_dns = np.asarray(metrics_per_t[mid]["E_dns_k"])
            k = np.arange(len(E_pred))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(dns_t, ke_t_pred, "-o", label="pred", markersize=3)
            axes[0].plot(dns_t, ke_t_dns, "-s", label="DNS", markersize=3)
            axes[0].set_xlabel("t"); axes[0].set_ylabel("KE(t)"); axes[0].legend()
            axes[0].set_title("Mean KE time series")
            axes[1].loglog(k[1:], E_pred[1:] + 1e-20, label="pred")
            axes[1].loglog(k[1:], E_dns[1:] + 1e-20, label="DNS")
            axes[1].set_xlabel("k"); axes[1].set_ylabel("E(k)")
            axes[1].set_title(f"Energy spectrum at t={metrics_per_t[mid]['t']:.2f}")
            axes[1].legend()
            fig.suptitle(
                f"EXP-245 JAX  step={restored_step}  "
                f"KE_err={metrics_mean['ke_rel_err']:.4f}  "
                f"ω_err={metrics_mean['omega_rel_err']:.4f}"
            )
            fig.tight_layout()
            plot_path = out_dir / "ke_and_spectrum.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"[out] ke_and_spectrum.png → {plot_path}")

            if eval_p:
                from pi_lnn_jax.evaluate import reconstruct_field
                mid_i = len(dns_t) // 2
                mid_t = float(dns_t[mid_i])
                Nx, Ny = dns_u.shape[1], dns_u.shape[2]
                xg = np.linspace(0.0, 1.0, Nx, endpoint=False).astype(np.float32)
                yg = np.linspace(0.0, 1.0, Ny, endpoint=False).astype(np.float32)
                _, _, p_pred = reconstruct_field(
                    model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
                    norm_stats, xg, yg, mid_t, return_p=True,
                )
                p_dns_mid = np.asarray(dns_p[mid_i])
                pp = p_pred - p_pred.mean()        # gauge-corrected
                pd = p_dns_mid - p_dns_mid.mean()
                # pred/DNS 共用色階，肉眼比對才不被各自 autoscale 誤導
                vmax = float(max(np.abs(pp).max(), np.abs(pd).max()))
                figp, axp = plt.subplots(1, 3, figsize=(15, 4))
                im0 = axp[0].imshow(pp, origin="lower", vmin=-vmax, vmax=vmax); axp[0].set_title("p_pred (gauge-corrected)")
                figp.colorbar(im0, ax=axp[0])
                im1 = axp[1].imshow(pd, origin="lower", vmin=-vmax, vmax=vmax); axp[1].set_title("p_DNS (gauge-corrected)")
                figp.colorbar(im1, ax=axp[1])
                im2 = axp[2].imshow(pp - pd, origin="lower"); axp[2].set_title("error")
                figp.colorbar(im2, ax=axp[2])
                figp.suptitle(f"Pressure field at t={mid_t:.2f}  p_rel_err={metrics_mean.get('p_rel_err', float('nan')):.4f}")
                figp.tight_layout()
                p_plot_path = out_dir / "pressure_field.png"
                figp.savefig(p_plot_path, dpi=120)
                plt.close(figp)
                print(f"[out] pressure_field.png → {p_plot_path}")
        except Exception as e:
            print(f"[WARN] plot 失敗 ({type(e).__name__}: {e})；可加 --no-plot 跳過")

    # ── 最終 summary print ──
    print("\n" + "=" * 80)
    print(f"=== Mean over T={T} (Re={re_value}, step={restored_step}) ===")
    for k_, v_ in metrics_mean.items():
        print(f"  {k_:<22s} = {v_:.4e}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
