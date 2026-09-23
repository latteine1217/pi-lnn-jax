#!/usr/bin/env python
"""從 ckpt 讀出 learned forcing (A, k_f)，回答 forcing 是否可辨識。

Why a separate script:
    learned forcing 會餵進 NS residual（`pipeline/kolmogorov/run.py:770`），但訓練端
    不把它寫進 `summary.json` 也不印。補在訓練端會動到 `summary.json` 的欄位集，
    那是 bit-identical A/B 對拍的比對面（CLAUDE.md §7.2 第 2 點），會造成假紅。
    所以改在這裡從 ckpt 取——參數樹裡是 `log_A` 與 `raw_k_f`，經 `get_forcing`
    還原成物理值（A=exp(log_A)；k_f=sigmoid(raw)·(k_max−k_min)+k_min）。

What it answers:
    在感測重建已達 ~5.7% KE 的前提下，優化器能否把遠離真值的 forcing 初值拉回真值。
    未被學的那一維由 config 釘在真值，所以 `--truth-*` 必須由呼叫端明示——
    腳本不猜真值，猜錯會把「塌回初值」誤讀成「收斂到真值」。

Usage（lab-server GPU node；ckpt 由 GPU 寫出，CPU 上 orbax 推不出 sharding）:
    uv run python scripts/diag_forcing_recovery.py \
        --config configs/exp_forcing_both.toml --truth-A 0.1 --truth-kf 2.0 \
        --output artifacts/diag/forcing_recovery_both.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--arch", default="liquid")
    ap.add_argument("--truth-A", type=float, required=True,
                    help="DNS forcing 的真實振幅。必填：腳本不猜真值")
    ap.add_argument("--truth-kf", type=float, required=True,
                    help="DNS forcing 的真實波數。必填：腳本不猜真值")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import jax
    from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params
    from pi_lnn_jax.config import load_config
    from pi_lnn_jax.data import load_sensors_from_path
    from pi_lnn_jax.model_factory import build_model

    cfg = load_config(a.config)
    dk, mk = cfg["data_kwargs"], cfg["model_kwargs"]
    artifacts_dir = Path(
        a.artifacts_dir if a.artifacts_dir is not None
        else cfg["train_kwargs"]["artifacts_dir"]).resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[fail-fast] 找不到 checkpoints：{ckpt_dir}")
    if not dk.get("sensor_jsons"):
        raise SystemExit(f"[fail-fast] {a.config} 的 data_kwargs.sensor_jsons 為空")

    s = load_sensors_from_path(dk["sensor_jsons"][0])
    sv = np.asarray(s["sensor_vals"])
    sp = np.asarray(s["sensor_pos"])
    st = np.asarray(s["sensor_time"])

    model, model_name = build_model(a.arch, mk, K_sensors=int(sv.shape[1]))
    params = reference_params_for(model, sv, sp, st)
    params, step, prov = restore_eval_params(
        ckpt_dir, a.ckpt, reference_params=params, model=model)

    A, k_f = model.apply(params, method=model.__class__.get_forcing)
    A, k_f = float(A), float(k_f)

    init_A, init_kf = float(mk["forcing_A_init"]), float(mk["forcing_k_f_init"])
    learn_A, learn_kf = bool(mk["learn_forcing_A"]), bool(mk["learn_forcing_k_f"])

    def _move(learned: float, init: float, truth: float) -> float:
        """學到的值走了初值→真值這段距離的幾分之幾。1.0 = 完全復原，0.0 = 完全沒動。"""
        gap = truth - init
        return float("nan") if gap == 0.0 else (learned - init) / gap

    out = {
        "provenance": {
            "script": "scripts/diag_forcing_recovery.py",
            "config": str(Path(a.config).resolve()),
            "ckpt_dir": str(ckpt_dir), "ckpt_step": int(step),
            "arch": a.arch, "model": model_name,
            "ckpt_provenance": {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                                    else str(v)) for k, v in dict(prov).items()},
            "jax_version": jax.__version__,
            "k_f_band": [float(mk.get("forcing_k_f_min", 1.0)),
                         float(mk.get("forcing_k_f_max", 8.0))],
            "k_f_band_note": ("ForcingPrior 用 sigmoid 把 k_f 鎖在此區間；區間外的初值"
                              "會在建構期 ValueError。可辨識性的結論必須帶上這個先驗"),
        },
        "A":   {"learned": A,   "init": init_A,  "truth": a.truth_A,
                "is_learned": learn_A, "recovered_fraction": _move(A, init_A, a.truth_A)},
        "k_f": {"learned": k_f, "init": init_kf, "truth": a.truth_kf,
                "is_learned": learn_kf, "recovered_fraction": _move(k_f, init_kf, a.truth_kf)},
    }
    op = Path(a.output); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n[forcing] {model_name}  step={step}")
    for nm, d in (("A", out["A"]), ("k_f", out["k_f"])):
        tag = "learned" if d["is_learned"] else "FIXED  "
        print(f"  {nm:4s} [{tag}] init={d['init']:.6g}  ->  {d['learned']:.6g}"
              f"   truth={d['truth']:.6g}   recovered={d['recovered_fraction']:.3f}")
    print(f"\n[out] {op}")


if __name__ == "__main__":
    main()
