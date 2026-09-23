#!/usr/bin/env python3
"""latent 的線性可解碼性：降秩線性探針 vs decoder，交叉驗證。

What:
    取 encoder 輸出 h_states [T, K, d_model]，投影到前 r 個 POD 模態後，
    用**最小平方**擬合 DNS 場，在**未用於擬合的幀**上算 uv 相對 L2。
    掃 r，並與同一批測試幀上 decoder 的誤差對照。

Why:
    `diag_latent_pod.py` 量到 latent 的跨時間秩遠高於重建場
    （K=100：r_99 為 43 對 22）。但高維空間的低能量噪聲方向也會抬高秩，
    所以「latent 有更多**有用**自由度」還沒被證實。

    這支腳本關掉那個競爭假設：若 latent 的前 r 個模態經**線性**映射就能在
    未見幀上勝過 decoder，那多出來的秩承載的是真資訊，而 decoder 沒用到它。

⚠️ **這是單向判別，不要反向讀。**
    線性探針勝過 decoder → 結論強（連線性都更好）。
    線性探針輸給 decoder → **不能下結論**：decoder 是非線性的，本來就可能更強。

⚠️ **探針用 DNS 場當擬合目標，所以它是 oracle，不是可部署的方法。**
    它回答的是「latent 裡有沒有這些資訊」，不是「可以這樣做重建」。
    與 job 5820 的 POD-margin oracle 同性質。

⚠️ **必須看 control，不要只看絕對數字。**
    本腳本一定同時跑 DNS self-probe（拿 DNS 場自己當輸入）當**可達上界**。
    理由是實測教訓：初版用「前半 fit、後半 test」的連續區塊分割，latent 探針得 90%，
    看起來像 latent 沒資訊——但同一分割下 **DNS self-probe 也只有 86-92%**。
    2.5 秒的間隔已超過該流的時間相關長度（KE 前後半差 14%），前半學的 POD 基底
    涵蓋不到後半，那個分割對**任何**輸入都沒有判別力。

    因此可讀的是 **latent 探針 / control 的比值**，不是探針的絕對值；
    分割方式對兩者的影響相同，比值把它消掉。

    ⚠️ **連續區塊分割已於 2026-09-20 移除**（原 `--split block`）。它對本問題無判別力，
    留著只會讓人再走一次那條路。要重現該教訓見上述數字與
    `knowledge/experiments/kolmogorov-pod-rank-latent-2026-09-20.md`。

Usage（sbatch 到 r740）：
    uv run python scripts/diag_latent_linear_probe.py \
        --config configs/exp_main5_s42.toml --arch liquid \
        --protocol follow_training \
        --pred artifacts/kolmogorov/main5_s42/final_eval/fields.npz \
        --out artifacts/diag/latent_probe_main5_s42.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params  # noqa: E402
from pi_lnn_jax.config import DATA_SCHEMA, load_config  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode, load_for_evaluation, resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.model_factory import build_model  # noqa: E402
from pi_lnn_jax.models import LiquidOperator  # noqa: E402

RANKS = (5, 10, 20, 40, 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="latest")
    p.add_argument("--arch", required=True, choices=["liquid"])
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode])
    p.add_argument("--protocol-reason", default=None)
    p.add_argument("--time-stride", type=int, default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--pred", required=True,
                   help="decoder 輸出的 fields.npz，作為對照組。缺它就沒有比較基準，故必填")
    p.add_argument("--split", default="gap2", choices=("interleaved", "gap2"),
                   help="gap2=每 4 幀取 1 fit、1 test 且相隔 2 幀（0.1s），預設；"
                        "interleaved=偶/奇幀（天花板較低但洩漏只污染探針不污染 decoder，"
                        "故該模式不判定與 decoder 的勝負）")
    p.add_argument("--out", required=True)
    return p.parse_args()


def uv_rel_l2(u_p: np.ndarray, v_p: np.ndarray,
              u_r: np.ndarray, v_r: np.ndarray) -> float:
    """與主指標同定義：sqrt(Σ|Δu|²+|Δv|²) / sqrt(Σ|u|²+|v|²)，跨給定幀彙總。"""
    num = np.square(u_p - u_r).sum() + np.square(v_p - v_r).sum()
    den = np.square(u_r).sum() + np.square(v_r).sum()
    return float(np.sqrt(num / den) * 100.0)


def main() -> int:
    a = parse_args()
    cfg = load_config(a.config)
    train_kwargs, data_kwargs, model_kwargs = (
        cfg["train_kwargs"], cfg["data_kwargs"], cfg["model_kwargs"])
    artifacts_dir = Path(a.artifacts_dir if a.artifacts_dir is not None
                         else train_kwargs.get("artifacts_dir", "artifacts/run")).resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir 不存在: {ckpt_dir}")

    sensor_json = data_kwargs.get("sensor_jsons", [])
    dns_paths = data_kwargs.get("dns_paths", [])
    re_values = data_kwargs.get("re_values", [])
    for name, val in (("sensor_jsons", sensor_json), ("dns_paths", dns_paths),
                      ("re_values", re_values)):
        if not val:
            raise ValueError(f"TOML data_kwargs.{name} 為空")
    re_value = float(re_values[0])
    nu = 1.0 / re_value
    rns = (float(data_kwargs["re_norm_scale"]) if "re_norm_scale" in data_kwargs
           else float(DATA_SCHEMA["re_norm_scale"][1]))
    re_norm = float(np.log(re_value) / np.log(rns))

    protocol = resolve_protocol(
        mode=a.protocol,
        training_time_strides=training_time_strides_from_config(a.config),
        cli_time_stride=a.time_stride, reason=a.protocol_reason)
    aligned = load_for_evaluation(sensor_json[0], Path(dns_paths[0]), protocol=protocol,
                                  viscosity=nu, with_pressure=False)
    sensor_vals, sensor_pos = aligned.sensor_vals_normalized, aligned.sensor_pos
    sensor_time = aligned.sensor_time
    dns_u = np.asarray(aligned.dns_u_eval, dtype=np.float64)
    dns_v = np.asarray(aligned.dns_v_eval, dtype=np.float64)
    K = int(sensor_vals.shape[1])
    T = dns_u.shape[0]
    print(f"[data] Re={re_value:g} K={K} frames={T} grid={dns_u.shape[1]}x{dns_u.shape[2]}")

    # decoder 對照：既有的 fields.npz，避免重跑 19 分鐘的全場重建
    z = np.load(a.pred)
    for k in ("u_pred", "v_pred", "t"):
        if k not in z.files:
            raise SystemExit(f"[fail] {a.pred} 缺 {k}")
    u_dec = np.asarray(z["u_pred"], dtype=np.float64)
    v_dec = np.asarray(z["v_pred"], dtype=np.float64)
    if u_dec.shape != dns_u.shape:
        raise SystemExit(f"[fail] decoder 場 {u_dec.shape} 與 DNS {dns_u.shape} 形狀不符；"
                         f"不做截短或內插")
    dt = np.abs(np.asarray(z["t"], dtype=np.float64)
                - np.asarray(aligned.dns_t_eval, dtype=np.float64))
    if dt.max() > 1e-6:
        raise SystemExit(f"[fail] decoder 場與 DNS 時間戳不吻合，最大偏差 {dt.max():.3e}")
    print(f"[dec ] {a.pred} 對齊通過")

    model, model_name = build_model(a.arch, model_kwargs, K_sensors=K)
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    params, step, prov = restore_eval_params(ckpt_dir, a.ckpt,
                                             reference_params=params, model=model)
    print(f"[ckpt] step={step} fingerprint_verified={prov.get('fingerprint_verified')}")

    h = np.asarray(model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                               method=LiquidOperator.encode), dtype=np.float64)
    if h.shape[0] != T:
        raise SystemExit(f"[fail] latent 幀數 {h.shape[0]} != DNS 幀數 {T}")
    X = h.reshape(T, -1)
    print(f"[lat ] h_states {h.shape} → X {X.shape}")

    # 連續區塊分割：相鄰幀動力學相關，隨機分割會洩漏
    if a.split == "interleaved":
        idx_fit, idx_test = np.arange(0, T, 2), np.arange(1, T, 2)
        note = "偶/奇幀（相鄰 0.05s，洩漏最重；僅供上下界對照）"
    else:  # gap2
        idx_fit = np.arange(0, T, 4)
        idx_test = np.arange(2, T, 4)
        note = "每 4 幀取 1 fit、1 test，兩者相隔 2 幀（0.1s）以降低洩漏"
    if len(idx_fit) < 4 or len(idx_test) < 4:
        raise SystemExit(f"[fail] 分割後 fit={len(idx_fit)} test={len(idx_test)}，兩側都需 >=4 幀")
    print(f"[split] {a.split}: fit={len(idx_fit)} 幀 test={len(idx_test)} 幀 — {note}")

    Y = np.concatenate([dns_u.reshape(T, -1), dns_v.reshape(T, -1)], axis=1)
    Xm = X[idx_fit].mean(axis=0, keepdims=True)
    Ym = Y[idx_fit].mean(axis=0, keepdims=True)
    Xc, Yc = X - Xm, Y - Ym

    # 在 fit 區塊上取 latent 的 POD 基底，再把全段投影上去
    U, S, Vt = np.linalg.svd(Xc[idx_fit], full_matrices=False)
    n_field = dns_u.shape[1] * dns_u.shape[2]

    dec_test = uv_rel_l2(u_dec[idx_test], v_dec[idx_test],
                         dns_u[idx_test], dns_v[idx_test])
    dec_all = uv_rel_l2(u_dec, v_dec, dns_u, dns_v)
    print(f"[dec ] decoder uv_rel_err: 全段 {dec_all:.3f}%  測試段 {dec_test:.3f}%")

    def run_probe(Xsrc: np.ndarray, label: str) -> list:
        Xm_ = Xsrc[idx_fit].mean(axis=0, keepdims=True)
        Xc_ = Xsrc - Xm_
        _, S_, Vt_ = np.linalg.svd(Xc_[idx_fit], full_matrices=False)
        res = []
        for r in RANKS:
            r_eff = int(min(r, len(S_), len(idx_fit) - 1))
            Z = Xc_ @ Vt_[:r_eff].T
            W, *_ = np.linalg.lstsq(Z[idx_fit], Yc[idx_fit], rcond=None)
            Yh = Z[idx_test] @ W + Ym
            u_h = Yh[:, :n_field].reshape(len(idx_test), *dns_u.shape[1:])
            v_h = Yh[:, n_field:].reshape(len(idx_test), *dns_u.shape[1:])
            e = uv_rel_l2(u_h, v_h, dns_u[idx_test], dns_v[idx_test])
            res.append({"rank_requested": r, "rank_effective": r_eff, "uv_rel_err_test": e})
            print(f"[{label:9s}] r={r_eff:<3d} test uv_rel_err = {e:7.3f}%")
        return res

    # CONTROL：用 DNS 場自己當輸入。它含 100% 資訊，是這個分割下的**可達上界**。
    # 沒有它就無法分辨「latent 沒資訊」與「分割本身沒有判別力」——實測 block 分割
    # 下 control 只有 86-92%，先前的結論因此作廢。
    ctrl_rows = run_probe(Y.copy(), "control")
    rows = run_probe(X, "probe")

    best = min(rows, key=lambda d: d["uv_rel_err_test"])
    best_ctrl = min(ctrl_rows, key=lambda d: d["uv_rel_err_test"])
    beats = best["uv_rel_err_test"] < dec_test
    gap_to_ctrl = best["uv_rel_err_test"] / best_ctrl["uv_rel_err_test"]
    print(f"\n[verdict] 最佳 latent 探針 r={best['rank_effective']} → {best['uv_rel_err_test']:.3f}%")
    print(f"          上界 control (DNS self) r={best_ctrl['rank_effective']} → "
          f"{best_ctrl['uv_rel_err_test']:.3f}%   → 比值 {gap_to_ctrl:.2f}×")
    print(f"          decoder 於同一批測試幀 → {dec_test:.3f}%")
    if best_ctrl["uv_rel_err_test"] > 50.0:
        print("  ⚠️ control 自己就 >50%，**本分割無判別力**，不要解讀 latent 的數字。")
    elif a.split == "interleaved":
        # 洩漏不對稱：探針的測試幀鄰居（0.05s）在訓練集裡，decoder 沒有這個便宜。
        # 所以本模式下「探針勝過 decoder」不成立，可讀的只有 probe/control 比值。
        print("  ⚠️ interleaved 有洩漏且**只污染探針、不污染 decoder**，"
              "故此模式下不判定與 decoder 的勝負。")
        print(f"  → 可讀的是 probe/control = {gap_to_ctrl:.3f}"
              f"（越接近 1 表示 latent 的線性可解碼資訊越接近 DNS 場本身）。")
    elif beats:
        print("  → 線性探針勝過 decoder：latent 多出來的自由度承載真資訊，decoder 沒用到它。")
    else:
        print("  → 線性探針未勝過 decoder。**這不構成反向結論**（decoder 非線性）；"
              "可讀的是 latent 相對 control 的比值。")

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = ""
    out = {
        "probe": rows,
        "control_dns_self_probe": ctrl_rows,
        "decoder": {"uv_rel_err_test": dec_test, "uv_rel_err_all": dec_all},
        "verdict": {"probe_beats_decoder": (None if a.split == "interleaved" else bool(beats)),
                    "beats_undefined_reason": ("interleaved 的洩漏只污染探針不污染 decoder"
                                               if a.split == "interleaved" else None),
                    "best": best, "best_control": best_ctrl,
                    "probe_over_control_ratio": gap_to_ctrl,
                    "one_sided_note": "只有『探針勝出』有結論力；未勝出不代表 latent 沒資訊"},
        "split": {"mode": a.split, "n_fit": int(len(idx_fit)), "n_test": int(len(idx_test)),
                  "note": note},
        "oracle_note": "探針以 DNS 場為擬合目標，屬 oracle，不是可部署方法",
        "provenance": {
            "config": str(a.config), "arch": a.arch, "model_name": model_name,
            "ckpt_dir": str(ckpt_dir), "ckpt_step": int(step), "ckpt_provenance": prov,
            "pred": str(a.pred), "protocol": protocol.mode.value,
            "K": K, "T": int(T), "latent_dim": int(X.shape[1]),
            "re_value": re_value, "git_head": rev,
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
