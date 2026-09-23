#!/usr/bin/env python3
"""模型內部 latent（encoder 輸出 h_states）的跨時間 POD 秩。

What:
    只跑 encoder：`model.apply(params, sv, sp, re_norm, st, method=LiquidOperator.encode)`
    取 h_states [T, K, d_model]，展平成 [T, K*d_model] 後做 economy SVD，
    量它**跨時間**的有效秩 r_lv。

Why:
    `diag_pod_spectrum_compare.py` 已量出 DNS 場與**重建場**的跨時間有效秩
    （K=100 主線 n=5：r_99 為 25 對 22、r_99.9 為 53 對 40）。
    那告訴我們輸出端丟了自由度，但**丟在哪一段沒有判別**。

    這支腳本補上判別觀測——同一組幀、同一份 ckpt 的 latent 秩：

      r_latent ≈ r_recon  →  資訊在 **encoder** 就丟了（latent 本身就沒有那些自由度）
      r_latent ≫ r_recon  →  latent 保有自由度而 **decoder/query 路徑**沒把它表達出來

    兩者的修法完全不同，所以這個分叉值得一個 job。

⚠️ 三條判讀限制：

  1. **latent 與場活在不同空間，秩的絕對值不可直接相比。** 可比的是「跨時間有幾個
     自由度」這個問題，以及各自相對於 DNS 場秩（25 / 53）的**位置**。
  2. **r_99.99 快照數敏感**（同一份 DNS 在 201 幀下 r_99.99=92、101 幀下 82，而
     r_90/r_99/r_99.9 不變）。比就比 r_90/r_99/r_99.9。
  3. **d_model 是秩的硬上限**，K*d_model 遠大於 T，所以實際上限仍是快照數 T。
     若 r 逼近 T，量到的是取樣不足，腳本會標記。

## 試過而移除的兩個讀數（2026-09-20）
    participation ratio 與冪律指數 α 於 2026-09-20 一併移除，理由見
    `diag_pod_spectrum_compare.py` 的同名小節與
    `knowledge/experiments/kolmogorov-pod-rank-latent-2026-09-20.md`。

Usage（一定要 sbatch 到 r740；GPU 寫的 ckpt 在 head node 的 CPU JAX 上
orbax 推不出 sharding）：
    uv run python scripts/diag_latent_pod.py \
        --config configs/exp_ksweep_k200_s42.toml --arch liquid \
        --protocol follow_training \
        --out artifacts/diag/latent_pod_k200_s42.json
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

from diag_dns_pod_rank import ENERGY_LEVELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="訓練用的 TOML")
    p.add_argument("--ckpt", default="latest", help="'latest' 或 step 數")
    # arch 刻意 required：它是 sbatch env、不進 config——config 相同不代表同一個模型。
    p.add_argument("--arch", required=True, choices=["liquid"],
                   help="只支援 liquid：vanilla/pinn 沒有 encode/h_states 這條路")
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode])
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--time-stride", type=int, default=None)
    p.add_argument("--artifacts-dir", default=None, help="覆寫 config 的 artifacts_dir")
    p.add_argument("--out", required=True, help="JSON 產物落點")
    return p.parse_args()


def rank_at(lam: np.ndarray, levels) -> dict:
    cum = np.cumsum(lam) / lam.sum()
    return {str(lv): int(np.searchsorted(cum, lv) + 1) for lv in levels}


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
    if "re_norm_scale" in data_kwargs:
        re_norm_scale, rns_src = float(data_kwargs["re_norm_scale"]), "config"
    else:
        re_norm_scale, rns_src = float(DATA_SCHEMA["re_norm_scale"][1]), "DATA_SCHEMA default"
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))

    print("=" * 78)
    print(f"diag_latent_pod — config={Path(a.config).name}")
    print(f"  arch={a.arch}  ckpt={a.ckpt}  ckpt_dir={ckpt_dir}")
    print("=" * 78)

    protocol = resolve_protocol(
        mode=a.protocol,
        training_time_strides=training_time_strides_from_config(a.config),
        cli_time_stride=a.time_stride, reason=a.protocol_reason)
    print(f"[protocol] {protocol.mode.value}  sensor_stride={protocol.sensor_time_stride} "
          f"dns_stride={protocol.dns_time_stride}  {protocol.basis}")

    aligned = load_for_evaluation(sensor_json[0], Path(dns_paths[0]), protocol=protocol,
                                  viscosity=nu, with_pressure=False)
    sensor_vals, sensor_pos = aligned.sensor_vals_normalized, aligned.sensor_pos
    sensor_time = aligned.sensor_time
    K = int(sensor_vals.shape[1])
    print(f"[data] Re={re_value:g}  K={K}  sensor_frames={sensor_vals.shape[0]}  "
          f"re_norm={re_norm:.4f} (re_norm_scale={re_norm_scale:g} ← {rns_src})")

    model, model_name = build_model(a.arch, model_kwargs, K_sensors=K)
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[model] {model_name}  params={n_params:,}")
    params, restored_step, ckpt_prov = restore_eval_params(
        ckpt_dir, a.ckpt, reference_params=params, model=model)
    print(f"[ckpt] step={restored_step}  "
          f"fingerprint_verified={ckpt_prov.get('fingerprint_verified')}")

    h = model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                    method=LiquidOperator.encode)
    h = np.asarray(h, dtype=np.float64)
    print(f"[latent] h_states shape={h.shape}  (T, K, d_model)")
    if h.ndim != 3:
        raise ValueError(f"預期 h_states 為 3 維 [T,K,d_model]，得到 {h.shape}")
    T = h.shape[0]

    # 跨時間的 POD：把每一幀攤成一個向量，與場的處理對齊（同樣減時間平均取脈動）
    x = h.reshape(T, -1)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    lam = np.square(s)

    r = rank_at(lam, ENERGY_LEVELS)
    near_limit = {k: bool(int(v) >= 0.8 * T) for k, v in r.items()}

    print("[pod  ] " + "  ".join(f"r_{float(lv)*100:g}={r[str(lv)]}" for lv in ENERGY_LEVELS))
    near = [k for k, v in near_limit.items() if v]
    if near:
        print(f"[warn] r_{'/'.join(f'{float(k)*100:g}' for k in near)} 已達快照數 {T} 的 80%，"
              f"量到的可能是取樣不足而非 latent 的結構秩。")

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance 缺失不該讓診斷失敗
        rev = ""
    out = {
        "latent": {
            "shape": list(h.shape), "n_snapshots": T,
            "rank_at": r, "rank_near_snapshot_limit": near_limit,
        },
        # 參考基準寫進產物，讓判讀不必回頭翻別的檔
        "reference": {
            "dns_field_rank_at_101frames": {"0.9": 9, "0.99": 25, "0.999": 53},
            "recon_field_rank_at_101frames_K100_n5": {"0.9": 9, "0.99": 22, "0.999": 40},
            "note": "參考值來自 diag_pod_spectrum_compare（K=100 主線 main5, n=5）。"
                    "latent 與場不同空間，比的是相對位置不是絕對值。",
        },
        "provenance": {
            "config": str(a.config), "arch": a.arch, "model_name": model_name,
            "ckpt_dir": str(ckpt_dir), "ckpt_step": int(restored_step),
            "ckpt_provenance": ckpt_prov, "protocol": protocol.mode.value,
            "sensor_time_stride": protocol.sensor_time_stride,
            "re_value": re_value, "re_norm": re_norm, "re_norm_scale_source": rns_src,
            "K": K, "n_params": int(n_params), "git_head": rev,
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
