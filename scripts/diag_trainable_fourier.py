"""方向1 exp_508 的核心診斷：trainable-frequency embedding 訓練後的**頻率落點**。

What:
    從 ckpt 取出 TrainableFourierEmb 的頻率矩陣 B（shape [2, dim//2]，每一行是一個
    feature 的 2D 波向量），與**同一份 config、同一顆 seed 產生的 init B** 對照，報：
      - |B| = sqrt(kx²+ky²) 的分位數（訓練前 vs 訓練後）
      - 低於 sensor Nyquist k_s 的 feature 比例
      - d_lattice = |B − round(B)|：到整數格點的距離

Why:
    exp_508 問的不是「uv 有沒有變好」（前四臂都沒有），而是「網路自己會把頻率移到哪」。
    事前登錄的判讀規則、以及為什麼必須同時看 d_lattice（週期性壓力是「|B| 下降」的
    競爭假設），見 knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md。

    k_s 由 sensor 數導出（`diag_identifiability.sensor_band_edge`，與整條 identifiability
    工具鏈同一個定義），不寫死——換 K 的 config 若吃到寫死的 5.64，整份判讀會安靜對錯基準。

Usage:
    PYTHONPATH=. uv run python scripts/diag_trainable_fourier.py \\
        --config configs/exp_508_b3_trainablefourier.toml --ckpt latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flax.traverse_util import flatten_dict  # noqa: E402

from pi_lnn_jax.ckpt import restore_eval_params  # noqa: E402
from pi_lnn_jax.config import DATA_SCHEMA, load_config  # noqa: E402
from pi_lnn_jax.data import load_sensors_from_path  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.model_factory import build_model, model_fingerprint  # noqa: E402
from scripts.diag_identifiability import sensor_band_edge  # noqa: E402

_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp_508 trainable Fourier 頻率落點診斷")
    p.add_argument("--config", required=True, help="訓練用的 TOML config（必須與 ckpt 同一份）")
    p.add_argument("--ckpt", default="latest", help='"latest" 或明確 step')
    p.add_argument("--arch", choices=["liquid", "vanilla", "pinn"], default="liquid")
    p.add_argument("--artifacts_dir", default=None,
                   help="訓練若用 CLI 覆蓋 artifacts_dir，這裡必須帶同一個路徑")
    p.add_argument("--output-subdir", default="trainable_fourier_diag")
    p.add_argument("--k-sensor", type=float, default=None,
                   help="sensor Nyquist 波數；預設由 sensor 數導出 sqrt(K/pi)")
    return p.parse_args()


def _find_frequency_matrix(params) -> tuple[str, np.ndarray]:
    """在參數樹裡找 TrainableFourierEmb 的 B，找不到或不只一個都 fail-fast。"""
    hits = {
        "/".join(map(str, path)): np.asarray(leaf)
        for path, leaf in flatten_dict(params).items()
        if path[-1] == "B" and any("trainable_spatial_emb" in str(p) for p in path)
    }
    if not hits:
        raise SystemExit(
            "參數樹裡沒有 trainable_spatial_emb/B——這個 ckpt 不是用 "
            "use_trainable_fourier=true 訓練的，診斷無對象。")
    if len(hits) > 1:
        raise SystemExit(f"找到多個頻率矩陣，無法判定要診斷哪一個：{sorted(hits)}")
    path, B = next(iter(hits.items()))
    if B.ndim != 2 or B.shape[0] != 2:
        raise SystemExit(f"B 的形狀 {B.shape} 不是預期的 [2, n_features]")
    return path, B


def _trunk_in_kernel(params) -> np.ndarray:
    hits = [np.asarray(leaf) for path, leaf in flatten_dict(params).items()
            if path[-1] == "kernel" and any("trunk_in" in str(p) for p in path)]
    if len(hits) != 1:
        raise SystemExit(f"trunk_in kernel 不唯一（找到 {len(hits)} 個），無法量下游閘門")
    return hits[0]


def _downstream_gate(params, fourier_dim: int, trainable_dim: int) -> dict:
    """`trunk_in` 對兩塊座標特徵的 row norm —— 「下游關斷」假設的判別觀測。

    forward 的 base_inputs 以 `pos_enc = [spatial_emb, trainable_spatial_emb]` 開頭，
    所以 kernel 的 row [0:fourier_dim] 吃 baseline 特徵、[fourier_dim:+trainable_dim]
    吃 trainable Fourier 特徵。

    Why 要量這個：若 |B| 不動而結果變差，有兩個競爭假設——(P) B 收到的梯度微不足道
    （分支是死重），(G) 網路不調頻率而是把下游權重壓小、主動關斷整條分支。
    兩者都預測「B 不動」，但只有 (G) 預測 trainable 區塊的 row norm 相對 baseline 區塊
    顯著萎縮。
    """
    kernel = _trunk_in_kernel(params)
    if kernel.shape[0] < fourier_dim + trainable_dim:
        raise SystemExit(
            f"trunk_in kernel 只有 {kernel.shape[0]} 列，放不下 "
            f"{fourier_dim}+{trainable_dim} 的座標特徵——維度佈局假設不成立")
    base_rows = kernel[:fourier_dim]
    trainable_rows = kernel[fourier_dim:fourier_dim + trainable_dim]
    base_norm = float(np.linalg.norm(base_rows, axis=1).mean())
    trainable_norm = float(np.linalg.norm(trainable_rows, axis=1).mean())
    return {
        "trunk_in_rows": int(kernel.shape[0]),
        "base_block_row_norm_mean": base_norm,
        "trainable_block_row_norm_mean": trainable_norm,
        "trainable_over_base": trainable_norm / base_norm if base_norm > 0 else float("nan"),
    }


def _describe(B: np.ndarray, k_sensor: float) -> dict:
    """把一個頻率矩陣壓成可判讀的統計量。"""
    mag = np.linalg.norm(B, axis=0)                    # [n_features]
    d_lattice = np.abs(B - np.round(B))                # 到整數格點的距離（逐分量）
    return {
        "n_features": int(mag.size),
        "abs_B_quantiles": {f"p{int(q * 100)}": float(np.quantile(mag, q)) for q in _QUANTILES},
        "abs_B_mean": float(mag.mean()),
        "frac_below_k_sensor": float((mag < k_sensor).mean()),
        "frac_in_mid_band": float(((mag >= k_sensor) & (mag <= 16.0)).mean()),
        "d_lattice_mean": float(d_lattice.mean()),
        "d_lattice_median": float(np.median(d_lattice)),
        "frac_near_lattice_0p05": float((d_lattice < 0.05).mean()),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_kwargs, data_kwargs, model_kwargs = (
        cfg["train_kwargs"], cfg["data_kwargs"], cfg["model_kwargs"])

    if not model_kwargs.get("use_trainable_fourier", False):
        raise SystemExit(
            f"{args.config} 沒有開 use_trainable_fourier——本診斷無對象。")

    artifacts_dir = Path(
        args.artifacts_dir if args.artifacts_dir is not None
        else train_kwargs.get("artifacts_dir", "artifacts/run")
    ).resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    out_dir = artifacts_dir / args.output_subdir
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint 目錄不存在：{ckpt_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor_jsons = data_kwargs.get("sensor_jsons", [])
    if not sensor_jsons:
        raise SystemExit("TOML data_kwargs.sensor_jsons 為空")

    # sensor 只用來給 model.init 正確的 [T, K, ...] 形狀（B 的形狀與時間軸無關），
    # 但 stride 仍照訓練設定取——用「反正不影響」當理由去偏離訓練設定是壞習慣。
    protocol = resolve_protocol(
        mode="follow_training",
        training_time_strides=training_time_strides_from_config(args.config),
        cli_time_stride=None, reason=None)
    sensors = load_sensors_from_path(
        sensor_jsons[0], time_stride=protocol.sensor_time_stride)
    sensor_vals = jnp.asarray(sensors["sensor_vals"])
    sensor_pos = jnp.asarray(sensors["sensor_pos"])
    sensor_time = jnp.asarray(sensors["sensor_time"])
    K = int(sensor_pos.shape[0])

    # 與 diag_identifiability / 4D-Var oracle 共用同一個 k_s 定義（K=100 → 5.642）。
    k_sensor = args.k_sensor if args.k_sensor is not None else sensor_band_edge(K)

    re_values = data_kwargs.get("re_values", [])
    if not re_values:
        raise SystemExit("TOML data_kwargs.re_values 為空")
    # re_norm_scale 未寫進 TOML 時，訓練端吃的是 **schema 預設**（typed config 層套用，
    # 見 pipeline/kolmogorov/assembly.py:re_norm_scale_of）。因此這裡不能 fail——那會把
    # 「正當預設」誤判成「不知道」；但也不能寫死 10000.0，否則 schema 預設改了就靜默漂移。
    # 取同一個來源，並把實際用值與來源寫進報告。
    re_norm_scale_source = "config"
    if "re_norm_scale" not in data_kwargs:
        re_norm_scale_source = "DATA_SCHEMA default"
    re_norm_scale = float(data_kwargs.get("re_norm_scale", DATA_SCHEMA["re_norm_scale"][1]))
    re_norm = float(np.log(float(re_values[0])) / np.log(re_norm_scale))

    seed = int(train_kwargs.get("seed", 42))
    model, model_name = build_model(args.arch, model_kwargs, K_sensors=K)
    # 這裡刻意**不用** ckpt.reference_params_for：那支 adapter 產的是形狀模板，
    # 用固定 RNG，值不具意義。本腳本讀的是 init_params 的**值**（B_init、
    # gate_init，以及報告裡的 shift 三項差值），必須是訓練 seed 下的實際初始化。
    T_total = float(model_kwargs.get("T_total", 5.0))
    init_xy = jnp.asarray(np.random.RandomState(seed).uniform(0, 1, (8, 2)).astype(np.float32))
    init_t = jnp.asarray(np.random.RandomState(seed).uniform(0, T_total, (8,)).astype(np.float32))
    init_params = model.init(
        jax.random.split(jax.random.PRNGKey(seed))[1],
        sensor_vals, sensor_pos, re_norm, sensor_time, init_xy, init_t)

    print("=" * 78)
    print(f"diag_trainable_fourier — config={Path(args.config).name}  model={model_name}")
    print(f"  artifacts: {artifacts_dir}")
    print(f"  K={K}  k_sensor={k_sensor:.3f}  init_freq_scale="
          f"{model_kwargs.get('trainable_fourier_init_scale')}")
    print("=" * 78)

    trained_params, restored_step, provenance = restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=init_params, model=model)

    b_path, B_init = _find_frequency_matrix(init_params)
    _, B_trained = _find_frequency_matrix(trained_params)
    if B_init.shape != B_trained.shape:
        raise SystemExit(f"init 與 ckpt 的 B 形狀不符：{B_init.shape} vs {B_trained.shape}")

    report = {
        "config": str(Path(args.config).resolve()),
        "artifacts_dir": str(artifacts_dir),
        "ckpt_step": restored_step,
        "ckpt_provenance": provenance,
        "model_construction": model_fingerprint(model),
        "seed": seed,
        "K_sensors": K,
        "k_sensor": k_sensor,
        "k_sensor_source": "cli" if args.k_sensor is not None else "sensor_band_edge(K)",
        "re_norm_scale": re_norm_scale,
        "re_norm_scale_source": re_norm_scale_source,
        "re_norm": re_norm,
        "b_param_path": b_path,
        "sensor_time_stride": protocol.sensor_time_stride,
        "evaluation_protocol": protocol.mode.value,
        "init": _describe(B_init, k_sensor),
        "trained": _describe(B_trained, k_sensor),
        "gate_init": _downstream_gate(init_params, int(model.fourier_embed_dim),
                                      int(model.trainable_fourier_dim)),
        "gate_trained": _downstream_gate(trained_params, int(model.fourier_embed_dim),
                                         int(model.trainable_fourier_dim)),
    }
    report["shift"] = {
        "abs_B_median_delta":
            report["trained"]["abs_B_quantiles"]["p50"] - report["init"]["abs_B_quantiles"]["p50"],
        "frac_below_k_sensor_delta":
            report["trained"]["frac_below_k_sensor"] - report["init"]["frac_below_k_sensor"],
        "d_lattice_mean_delta":
            report["trained"]["d_lattice_mean"] - report["init"]["d_lattice_mean"],
    }

    out_json = out_dir / "trainable_fourier_frequencies.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez(out_dir / "trainable_fourier_B.npz", B_init=B_init, B_trained=B_trained)

    ini, tra = report["init"], report["trained"]
    print("\n|B| 分位數        init → trained")
    for q in _QUANTILES:
        key = f"p{int(q * 100)}"
        print(f"  {key:>4s}  {ini['abs_B_quantiles'][key]:8.3f} → {tra['abs_B_quantiles'][key]:8.3f}")
    print(f"\n低於 k_s={k_sensor:.3f} 的比例   "
          f"{ini['frac_below_k_sensor']:.3f} → {tra['frac_below_k_sensor']:.3f}")
    print(f"落在 mid-band (k_s, 16] 的比例  "
          f"{ini['frac_in_mid_band']:.3f} → {tra['frac_in_mid_band']:.3f}")
    print(f"到整數格點距離 d_lattice 均值   "
          f"{ini['d_lattice_mean']:.4f} → {tra['d_lattice_mean']:.4f}"
          f"   (隨機值理論均值 0.25)")
    gi, gt = report["gate_init"], report["gate_trained"]
    print("\n下游閘門（trunk_in row norm 均值）      init → trained")
    print(f"  baseline 區塊    {gi['base_block_row_norm_mean']:.4f} → {gt['base_block_row_norm_mean']:.4f}")
    print(f"  trainable 區塊   {gi['trainable_block_row_norm_mean']:.4f} → {gt['trainable_block_row_norm_mean']:.4f}")
    print(f"  比值 t/b         {gi['trainable_over_base']:.4f} → {gt['trainable_over_base']:.4f}")
    print(f"\n寫出：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
