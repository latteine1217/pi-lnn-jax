"""模型內部壓力 vs 由 DNS 速度解 Poisson 的參考壓力（重產 thesis 附錄的壓力表）。

What:
    對一個 Kolmogorov 主線 run 的 checkpoint，在評估協定決定的每一幀、DNS 網格上
    重建模型的 (u, v, p)，與「由同一幀 DNS 速度解 Poisson 得到的參考壓力」比較：
      grad_p_rel_l2    ‖∇p_pred − ∇p_ref‖₂ / ‖∇p_ref‖₂
      grad_p_rms_pred  √mean|∇p_pred|²
      grad_p_rms_ref   √mean|∇p_ref|²
      p_gr_rel_l2      各自減空間均值後的相對 L2（gauge-removed）
      uv_rel_err       速度向量相對 L2（內容指紋，見下）
    逐幀計算再做時間平均；逐幀陣列與時間平均一起寫進輸出 JSON。

Why 參考壓力不讀 DNS 的 `p`:
    DNS 檔的 `p` 出自生成器，而生成器把 Π = p + ½|u|² 那一項的符號寫反了：
    存檔 `p` 與正確壓力的相對差約 2，放進動量方程做最小平方擬合只得係數 ≈ −0.1
    （正確壓力為 +1.000）。`tests/test_pressure_poisson_reference.py` 釘住這兩件事。
    所以參考壓力一律由 DNS 的 (u, v) 解 Poisson 取得，`load_for_evaluation` 以
    `with_pressure=False` 呼叫——DNS 的 `p` 欄位在這支腳本裡從未被讀取。

Why 指標要嘛取梯度、要嘛先去均值:
    模型的 p 只經由動量殘差裡的 ∇p 受約束，loss 對 p → p + C(t) 不變（gauge 自由度）。
    未去均值的 p 值誤差量的是一個沒有被訓練約束的常數。

Why 需要 uv 指紋閘門:
    壓力數字本身沒有既有值可對，而「restore 到錯的 ckpt／config 或協定錯位」不會
    crash，只會給出看起來合理的壓力數字。同一次重建的速度場可以對：`--expect-metrics`
    是該 run 既有評估的 `metrics.json`，其 `uv_rel_err` 的時間平均與本次相差 > 2e-4
    （分數單位）即失敗。額外要求該 json 記載的 ckpt_dir／ckpt_step／config／評估時刻
    與本次一致：相鄰 seed 的 uv 可以只差 7e-6（pv_les_s2 vs s3），光靠數值擋不住
    把別的 run 的 metrics.json 傳進來。

Why 在腳本內直接呼叫 model 的 encode／decode_query:
    `evaluate.py` 的公開 API 拿不到 encode-once 的 p：`evaluate_time_series` 只收集
    (u, v) 場、`eval_p=True` 還要求 DNS 的 p（正是不能讀的那個）；`reconstruct_field`
    的 `return_p=True` 每次呼叫都重新 encode。不改 `evaluate.py`（共用 leaf module），
    在此依它的 encode-once 模式重寫查詢迴圈：encode 一次、jit decode、分塊規則相同。
    速度與 `evaluate_time_series` 走同一個 forward，由上面的 uv 閘門實測背書。

Conventions:
    陣列 [N, N]，axis 0 = x、axis 1 = y（與 `evaluate.reconstruct_field` 及
    `metric_artifact.compute_vorticity` 相同）。梯度一律週期二階中央差分，
    dx = L/N，與 `compute_vorticity` 同慣例。壓力正規化常數：主線 sensor 只有 (u, v)，
    norm_stats 無 p 統計量，訓練端 fallback 為 (p_mean, p_std) = (0, 1)
    （`pipeline/kolmogorov/assembly.py` 的 `ns.get("p_mean", 0.0)`／`ns.get("p_std", 1.0)`），
    本腳本斷言同樣的前提後補上同一組值——第三通道因此就是物理單位的 p。

Usage（lab-server，經 sbatch；腳本以未追蹤檔形式放在 job 目錄，`_common/periodic_fields.py`
須隨行放在同目錄的 `_common/` 下，由 `scripts/slurm/render_eval_pressure_limit.sh` 一併複製）:
    uv run python -u /path/to/eval_pressure_limit.py \\
        --arch liquid --protocol follow_training --arm K100_al \\
        --config /home/junyi/pi-lnn-jax/configs/exp_pv_les_s42.toml \\
        --artifacts-dir /home/junyi/pi-lnn-jax/artifacts/kolmogorov/pv_les_s42 \\
        --expect-metrics /home/junyi/pi-lnn-jax/artifacts/kolmogorov/pv_les_s42/eval_v2_chunk400k/metrics.json \\
        --out out/pv_les_s42.json
    多份輸出的 arm 聚合見 `scripts/aggregate_pressure_limit.py`。
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import pi_lnn_jax
from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params, verify_norm_stats
from pi_lnn_jax.config import DATA_SCHEMA, load_config
from pi_lnn_jax.data import _resolve_data_path
from pi_lnn_jax.evaluation_protocol import (
    ProtocolMode,
    encoder_time_stride_from_config,
    load_for_evaluation,
    resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.metric_artifact import compute_metrics
from pi_lnn_jax.model_factory import build_model
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.result_table import metric_definition_id, read_metric_value

PRODUCER = "scripts/eval_pressure_limit.py"
SCHEMA = "pressure_limit.v1"

#: uv_rel_err 時間平均的容許絕對差（分數單位）。既有評估與本次走同一個 forward；
#: 已知的合法差異只剩浮點累加順序：395bb32 前後同一 ckpt 的 headline 差 ≤ 5.4e-5。
#: 分塊大小不同會更大（evaluate.py 註解：d384/Re=1e6 下 chunk 1000 vs 2000 差到 3e-4），
#: 所以 sbatch 設與既有評估相同的 PILNN_EVAL_CHUNK_BUDGET，讓分塊也相同。
UV_GATE_ATOL = 2e-4

METRIC_KEYS = ("grad_p_rel_l2", "grad_p_rms_pred", "grad_p_rms_ref", "p_gr_rel_l2", "uv_rel_err")

#: 與訓練端 fallback 相同的壓力正規化常數（見 module docstring 的 Conventions）。
P_NORM_FALLBACK = {"p_mean": 0.0, "p_std": 1.0}


# ── 參考壓力與梯度算子 ────────────────────────────────────────────────────────

# 週期場運算與 oracle 佈點生成器共用；此處重新匯出以維持既有匯入路徑。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common.periodic_fields as _periodic_fields
from _common.periodic_fields import (
    periodic_gradient,
    reference_pressure,
)

# provenance 記的是實際被匯入的那份共用模組
PERIODIC_FIELDS_PATH = Path(_periodic_fields.__file__).resolve()


def pressure_frame_metrics(p_pred: np.ndarray, p_ref: np.ndarray, L: float) -> dict[str, float]:
    """單幀的四個壓力指標。不加 eps：參考場為零時讓 inf/nan 冒出來，由呼叫端的有限性檢查擋下。"""
    p_pred = np.asarray(p_pred, dtype=np.float64)
    p_ref = np.asarray(p_ref, dtype=np.float64)
    gxp, gyp = periodic_gradient(p_pred, L)
    gxr, gyr = periodic_gradient(p_ref, L)
    ref_sq = np.sum(gxr ** 2 + gyr ** 2)
    pp = p_pred - p_pred.mean()
    pr = p_ref - p_ref.mean()
    return {
        "grad_p_rel_l2": float(np.sqrt(np.sum((gxp - gxr) ** 2 + (gyp - gyr) ** 2) / ref_sq)),
        "grad_p_rms_pred": float(np.sqrt(np.mean(gxp ** 2 + gyp ** 2))),
        "grad_p_rms_ref": float(np.sqrt(ref_sq / gxr.size)),
        "p_gr_rel_l2": float(np.linalg.norm(pp - pr) / np.linalg.norm(pr)),
    }


# ── encode-once 重建（依 evaluate.py 的模式，見 module docstring）──────────────

@functools.partial(jax.jit, static_argnums=0)
def _encode(model, params, sensor_vals, sensor_pos, re_norm, sensor_time):
    return model.apply(params, sensor_vals, sensor_pos, re_norm, sensor_time,
                       method=LiquidOperator.encode)


@functools.partial(jax.jit, static_argnums=0)
def _decode(model, params, xy, t_q, h_states, sensor_time, sensor_pos):
    return model.apply(params, xy, t_q, h_states, sensor_time, sensor_pos,
                       method=LiquidOperator.decode_query)


def _query_chunk(K: int, n_total: int) -> int:
    """與 `evaluate._reconstruct` 相同的分塊規則（含 PILNN_EVAL_CHUNK_BUDGET 語意）。

    decode 的 cross-attention 中介 ∝ chunk×K×d_model，K>128 時 chunk=4096 會 OOM。
    查詢點逐點獨立，分塊只影響 XLA 的累加順序（~1e-5 相對量級），由 uv 閘門覆蓋。
    """
    budget = int(os.environ.get("PILNN_EVAL_CHUNK_BUDGET", "200000"))
    chunk = 4096 if K <= 128 else max(256, budget // K)
    return min(chunk, n_total)


def iter_uvp_frames(model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
                    norm_stats: dict, N: int, times):
    """encode 一次，逐幀 jit decode；yield 物理單位的 (u, v, p)，各 [N, N] float32。

    查詢格點 = linspace(0, 1, N, endpoint=False)，與 `evaluate.evaluate_time_series`
    相同（呼叫端已斷言 DNS 的 L = 1）。
    """
    if not isinstance(model, LiquidOperator):
        raise NotImplementedError(
            f"encode-once 只對 LiquidOperator 成立；收到 {type(model).__name__}")
    h_states = _encode(model, params, sensor_vals, sensor_pos, re_norm, sensor_time)
    grid = np.linspace(0.0, 1.0, N, endpoint=False).astype(np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    n_total = xy.shape[0]
    chunk = _query_chunk(int(sensor_pos.shape[0]), n_total)
    # 每塊固定 shape：尾塊補零點再截掉，jit 只編譯一次（同 evaluate._reconstruct）。
    xy = jnp.asarray(np.pad(xy, ((0, (-n_total) % chunk), (0, 0))))
    for t in times:
        t_q = jnp.full((chunk,), float(t), dtype=jnp.float32)
        pred = np.concatenate([
            np.asarray(_decode(model, params, xy[i:i + chunk], t_q, h_states,
                               sensor_time, sensor_pos))
            for i in range(0, xy.shape[0], chunk)
        ])[:n_total]
        u = pred[:, 0] * norm_stats["u_std"] + norm_stats["u_mean"]
        v = pred[:, 1] * norm_stats["v_std"] + norm_stats["v_mean"]
        p = pred[:, 2] * norm_stats["p_std"] + norm_stats["p_mean"]
        yield u.reshape(N, N), v.reshape(N, N), p.reshape(N, N)


# ── 輸入核對（全部在 GPU 工作之前）──────────────────────────────────────────

@dataclass(frozen=True)
class Expectation:
    """`--expect-metrics` 裡本腳本要對的東西。"""
    path: Path
    uv_rel_err: float
    ckpt_step: int
    times: tuple[float, ...]


def _same_path(a, b) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def read_expectation(path: Path, *, ckpt_dir: Path, config: Path) -> Expectation:
    """讀既有評估的 metrics.json，並確認它描述的正是這個 run。

    uv 值經 `result_table.read_metric_value`（語意層）讀取：鍵缺席時大聲失敗，
    不回 None。ckpt_dir／config／ckpt_step／metrics_per_t 是 provenance 不是 metric，直接讀。
    """
    payload = json.loads(path.read_text())
    uv = read_metric_value(payload, metric_definition_id("uv_rel_err"))
    if uv is None:
        raise ValueError(f"{path}: metrics_mean.uv_rel_err 為 null——無指紋可對")
    for key in ("ckpt_dir", "ckpt_step", "config", "metrics_per_t"):
        if key not in payload:
            raise KeyError(f"{path} 缺 {key!r}，無法確認它描述的是哪個 run")
    if not _same_path(payload["ckpt_dir"], ckpt_dir):
        raise ValueError(f"{path} 記載 ckpt_dir={payload['ckpt_dir']}，本次為 {ckpt_dir}")
    if not _same_path(payload["config"], config):
        raise ValueError(f"{path} 記載 config={payload['config']}，本次為 {config}")
    times = tuple(float(row["t"]) for row in payload["metrics_per_t"])
    return Expectation(path=path, uv_rel_err=float(uv),
                       ckpt_step=int(payload["ckpt_step"]), times=times)


def read_training_summary(artifacts_dir: Path, *, arch: str, config: Path) -> dict:
    """summary.json 記載的 arch 與 config 必須與 CLI 一致（ARCH 是 sbatch env、不進 config）。"""
    path = artifacts_dir / "summary.json"
    summary = json.loads(path.read_text())
    for key in ("arch", "config_path", "use_al"):
        if key not in summary:
            raise KeyError(f"{path} 缺 {key!r}")
    if summary["arch"] != arch:
        raise ValueError(f"{path} 記載 arch={summary['arch']!r}，CLI 為 {arch!r}")
    if not _same_path(summary["config_path"], config):
        raise ValueError(f"{path} 記載 config_path={summary['config_path']}，CLI 為 {config}")
    return summary


def git_sha(repo_root: Path) -> str:
    """`git rev-parse HEAD`；非零退出（不是 git 目錄等）記為 "unknown"，其餘錯誤照常拋出。"""
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(f"[provenance][WARN] git rev-parse 失敗（{res.stderr.strip()}）→ 記為 unknown")
        return "unknown"
    return res.stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # arch 必填：它是 sbatch env、不進 config，config 相同不代表同一個模型。
    # 只收 liquid：encode／decode_query 拆分只存在於 LiquidOperator。
    p.add_argument("--arch", required=True, choices=["liquid"])
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode])
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--time-stride", type=int, default=None,
                   help="給了是斷言；不給由協定決定")
    p.add_argument("--artifacts-dir", required=True, type=Path,
                   help="run 目錄（含 checkpoints/ 與 summary.json）")
    p.add_argument("--config", required=True, type=Path, help="訓練用的 TOML")
    p.add_argument("--expect-metrics", required=True, type=Path,
                   help="該 run 既有評估的 metrics.json（uv 指紋；其 ckpt_step 決定 restore 哪一步）")
    p.add_argument("--arm", required=True,
                   help="聚合用的組別標籤（例 K100_al）；aggregate_pressure_limit 依此分組")
    p.add_argument("--out", required=True, type=Path, help="輸出 JSON 路徑")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    t_start = time.time()
    config = args.config.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    ckpt_dir = artifacts_dir / "checkpoints"
    for path in (config, ckpt_dir, artifacts_dir / "summary.json", args.expect_metrics):
        if not path.exists():
            raise FileNotFoundError(f"不存在: {path}")

    summary = read_training_summary(artifacts_dir, arch=args.arch, config=config)
    expect = read_expectation(args.expect_metrics.resolve(), ckpt_dir=ckpt_dir, config=config)

    cfg = load_config(config)
    data_kwargs, model_kwargs = cfg["data_kwargs"], cfg["model_kwargs"]
    sensor_jsons, dns_paths = data_kwargs.get("sensor_jsons", []), data_kwargs.get("dns_paths", [])
    re_values = data_kwargs.get("re_values", [])
    if not (len(sensor_jsons) == len(dns_paths) == len(re_values) == 1):
        raise ValueError(f"只支援單-Re config：sensor_jsons={sensor_jsons} "
                         f"dns_paths={dns_paths} re_values={re_values}")
    enc_stride = encoder_time_stride_from_config(config)
    if enc_stride != 1:
        raise NotImplementedError(f"encoder_time_stride={enc_stride} 未支援（主線為 1）")
    re_value = float(re_values[0])
    # 單-Re config 普遍不寫 re_norm_scale；訓練端經同一個 schema 拿到 default。
    re_norm_scale = float(data_kwargs.get("re_norm_scale", DATA_SCHEMA["re_norm_scale"][1]))
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))

    protocol = resolve_protocol(
        mode=args.protocol, training_time_strides=training_time_strides_from_config(config),
        cli_time_stride=args.time_stride, reason=args.protocol_reason)
    print(f"[protocol] {protocol.mode.value}  sensor_stride={protocol.sensor_time_stride} "
          f"dns_stride={protocol.dns_time_stride}  {protocol.basis}")

    dns_path = _resolve_data_path(dns_paths[0])
    dns_L = float(np.load(dns_path, allow_pickle=True).item()["config"]["L"])
    if dns_L != 1.0:
        raise NotImplementedError(
            f"DNS 的 L={dns_L}：重建的查詢格點與模型座標都寫死 [0,1)，只支援 L=1")
    # with_pressure=False：DNS 的 p 欄位（生成器符號錯誤）從不載入。
    aligned = load_for_evaluation(sensor_jsons[0], dns_path, protocol=protocol,
                                  viscosity=1.0 / re_value, with_pressure=False)
    if aligned.s != 1:
        raise ValueError(f"評估網格必須是 DNS 網格本身，grid stride={aligned.s}")
    times = np.asarray(aligned.dns_t_eval, dtype=np.float64)
    if len(times) != len(expect.times) or not np.allclose(times, expect.times, atol=1e-6, rtol=0):
        raise ValueError(
            f"評估時刻與 {expect.path} 不一致：本次 {len(times)} 幀 "
            f"[{times[0]:.4f}, {times[-1]:.4f}]，既有評估 {len(expect.times)} 幀——協定錯位")
    N = int(aligned.N)
    print(f"[data] Re={re_value:g} re_norm={re_norm:.4f} K={aligned.K} T_sensor={aligned.T} "
          f"frames={len(times)} N={N} dns={dns_path}")

    sensor_vals = jnp.asarray(aligned.sensor_vals_normalized)
    sensor_pos = jnp.asarray(aligned.sensor_pos)
    sensor_time = jnp.asarray(aligned.sensor_time)
    model, model_name = build_model(args.arch, model_kwargs, K_sensors=aligned.K)
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    # restore 既有評估所用的那一步，不取 latest：指紋比的是那一次重建。
    params, restored_step, restore_prov = restore_eval_params(
        ckpt_dir, expect.ckpt_step, reference_params=params, model=model)
    norm_prov = verify_norm_stats(artifacts_dir, aligned.norm_stats)
    print(f"[ckpt] {model_name} params={n_params:,} step={restored_step} "
          f"fingerprint_verified={restore_prov['fingerprint_verified']} "
          f"norm_stats_verified={norm_prov['norm_stats_verified']}")

    stale = sorted(set(P_NORM_FALLBACK) & set(aligned.norm_stats))
    if stale:
        raise ValueError(
            f"sensor 的 norm_stats 含 {stale}：訓練端不會 fallback，p 的反正規化前提不成立")
    norm_stats = {**{k: float(v) for k, v in aligned.norm_stats.items()}, **P_NORM_FALLBACK}

    per_frame = {k: [] for k in METRIC_KEYS}
    frames = iter_uvp_frames(model, params, sensor_vals, sensor_pos, re_norm, sensor_time,
                             norm_stats, N, times)
    for i, (u_pred, v_pred, p_pred) in enumerate(frames):
        u_ref, v_ref = aligned.dns_u_eval[i], aligned.dns_v_eval[i]
        row = pressure_frame_metrics(p_pred, reference_pressure(u_ref, v_ref, dns_L), dns_L)
        # uv 走與既有評估相同的定義（compute_metrics 的 uv_rel_err），不另寫公式。
        row["uv_rel_err"] = compute_metrics(u_pred, v_pred, u_ref, v_ref)["uv_rel_err"]
        bad = [k for k, val in row.items() if not np.isfinite(val)]
        if bad:
            raise FloatingPointError(f"t={times[i]:.4f} 的 {bad} 非有限值：{row}")
        for k in METRIC_KEYS:
            per_frame[k].append(row[k])
        if i % 10 == 0 or i == len(times) - 1:
            print(f"  [{i + 1:3d}/{len(times)}] t={times[i]:.3f}  "
                  + "  ".join(f"{k}={row[k]:.4g}" for k in METRIC_KEYS))

    means = {k: float(np.mean(v)) for k, v in per_frame.items()}
    diff = abs(means["uv_rel_err"] - expect.uv_rel_err)
    gate = {"expect_metrics": str(expect.path), "expected": expect.uv_rel_err,
            "measured": means["uv_rel_err"], "abs_diff": diff, "atol": UV_GATE_ATOL,
            "ckpt_step": expect.ckpt_step, "passed": bool(diff <= UV_GATE_ATOL)}
    print(f"[gate] uv_rel_err 本次 {means['uv_rel_err']:.6f} vs 既有 {expect.uv_rel_err:.6f} "
          f"|Δ|={diff:.2e}（容許 {UV_GATE_ATOL:.0e}）→ {'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        raise SystemExit(
            f"[FAIL] uv 指紋不符：|Δ|={diff:.3e} > {UV_GATE_ATOL:.0e}。重建的不是既有評估的那個模型／"
            f"協定——先查 ckpt、config、norm_stats、協定，壓力數字不可用。未寫出 {args.out}")

    repo_root = Path(pi_lnn_jax.__file__).resolve().parent.parent
    return {
        "producer": PRODUCER,
        "schema": SCHEMA,
        "arm": args.arm,
        "run": {
            "artifacts_dir": str(artifacts_dir), "config": str(config),
            "arch": args.arch, "model_name": model_name, "n_params": n_params,
            "K": int(aligned.K), "seed": int(cfg["train_kwargs"]["seed"]),
            "use_al": bool(summary["use_al"]), "ckpt_dir": str(ckpt_dir),
            "ckpt_step": int(restored_step), "re_value": re_value, "re_norm": re_norm,
            "re_norm_scale": re_norm_scale,
        },
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": file_sha256(Path(__file__))},
        # 壓力與梯度的定義在共用模組；與本腳本分開記，兩者任一變動都看得出來
        "periodic_fields": {"path": str(PERIODIC_FIELDS_PATH),
                            "sha256": file_sha256(PERIODIC_FIELDS_PATH)},
        "repo": {"root": str(repo_root), "git_sha": git_sha(repo_root)},
        "protocol": protocol.to_provenance(),
        "restore_provenance": restore_prov,
        "fingerprint_verified": bool(restore_prov["fingerprint_verified"]),
        "norm_stats_provenance": norm_prov,
        "norm_stats_verified": bool(norm_prov["norm_stats_verified"]),
        "norm_stats_eval": norm_stats,
        "p_normalization": "sensor norm_stats 無 p 統計量 → (p_mean, p_std) = (0, 1)，同訓練端 fallback",
        "pressure_reference": {
            "method": "spectral Poisson from DNS (u, v): p_hat = -k_i k_j F(u_i u_j) / |k|^2, k=0 -> 0",
            "dns_path": str(dns_path), "domain_length": dns_L,
            "dns_p_field_used": False,
        },
        "gradient_operator": "periodic 2nd-order central difference (np.roll), dx = L/N, axis0=x, axis1=y",
        "grid": {"N": N, "query": "linspace(0, 1, N, endpoint=False), float32"},
        "query_chunk": _query_chunk(int(aligned.K), N * N),
        "times": times.tolist(),
        "uv_gate": gate,
        "metrics": {k: {"time_mean": means[k], "per_frame": per_frame[k]} for k in METRIC_KEYS},
        "wall_seconds": time.time() - t_start,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    out = run(args)
    text = json.dumps(out, indent=2, allow_nan=False)   # 先序列化，失敗不留半份檔
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    m = out["metrics"]
    print("=== pressure limit ===")
    for k in METRIC_KEYS:
        print(f"  {k:<16s} = {m[k]['time_mean']:.6g}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
