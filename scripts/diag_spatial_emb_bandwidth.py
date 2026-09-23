#!/usr/bin/env python3
"""diag_spatial_emb_bandwidth.py — 訓練後的空間嵌入能表達到多高的波數。

What:
    從 ckpt 取 `LearnableFourierEmb` 的頻率矩陣 B（`spatial_emb/kernel`，shape [4, dim//2]），
    與**同一份 config、同一顆 seed 的 init B** 對照，報每個 feature 的振幅
    R_x = ‖B[0:2, j]‖、R_y = ‖B[2:4, j]‖ 的分位數，以及由此導出的可達諧波階數。

Why:
    `LearnableFourierEmb` 的週期編碼只有**基頻**：
        period_enc = [sin(cx), cos(cx), sin(cy), cos(cy)]，c = 2π/L
        proj       = period_enc @ B          → R_x·sin(cx + φ_x) + R_y·sin(cy + φ_y)
        γ(x)       = [cos(proj), sin(proj)]
    所以高次諧波是 `cos(R sin θ)` 展開出來的（Jacobi–Anger），第 n 階振幅 = |J_n(R)|。
    R 決定這個嵌入的**頻寬上限**：R 小的時候 k≥16 的振幅是機器零。

    `chapter04.tex` §sec:spectrum 把 k≥16 的譜地板歸因於「感測極限與這個 operator
    萃取得到什麼」。trunk 座標基底本身的頻寬是同一觀測的**第二個解釋**，而且可分辨：
    看訓練後的 R 有沒有長大。init 的 R 中位數約 2.5（此時 |J_16| ~ 1e-11）；若訓練後
    仍在同一量級，這條競爭假設就是承重的，§sec:spectrum 的歸因需要加限制；
    若長到 8–16，可以直接排除。

    這是**診斷**不是 eval producer：不落 metric artifact。

Usage:
    PYTHONPATH=. uv run python scripts/diag_spatial_emb_bandwidth.py \\
        --config configs/exp_245_b3_les_T50.toml --ckpt latest

    需要 GPU 寫出的 ckpt → 走 Slurm（head node 的 CPU JAX 推不出 orbax sharding）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pi_lnn_jax import ckpt as ckpt_mod
from pi_lnn_jax.config import load_config
from pi_lnn_jax.data import load_sensors_from_path
from pi_lnn_jax.model_factory import build_model

# |J_n(R)| 低於此值視為「該諧波不存在」。1e-3 是相對於 |J_0|~O(1) 的三個數量級。
HARMONIC_FLOOR = 1e-3
MAX_HARMONIC = 64


# 訓練端的 fallback（單一來源：pipeline/kolmogorov/assembly.py:194 與
# pi_lnn_jax/config.py:306）。eval 紅線禁的是「**與訓練不同**的靜默預設」，
# 不是「config 省略了訓練自己也省略的鍵」——主線 exp_main5_* 兩個鍵都沒設，
# 硬失敗會讓診斷在它唯一該用的 run 上跑不起來。所以照抄訓練端的解析，
# 但把來源印出來、也寫進輸出，讓「用了 fallback」永遠可稽核。
TRAIN_TIME_STRIDE_FALLBACK = 2       # assembly.py:194  `if time_strides else 2`
TRAIN_RE_NORM_SCALE_FALLBACK = 1e4   # config.py:306    schema default


def resolve_training_axes(data_kwargs: dict) -> tuple[int, float, dict]:
    """回 (sensor_stride, re_norm_scale, provenance)；provenance 記每個值的來源。"""
    strides = data_kwargs.get("time_strides") or []
    if strides:
        stride, stride_src = int(strides[0]), "config.time_strides[0]"
    else:
        stride, stride_src = TRAIN_TIME_STRIDE_FALLBACK, "training fallback (assembly.py:194)"
    if "re_norm_scale" in data_kwargs:
        scale, scale_src = float(data_kwargs["re_norm_scale"]), "config.re_norm_scale"
    else:
        scale, scale_src = TRAIN_RE_NORM_SCALE_FALLBACK, "schema default (config.py:306)"
    return stride, scale, {"sensor_stride_source": stride_src, "re_norm_scale_source": scale_src}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="latest", help="'latest' 或 step 整數")
    p.add_argument("--arch", default="liquid")
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _reach(r_max: float) -> int:
    """最大的 n 使 |J_n(r_max)| >= HARMONIC_FLOOR，即這個 R 能表達到第幾階諧波。"""
    from scipy.special import jv

    n = np.arange(MAX_HARMONIC + 1)
    ok = np.abs(jv(n, r_max)) >= HARMONIC_FLOOR
    return int(n[ok].max()) if ok.any() else 0


def _stats(B: np.ndarray) -> dict:
    """B: [4, F]。回 x/y 兩個方向的振幅分位數與可達諧波階。"""
    rx = np.linalg.norm(B[0:2], axis=0)      # sin(cx), cos(cx)
    ry = np.linalg.norm(B[2:4], axis=0)      # sin(cy), cos(cy)
    out = {}
    for lab, r in (("x", rx), ("y", ry)):
        out[lab] = {
            "median": float(np.median(r)),
            "p90": float(np.percentile(r, 90)),
            "max": float(r.max()),
            "harmonic_reach_at_max": _reach(float(r.max())),
        }
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_kwargs, data_kwargs, model_kwargs = (
        cfg["train_kwargs"], cfg["data_kwargs"], cfg["model_kwargs"])

    if model_kwargs.get("use_trainable_fourier", False):
        raise SystemExit(
            f"{args.config} 開了 use_trainable_fourier——那條走 TrainableFourierEmb，"
            "頻率矩陣的語意不同（直接是波向量，不是 Jacobi–Anger 的振幅）。"
            "該臂請用 scripts/diag_trainable_fourier.py。")
    if not model_kwargs.get("periodic_domain", True):
        raise SystemExit(
            f"{args.config} 是非週期域——走的是 FourierEmbs（真 RFF），"
            "本診斷的 Jacobi–Anger 推導不適用。")

    sensor_jsons = data_kwargs.get("sensor_jsons") or []
    if not sensor_jsons:
        raise SystemExit(f"{args.config}: data_kwargs.sensor_jsons 為空")
    sensor_stride, re_norm_scale, axes_prov = resolve_training_axes(data_kwargs)
    print(f"[axes] sensor stride = {sensor_stride}  ({axes_prov['sensor_stride_source']})")

    artifacts_dir = Path(args.artifacts_dir or train_kwargs.get("artifacts_dir") or "").resolve()
    if not artifacts_dir.name:
        raise SystemExit("無 artifacts_dir：config 未設且未給 --artifacts-dir")
    ckpt_dir = artifacts_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint 目錄不存在：{ckpt_dir}")

    # sensor 只用來給參數樹模板正確的形狀（唯一影響參數樹的是末維通道數）
    d = load_sensors_from_path(sensor_jsons[0], time_stride=sensor_stride)
    sensor_vals, sensor_pos = d["sensor_vals"], d["sensor_pos"]
    sensor_time = np.asarray(d["sensor_time"])

    model, model_name = build_model(args.arch, model_kwargs, K_sensors=int(sensor_pos.shape[0]))

    # init 對照必須是**訓練 seed 下的實際初始化值**，不能用 ckpt.reference_params_for——
    # 那支用固定 RNG，只是形狀模板，值不具意義（scripts/CLAUDE.md §2 明列此例外）。
    # 這裡逐字對齊訓練端 pipeline/kolmogorov/run.py:567-573 的 init 路徑。
    seed = int(train_kwargs.get("seed", 42))
    T_total = float(data_kwargs.get("T_total", train_kwargs.get("T_total", 5.0)))
    init_xy = jnp.asarray(
        np.random.RandomState(seed).uniform(0, 1, (8, 2)).astype(np.float32))
    init_t = jnp.asarray(
        np.random.RandomState(seed).uniform(0, T_total, (8,)).astype(np.float32))
    init = model.init(
        jax.random.split(jax.random.PRNGKey(seed))[1],
        jnp.asarray(sensor_vals), jnp.asarray(sensor_pos),
        float(np.log(float((data_kwargs.get("re_values") or [1e4])[0])) / np.log(re_norm_scale)),
        jnp.asarray(sensor_time), init_xy, init_t)

    trained, step, provenance = ckpt_mod.restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=init, model=model)
    print(f"[ckpt] {model_name} step={step}  seed={seed}  fingerprint_verified="
          f"{provenance.get('fingerprint_verified')}")

    def grab(tree, where: str) -> np.ndarray:
        node = tree["params"]
        for key in (where, "spatial_emb", "kernel"):
            if key not in node:
                raise SystemExit(
                    f"參數樹缺 {where}/spatial_emb/kernel——此 arch 的空間嵌入不在預期位置，"
                    f"目前有：{sorted(node)}")
            node = node[key]
        return np.asarray(node)

    report = {"config": str(args.config), "ckpt_step": int(step),
              "harmonic_floor": HARMONIC_FLOOR, "axes_provenance": axes_prov, "provenance": provenance, "modules": {}}
    for where in ("query_decoder", "spatial_encoder"):
        B_tr, B_in = grab(trained, where), grab(init, where)
        if B_tr.shape != B_in.shape:
            raise SystemExit(f"{where}: trained {B_tr.shape} vs init {B_in.shape} 形狀不符")
        s_tr, s_in = _stats(B_tr), _stats(B_in)
        report["modules"][where] = {"shape": list(B_tr.shape), "trained": s_tr, "init": s_in}
        print(f"\n=== {where}/spatial_emb/kernel  {B_tr.shape} ===")
        print(f"{'方向':>4s} {'':>8s} {'median':>9s} {'p90':>9s} {'max':>9s} {'可達諧波階':>10s}")
        for ax in ("x", "y"):
            for lab, s in (("init", s_in), ("trained", s_tr)):
                v = s[ax]
                print(f"{ax:>4s} {lab:>8s} {v['median']:9.3f} {v['p90']:9.3f} "
                      f"{v['max']:9.3f} {v['harmonic_reach_at_max']:10d}")
            g = s_tr[ax]["median"] / max(s_in[ax]["median"], 1e-12)
            print(f"{'':>4s} {'成長':>8s} {g:9.2f}x")

    out = Path(args.out) if args.out else artifacts_dir / "diag_spatial_emb_bandwidth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
