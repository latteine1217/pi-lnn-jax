"""模型的 PDE residual 頻譜 vs 真值地板（階段 2）。

Why 不直接讀階段 1 的地板 JSON：地板只在「與模型端**完全相同的離散化**」下才可比。
存檔間隔 dt=0.025 讓時間導數的截斷誤差成為地板的主要來源（階段 1 實測：
stride×2 使渦量殘差 +79%，4 階時間差分只降 15% ⇒ 誤差不在漸近區）。這個誤差對
truth 與 pred 是**共模**的——只要兩邊用同一組 frame、同一階時間差分、同一套空間
算子，它就會在相減時抵消。若圖省事用 autodiff 取模型的精確 u_t，模型端沒有這個
誤差、地板端有，兩者不可比。所以本腳本在同一次執行內對 truth 與 pred **各算一次**，
共用 `diag_residual_spectrum` 的算子——一致性由程式碼結構保證，不靠人記得。

診斷的是渦量傳輸式（消去壓力）與連續方程，不是訓練 loss 的動量式：
主線 DNS 的 `p` 與 (u,v) 不自洽（TD-29），含壓力的動量殘差沒有真值地板。
渦量式是動量式的 curl，兩者不等價——任何結論都要帶這個限定。

輸出的是**聯合平面**而非單一 R(k)：R(k) 單獨看分不出「物理沒學到」與「算符本來
就放大高頻」。R(k) 與 γ(k) 的組合才有判別力——
  R 高 ∧ γ 低 → 殘差只是誤差的代理
  R ≈ 地板 ∧ γ 低 → 模型滿足 PDE 卻仍然錯 = 該尺度上 PDE 約束有 null space
  R 高 ∧ γ 高 → 最佳化/權重取捨，不是物理沒學到
外加 Parseval 分解：PDE loss 的頻帶佔比 vs 能量佔比，回答「physics 約束的梯度
實際花在哪個尺度」。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_residual_spectrum import (  # noqa: E402
    BANDS, Ops, _shell_index, _shell_power, _assert_shell_matches_repo,
    band_summary, residual_terms, spectra_of,
)
from pi_lnn_jax.ckpt import reference_params_for, restore_eval_params  # noqa: E402
from pi_lnn_jax.config import DATA_SCHEMA, load_config  # noqa: E402
from pi_lnn_jax.evaluate import evaluate_time_series  # noqa: E402
from pi_lnn_jax.evaluation_protocol import (  # noqa: E402
    ProtocolMode, load_for_evaluation, resolve_protocol,
    training_time_strides_from_config,
)
from pi_lnn_jax.data import _resolve_data_path, load_sensors_from_path  # noqa: E402
from pi_lnn_jax.metric_artifact import spectral_coherence  # noqa: E402
from pi_lnn_jax.model_factory import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="訓練用的 TOML")
    p.add_argument("--ckpt", default="latest", help="'latest' 或 step 數")
    # arch 刻意 required：它是 sbatch env（預設 liquid）、不進 config 也不進
    # summary.json——config 相同不代表同一個模型。逼呼叫端把假設寫出來。
    p.add_argument("--arch", required=True, choices=["liquid", "vanilla", "pinn"])
    p.add_argument("--protocol", required=True, choices=[m.value for m in ProtocolMode])
    p.add_argument("--protocol-reason", default=None, help="fixed_grid 必填")
    p.add_argument("--time-stride", type=int, default=None)
    p.add_argument("--artifacts_dir", default=None,
                   help="訓練若覆蓋過就必須帶同一個，否則讀到別的 run 的 ckpt")
    p.add_argument("--max-frames", type=int, default=0,
                   help="只取前 N 個**連續** frame（0=全部）。時間導數需要連續 frame，"
                        "故此處是截斷而非子採樣。")
    p.add_argument("--time-order", type=int, default=2, choices=[2, 4])
    p.add_argument("--allow-forcing-mismatch", action="store_true",
                   help="模型 forcing 與 DNS forcing 不符時仍繼續（預設 fail-fast）")
    p.add_argument("--out", required=True, help="輸出 JSON 路徑")
    return p.parse_args()


def _gate_forcing(model, params, dns_cfg: dict, allow: bool) -> tuple[float, float]:
    """模型自己的 forcing 必須與 DNS 的一致，否則 pred 與 truth 滿足的是不同方程。

    訓練端的 residual 吃的是 `get_forcing` 的輸出（見 pipeline/kolmogorov/assembly.py），
    不是 config 的 `kolmogorov_A`/`kolmogorov_k_f`——後者是 DEPRECATED inert 鍵。
    forcing 可訓練時（learn_forcing_A/k_f）兩者會分岔，那時比較不再公平。
    """
    from pi_lnn_jax.models import LiquidOperator
    if not isinstance(model, LiquidOperator):
        raise NotImplementedError(
            f"forcing 取法只對 liquid 實作；{type(model).__name__} 走的是 "
            f"make_ns_residual_fn_baseline，其 forcing 來源未經本腳本驗證。"
            f"要支援先確認 baseline 路徑的 A/k_f 從哪裡來，不要猜。")
    A_m, kf_m = model.apply(params, method=LiquidOperator.get_forcing)
    A_m, kf_m = float(A_m), float(kf_m)
    A_d, kf_d = float(dns_cfg["A"]), float(dns_cfg["k_f"])
    ok = np.isclose(A_m, A_d, rtol=1e-6) and np.isclose(kf_m, kf_d, rtol=1e-6)
    msg = (f"model forcing (A={A_m:.6g}, k_f={kf_m:.6g}) vs "
           f"DNS forcing (A={A_d:.6g}, k_f={kf_d:.6g})")
    if not ok and not allow:
        raise AssertionError(
            f"forcing 不一致：{msg}\n"
            f"  pred 與 truth 會滿足不同的方程，殘差不可比。\n"
            f"  確認 model_kwargs.forcing_A_init/forcing_k_f_init 與 learn_forcing_*；"
            f"  確知要比就加 --allow-forcing-mismatch。")
    print(f"[gate] forcing {'一致' if ok else '不一致（已放行）'}：{msg}")
    return A_m, kf_m


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    train_kwargs, data_kwargs, model_kwargs = (
        cfg["train_kwargs"], cfg["data_kwargs"], cfg["model_kwargs"])

    artifacts_dir = Path(args.artifacts_dir if args.artifacts_dir is not None
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
    # re_norm_scale 缺鍵 ≠ 缺資訊：DATA_SCHEMA 有 default，訓練端也是經同一個
    # schema 拿到它（assembly.re_norm_scale_of 讀 typed config 的同一欄）。
    # 讀 schema 的 default 而非硬編 10000.0——單-Re config 普遍不寫這個鍵。
    if "re_norm_scale" in data_kwargs:
        re_norm_scale, rns_src = float(data_kwargs["re_norm_scale"]), "config"
    else:
        re_norm_scale, rns_src = float(DATA_SCHEMA["re_norm_scale"][1]), "DATA_SCHEMA default"
    re_norm = float(np.log(re_value) / np.log(re_norm_scale))

    print("=" * 78)
    print(f"diag_residual_spectrum_model — config={Path(args.config).name}")
    print(f"  arch={args.arch}  ckpt={args.ckpt}  ckpt_dir={ckpt_dir}")
    print("=" * 78)

    protocol = resolve_protocol(
        mode=args.protocol,
        training_time_strides=training_time_strides_from_config(args.config),
        cli_time_stride=args.time_stride, reason=args.protocol_reason)
    print(f"[protocol] {protocol.mode.value}  sensor_stride={protocol.sensor_time_stride} "
          f"dns_stride={protocol.dns_time_stride}  {protocol.basis}")

    probe = load_sensors_from_path(sensor_json[0], time_stride=protocol.sensor_time_stride)
    aligned = load_for_evaluation(sensor_json[0], Path(dns_paths[0]), protocol=protocol,
                                  viscosity=nu, with_pressure=False)
    sensor_vals, sensor_pos = aligned.sensor_vals_normalized, aligned.sensor_pos
    sensor_time, norm_stats = aligned.sensor_time, aligned.norm_stats
    dns_u, dns_v, dns_t = aligned.dns_u_eval, aligned.dns_v_eval, aligned.dns_t_eval
    K = sensor_vals.shape[1]

    if args.max_frames:
        n = int(args.max_frames)
        if n > len(dns_t):
            raise ValueError(f"--max-frames {n} > 可用 frame 數 {len(dns_t)}")
        print(f"[trunc] 明示截斷到前 {n} 個連續 frame（全部 {len(dns_t)}）")
        dns_u, dns_v, dns_t = dns_u[:n], dns_v[:n], dns_t[:n]
    if len(dns_t) < 6:
        raise ValueError(f"frame 數 {len(dns_t)} < 6，4 階時間差分無意義")
    dt = np.diff(np.asarray(dns_t, dtype=np.float64))
    spread = float((dt.max() - dt.min()) / dt.mean())
    if spread > 1e-4:
        raise ValueError(
            f"DNS 時間軸非均勻（相對離散 {spread:.3e} > 1e-4），時間導數不可用")
    print(f"[data] Re={re_value:g}  K={K}  frames={len(dns_t)}  dt={dt[0]:.6g}  "
          f"t∈[{float(dns_t[0]):.3f}, {float(dns_t[-1]):.3f}]  "
          f"dt 相對離散={spread:.2e}（float32 捨入）")
    print(f"[data] re_norm={re_norm:.4f}  (re_norm_scale={re_norm_scale:g} ← {rns_src})")

    model, model_name = build_model(args.arch, model_kwargs, K_sensors=K)
    params = reference_params_for(model, sensor_vals, sensor_pos, sensor_time)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[model] {model_name}  params={n_params:,}")
    params, restored_step, ckpt_prov = restore_eval_params(
        ckpt_dir, args.ckpt, reference_params=params, model=model)
    print(f"[ckpt] step={restored_step}  fingerprint_verified="
          f"{ckpt_prov.get('fingerprint_verified')}")

    dns_cfg = np.load(_resolve_data_path(dns_paths[0]),
                      allow_pickle=True).item()["config"]
    A, k_f = _gate_forcing(model, params, dns_cfg, args.allow_forcing_mismatch)

    L = float(dns_cfg.get("L", 1.0))
    if not np.isclose(L, 1.0):
        raise NotImplementedError(
            f"evaluate.reconstruct_field 的 query grid 硬寫 [0,1)，L={L} 不支援")

    print("\n[eval] 重建全部 frame（走既有 evaluate_time_series，非自寫迴圈）…")
    res = evaluate_time_series(model, params, sensor_vals, sensor_pos, re_norm,
                               sensor_time, norm_stats, dns_u, dns_v, dns_t,
                               verbose=False, nu=nu, collect_fields=True)
    u_pred = np.asarray(res["u_fields"], dtype=np.float64)
    v_pred = np.asarray(res["v_fields"], dtype=np.float64)
    mm = res["metrics_mean"]
    print(f"[sanity] uv_rel_err={mm.get('uv_rel_err', float('nan')):.4f}  "
          f"omega_rel_err={mm.get('omega_rel_err', float('nan')):.4f}  "
          f"（對回該 run 已知的 headline，數字不符先查 ckpt/protocol，不要往下判讀）")

    u_true = np.asarray(dns_u, dtype=np.float64)
    v_true = np.asarray(dns_v, dtype=np.float64)
    t = np.asarray(dns_t, dtype=np.float64)
    N = u_true.shape[-1]
    y = np.arange(N) * (L / N)
    idx, n_bins = _shell_index(N)
    _assert_shell_matches_repo(u_true[0], v_true[0], idx, n_bins)
    k_arr = np.arange(n_bins)
    ops = Ops("spectral", L, N)
    zero = np.zeros_like(u_true)

    # truth 與 pred 走完全相同的算子 / frame / 時間階數 → 離散化誤差共模
    def spectra(uu, vv):
        terms = residual_terms(uu, vv, zero, t, y, nu, A, k_f, ops,
                               time_order=args.time_order, with_mom=False)
        return {eq: spectra_of(tm, idx, n_bins) for eq, tm in terms.items()}

    print("[calc] 殘差譜：truth（地板）…")
    floor = spectra(u_true, v_true)
    print("[calc] 殘差譜：pred…")
    pred = spectra(u_pred, v_pred)

    E_true = np.mean([0.5 * (_shell_power(u_true[i], idx, n_bins)
                             + _shell_power(v_true[i], idx, n_bins))
                      for i in range(len(t))], axis=0)
    # spectral_coherence 回傳 (k_bins, gamma)——取 [1]，別把 k 軸當成 γ
    gam = np.nanmean([spectral_coherence(u_pred[i], v_pred[i],
                                         u_true[i], v_true[i])[1]
                      for i in range(len(t))], axis=0)
    gam = np.asarray(gam).ravel()

    # band 聚合刻意**不用逐 shell 算術平均**：k 的殼層能量跨 20 個數量級，
    # 算術平均讓 E~1e-20 的空 shell 與主導 shell 等權。實測兩種聚合對真值
    # low band 的 cancellation 差 84 倍（0.168 vs 0.0020），且會**翻轉**
    # 「哪個 band 相對地板最差」的排序。故：
    #   γ      → 能量加權（與既有 gamma_k 的報法一致，否則兩邊數字無法並排）
    #   cancel → band 內總和比 Σ R(k) / Σ terms(k)（無量綱量的自然推廣）
    def _cancel(src, sel):
        v = src["vort"]
        R = np.asarray(v["residual"])[sel].sum()
        T = sum(np.asarray(x) for key, x in v.items() if key.startswith("term::"))
        return float(R / T[sel].sum())

    print(f"\n--- 聯合平面（渦量傳輸式，time_order={args.time_order}）---")
    print(f"  {'band':<6}{'E frac':>9}{'γ(k)':>8}{'R_pred':>11}{'R_floor':>11}"
          f"{'p/f':>8}{'cancel_p':>11}{'cancel_f':>11}{'cancel比':>10}{'headroom':>10}")
    joint = {}
    for b, (lo, hi) in BANDS.items():
        sel = (k_arr > lo) & (k_arr <= hi) if lo > 0 else (k_arr <= hi)
        rp = float(np.asarray(pred["vort"]["residual"])[sel].sum())
        rf = float(np.asarray(floor["vort"]["residual"])[sel].sum())
        cp, cf = _cancel(pred, sel), _cancel(floor, sel)
        Ew = E_true[sel]
        g = float(np.nansum(gam[sel] * Ew) / Ew.sum()) if Ew.sum() > 0 else float("nan")
        ef = band_summary(k_arr, E_true.tolist())[b]["frac"]
        # headroom 關閉率：抵消空間是 1−cancel（cancel 上限 1 = 完全不抵消），
        # 問的是「地板留下的抵消空間，模型丟掉了多少比例」。
        #
        # 兩個指標的偏誤方向**相反**，故並列而不擇一：
        #   cancel 比值 cp/cf —— 地板逼近 1 時分母飽和，對已經很爛的 band 低估
        #   headroom 關閉率  —— 地板逼近 1 時 hr_f→0，對同一個 band 高估
        # 排序不一致是預期中的，判讀時不得只挑一個講。
        hr_f, hr_p = 1.0 - cf, 1.0 - cp
        hr = (hr_f - hr_p) / hr_f if hr_f > 0 else float("nan")
        joint[b] = {"E_frac": ef, "gamma_energy_weighted": g, "R_pred": rp, "R_floor": rf,
                    "ratio": rp / rf if rf > 0 else float("nan"),
                    "cancel_pred": cp, "cancel_floor": cf,
                    "cancel_ratio": cp / cf if cf > 0 else float("nan"),
                    "headroom_closure": hr}
        print(f"  {b:<6}{ef:>8.2%}{g:>8.3f}{rp:>11.3e}{rf:>11.3e}"
              f"{joint[b]['ratio']:>8.2f}{cp:>11.3e}{cf:>11.3e}"
              f"{joint[b]['cancel_ratio']:>10.2f}{hr:>9.1%}")

    print("\n--- Parseval 分解：殘差的頻帶佔比 vs 能量佔比 ---")
    print(f"  {'eq':<6}{'source':<8}{'low':>9}{'mid':>9}{'high':>9}")
    for eq in ("vort", "cont"):
        for nm, src in (("pred", pred), ("floor", floor)):
            bs = band_summary(k_arr, src[eq]["residual"])
            print(f"  {eq:<6}{nm:<8}" + "".join(f"{bs[b]['frac']:>8.2%}" for b in BANDS))
    print(f"  {'energy':<14}" + "".join(
        f"{band_summary(k_arr, E_true.tolist())[b]['frac']:>8.2%}" for b in BANDS))

    out = {
        "provenance": {
            "config": str(Path(args.config).resolve()), "arch": args.arch,
            "model_name": model_name, "n_params": int(n_params),
            "ckpt_dir": str(ckpt_dir), "ckpt_step": int(restored_step),
            "ckpt_provenance": ckpt_prov,
            "dns_path": str(dns_paths[0]), "re": re_value, "nu": nu,
            "re_norm": re_norm, "re_norm_scale": re_norm_scale,
            "re_norm_scale_source": rns_src,
            "forcing_A": A, "forcing_k_f": k_f, "domain_length": L,
            "protocol": protocol.mode.value,
            "sensor_time_stride": protocol.sensor_time_stride,
            "dns_time_stride": protocol.dns_time_stride,
            "frames": int(len(t)), "dt": float(np.mean(dt)),
            "dt_relative_spread": spread, "N": int(N),
            "time_order": args.time_order, "K": int(K),
            "bands": {k: list(v) for k, v in BANDS.items()},
            "equations": "vort(渦量傳輸，消去壓力) + cont；非訓練 loss 的動量式（見 TD-29）",
        },
        "sanity_metrics_mean": {k: float(v) for k, v in mm.items()
                                if isinstance(v, (int, float))},
        "k": k_arr.tolist(), "E_k_true": E_true.tolist(), "gamma_k": gam.tolist(),
        "pred": pred, "floor": floor, "joint_bands": joint,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
