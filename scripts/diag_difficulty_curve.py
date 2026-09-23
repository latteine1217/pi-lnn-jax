#!/usr/bin/env python3
"""難度曲線：從 K 個點重建這個流場，隨時間變得多容易？

What:
    在每個 DNS 時刻，用**該時刻的**真值在 sensor 座標上取樣，交給 training-free 的
    散點內插重建全場，量逐時相對誤差。同時算三個流場複雜度指標（enstrophy、
    能譜 participation ratio、sensor Nyquist 之上的能量佔比）。

Why:
    時間外推實驗的 τ 與流場難度完全糾纏——τ 越大時刻越晚，而本軌跡是衰減暫態，
    越晚越低維。這支給出一條**與模型無關**的難度尺：任何「模型在晚期比較好」的
    觀察都能先除掉難度成分再解讀。

    刻意用 training-free 的 interp（不是 gappy-POD）：後者要先建 POD basis，
    basis 取自哪段時間會把訓練資料的分佈偷渡進「難度」裡。

限制（判讀時必須說出來）：
    這量的是**同時刻重建**的難度，不是**往前預測**的難度。它是一階代理，不是
    外推難度本身——外推還要面對混沌放大，那由 `diag_chaos_budget.py` 量。
    它也是 oracle：t>資料末端的 sensor 值取自真值，模型在那段沒有這些觀測。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo 內 scripts/journal_style.py
from journal_style import figwidth, save_figure, setup_style

from pi_lnn_jax.baselines import InterpBaseline
from pi_lnn_jax.data import load_dns_from_path, load_sensors_from_path
from pi_lnn_jax.metric_artifact import compute_energy_spectrum, sparsity_yardsticks


def flow_complexity(u: np.ndarray, v: np.ndarray, k_sensor: float) -> dict:
    """單一快照的三個複雜度指標。回 dict（enstrophy / 有效模態數 / 高頻能量佔比）。"""
    N = u.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    omega = np.fft.ifft2(1j * KX * vh - 1j * KY * uh).real

    # 能譜走**權威實作**（`metric_artifact.compute_energy_spectrum`），不自己抄一份：
    # 它蓋到 Fourier 方盒角落 |k|max=(N/2)√2 **且從 k=0 起算**，Parseval 精確。
    # 本檔早先的抄本從 k=1 起算 → 丟掉 DC，對零均值場無害、對有均流的場會少算分母。
    bins, spec = compute_energy_spectrum(u, v)
    total = spec.sum()
    if total <= 0:
        raise ValueError("能譜總和為零，快照可能全為零")
    p = spec / total
    return {
        "enstrophy": float(0.5 * np.mean(omega ** 2)),
        # shell-level participation ratio：三個落在同一 |k| shell 的相異模態會讀 1，
        # 不是「模態數」。名稱誠實化為 effective_shells。
        "effective_shells": float(1.0 / np.sum(p ** 2)),
        "energy_above_k_sensor": float(spec[bins > k_sensor].sum() / total),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", required=True, help="真值 DNS .npy（涵蓋整個要量的時窗）")
    ap.add_argument("--sensor-json", required=True, help="取 sensor 座標用（值一律取自 DNS）")
    ap.add_argument("--time-stride", type=int, default=2, help="DNS 幀取樣間隔")
    ap.add_argument("--method", default="linear", choices=("linear", "cubic", "nearest"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dns_u, dns_v, dns_t = load_dns_from_path(args.dns, time_stride=args.time_stride)
    sensors = load_sensors_from_path(args.sensor_json, time_stride=1)
    pos = np.asarray(sensors["sensor_pos"], dtype=np.float64)     # [K,2] ∈ [0,1]²
    K, N = pos.shape[0], int(dns_u.shape[1])
    # sensor Nyquist：取自權威實作，不在此重寫（本專案慣例是 √(K/π)，2D 模態計數
    # πk²=K；均勻格點的 √K/2 對 K=100 是 5.00 vs 5.64，混用會讓不同表不可比）。
    k_sensor = sparsity_yardsticks(K)["nyquist_kmax"]
    print(f"[setup] K={K}  N={N}  frames={len(dns_t)}  "
          f"t ∈ [{dns_t[0]:.2f}, {dns_t[-1]:.2f}]  k_sensor≈{k_sensor:.2f}")

    # sensor 座標 → 格點索引。**週期域必須 wrap 不是 clip**：x≈1⁻ 的點屬於 index 0，
    # clip 會把它壓到 N−1（本批座標剛好都落在格點上、兩者一致，但換一組就會咬人）。
    idx = np.rint(pos * N).astype(int) % N
    interp = InterpBaseline(method=args.method, periodic=True)

    rows = []
    for i, t in enumerate(dns_t):
        u_t, v_t = np.asarray(dns_u[i], dtype=np.float64), np.asarray(dns_v[i], dtype=np.float64)
        # oracle 取樣：**該時刻**的真值在 sensor 位置上的值
        sv = np.stack([u_t[idx[:, 0], idx[:, 1]], v_t[idx[:, 0], idx[:, 1]]], axis=-1)
        u_p, v_p = interp.reconstruct(sv[None], pos, (N, N))
        num = np.sqrt(np.sum((u_p[0] - u_t) ** 2) + np.sum((v_p[0] - v_t) ** 2))
        den = np.sqrt(np.sum(u_t ** 2) + np.sum(v_t ** 2))
        rows.append({"t": float(t), "interp_uv_rel_err": float(num / den),
                     **flow_complexity(u_t, v_t, k_sensor)})

    t = np.array([r["t"] for r in rows])
    err = np.array([r["interp_uv_rel_err"] for r in rows])
    corr = {k: float(np.corrcoef([r[k] for r in rows], err)[0, 1])
            for k in ("enstrophy", "effective_shells", "energy_above_k_sensor")}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "difficulty_curve.json").write_text(json.dumps({
        "dns": str(args.dns), "sensor_json": str(args.sensor_json),
        "K": K, "grid": N, "method": args.method, "k_sensor": float(k_sensor),
        "rows": rows, "corr_with_interp_err": corr,
        "reading": (
            "interp_uv_rel_err 是 training-free 的同時刻重建誤差，當作與模型無關的"
            "難度尺。它量的是重建難度、不是預測難度（後者見 diag_chaos_budget.py），"
            "且 t>資料末端的 sensor 值取自真值，屬 oracle。"),
    }, indent=2))

    setup_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(figwidth("iclr", "single"), 2.6))
    ax.plot(t, err, lw=1.4, color="#2a78d6", label="Interp reconstruction error")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Relative $L_2$ error of $(u,v)$", color="#2a78d6")
    ax.tick_params(axis="y", labelcolor="#2a78d6")
    ax2 = ax.twinx()
    ax2.plot(t, [r["effective_shells"] for r in rows], lw=1.2, ls="--", color="#eb6834")
    ax2.set_ylabel("Effective modes (participation ratio)", color="#eb6834")
    ax2.tick_params(axis="y", labelcolor="#eb6834")
    save_figure(fig, str(out_dir / "difficulty_curve"))

    print(f"\n{'t':>6} {'interp err':>11} {'enstrophy':>10} {'eff.shells':>10} {'E(k>ks)':>9}")
    for r in rows[::max(1, len(rows) // 12)]:
        print(f"{r['t']:>6.2f} {r['interp_uv_rel_err']:>11.4f} {r['enstrophy']:>10.2f} "
              f"{r['effective_shells']:>10.2f} {r['energy_above_k_sensor']:>9.4f}")
    print("\ncorr(interp_err, ·):", {k: round(v, 3) for k, v in corr.items()})
    print(f"[out] {out_dir / 'difficulty_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
