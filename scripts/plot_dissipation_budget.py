#!/usr/bin/env python3
"""Enstrophy 與耗散率的時間軌跡（thesis fig:dissipation_budget）。

What:
    左軸 enstrophy Z(t) = <ω²/2> [1/s²]，右軸耗散率 ε = 2νZ [m²/s³]——後者是
    前者的常數縮放，同一條曲線兩種讀法。DNS 實線，PI-CON 為 n-seed 的
    mean ± 1σ，個別 seed 以細線疊上。

Why 不從 series.npz 的 enstrophy_rel_err 反推:
    那是 |Z_pred − Z_dns| / Z_dns，取過絕對值，符號已經丟失——反推會在
    Z_pred 低於 DNS 的時段把曲線鏡射到上方。故直接從場算。

Usage:
    uv run python scripts/plot_dissipation_budget.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, figwidth, save_figure, DNS, PICON, DNS_LS, PICON_LS  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pi_lnn_jax.data import load_dns_from_path  # noqa: E402
from pi_lnn_jax.metric_artifact import compute_vorticity  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)


def _enstrophy(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Z(t) = <ω²/2>，對空間平均。u/v 形狀 [T, Nx, Ny]。"""
    return np.array([0.5 * np.mean(compute_vorticity(u[i], v[i]) ** 2)
                     for i in range(u.shape[0])])


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", default=str(repo / "artifacts" / "kolmogorov"))
    ap.add_argument("--run-stem", default="main5_s",
                    help="artifacts 目錄前綴（K=200 主線為 ksweep_k200_s）")
    ap.add_argument("--eval-subdir", default="final_eval",
                    help="讀哪個評估子目錄（帶 --export-fields 的那次）")
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "results"))
    ap.add_argument("--require-seeds", type=int, default=len(SEEDS),
                    help="少於這個數目的 seed 就硬失敗；thesis caption 宣告 n=5，"
                         "少畫幾條而不聲明會讓 ±1σ 帶的意義悄悄改變")
    args = ap.parse_args()
    root = Path(args.artifacts_root)

    Z_pred, found, missing = [], [], []
    dns_path = None
    for s in SEEDS:
        fp = root / f"{args.run_stem}{s}" / args.eval_subdir / "fields.npz"
        if not fp.is_file():
            missing.append(s)
            continue
        F = np.load(fp)
        Z_pred.append(_enstrophy(F["u_pred"], F["v_pred"]))
        t = F["t"]
        dns_path = dns_path or str(F["dns_path"])
        found.append(s)

    if missing:
        print(f"[warn] 缺 fields.npz 的 seed: {missing}"
              f"（需 evaluate_exp245 --export-fields）")
    if len(found) < args.require_seeds:
        raise SystemExit(
            f"只有 {len(found)}/{args.require_seeds} 個 seed 有場資料 {found}；"
            "要先畫可用的請明確給 --require-seeds")

    dns_u, dns_v, dns_t = load_dns_from_path(dns_path, time_stride=1)
    if dns_u.shape[0] != t.size:
        stride = (dns_u.shape[0] - 1) // (t.size - 1)
        dns_u, dns_v, dns_t = dns_u[::stride], dns_v[::stride], dns_t[::stride]
    if not np.allclose(dns_t, t, rtol=1e-4, atol=1e-6):
        raise ValueError(f"時間軸不符: dns[:3]={dns_t[:3]} fields[:3]={t[:3]}")
    Z_dns = _enstrophy(dns_u, dns_v)

    # ν 由 config 的 Re 決定（Kolmogorov 無因次化下 ν = 1/Re）
    meta = json.loads((root / f"{args.run_stem}{found[0]}" / args.eval_subdir
                       / "metrics.json").read_text())
    nu = 1.0 / float(meta["re_value"])

    setup_style("thesis")
    fig, ax = plt.subplots(figsize=(figwidth("thesis", "single"), 3.2))
    arr = np.stack(Z_pred)
    m, sd = arr.mean(0), arr.std(0)
    ax.plot(t, Z_dns, color=DNS, ls=DNS_LS, label="DNS")
    for c in arr:
        ax.plot(t, c, color=PICON, lw=0.4, alpha=0.35)
    ax.fill_between(t, m - sd, m + sd, color=PICON, alpha=0.18, lw=0)
    ax.plot(t, m, color=PICON, ls=PICON_LS, label=f"PI-CON ($n={len(found)}$)")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"enstrophy $\mathcal{Z}(t)$ (1/s$^2$)")
    ax.legend(loc="upper right")

    # 右軸：ε = 2νZ，純常數縮放，故與左軸共用資料範圍
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(2 * nu * lo, 2 * nu * hi)
    ax2.set_ylabel(r"dissipation rate $\varepsilon = 2\nu\mathcal{Z}$ (m$^2$/s$^3$)")
    ax2.grid(False)

    fig.tight_layout()
    out = Path(args.out)
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'dissipation_budget'))]}")
    print(f"[data] seeds={found}  nu={nu:.3g}  Z_dns(t=5)={Z_dns[-1]:.4g}  "
          f"Z_pred(t=5)={m[-1]:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
