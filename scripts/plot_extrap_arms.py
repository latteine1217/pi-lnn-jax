#!/usr/bin/env python3
"""跨臂合成圖：外推 lead-time 曲線 + baseline 的 seed 雜訊帶。

What:
    吃多份 `extrapolation_leadtime.json`（`plot_extrapolation_leadtime.py` 的產物，
    內含 `curves`），把 baseline 的多個 seed 畫成**帶**、其餘各臂畫成線，
    並輸出跨臂統計表（水平線、τ=5、勝過 persistence 的 τ 佔比）。

Why:
    單臂圖看不出「差異是否超出 seed 雜訊」。baseline 的 3-seed 全距只有一格
    （水平線 1.05–1.10），把它畫成帶，其他臂在不在帶外一眼可判。

    另外報 `win_fraction`（τ>0 中模型優於 persistence 的比例）：主判準
    `useful_horizon` 定義成「從 τ=0⁺ **連續**贏」，預設了誤差單調變壞——
    自迴歸臂前段輸、後段全贏，那個定義會判它 0.45 而完全看不到後段最好。
    兩個量一起報才不會漏掉這種形狀。

Usage:
    uv run python scripts/plot_extrap_arms.py \\
      --baseline lt_530.json lt_530s1.json lt_530s2.json \\
      --arm "EXP-532 dt>0 supervision"=lt_532.json \\
      --arm "EXP-533 autoregressive"=lt_533.json \\
      --out-dir artifacts/kolmogorov/_compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo 內 scripts/journal_style.py
from journal_style import figwidth, save_figure, setup_style  # noqa: E402

UNCORRELATED_SQRT2 = float(np.sqrt(2.0))


def load_curves(path: str) -> dict:
    """讀一份 lead-time JSON 的曲線；缺 `curves` 就 fail-fast（舊版產物）。"""
    d = json.loads(Path(path).read_text())
    if "curves" not in d:
        raise KeyError(
            f"{path} 沒有 `curves` —— 那是舊版 plot_extrapolation_leadtime.py 的產物，"
            "請以現版重跑（slurm/replot_extrap.sbatch）")
    c = d["curves"]
    if c.get("persistence_model") is None:
        raise KeyError(f"{path} 缺 persistence_model —— eval 需帶 --export-fields 重跑")
    return {
        "tau": np.asarray(c["tau"], dtype=float),
        "model": np.asarray(c["model"], dtype=float),
        "persistence": np.asarray(c["persistence_model"], dtype=float),
        "horizon": d.get("useful_horizon_vs_persistence_model"),
        "at_lead": d.get("at_lead_times", {}),
    }


def win_fraction(tau, model, persistence) -> float:
    """τ>0 中模型優於 persistence 的比例（不要求連續）。"""
    ext = tau > 0
    return float((model[ext] < persistence[ext]).mean())


def _common_grid(curves: list[dict]) -> np.ndarray:
    """各臂的 τ 格點必須一致；不一致就 raise（不內插湊合）。"""
    base = curves[0]["tau"]
    for c in curves[1:]:
        if c["tau"].shape != base.shape or not np.allclose(c["tau"], base, atol=1e-6):
            raise ValueError("各臂的 τ 格點不一致——拒絕內插湊合，請以同一個協定重跑 eval")
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="baseline 各 seed 的 lead-time JSON（≥2 份才有帶）")
    ap.add_argument("--baseline-label", default="EXP-530 baseline")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                    help="對照臂，可重複；格式 '標籤=路徑'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="extrap_arms")
    args = ap.parse_args()

    base = [load_curves(p) for p in args.baseline]
    if len(base) < 2:
        raise ValueError("baseline 至少要兩個 seed，否則畫不出雜訊帶")
    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise ValueError(f"--arm 需為 '標籤=路徑'，收到 {spec!r}")
        label, path = spec.split("=", 1)
        arms.append((label, load_curves(path)))

    tau = _common_grid(base + [c for _, c in arms])
    stack = np.stack([c["model"] for c in base])
    lo, hi, mean = stack.min(0), stack.max(0), stack.mean(0)
    pers = base[0]["persistence"]

    rows = [{
        "arm": args.baseline_label,
        "n_seeds": len(base),
        "horizon": [c["horizon"] for c in base],
        "tau5": [float(np.interp(5.0, tau, c["model"])) for c in base],
        "win_fraction": [win_fraction(tau, c["model"], c["persistence"]) for c in base],
    }]
    for label, c in arms:
        rows.append({
            "arm": label, "n_seeds": 1, "horizon": [c["horizon"]],
            "tau5": [float(np.interp(5.0, tau, c["model"]))],
            "win_fraction": [win_fraction(tau, c["model"], c["persistence"])],
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.stem}.json").write_text(json.dumps({
        "rows": rows,
        "baseline_band": {"tau": tau.tolist(), "lo": lo.tolist(), "hi": hi.tolist()},
        "reading": (
            "baseline 帶 = 各 seed 的逐點 min–max。落在帶內的臂與 baseline 無法區分。"
            "win_fraction 與 horizon 一起看：horizon 要求從 τ=0⁺ 連續贏過 persistence，"
            "對『前段輸、後段全贏』的形狀會嚴重低估。"),
    }, indent=2))

    setup_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(figwidth("iclr", "single"), 2.9))
    ext = tau >= 0
    ax.fill_between(tau[ext], lo[ext], hi[ext], color="#2a78d6", alpha=0.22, lw=0,
                    label=f"{args.baseline_label} ({len(base)} seeds)")
    ax.plot(tau[ext], mean[ext], color="#2a78d6", lw=1.3)
    for (label, c), color in zip(arms, ("#eb6834", "#1baf7a", "#eda100")):
        ax.plot(tau[ext], c["model"][ext], color=color, lw=1.4, label=label)
    ax.plot(tau[ext], pers[ext], ls="-.", lw=1.1, color="0.45", label="Persistence (model)")
    ax.axhline(1.0, lw=0.9, color="0.6", zorder=1)
    ax.text(0.12, 1.012, "zero prediction", ha="left", va="bottom", fontsize=6.5, color="0.4")
    ax.set_xlabel(r"Lead time $\tau = t - t_{\mathrm{data}}$")
    ax.set_ylabel("Relative $L_2$ error of $(u,v)$")
    ax.set_xlim(0.0, float(tau.max()))
    ax.set_ylim(0.0, max(1.45, float(hi.max()) * 1.05))
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save_figure(fig, str(out_dir / args.stem))

    print(f"{'arm':<28} {'horizon':>16} {'tau=5':>16} {'win frac':>12}")
    for r in rows:
        def fmt(v):
            return f"{v[0]:.3f}" if len(v) == 1 else f"{np.mean(v):.3f} [{min(v):.2f},{max(v):.2f}]"
        print(f"{r['arm']:<28} {fmt(r['horizon']):>16} {fmt(r['tau5']):>16} "
              f"{fmt(r['win_fraction']):>12}")
    print(f"[out] {out_dir / (args.stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
