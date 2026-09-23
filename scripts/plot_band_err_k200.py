"""從 metric_artifact.json 直接畫 band 誤差圖（k_eta 頻帶定義），不需重跑 eval。

series.npz 是舊 eval 路徑的產物，K=200 那批走的是 metric_artifact 管線，沒有它。
但逐時的 band 相對誤差就在 measurements 裡（`spectrum.energy.rel_l2.{low,mid,high}.v1`），
定義是 k_eta 的分數：low (0, 0.1k_eta]、mid (0.1k_eta, 0.4k_eta]、high (0.4k_eta, 1.0k_eta]。

樣式沿用 plot_thesis_series_figures.fig_band_energy，以免與同批圖不一致。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, figwidth, save_figure, MUTED, STYLE_CYCLE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)
BANDS = [("spectrum.energy.rel_l2.low.v1", r"low band ($k \leq 0.1 k_\eta$)"),
         ("spectrum.energy.rel_l2.mid.v1", r"mid band ($0.1 k_\eta < k \leq 0.4 k_\eta$)"),
         ("spectrum.energy.rel_l2.high.v1", r"high band ($0.4 k_\eta < k \leq k_\eta$)")]


def _series(measurements, did):
    e = sorted((x["time"], x["value"]) for x in measurements
               if x.get("definition_id") == did)
    if not e:
        return None, None
    return (np.array([a for a, _ in e], dtype=float),
            np.array([b for _, b in e], dtype=float))


def _envelope(ax, t, curves, colour, ls, label):
    a = np.asarray(curves, dtype=float)
    m, s = a.mean(0), a.std(0, ddof=1)
    ax.fill_between(t, m - s, m + s, color=colour, alpha=0.18, lw=0)
    for row in a:
        ax.plot(t, row, color=colour, ls=":", lw=0.5, alpha=0.55)
    ax.plot(t, m, color=colour, ls=ls, lw=1.4, label=label)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--run-stem", default="ksweep_k200_s",
                   help="artifacts 目錄前綴，seed 接在後面")
    p.add_argument("--artifacts-root", default=str(repo / "artifacts" / "kolmogorov"))
    p.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                        / "figures" / "results"))
    p.add_argument("--drop-t0", action="store_true", default=True,
                   help="排除 t=0（高頻能量近零使相對誤差奇異）")
    p.add_argument("--venue", default="thesis",
                   help="journal_style venue：決定字體、圖寬與框線樣式（thesis / tmlr）")
    args = p.parse_args()

    root = Path(args.artifacts_root)
    per_seed = []
    for s in SEEDS:
        cand = sorted((root / f"{args.run_stem}{s}").glob("*/metric_artifact.json"))
        if not cand:
            raise FileNotFoundError(f"缺 seed {s}: {root / (args.run_stem + str(s))}")
        per_seed.append(json.load(open(cand[-1]))["measurements"])

    setup_style(args.venue)
    fig, ax = plt.subplots(figsize=(figwidth(args.venue, "single"), 3.0))
    t_ref = None
    stats = {}
    for i, (did, lab) in enumerate(BANDS):
        curves = []
        for ms in per_seed:
            t, v = _series(ms, did)
            if t is None:
                break
            if args.drop_t0:
                keep = t > 0
                t, v = t[keep], v[keep]
            if t_ref is None:
                t_ref = t
            elif t.size != t_ref.size:
                raise ValueError(f"{did}: 各 seed 時間點數不一致")
            curves.append(100 * v)
        if len(curves) != len(per_seed):
            print(f"[warn] 缺 {did}，跳過")
            continue
        colour, _, ls = STYLE_CYCLE[i]
        _envelope(ax, t_ref, curves, colour, ls, lab)
        m = np.asarray(curves).mean(0)
        stats[did] = (float(np.median(m)), float(m.min()), float(m.max()))

    ax.axhline(10.0, color=MUTED, ls=":", lw=0.8)
    ax.text(t_ref[-1], 10.5, r"10% target", ha="right", va="bottom",
            fontsize=7, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"band energy relative error [%]")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = save_figure(fig, str(out / "band_energy_rel_error_vs_time"))
    plt.close(fig)
    print(f"[out] {[str(q) for q in paths]}")
    print(f"[data] {len(per_seed)} seeds, T={t_ref.size}, t=[{t_ref[0]:.2f},{t_ref[-1]:.2f}]")
    for did, (med, lo, hi) in stats.items():
        print(f"  {did:38s} median={med:6.2f}%  range={lo:.2f}–{hi:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
