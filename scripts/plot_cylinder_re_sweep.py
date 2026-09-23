"""畫 cylinder reconstruction error vs Reynolds number（§4.15 Nyquist 定量曲線）。

讀 CSV（欄位 Re,u_rel,v_rel,ke_rel,omega,ref），畫各 metric vs Re。動機：把 §4.15 的
「Re 決定稀疏重建可行性」從兩點對照升級為連續曲線，定位 laminar→transitional 轉折。

ref 欄：0 = every-5th sweep 點（連線），1 = 離群參考點（如 Re=10031 湍流，獨立星號不連線）。

用法：
  uv run python scripts/plot_cylinder_re_sweep.py --csv data/cylinder_re_sweep.csv \
    --out docs/figures/cylinder_re_sweep.png
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_style import OKABE_ITO, figwidth, save_figure, setup_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None,
                    help="圖上標題。**論文用圖不要給**——說明走 caption，圖上標題會與它重複")
    ap.add_argument("--venue", default="thesis", help="journal_style 的 rcParams profile")
    ap.add_argument("--annotate-ref", action="store_true",
                    help="在參考點旁畫箭頭註解。預設關：星號已足以辨識，"
                         "說明走 caption；開著會與右上的資料與圖例重疊")
    args = ap.parse_args()
    setup_style(args.venue)

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(filter(lambda line: not line.lstrip().startswith("#"), f)):
            rows.append(r)
    rows.sort(key=lambda r: float(r["Re"]))
    sweep = [r for r in rows if int(r.get("ref", "0")) == 0]
    ref = [r for r in rows if int(r.get("ref", "0")) == 1]

    def col(rs, k):
        return np.array([float(r[k]) for r in rs])

    Re_s = col(sweep, "Re")
    _OMEGA_C = OKABE_ITO["purple"]
    series = [("v_rel", "v rel-$L_2$", OKABE_ITO["vermillion"], "o"),
              ("u_rel", "u rel-$L_2$", OKABE_ITO["blue"], "^"),
              ("ke_rel", "KE rel-err", OKABE_ITO["green"], "v")]

    fig, ax = plt.subplots(figsize=(figwidth(args.venue, "single"), 3.4),
                           constrained_layout=True)
    for key, label, c, m in series:
        ax.plot(Re_s, 100.0 * col(sweep, key), marker=m, color=c, label=label, lw=1.7, ms=6)
        for r in ref:  # 離群參考點：獨立星號
            ax.scatter(float(r["Re"]), 100.0 * float(r[key]), marker="*", s=130, color=c,
                       edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("velocity / KE relative error (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", frameon=True)

    # omega RMSE（量綱不同）→ 右軸
    ax2 = ax.twinx()
    ax2.plot(Re_s, col(sweep, "omega"), marker="d", color=_OMEGA_C,
             label="$\\omega$ RMSE", lw=1.3, ms=5, ls="--", alpha=0.85)
    for r in ref:
        ax2.scatter(float(r["Re"]), float(r["omega"]), marker="*", s=130,
                    color=_OMEGA_C, edgecolors="k", linewidths=0.5, zorder=5)
    ax2.set_ylabel("$\\omega$ RMSE (1/s)", color=_OMEGA_C)
    ax2.tick_params(axis="y", labelcolor=_OMEGA_C)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="lower right", frameon=True)

    # 標註 laminar→transitional 轉折（1781→2343 陡跳）與離群參考
    if len(sweep) >= 2:
        ax.annotate("laminar\n(periodic)", xy=(Re_s[0], 100.0 * col(sweep, "v_rel")[0]),
                    xytext=(Re_s[0] + 200, 100.0 * col(sweep, "v_rel")[0] + 5.0),
                    fontsize=9, color=OKABE_ITO["vermillion"],
                    arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillion"], lw=0.8))
    if args.annotate_ref:
        for r in ref:
            ax.annotate(f"Re={int(float(r['Re']))}\n(turbulent ref)",
                        xy=(float(r["Re"]), 100.0 * float(r["v_rel"])),
                        xytext=(float(r["Re"]) - 2100, 100.0 * float(r["v_rel"]) - 8.5),
                        fontsize=8.5, color="0.3", ha="center",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))

    if args.title:
        ax.set_title(args.title)
    for w in save_figure(fig, args.out):
        print(f"[out] {w}")
    print(f"sweep {len(sweep)} pts {Re_s.min():.0f}-{Re_s.max():.0f} + {len(ref)} ref")


if __name__ == "__main__":
    main()
