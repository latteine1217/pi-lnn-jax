#!/usr/bin/env python3
"""thesis fig:spectrum_K{200,400} —— 不同 sensor 預算下的 t=5 能譜。

與 K=100 那張（由 plot_thesis_series_figures 產出）共用同一個繪圖函式，
只換資料源與 sensor 取樣帶邊界 k_max=sqrt(K/π)。三張因此保證同版面、
同標註規則——各寫一份遲早在其中一張漏掉某條參考線。

單 seed（thesis caption 明寫 seed 42）。

Usage:
    uv run python scripts/plot_spectrum_ksweep.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style  # noqa: E402
from plot_thesis_series_figures import fig_energy_spectrum  # noqa: E402

import numpy as np  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", default="200,400")
    ap.add_argument("--artifacts-root", default=str(repo / "artifacts" / "kolmogorov"))
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "results"))
    ap.add_argument("--venue", default="thesis",
                    help="journal_style venue：決定字體、圖寬與框線樣式（thesis / tmlr）")
    args = ap.parse_args()

    setup_style(args.venue)
    out = Path(args.out)
    for K in (int(x) for x in args.ks.split(",")):
        p = Path(args.artifacts_root) / f"ksweep_K{K}_b3" / "final_eval" / "series.npz"
        if not p.is_file():
            raise FileNotFoundError(
                f"缺 K={K} 的 series.npz: {p}（需該 config 訓練完成且 eval 帶 --export-arrays）")
        data = [{k: v for k, v in np.load(p).items()}]
        fig_energy_spectrum(data, out, K=K, stems=(f"spectrum_K{K}_nyquist",),
                            venue=args.venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
