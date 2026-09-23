"""把 gpu_peak_mem.py 產的 JSONL 聚合成三變體對照表（markdown）。

以現行 production fof (forward-over-forward) 為基準，列 for / ror 的 peak/step 比值。
Usage: uv run python -m bench.summarize_mem logs/bench_fwd_mem_<jobid>.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

VARIANTS = ["fof", "for", "ror"]
LABEL = {"fof": "fwd-over-fwd", "for": "fwd-over-rev", "ror": "rev-over-rev"}


def main():
    path = sys.argv[1]
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    grp = defaultdict(dict)   # (mode,dtype,N) -> {variant: rec}
    for r in recs:
        grp[(r["mode"], r["dtype"], r["n_collo"])][r["variant"]] = r

    dev = recs[0]["device"] if recs else "?"
    print("\n=== AD-mode 三方對照（基準 = fof / forward-over-forward, production）===")
    print(f"device={dev}  | 變體: " + ", ".join(f"{v}={LABEL[v]}" for v in VARIANTS) + "\n")

    hdr = ("mode", "dtype", "N",
           "fof_peak_MB", "for_peak_MB", "ror_peak_MB", "peak for/fof", "peak ror/fof",
           "fof_ms", "for_ms", "ror_ms")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")

    def g(d, v, key):
        return d[v][key] if v in d else float("nan")

    for key in sorted(grp, key=lambda k: (k[0], k[1], k[2])):
        mode, dtype, n = key
        d = grp[key]
        fof_pk = g(d, "fof", "peak_mb"); for_pk = g(d, "for", "peak_mb"); ror_pk = g(d, "ror", "peak_mb")
        fof_ms = g(d, "fof", "step_ms"); for_ms = g(d, "for", "step_ms"); ror_ms = g(d, "ror", "step_ms")
        pr_for = (for_pk / fof_pk) if fof_pk else float("nan")
        pr_ror = (ror_pk / fof_pk) if fof_pk else float("nan")
        print(f"| {mode} | {dtype} | {n} | {fof_pk:.1f} | {for_pk:.1f} | {ror_pk:.1f} | "
              f"{pr_for:.2f}× | {pr_ror:.2f}× | {fof_ms:.3f} | {for_ms:.3f} | {ror_ms:.3f} |")

    print("\nratio < 1 表示比 fof 更省記憶體；> 1 表示更耗。"
          "\n預期（小輸入/輸出維度）：fof 最省（無 reverse tape）≤ for < ror。")


if __name__ == "__main__":
    main()
