"""把多份 `eval_pressure_limit.py` 輸出依 arm 聚合成 mean ± sd（ddof=1）。

What:
    讀 N 份輸出 JSON，依各檔記載的 `arm` 分組，對每個指標的時間平均報
    mean、sd（樣本標準差，ddof=1）、n 與逐 run 值；結果印成表並寫成 JSON。

Why fail-fast 而非「有幾份算幾份」:
    同一 arm 內混進不同 K、不同 AL 設定或重複 seed，平均出來的數字仍然長得合理。
    這裡把三者都當成錯誤；缺 seed 則由呼叫端決定能不能宣稱（n 原樣報出，
    n=1 時 sd 寫 null，不寫 0 也不寫 NaN）。只收 uv 閘門通過的輸出——
    eval 腳本在閘門失敗時不寫檔，這裡再驗一次，擋手動拼湊的檔案。

Usage:
    uv run python scripts/aggregate_pressure_limit.py out/*.json --out summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PRODUCER = "scripts/eval_pressure_limit.py"
SCHEMA = "pressure_limit.v1"
METRIC_KEYS = ("grad_p_rel_l2", "grad_p_rms_pred", "grad_p_rms_ref", "p_gr_rel_l2", "uv_rel_err")


def _load(path: Path) -> dict:
    d = json.loads(path.read_text())
    if d.get("producer") != PRODUCER or d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: 不是 {PRODUCER} 的 {SCHEMA} 輸出"
                         f"（producer={d.get('producer')!r} schema={d.get('schema')!r}）")
    if not d["uv_gate"]["passed"]:
        raise ValueError(f"{path}: uv 閘門未通過，不得進聚合")
    return d


def aggregate(payloads: list[tuple[Path, dict]]) -> dict:
    groups: dict[str, list[tuple[Path, dict]]] = {}
    for path, d in payloads:
        groups.setdefault(d["arm"], []).append((path, d))

    out = {}
    for arm, members in sorted(groups.items()):
        for field in ("K", "use_al", "arch"):
            vals = {m["run"][field] for _, m in members}
            if len(vals) != 1:
                raise ValueError(f"arm {arm!r} 混了不同的 {field}：{sorted(map(str, vals))}")
        seeds = [m["run"]["seed"] for _, m in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"arm {arm!r} 有重複 seed：{seeds}")
        n = len(members)
        metrics = {}
        for k in METRIC_KEYS:
            vals = np.asarray([m["metrics"][k]["time_mean"] for _, m in members], dtype=float)
            metrics[k] = {"mean": float(vals.mean()),
                          "sd": float(vals.std(ddof=1)) if n > 1 else None,
                          "per_run": vals.tolist()}
        first = members[0][1]["run"]
        out[arm] = {
            "n": n, "seeds": seeds, "K": first["K"], "use_al": first["use_al"],
            "runs": [m["run"]["artifacts_dir"] for _, m in members],
            "inputs": [str(p) for p, _ in members],
            "fingerprint_verified": [m["fingerprint_verified"] for _, m in members],
            "norm_stats_verified": [m["norm_stats_verified"] for _, m in members],
            "script_sha256": sorted({m["script"]["sha256"] for _, m in members}),
            "repo_git_sha": sorted({m["repo"]["git_sha"] for _, m in members}),
            "metrics": metrics,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="eval_pressure_limit.py 的輸出 JSON")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    result = aggregate([(path, _load(path)) for path in args.inputs])
    text = json.dumps({"producer": "scripts/aggregate_pressure_limit.py", "arms": result},
                      indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    print(f"{'arm':<12}{'n':>3}  " + "  ".join(f"{k:>22}" for k in METRIC_KEYS))
    for arm, a in result.items():
        cells = []
        for k in METRIC_KEYS:
            m = a["metrics"][k]
            cells.append(f"{m['mean']:>11.4g} ± {m['sd']:<8.2g}" if m["sd"] is not None
                         else f"{m['mean']:>11.4g} (n=1)   ")
        print(f"{arm:<12}{a['n']:>3}  " + "  ".join(cells))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
