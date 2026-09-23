#!/usr/bin/env python3
"""K–Δt 取樣平面的可分離性：主效應、交互作用，與交互作用的 seed 不確定度。

What:
    在乾淨格上擬 `ln err ≈ a(K) + b(Δt)`，報三件事：
      1. 兩個主效應在 ln 空間的跨度（換算成倍率）；
      2. 交互作用 = 擬合殘差，逐格以「比可分離模型差幾 %」表示；
      3. 每一格殘差的 **seed bootstrap ±1 sd**——逐列重抽 seed 標籤
         （seed 在同一列內是配對的），重算格均值、重擬、重取殘差。

Why:
    殘差圖很容易被逐格解讀，但它的每一格都是由整張表的擬合導出的，
    而各列的 seed 數不同（K=200 是 n=5、其餘 n=3、K=400/Δt=0.025 只有 n=2），
    **殘差的精度因此跨列不同**。沒有這個數，讀者無從判斷一格 ±2% 的起伏
    是結構還是噪聲。第 (3) 項另外標出「不可獨立估計」的格：某欄只剩兩格時，
    可分離模型會把它們的 ln 殘差強制成等值反號，那不是量到的交互作用。

Usage:
    uv run python scripts/diag_sampling_plane_separability.py \
        --grid docs/figures/sampling_plane_grid_data.json \
        --out docs/figures/sampling_plane_separability.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

_N_BOOT = 4000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, help="sampling_plane_grid_data.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=_N_BOOT)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed（結果要可重跑）")
    p.add_argument("--fit", choices=("balanced-core", "all-clean"), default="balanced-core",
                   help="balanced-core（預設）只擬最大的完全交叉子塊——每一格的殘差都獨立"
                        "可估。all-clean 擬所有乾淨格，覆蓋較廣但會有整列／整欄被強制"
                        "等值反號的退化格（輸出的兩個 *_independently_estimable 旗標會標出來）。")
    return p.parse_args()


def _largest_balanced_block(clean: list[tuple[int, float]]) -> set[tuple[int, float]]:
    """最大的完全交叉子塊（列 × 欄全滿）。格數不多，直接枚舉列的子集。

    為什麼要它：可加模型在只剩兩格的列或欄上會把 ln 殘差強制成等值反號，那不是量到的
    交互作用。完全交叉的子塊裡每一格都由別的列與欄共同約束，殘差才讀得出東西。"""
    from itertools import combinations
    rows = sorted({c[0] for c in clean})
    have = set(clean)
    best: set[tuple[int, float]] = set()
    best_score = (0, 0)
    # 目標是**格數**最多，不是列數最多——4 列 × 2 欄只有 8 格，3 列 × 3 欄有 9 格，
    # 而後者能估的交互作用自由度也較多。所以全部枚舉完再挑，不可提早 break。
    for k in range(2, len(rows) + 1):
        for rsub in combinations(rows, k):
            cols = [d for d in sorted({c[1] for c in clean})
                    if all((r, d) in have for r in rsub)]
            if len(cols) < 2:
                continue
            block = {(r, d) for r in rsub for d in cols}
            # 同格數時偏好較方正的：(列−1)×(欄−1) 就是交互作用的自由度
            score = (len(block), (k - 1) * (len(cols) - 1))
            if score > best_score:
                best, best_score = block, score
    if not best:
        raise ValueError("找不到任何 2×2 以上的完全交叉子塊")
    return best


def _fit(cells: list[tuple[int, float]], ks: list[int], dts: list[float],
         means: dict[tuple[int, float], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ln err ≈ a(K) + b(Δt) 的最小平方。設計矩陣秩虧（差一個常數），
    但殘差是對行空間的投影，與係數的 gauge 無關，故直接取 lstsq 的殘差。"""
    A = np.zeros((len(cells), len(ks) + len(dts)))
    y = np.zeros(len(cells))
    for i, c in enumerate(cells):
        A[i, ks.index(c[0])] = 1.0
        A[i, len(ks) + dts.index(c[1])] = 1.0
        y[i] = np.log(means[c])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef, coef[:len(ks)], coef[len(ks):]


def main() -> None:
    a = parse_args()
    grid = json.loads(Path(a.grid).read_text())

    per_seed: dict[tuple[int, float], dict[str, float]] = {}
    excluded: dict[tuple[int, float], str] = {}
    for row in grid["rows"]:
        key = (int(row["K"]), float(row["dt"]))
        per_seed[key] = {k: float(v) for k, v in row["per_seed"].items()}
        # 兩種排除理由都讓該格「換了一個變因」，擬合不得納入
        if row.get("sensor_query_downgraded"):
            excluded[key] = "sensor_query 降級為全取"
        elif row.get("grad_accum_mismatch"):
            excluded[key] = f"grad_accum_chunks={row.get('grad_accum_chunks')} 與同列其他格不同"
    downgraded = set(excluded)

    clean = sorted(c for c in per_seed if c not in downgraded)
    if a.fit == "balanced-core":
        core = _largest_balanced_block(clean)
        for c in clean:
            if c not in core:
                excluded[c] = "不在最大完全交叉子塊內"
        clean = sorted(core)
        downgraded = set(excluded)
    # 水準必須由**乾淨格**推導。用全部的格會留下沒有任何觀測的水準（例如整欄被排除的
    # Δt），那個係數不受任何資料約束，lstsq 的最小範數解會隨便給它一個值，而主效應的
    # 「跨度」是 max−min——於是報出一個純屬人造的數字。實測：Δt 跨度會從 0.19 變 1.39。
    ks = sorted({c[0] for c in clean})
    dts = sorted({c[1] for c in clean})
    for lvl, axis in [(ks, "K"), (dts, "Δt")]:
        if len(lvl) < 2:
            raise ValueError(f"{axis} 只剩 {len(lvl)} 個水準有乾淨格——主效應無從估計")
    if not clean:
        raise ValueError("沒有乾淨的格可擬——檢查 grid JSON 的排除旗標")

    # 每欄／每列剩幾格：只剩兩格時該欄（列）的 ln 殘差被強制等值反號，
    # 不是獨立估計出來的。**兩個方向都要檢查**——排除整欄之後換成列會退化。
    per_col, per_row = defaultdict(list), defaultdict(list)
    for c in clean:
        per_col[c[1]].append(c[0])
        per_row[c[0]].append(c[1])

    means = {c: float(np.mean(list(v.values()))) for c, v in per_seed.items()}
    resid, a_k, b_dt = _fit(clean, ks, dts, means)

    rng = np.random.default_rng(a.seed)
    row_seeds = {K: sorted({s for c in clean if c[0] == K for s in per_seed[c]}) for K in ks}
    # seed 在同一列內是配對的，故預設逐列重抽同一組標籤。但有的格缺 seed
    # （K=400/Δt=0.025 只跑成 2 支），列的標籤可能一個都不落在它上面——
    # 那種格改在它自己的 seed 集合上獨立重抽，並在輸出裡標明配對已斷。
    unpaired = {c for c in clean if set(per_seed[c]) != set(row_seeds[c[0]])}
    draws: dict[tuple[int, float], list[float]] = defaultdict(list)
    for _ in range(a.n_boot):
        pick = {K: rng.choice(row_seeds[K], size=len(row_seeds[K]), replace=True) for K in ks}
        m = {}
        for c in clean:
            v = per_seed[c]
            own = sorted(v)
            sel = ([v[s] for s in rng.choice(own, size=len(own), replace=True)]
                   if c in unpaired else [v[s] for s in pick[c[0]] if s in v])
            if not sel:
                raise ValueError(f"{c} 在重抽後沒有任何 seed——seed 集合的推導有誤")
            m[c] = float(np.mean(sel))
        r, _, _ = _fit(clean, ks, dts, m)
        for i, c in enumerate(clean):
            draws[c].append(float(np.expm1(r[i]) * 100.0))

    rows = []
    for i, c in enumerate(clean):
        d = np.asarray(draws[c])
        rows.append({
            "K": c[0], "dt": c[1],
            "residual_pct": float(np.expm1(resid[i]) * 100.0),
            "residual_ln": float(resid[i]),
            "bootstrap_sd_pp": float(d.std(ddof=1)),
            "n_seeds": len(per_seed[c]),
            "column_independently_estimable": len(per_col[c[1]]) > 2,
            "row_independently_estimable": len(per_row[c[0]]) > 2,
            "seed_paired_with_row": c not in unpaired,
        })

    ak = a_k - a_k.mean()
    bd = b_dt - b_dt.mean()
    payload = {
        "producer": "scripts/diag_sampling_plane_separability.py",
        "model": "ln(uv_rel_err) ~ a(K) + b(dt)",
        "fit_scope": a.fit,
        "provenance": {
            "grid": str(a.grid),
            "code_revision": _rev(),
            "n_boot": a.n_boot, "bootstrap_seed": a.seed,
            "bootstrap": "逐列重抽 seed 標籤（seed 在列內配對），重算格均值後重擬",
            "clean_cells": len(clean),
            "excluded": [{"K": c[0], "dt": c[1], "why": excluded[c]}
                         for c in sorted(excluded)],
        },
        "main_effects_ln": {
            "K": {str(K): float(v) for K, v in zip(ks, ak)},
            "dt": {str(d): float(v) for d, v in zip(dts, bd)},
            "K_span": float(ak.max() - ak.min()), "K_span_ratio": float(np.exp(ak.max() - ak.min())),
            "dt_span": float(bd.max() - bd.min()), "dt_span_ratio": float(np.exp(bd.max() - bd.min())),
        },
        "interaction_rms_ln": float(np.sqrt((resid ** 2).mean())),
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    print(f"{'K':>5}{'dt':>8}{'殘差%':>10}{'±sd(pp)':>10}{'n':>4}  獨立估計（欄/列）")
    for r in rows:
        mark = lambda b: '是' if b else '否'
        print(f"{r['K']:>5}{r['dt']:>8}{r['residual_pct']:>10.2f}{r['bootstrap_sd_pp']:>10.2f}"
              f"{r['n_seeds']:>4}  {mark(r['column_independently_estimable'])}"
              f" / {mark(r['row_independently_estimable'])}")
    me = payload["main_effects_ln"]
    print(f"\nK 主效應 {me['K_span']:.3f} ln（{me['K_span_ratio']:.2f}×）、"
          f"Δt {me['dt_span']:.3f} ln（{me['dt_span_ratio']:.2f}×）、"
          f"交互作用 RMS {payload['interaction_rms_ln']:.3f} ln")
    print(f"[out] {out}")


def _rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 —— 非 git checkout 下仍應能產出
        return None


if __name__ == "__main__":
    main()
