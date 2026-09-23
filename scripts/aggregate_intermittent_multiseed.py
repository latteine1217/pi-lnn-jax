#!/usr/bin/env python3
"""intermittent multi-seed 聚合（thesis §7.2 的 n=5 升級）。

What:
    把 gap{70,90}_{b3,b0cap} 的 5 個 seed 依 gap 長度分箱後聚合，並跑
    `knowledge/experiments/kolmogorov-intermittent-multiseed-2026-08.md`
    事前登錄的兩條判準。

Why 匯入 plot_intermittent_staleness 而不是自己算:
    分箱邊界（BIN_EDGES）、gap 的 frame 定義（以 sensor 序列最小間隔為單位）
    與誤差來源（metrics_per_t 的 ke_rel_err）必須與 thesis 那張圖**逐字相同**，
    否則跨 n=1 / n=5 的比較會混進「換了分箱」這個變因。事前登錄已寫死這點。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_intermittent_staleness import (  # noqa: E402
    BIN_EDGES, SENSOR_STEM, _gap_frames, load_metrics,
)

REPO = Path(__file__).resolve().parent.parent
SEEDS = ["42", "1", "2", "3", "4"]


def _run_dir(root: Path, rate: int, arm: str, seed: str) -> Path:
    """seed 42 是既有 run（無後綴），其餘是本次新增。"""
    return root / (f"gap{rate}_{arm}" if seed == "42" else f"gap{rate}_{arm}_s{seed}")


MIN_SEEDS_PER_BIN = 3      # bin 內少於此數就不報該 bin——寧可缺格也不要偽裝成 n=5
MIN_QUERY_TIMES = 50       # 完整格點應有 ~101 點；只有保留幀（gap 恆 0）會遠少於此


def _binned(root: Path, rate: int, arm: str, seed: str, sensor_t: np.ndarray, subdir: str):
    p = _run_dir(root, rate, arm, seed) / subdir / "metrics.json"
    if not p.is_file():
        return None, f"缺 {p}"
    per_t = json.loads(p.read_text())["metrics_per_t"]  # 只用來數 query 點；誤差走 load_metrics
    # 協定閘門：staleness 分析要求 query 走完整 DNS 格點（thesis §7.2 的
    # sensor_time_independent）。若拿訓練內建 final_eval（--protocol follow_training），
    # query 時刻 = 保留幀 → gap 恆 0，高 gap 的 bin 全空，而聚合會靜默地
    # 用少數 seed 充當全部。2026-08-22 就是這樣得到一張看似合理的廢表。
    if len(per_t) < MIN_QUERY_TIMES:
        return None, (f"{p} 只有 {len(per_t)} 個 query 時刻（<{MIN_QUERY_TIMES}）"
                      f"——多半是 follow_training 的產物，不能用於 staleness 分析")
    # 誤差定義的唯一來源是 plot_intermittent_staleness.load_metrics —— 直接讀
    # per_t["ke_rel_err"] 會拿到逐點場誤差，與 thesis 那張圖不是同一個量。
    qt, err = load_metrics(p)
    gaps = _gap_frames(qt, sensor_t)
    # 分箱閘門：BIN_EDGES 是整數幀邊界，只有在 query 對齊 sensor 幀（gap 為整數）
    # 時 bin 組成才與 thesis 那張圖相同。stride 取太細會讓 gap=0.5 這種半幀點
    # 落進 [0,1) 箱，數字看起來正常但量的是別的東西。
    # 門檻 1e-3：實測對齊良好時偏差 ~8e-5（時間軸的浮點表示雜訊），
    # 而真正要擋的錯配是半幀偏移（0.5）。1e-6 會把合法的 run 也擋掉。
    frac = np.abs(gaps - np.round(gaps))
    if frac.max() > 1e-3:
        return None, (f"{p} 的 query 未對齊 sensor 幀（最大非整數 gap 偏差 "
                      f"{frac.max():.3f} 幀）——stride 需使 query 落在 sensor cadence 上")
    # 落在第一筆 sensor 之前的 query（dropout 後首筆保留幀不在 t=0）gap 為負，
    # 不屬任何 bin。thesis 那張圖同樣丟棄它們，行為一致——但要出聲，不靜默。
    n_before = int((gaps < 0).sum())
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        sel = (gaps >= lo) & (gaps < hi)
        out.append(float(err[sel].mean()) if sel.any() else math.nan)
    if n_before:
        print(f"     · {_run_dir(root, rate, arm, seed).name}: {n_before}/{len(gaps)} 個 query "
              f"早於首筆 sensor，不入箱（與 thesis 圖一致）")
    return np.asarray(out), None


def _ms(x):
    x = [v for v in x if not math.isnan(v)]
    if not x:
        return math.nan, math.nan
    m = sum(x) / len(x)
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else math.nan
    return m, s


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-subdir", default="eval_staleness",
                    help="讀哪個評估子目錄（需為 sensor_time_independent 的產物）")
    args = ap.parse_args()
    root = REPO / "artifacts" / "kolmogorov"
    labels = [f"[{lo},{hi})" if hi < 10**5 else f">={lo}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    summary = {}
    for rate in (70, 90):
        npz = REPO / f"{SENSOR_STEM}_gap{rate}_dns_values.npz"
        sensor_t = np.load(npz)["time"]
        print(f"\n=== {rate}% dropout（KE rel-err %，mean ± sd over n=5 seeds）===")
        print("%-22s %s" % ("arm \\ gap(frames)", "  ".join("%12s" % l for l in labels)))
        for arm in ("b3", "b0cap"):
            per_seed, miss = [], []
            for s in SEEDS:
                b, err = _binned(root, rate, arm, s, sensor_t, args.eval_subdir)
                (miss if b is None else per_seed).append(err if b is None else b)
            if miss:
                print("  ⚠️ %s 有 %d 個 seed 不可用：" % (arm, len(miss)))
                for m in miss:
                    print("     -", m)
            if not per_seed:
                print("%-22s (全缺)" % arm); continue
            arr = np.vstack(per_seed)
            cells = []
            for j in range(arr.shape[1]):
                col = [v for v in arr[:, j] if not math.isnan(v)]
                if len(col) < MIN_SEEDS_PER_BIN:
                    # 不足就標明實際 n，不給一個看起來像 n=5 的數字
                    cells.append("n=%d 略" % len(col))
                    continue
                m, sd = _ms(list(arr[:, j]))
                cells.append("%5.2f±%.2f(n=%d)" % (m * 100, (sd or 0) * 100, len(col)))
            print("%-22s %s%s" % (arm, "  ".join("%12s" % c for c in cells),
                                  ("  ⚠️缺%d" % len(miss)) if miss else ""))
            summary[(rate, arm)] = arr
    # ── 事前登錄的兩條判準（只對 90% 組）──
    print("\n=== 事前登錄判準（90% dropout）===")
    for arm, want in (("b3", "平坦"), ("b0cap", "單調上升")):
        arr = summary.get((90, arm))
        if arr is None:
            print("  %-6s (無資料)" % arm); continue
        means = [_ms(list(arr[:, j]))[0] for j in range(arr.shape[1])]
        sds = [_ms(list(arr[:, j]))[1] for j in range(arr.shape[1])]
        ok = [(m, s) for m, s in zip(means, sds) if not math.isnan(m)]
        mono = all(ok[i][0] <= ok[i + 1][0] + 1e-12 for i in range(len(ok) - 1))
        spread = (max(m for m, _ in ok) - min(m for m, _ in ok)) * 100
        sd_typ = np.nanmean([s for _, s in ok if not math.isnan(s)]) * 100
        print("  %-6s 期望=%-6s  單調上升=%-5s  極差=%.2fpp  典型σ=%.2fpp  3σ=%.2fpp → %s"
              % (arm, want, mono, spread, sd_typ, 3 * sd_typ,
                 "極差<3σ（平坦）" if spread < 3 * sd_typ else "極差>3σ（不平坦）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
