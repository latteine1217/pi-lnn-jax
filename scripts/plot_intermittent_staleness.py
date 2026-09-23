#!/usr/bin/env python3
"""重建誤差 vs sensor gap 長度（thesis §7.2 fig:intermittent_staleness）。

What:
    對每個評估時刻，算它距離「最近一筆保留的 sensor frame」有多遠（staleness），
    依 gap 長度分箱後比較 B3（CfC）與容量匹配的 B0（vanilla）。

Why 這張圖是連續時間主張的關鍵證據:
    decoder 內部就是用 dt_to_query = t_q - sensor_time[idx] 前進狀態
    （models.py），所以 staleness 是模型實際看得見的量。CfC 把它當積分步長，
    vanilla 分支只能保持最近一筆輸入不變。若兩者對 gap 長度的敏感度沒有差別，
    連續時間的論點就沒有實證支撐。

    誤差沿用 evaluate 的逐時 ke_rel_err（metrics_per_t），gap 長度由 sensor
    自身的時間軸算出——兩者都不是重算的，避免與主表用不同定義。

Usage:
    uv run python scripts/plot_intermittent_staleness.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import setup_style, figwidth, save_figure, PICON, BASELINE, PICON_LS  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SENSOR_STEM = ("data/kolmogorov_sensors/re10000/"
               "sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone")
# thesis 主文用 70% 與 90% 兩組（gap 夠長才分得出差異）
RATES = (70, 90)
ARMS = (("b3", "Full PI-CON (B3)", PICON, "o", PICON_LS),
        ("b0cap", "Vanilla DeepONet (capacity-matched)", BASELINE, "^", "-"))
# gap 分箱（frames）；最後一箱吃到最長 gap
BIN_EDGES = [0, 1, 2, 3, 4, 6, 10, 10**6]
# n=5 training seeds {42,1,2,3,4}。seed 42 的 artifacts_dir 無後綴（既有慣例），
# 其餘為 _s{n}。mask 每個 rate 固定一份，故跨 seed 的散布是 **訓練變異**，
# 不是 mask 變異——圖上的 envelope 只能這樣讀。
SEED_SUFFIXES = ("", "_s1", "_s2", "_s3", "_s4")


def _gap_frames(query_t: np.ndarray, sensor_t: np.ndarray) -> np.ndarray:
    """每個 query 時刻距最近一筆（<=）sensor 的 frame 數。

    frame 以 sensor 序列的**最小**間隔為單位——那是原始儲存 cadence，也是
    thesis 表格的計數單位。用平均間隔會讓丟得越多、frame 越長，跨 rate 不可比。
    """
    st = np.sort(np.asarray(sensor_t, dtype=float))
    dt = float(np.min(np.diff(st)))
    idx = np.searchsorted(st, query_t, side="right") - 1
    idx = np.clip(idx, 0, st.size - 1)
    gaps = (query_t - st[idx]) / dt
    # 量化到 1e-3 再回傳。sensor 時間軸以 **float32** 存（0.45 落地成
    # 0.44999998807907104），query 時間軸是 float64，於是 dt 偏離名目 0.05 約 1e-8，
    # 商跟著偏離整數 ~1e-6：gap=1 會算成 0.999999 而掉進 [0,1) 箱。
    # 實測 90% 組因此把第一箱從 10 個灌成 16 個——thesis 舊表的 16 正是這麼來的。
    # 用 round 而非 rint：`sensor_time_independent` 下若 query 解析度細於 sensor
    # cadence，gap=0.5／1.5 這類半幀點是合法的，rint 會把它們併掉。
    return np.round(gaps, 3)


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """從一份 metrics.json 取 (query 時刻, 逐時誤差)。

    誤差是**空間平均 KE 的相對誤差**——即 `ke_t_mape_spatialmean` 的逐時項，
    與 thesis 表格同語意。per_t 也有一個 `ke_rel_err`，那是**逐點場誤差**
    （主線 ~19% 而非 ~5.7%）；拿它會讓整張圖比 thesis 高一個檔次而看不出
    是換了量。本函式是這個定義的唯一來源，multi-seed 聚合也匯入它。
    """
    if not path.is_file():
        raise FileNotFoundError(f"缺 {path}")
    per_t = json.loads(path.read_text())["metrics_per_t"]
    t = np.array([m["t"] for m in per_t], dtype=float)
    ke_p = np.array([m["ke_pred_mean"] for m in per_t], dtype=float)
    ke_d = np.array([m["ke_dns_mean"] for m in per_t], dtype=float)
    return t, np.abs(ke_p - ke_d) / (np.abs(ke_d) + 1e-12)


def _load_arm_seeds(root: Path, rate: int, arm: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """讀該臂全部 seed 的逐時誤差。

    走 `eval_staleness/` 而非訓練內建的 `final_eval/`：後者寫死
    `--protocol follow_training`，對 intermittent sensor 不保證 query 時刻與
    staleness 分箱相容（實驗紀錄 kolmogorov-intermittent-multiseed-2026-08）。
    兩者在本批資料上 query 時刻恰好相同，但依賴那個巧合是脆的。

    缺任何一個 seed 就 raise——靜默少算一個 seed 會讓 envelope 假性變窄。
    """
    out = []
    for suf in SEED_SUFFIXES:
        out.append(load_metrics(root / f"gap{rate}_{arm}{suf}"
                                / "eval_staleness" / "metrics.json"))
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", default=str(repo / "artifacts" / "kolmogorov"))
    ap.add_argument("--out", default=str(repo / "paper" / "thesis-format"
                                         / "figures" / "results"))
    args = ap.parse_args()
    root = Path(args.artifacts_root)

    setup_style("thesis")
    fig, axes = plt.subplots(1, len(RATES), sharey=True,
                             figsize=(figwidth("thesis", "single"), 3.0))

    for ax, rate in zip(np.atleast_1d(axes), RATES):
        npz = repo / f"{SENSOR_STEM}_gap{rate}_dns_values.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"缺 intermittent sensor 時間軸: {npz}")
        sensor_t = np.load(npz)["time"]

        # 軸以秒呈現：frame 是實作單位（sensor 序列的最小間隔），讀者要讀的是
        # 「資訊有多舊」。分箱仍在 frame 上做，只在畫之前換算。
        dt_s = float(np.min(np.diff(np.sort(np.asarray(sensor_t, dtype=float)))))
        for arm, label, colour, marker, ls in ARMS:
            seeds = _load_arm_seeds(root, rate, arm)
            gaps = _gap_frames(seeds[0][0], sensor_t)
            xs, mu, sd, ns = [], [], [], []
            for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                sel = (gaps >= lo) & (gaps < hi)
                if sel.sum() == 0:
                    continue
                # 先在每個 seed 內對該箱取平均，再跨 seed 算散布：
                # 反過來（把所有 seed 的時刻混在一起）會把訓練變異算進箱內散布。
                per_seed = np.array([100 * err[sel].mean() for _, err in seeds])
                xs.append(gaps[sel].mean() * dt_s)
                mu.append(per_seed.mean())
                sd.append(per_seed.std(ddof=1))
                ns.append(int(sel.sum()))
            xs, mu, sd = np.array(xs), np.array(mu), np.array(sd)
            ax.fill_between(xs, mu - sd, mu + sd, color=colour, alpha=0.18, linewidth=0)
            ax.plot(xs, mu, color=colour, marker=marker, linestyle=ls, label=label)
            if arm == "b3":
                print(f"  gap{rate}: bins n={ns}  (n_seed={len(seeds)})")
        ax.set_title(f"{rate}% temporal dropout", fontsize=8, pad=3)
        ax.set_xlabel("time since last sensor sample (s)")
    np.atleast_1d(axes)[0].set_ylabel("KE relative error (%)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.10), frameon=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = Path(args.out)
    print(f"[out] {[str(q) for q in save_figure(fig, str(out / 'intermittent_staleness'))]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
