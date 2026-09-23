#!/usr/bin/env python3
"""時間外推的 error-vs-lead-time 曲線（EXP-530）。

What:
    讀 `evaluate_exp245.py --export-arrays` 的 series.npz 與同一份延長版 DNS，
    畫模型誤差對 lead time τ = t − t_data_end 的曲線，並疊兩條參考線：
      persistence  —— 把最後一個有資料的時刻 t_end 的**真值場**凍結不動的誤差。
                      模型若贏不過它，代表外推段沒有帶進任何動力學資訊。
      decorrelation —— 兩個時間相距夠遠的真值場之間的誤差（同一條軌跡上取樣）。
                      這是「完全失相關」的水平；碰到它就等於這個時刻沒有任何
                      逐點預測能力，此後曲線的高低不再有意義。

Why:
    Re=10⁴ 是混沌流，逐點誤差必然隨 τ 上升並飽和。只報 τ=5 的單點數字無法區分
    「稍微能外推」與「完全失效」，也無法誠實地說出有效預測時間有多長。判準因此
    事前定為：模型曲線在哪個 τ 越過 persistence、在哪個 τ 越過去相關水平的一半。

Usage:
    uv run python scripts/plot_extrapolation_leadtime.py \\
      --series artifacts/kolmogorov/exp530_b3_extrap_T10/extrap_T10/series.npz \\
      --dns data/dns/kolmogorov_dns_..._T10_..._ds4_cuda.npy \\
      --t-data-end 5.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo 內 scripts/journal_style.py
from journal_style import figwidth, save_figure, setup_style

from pi_lnn_jax.data import load_dns_from_path

#: 相距至少這麼久的兩個時刻視為已失相關（Kolmogorov Re=10⁴，單位為模擬時間）
DECORRELATION_LAG = 4.0

#: 兩個零均值、等能量、統計獨立的場，其相對 L2 的期望值。推導而非量測：
#: E‖a−b‖² = ‖a‖²+‖b‖² = 2‖b‖² → √2。
#:
#: **量測值偏離 √2 由兩個方向相反的效應決定**（2026-08-21 實測，91 對、lag≥2）：
#:   1. **能量不等**（本軌跡是衰減暫態，E_i/E_j=1.19）把零互相關期望推高到
#:      √(1+E_i/E_j)=1.479；
#:   2. **真實的場相關**（forcing 撐的大尺度結構為兩張快照共用）再往下拉 6.2%，
#:      量測落在 1.387。
#: 早期註解把整個偏離歸給第 2 項、漏了反向的第 1 項——實際上哪一項勝出取決於
#: 取樣的幀對，換一組 lag 就可能高於 √2。
#:
#: 另外，預測振幅若高於真值（本實驗後段正是如此）誤差可超過 √2 → 它不是上界。
#: 故 √2 只是理想化漸近線，既非本流場的準確去相關值、也不是誤差上界。
UNCORRELATED_SQRT2 = float(np.sqrt(2.0))


def _uv_rel(u_a, v_a, u_b, v_b):
    """與 metric_artifact.uv_rel_err 同一個定義（u,v 合併的相對 L2）。"""
    num = np.sqrt(np.sum((u_a - u_b) ** 2) + np.sum((v_a - v_b) ** 2))
    den = np.sqrt(np.sum(u_b ** 2) + np.sum(v_b ** 2))
    return float(num / den)


def persistence_curve(dns_u, dns_v, dns_t, t_end):
    """把 t_end 的真值場凍結，對每個 t 算誤差。回 (t, err)。"""
    i_end = int(np.argmin(np.abs(dns_t - t_end)))
    if abs(float(dns_t[i_end]) - t_end) > 1e-6:
        raise ValueError(
            f"DNS 時間軸上沒有 t={t_end}（最近的是 {float(dns_t[i_end])}）；"
            "不做寬鬆對齊——參考線錯位會讓整張圖的解讀失真")
    u0, v0 = dns_u[i_end], dns_v[i_end]
    err = np.array([_uv_rel(u0, v0, dns_u[i], dns_v[i]) for i in range(len(dns_t))])
    return dns_t, err


def decorrelation_level(dns_u, dns_v, dns_t, lag=DECORRELATION_LAG):
    """相距 ≥ lag 的所有幀對之間的 uv 相對誤差 → (mean, std, n_pairs)。"""
    idx = np.arange(0, len(dns_t), max(1, len(dns_t) // 40))  # 取樣避免 O(T²) 全配對
    pairs = [(i, j) for i in idx for j in idx if dns_t[j] - dns_t[i] >= lag]
    if not pairs:
        raise ValueError(f"DNS 時間跨度不足 {lag}，無法估計去相關水平")
    vals = [_uv_rel(dns_u[i], dns_v[i], dns_u[j], dns_v[j]) for i, j in pairs]
    return float(np.mean(vals)), float(np.std(vals)), len(pairs)


def _first_crossing(tau, model, level):
    """model 首次 ≥ level 的 τ（level 可為陣列或純量）；沒越過回 None。"""
    over = model >= level
    if not np.any(over):
        return None
    return float(tau[int(np.argmax(over))])


def useful_horizon(tau, model, reference):
    """模型持續優於 reference 的最大 τ；一開始就輸則回 0.0。

    Why 不用 `_first_crossing`：oracle-persistence 在 τ=0 依定義為 0，任何模型都
    「越過」它，那個交叉點恆為 0、沒有資訊。這裡問的是可用的預測水平線——
    模型從 τ=0⁺ 起連續贏到哪裡。
    """
    order = np.argsort(tau)
    tau_s, model_s, ref_s = tau[order], model[order], reference[order]
    horizon = 0.0
    for tv, mv, rv in zip(tau_s, model_s, ref_s):
        if tv <= 0.0:
            continue
        if mv >= rv:
            break
        horizon = float(tv)
    return horizon


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True, help="evaluate_exp245.py 的 series.npz")
    ap.add_argument("--dns", required=True, help="評估用的延長版 DNS .npy（與 eval 同一份）")
    ap.add_argument("--t-data-end", type=float, required=True,
                    help="sensor 資料的最後時刻（外推起點）")
    ap.add_argument("--metric", default="uv_rel_err",
                    help="series.npz 內的逐時欄位（預設 uv_rel_err = primary field fidelity）")
    ap.add_argument("--fields", default=None,
                    help="evaluate_exp245.py --export-fields 的 fields.npz。給了就多算一條"
                         "model-persistence（凍結模型自己在 t_data_end 的重建場）——"
                         "那才是公平的『時間演化有沒有貢獻』對照，oracle-persistence "
                         "用的是模型拿不到的真值場")
    ap.add_argument("--chaos-budget", default=None,
                    help="diag_chaos_budget.py 的 chaos_budget.json。給了就多畫一條"
                         "物理下限（把模型在 t_data_end 的場交給真實求解器往前積），"
                         "並報模型誤差與該下限的比值")
    ap.add_argument("--out-dir", default=None, help="輸出落點；預設與 series.npz 同目錄")
    args = ap.parse_args()

    series_path = Path(args.series)
    if not series_path.exists():
        raise FileNotFoundError(f"series.npz 不存在：{series_path}")
    series = np.load(series_path)
    for key in ("t", args.metric):
        if key not in series:
            raise KeyError(f"series.npz 缺欄位 {key!r}（有：{sorted(series.files)}）")
    t_model = np.asarray(series["t"], dtype=float)
    err_model = np.asarray(series[args.metric], dtype=float)

    dns_u, dns_v, dns_t = load_dns_from_path(args.dns, time_stride=1)
    dns_t = np.asarray(dns_t, dtype=float)
    if t_model.max() > dns_t.max() + 1e-9:
        raise ValueError(
            f"series 的時間軸到 {t_model.max()}，超過 DNS 的 {dns_t.max()}；"
            "兩者不是同一次評估的產物")

    t_pers, err_pers = persistence_curve(dns_u, dns_v, dns_t, args.t_data_end)
    floor_mean, floor_std, n_pairs = decorrelation_level(dns_u, dns_v, dns_t)

    tau = t_model - args.t_data_end
    ext = tau >= 0.0
    # persistence 插到模型的時間格點上比較（兩者格點可能不同）
    pers_on_model = np.interp(t_model, t_pers, err_pers)

    # model-persistence：凍結模型自己在 t_data_end 的重建場。oracle-persistence 用的是
    # 模型拿不到的真值場，只能當「流場本身變多快」的尺；要問「時間演化有沒有貢獻」，
    # 對照必須是模型自己的最後一次估計。
    mp_on_model = None
    if args.fields:
        f = np.load(args.fields)
        for key in ("u_pred", "v_pred", "t"):
            if key not in f:
                raise KeyError(f"fields.npz 缺欄位 {key!r}（有：{sorted(f.files)}）")
        f_t = np.asarray(f["t"], dtype=float)
        i_end = int(np.argmin(np.abs(f_t - args.t_data_end)))
        if abs(float(f_t[i_end]) - args.t_data_end) > 1e-6:
            raise ValueError(
                f"fields.npz 的時間軸上沒有 t={args.t_data_end}；不做寬鬆對齊")
        u0, v0 = f["u_pred"][i_end], f["v_pred"][i_end]
        # 對真值算誤差（與模型曲線同一把尺），DNS 需插到 fields 的格點上
        idx = [int(np.argmin(np.abs(dns_t - tt))) for tt in f_t]
        if max(abs(dns_t[i] - tt) for i, tt in zip(idx, f_t)) > 1e-6:
            raise ValueError("fields.npz 與 DNS 的時間格點不一致；拒絕寬鬆對齊")
        mp = np.array([_uv_rel(u0, v0, dns_u[i], dns_v[i]) for i in idx])
        mp_on_model = np.interp(t_model, f_t, mp)

    # 物理下限：同一個起點交給真實求解器的結果。控制組（求解器對 DNS 的保真度）
    # 若不遠小於它，整份 chaos_budget 無效——這裡直接擋，不讓無效的下限進圖。
    budget_on_model = None
    budget_meta = None
    if args.chaos_budget:
        cb = json.loads(Path(args.chaos_budget).read_text())
        cb_tau = np.asarray(cb["tau"], dtype=float)
        cb_val = np.asarray(cb["solver_from_model"], dtype=float)
        ctrl_max = float(np.max(np.abs(cb["solver_from_dns"])))
        if ctrl_max > 0.1 * float(np.min(cb_val[cb_tau > 0])):
            raise ValueError(
                f"chaos_budget 的求解器保真度控制組太大（{ctrl_max:.3e}），"
                "下限不可信，拒絕入圖")
        budget_on_model = np.interp(tau, cb_tau, cb_val, left=np.nan, right=np.nan)
        budget_meta = {"solver_fidelity_control_max": ctrl_max, "source": args.chaos_budget}

    lead_times = [lt for lt in (0.0, 0.5, 1.0, 2.0, 5.0)
                  if args.t_data_end + lt <= t_model.max()]
    at_lead = {}
    for lt in lead_times:
        row = {
            "model": float(np.interp(args.t_data_end + lt, t_model, err_model)),
            "persistence_oracle": float(np.interp(args.t_data_end + lt, t_pers, err_pers)),
        }
        if mp_on_model is not None:
            row["persistence_model"] = float(
                np.interp(args.t_data_end + lt, t_model, mp_on_model))
        if budget_on_model is not None:
            b = float(np.interp(args.t_data_end + lt, t_model, budget_on_model))
            row["chaos_budget"] = b
            row["model_over_budget"] = float(row["model"] / b) if b > 0 else None
        at_lead[f"{lt:g}"] = row

    report = {
        "t_data_end": args.t_data_end,
        "metric": args.metric,
        "dns": str(args.dns),
        "in_window_mean": float(err_model[~ext].mean()) if np.any(~ext) else None,
        "at_lead_times": at_lead,
        # 附註，不是判準：兩個零均值等能量無關場的相對 L2 期望值就是 √2≈1.414，
        # 這個量測值只是它（略低是因為 forcing 撐的平均剪切為兩張場共用）。
        # 且本軌跡非統計穩態，遠距幀對混了「相位無關」與「能量狀態不同」。
        "decorrelation_level": {
            "mean": floor_mean, "std": floor_std, "n_pairs": n_pairs,
            "note": "informational only; ≈ sqrt(2) by construction, not a criterion",
        },
        # 有效預測水平線：模型從 τ=0⁺ 起連續優於該基線的最大 τ；0.0 = 一開始就輸。
        "useful_horizon_vs_persistence_oracle": useful_horizon(
            tau[ext], err_model[ext], pers_on_model[ext]),
        "useful_horizon_vs_persistence_model": (
            None if mp_on_model is None
            else useful_horizon(tau[ext], err_model[ext], mp_on_model[ext])),
        # 附註，不是判準：0.5× 這個門檻沒有依據，且它所依據的去相關水平見上。
        "tau_model_reaches_half_decorrelation": _first_crossing(
            tau[ext], err_model[ext], 0.5 * floor_mean),
        # 有依據的第二個標記：越過它代表比「整張場輸出零」還差。
        "tau_model_exceeds_zero_prediction": _first_crossing(
            tau[ext], err_model[ext], 1.0),
        # 理想化去相關漸近線（推導值，非量測；前提見 UNCORRELATED_SQRT2 註解）。
        # 量測到的 decorrelation_level 是它的一個有散佈的估計，兩者應互相印證。
        "uncorrelated_reference_sqrt2": UNCORRELATED_SQRT2,
        # 三條曲線本來就算完了；一起落盤讓別的呈現（重畫、講解、跨臂疊圖）不必重跑
        # 一次要載 800 MB DNS 的分析，也不必從圖上讀數字。
        "chaos_budget": budget_meta,
        "curves": {
            "tau": tau.tolist(),
            "model": err_model.tolist(),
            "persistence_oracle": pers_on_model.tolist(),
            "persistence_model": (None if mp_on_model is None else mp_on_model.tolist()),
            "chaos_budget": (None if budget_on_model is None else budget_on_model.tolist()),
        },
        "reading": (
            "useful_horizon_* 是有效預測水平線：模型從 τ=0⁺ 起連續贏過該基線到哪個 τ；"
            "0.0 代表外推一開始就不如把場凍住。oracle 版凍的是真值場（模型拿不到，"
            "只能當『流場本身變多快』的尺），model 版凍的是模型自己在 t_data_end 的"
            "重建場——那才是『時間演化有沒有貢獻』的公平對照。"
            "tau_model_exceeds_zero_prediction 之後，模型輸出比整張場填零還差"
            "（振幅／能量已跑掉）。decorrelation_level 與 "
            "tau_model_reaches_half_decorrelation 只是附註：前者 ≈ √2 是度量本身的"
            "性質、非本流場的資訊，後者的 0.5× 門檻無依據。"
        ),
    }

    out_dir = Path(args.out_dir) if args.out_dir else series_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extrapolation_leadtime.json").write_text(json.dumps(report, indent=2))

    setup_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(figwidth("iclr", "single"), 2.9))
    ax.plot(tau, err_model, label="PI-CON", lw=1.4, zorder=4)
    # persistence 只畫 τ≥0：τ<0 是拿「凍結在 t_data_end 的場」去比更早的時刻，
    # 那個誤差不是任何人的預測誤差，畫出來只會讓圖多兩條沒有意義的高聳曲線。
    pm = (t_pers - args.t_data_end) >= 0.0
    ax.plot((t_pers - args.t_data_end)[pm], err_pers[pm], ls="--", lw=1.1,
            label="Persistence (truth)", zorder=3)
    if mp_on_model is not None:
        ax.plot(tau[ext], mp_on_model[ext], ls="-.", lw=1.1,
                label="Persistence (model)", zorder=3)
    if budget_on_model is not None:
        ax.plot(tau[ext], budget_on_model[ext], lw=1.4, color="#1baf7a",
                label="Physics from model state", zorder=4)
    # 兩條參考水平，角色不同：
    #   1.0  —— 不帶假設的硬地標（輸出全零的分數恰為 1，恆成立）；越過 = 比不預測還差。
    #   √2   —— 理想化的去相關漸近線（前提見 UNCORRELATED_SQRT2）；看後段平台是否到頂。
    # 量測到的 decorrelation level 刻意不畫：它就是 √2 的一個有散佈的估計，
    # 三條線疊在一起只會讓圖更難讀。數值仍落在 JSON 供對照。
    ax.axhline(1.0, lw=0.9, color="0.55", zorder=1)
    ax.axhline(UNCORRELATED_SQRT2, lw=0.9, ls=(0, (4, 3)), color="0.55", zorder=1)
    # 標籤放左段：τ 小時所有曲線都遠低於 1.0，那裡是唯一不會被壓到的空白
    ax.text(0.15, 1.012, "zero prediction", ha="left", va="bottom",
            fontsize=6.5, color="0.4")
    ax.text(0.15, UNCORRELATED_SQRT2 + 0.012, r"uncorrelated ($\sqrt{2}$)",
            ha="left", va="bottom", fontsize=6.5, color="0.4")
    ax.axvspan(tau.min(), 0.0, color="0.92", zorder=0)
    ax.axvline(0.0, lw=0.8, color="0.55", zorder=2)
    ax.set_xlabel(r"Lead time $\tau = t - t_{\mathrm{data}}$"
                  "\n"
                  r"($\tau<0$: sensor data available   |   $\tau>0$: extrapolation)")
    ax.set_ylabel("Relative $L_2$ error of $(u,v)$")
    ax.set_xlim(tau.min(), tau.max())
    ax.set_ylim(top=max(1.55, float(np.nanmax(err_model)) * 1.08))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_figure(fig, str(out_dir / "extrapolation_leadtime"))

    print("=== extrapolation lead-time ===")
    print(f"in-window mean ({args.metric}) : {report['in_window_mean']}")
    header = f"  {'tau':>5}  {'model':>8}  {'pers(truth)':>11}"
    if mp_on_model is not None:
        header += f"  {'pers(model)':>11}"
    if budget_on_model is not None:
        header += f"  {'budget':>8}  {'ratio':>6}"
    print(header)
    for lt, row in report["at_lead_times"].items():
        line = f"  {lt:>5}  {row['model']:8.4f}  {row['persistence_oracle']:11.4f}"
        if "persistence_model" in row:
            line += f"  {row['persistence_model']:11.4f}"
        if "chaos_budget" in row:
            # 兩欄各自成立：ratio 為 None 時只有它顯示 n/a，budget 欄不能跟著消失
            # （原本 `line += A if ratio else ""` 把兩欄綁在一起，表頭仍在 → 欄位錯位）
            ratio = row["model_over_budget"]
            line += f"  {row['chaos_budget']:8.4f}  "
            line += f"{ratio:6.2f}×" if ratio else f"{'n/a':>6}"
        print(line)
    print(f"useful horizon vs pers(model): "
          f"{report['useful_horizon_vs_persistence_model']}   ← 主判準")
    print(f"tau where model > zero pred  : {report['tau_model_exceeds_zero_prediction']}")
    print(f"[note] uncorrelated asymptote √2 = {UNCORRELATED_SQRT2:.4f}（推導）；"
          f"量測 decorrelation level {floor_mean:.4f} ± {floor_std:.4f} "
          f"({n_pairs} pairs)。兩者互相印證，皆非判準")
    print(f"[out] {out_dir / 'extrapolation_leadtime.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
