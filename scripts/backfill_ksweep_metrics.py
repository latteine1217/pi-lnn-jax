#!/usr/bin/env python3
"""backfill_ksweep_metrics.py — backfill ke_t_errors + spectrum_rel_err 進舊版 K-sweep metrics.json。

What:
    K=50/100 的 final_eval/metrics.json 用較舊的 evaluate.py 跑，top-level `ke_t_errors` 為 null、
    且 metrics_mean / metrics_per_t 缺 `spectrum_rel_err`（K=200/400/800 才有）。本腳本從各檔既有的
    metrics_per_t（ke_pred_mean/ke_dns_mean = 逐幀 E(t)、E_pred_k/E_dns_k = 逐幀能譜）離線重算這些
    欄位，公式逐字沿用 pi_lnn_jax.evaluate.energy_timeseries_errors / spectrum_rel_error（純 python
    實作、無 numpy 依賴，便於在 head node 直接跑）。

Why:
    讓 K-sweep 5 點 metrics.json 欄位齊一、可重現，不再靠繪圖腳本臨時離線算 headline ke_t_mape。

Safety（eval 嚴格度，CLAUDE.md）:
    1. gate：先用已含欄位的參考檔（K=200/400/800）驗證「重算 == stored」（rtol 1e-9），任一不過即
       abort（公式漂移保護）；無任何參考可驗也 abort。
    2. 目標檔缺 E_pred_k/E_dns_k → fail-fast；已備齊欄位 → skip；都不靜默覆蓋。
    3. 寫入前 shutil.copy2 備份 .bak（真原始 bytes）。DRY=1（預設）只報告，DRY=0 才寫。

Run（lab-server，純 stdlib）:
    ssh lab-server "DRY=1 python3 -" < scripts/backfill_ksweep_metrics.py   # 預演
    ssh lab-server "DRY=0 python3 -" < scripts/backfill_ksweep_metrics.py   # 套用
"""
import glob
import json
import math
import os
import shutil


def energy_timeseries_errors(ep, et, eps: float = 1e-12) -> dict:
    """E(t) 時序相對誤差（沿用 pi_lnn_jax.evaluate 同名函式公式）。"""
    n = len(ep)
    return {
        "ke_t_mape": sum(abs(ep[i] - et[i]) / (abs(et[i]) + eps) for i in range(n)) / n,
        "ke_t_rel_l2": math.sqrt(sum((ep[i] - et[i]) ** 2 for i in range(n)))
        / (math.sqrt(sum(e * e for e in et)) + eps),
        "ke_t_rel_linf": max(abs(ep[i] - et[i]) for i in range(n)) / (max(abs(e) for e in et) + eps),
        "ke_t_final_rel": abs(ep[-1] - et[-1]) / (abs(et[-1]) + eps),
    }


def spectrum_rel_error(ep, ed, eps: float = 1e-12) -> float:
    """E(k) 全 k-shell relative-L2（沿用 pi_lnn_jax.evaluate 同名函式公式）。"""
    if len(ep) != len(ed) or len(ep) == 0:
        raise ValueError(f"E(k) 長度不符或為空: pred={len(ep)} dns={len(ed)}")
    return math.sqrt(sum((ep[i] - ed[i]) ** 2 for i in range(len(ep)))) / (
        math.sqrt(sum(e * e for e in ed)) + eps
    )


def recompute(d: dict):
    """從 metrics_per_t 重算 (ke_t_errors, 逐幀 spectrum_rel_err, spectrum 均值)。"""
    pt = d["metrics_per_t"]
    ep = [m["ke_pred_mean"] for m in pt]  # == ke_t_pred（0.5<u^2+v^2>_space, mask=None）
    et = [m["ke_dns_mean"] for m in pt]
    kte = energy_timeseries_errors(ep, et)
    spec_pt = [spectrum_rel_error(m["E_pred_k"], m["E_dns_k"]) for m in pt]
    return kte, spec_pt, sum(spec_pt) / len(spec_pt)


def close(a, b, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def main() -> None:
    H = os.path.expanduser("~")
    ref = {
        200: H + "/pi-lnn-jax-multire/artifacts/kolmogorov/exp245k200_b3_les_T50_jax_20k/final_eval/metrics.json",
        400: H + "/pi-lnn-jax-multire/artifacts/kolmogorov/exp245k400_b3_les_T50_jax_20k/final_eval/metrics.json",
        800: H + "/pi-lnn-jax-multire/artifacts/kolmogorov/exp245k800_b3_les_T50_jax_20k/final_eval/metrics.json",
    }
    target = {
        50: glob.glob(H + "/pi-lnn-jax*/artifacts/kolmogorov/exp245k50_*/final_eval/metrics.json"),
        100: [H + "/pi-lnn-jax/artifacts/kolmogorov/exp245_b3_les_T50_jax_20k/final_eval/metrics.json"],
    }
    dry = os.environ.get("DRY", "1") != "0"

    print("=== VERIFY formula vs stored (K=200/400/800) ===")
    nref = 0
    for K in sorted(ref):
        p = ref[K]
        if not os.path.exists(p):
            print(f"K={K}: ref not found, skip")
            continue
        d = json.load(open(p))
        s_kte = d.get("ke_t_errors")
        s_spec = d.get("metrics_mean", {}).get("spectrum_rel_err")
        if not s_kte or s_spec is None:
            print(f"K={K}: ref lacks stored fields, skip")
            continue
        kte, spec_pt, spec_mean = recompute(d)
        # 2026-08-01 起 `ke_t_mape` 在儲存的 ke_t_errors 裡改指 **pointwise** 定義，
        # 而本檔重算的是空間平均版——那個量現在存在 `ke_t_mape_spatialmean`。
        # 直接同名比對會拿兩個不同的量互比，然後以 "formula mismatch" 中止：
        # 看起來像公式壞了，實際只是鍵名換了。這裡做顯式映射。
        stored_key = {"ke_t_mape": "ke_t_mape_spatialmean"}
        missing = [k for k in kte if stored_key.get(k, k) not in s_kte]
        if missing:
            raise SystemExit(
                f"ABORT: K={K} 的 ke_t_errors 缺 {missing}（實有 {sorted(s_kte)}）"
                "——參考檔的欄位集與本檔預期不符，不得當作通過")
        ok = (all(close(kte[k], s_kte[stored_key.get(k, k)]) for k in kte)
              and close(spec_mean, s_spec))
        print(
            f"K={K}: ke_t_mape(spatial-mean) {kte['ke_t_mape']:.6f} vs "
            f"{s_kte[stored_key['ke_t_mape']]:.6f} | "
            f"spec_mean {spec_mean:.6f} vs {s_spec:.6f} | {'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            raise SystemExit(f"ABORT: formula mismatch at K={K}")
        if "spectrum_rel_err" in d["metrics_per_t"][0]:
            dmax = max(abs(spec_pt[i] - d["metrics_per_t"][i]["spectrum_rel_err"]) for i in range(len(spec_pt)))
            print(f"      per-frame spectrum max|delta| = {dmax:.2e}")
            if dmax > 1e-9:
                raise SystemExit(f"ABORT: per-frame spectrum mismatch K={K}")
        nref += 1
    if nref == 0:
        raise SystemExit("ABORT: no reference file validated the formula")
    print(f"verify PASSED on {nref} reference file(s)\n")

    print(f"=== BACKFILL (DRY={dry}) ===")
    for K in sorted(target):
        paths = [p for p in target[K] if os.path.exists(p)]
        if not paths:
            raise SystemExit(f"ABORT: K={K} metrics.json not found")
        p = paths[0]
        d = json.load(open(p))
        has_kte = bool(d.get("ke_t_errors"))
        has_spec = d.get("metrics_mean", {}).get("spectrum_rel_err") is not None
        if has_kte and has_spec:
            print(f"K={K}: already complete, SKIP ({p.replace(H, '~')})")
            continue
        pt = d["metrics_per_t"]
        if not all(("E_pred_k" in m and "E_dns_k" in m) for m in pt):
            raise SystemExit(f"ABORT: K={K} per_t missing E_pred_k/E_dns_k, cannot recompute spectrum")
        kte, spec_pt, spec_mean = recompute(d)
        print(
            f"K={K}: ke_t_mape={kte['ke_t_mape']:.6f} rel_l2={kte['ke_t_rel_l2']:.6f} "
            f"linf={kte['ke_t_rel_linf']:.6f} final={kte['ke_t_final_rel']:.6f} | "
            f"spectrum_rel_err(mean)={spec_mean:.6f}  [{p.replace(H, '~')}]"
        )
        if dry:
            print("      DRY: not written")
            continue
        bak = p + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        # 用庫現行的鍵名寫入。本檔算得出的只有空間平均版；headline `ke_t_mape`
        # 自 2026-08-01 起指 pointwise，需要逐點 KE 場，而這裡只有 metrics_per_t
        # 的聚合量，算不出來。
        #
        # 所以**刻意不寫** `ke_t_mape`：寫一個名字與語意不符的鍵，會讓 backfill 過的
        # 檔案與新評估的檔案在同一個 K-sweep 裡混著兩種語意，而讀取端
        # （plot_ksweep.py:32 就是零檢查直接索引）分辨不出來。
        # 缺鍵會讓下游 KeyError——大聲失敗，勝過安靜給錯數字。
        d["ke_t_errors"] = {
            "ke_t_mape_spatialmean" if k == "ke_t_mape" else k: v
            for k, v in kte.items()
        }
        d["ke_t_errors"]["ke_mape_def"] = "spatialmean_only（backfill 算不出 pointwise）"
        for i, m in enumerate(pt):
            m["spectrum_rel_err"] = spec_pt[i]
        d.setdefault("metrics_mean", {})["spectrum_rel_err"] = spec_mean
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
        print(f"      WROTE (backup {bak.replace(H, '~')})")
    print("done")


if __name__ == "__main__":
    main()
