#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pywavelets"]
# ///
"""appendix06 的 wavelet 稀疏度診斷 — Gini 係數與 99%-能量係數數。

Why this exists:
    該表的三組數字（Gini、n99、CS bound 的 s）原本無生成碼——`claims-ledger` 的
    evidence 欄指回 .tex 本身，全庫無 wavelet 程式，`POC_RESULTS.md` 直接記載
    「Wavelet analysis 未做」。它是 claim「三個單快照界設定 K=100 解析度尺度」
    的證據之一，卻無法重現。本腳本補上生成端。

    重現狀況（2026-09-01）：以 db4 / level 5 / symmetric、評估窗中段那一幀重算，
    **Gini 三個值與論文逐位相同**（0.983 / 0.985 / 0.942），但 n99 對不上
    （得 357/332/2070，論文 326/330/1917）。掃過 periodization 模式、level 3/4/6、
    |c| 累積、僅 detail 係數等定義，皆無法同時滿足兩者。論文那組 n99 很可能產於
    PyTorch 世代的另一份 DNS。因此表中數字改用本腳本輸出，並記錄完整 provenance。

Why pywavelets is declared inline:
    只有這支診斷需要它。用 PEP 723 inline metadata 讓 `uv run` 自行處理，
    不動 pyproject／uv.lock——後者在本專案會牽動 GPU venv 的 out-of-band wheels
    （見 CLAUDE.md §6）。

Usage:
    uv run scripts/diag_wavelet_sparsity.py \
        --dns-path data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
        --output artifacts/diag/wavelet_sparsity_re10000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pywt

FIELDS = ("u", "v", "omega")


def gini(x: np.ndarray) -> float:
    """Gini 係數，0 = 均勻、1 = 全部能量集中在一個係數。

    以升序排列的 |c| 計算（標準定義）；對有號的小波係數必須取絕對值,
    否則負係數會讓積分失去單調性。
    """
    a = np.sort(np.abs(x).ravel())
    n = a.size
    total = a.sum()
    if total <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * total))


def n_energy(x: np.ndarray, frac: float) -> int:
    """達到 `frac` 能量所需的最少係數數（降序累積 c²）。"""
    e = np.sort(np.abs(x).ravel() ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)


def flat_coeffs(field: np.ndarray, wavelet: str, level: int, mode: str) -> np.ndarray:
    """2D 多階分解後攤平成一維係數向量（approximation + 全部 detail）。"""
    C = pywt.wavedec2(field, wavelet, level=level, mode=mode)
    return np.concatenate([C[0].ravel()] + [a.ravel() for lv in C[1:] for a in lv])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns-path", required=True)
    ap.add_argument("--frame", default="mid",
                    help="'mid'（評估窗中段，預設）、'all'（逐幀並報 mean±sd），或整數索引")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--mode", default="symmetric",
                    help="pywt 邊界模式。symmetric（預設）會因延拓使係數數 > N²；"
                         "periodization 使係數數恰為 N²，但在本場上與論文的 Gini 不符")
    ap.add_argument("--energy-frac", type=float, default=0.99)
    ap.add_argument("--output", default="artifacts/diag/wavelet_sparsity.json")
    a = ap.parse_args()

    raw = np.load(a.dns_path, allow_pickle=True)
    d = raw.item() if raw.dtype == object else raw
    fields = {k: np.asarray(d[k], dtype=np.float64) for k in FIELDS}
    times = np.asarray(d["time"], dtype=np.float64)
    T, N, _ = fields["u"].shape

    if a.frame == "mid":
        idx = [T // 2]
    elif a.frame == "all":
        idx = list(range(T))
    else:
        idx = [int(a.frame)]

    per_field: dict[str, dict] = {}
    for name, arr in fields.items():
        g = [gini(flat_coeffs(arr[i], a.wavelet, a.level, a.mode)) for i in idx]
        s = [n_energy(flat_coeffs(arr[i], a.wavelet, a.level, a.mode), a.energy_frac) for i in idx]
        per_field[name] = {
            "gini_mean": float(np.mean(g)),
            "gini_sd": float(np.std(g, ddof=1)) if len(g) > 1 else None,
            "n_energy_mean": float(np.mean(s)),
            "n_energy_sd": float(np.std(s, ddof=1)) if len(s) > 1 else None,
            "n_energy_min": int(np.min(s)),
            "n_energy_max": int(np.max(s)),
        }

    n_coeffs = int(flat_coeffs(fields["u"][idx[0]], a.wavelet, a.level, a.mode).size)
    out = {
        "provenance": {
            "script": "scripts/diag_wavelet_sparsity.py",
            "dns_path": str(Path(a.dns_path).resolve()),
            "grid": N,
            "n_stored_frames": T,
            "frames_used": [int(i) for i in idx],
            "times_used": [float(times[i]) for i in idx],
            "wavelet": a.wavelet,
            "level": a.level,
            "boundary_mode": a.mode,
            "energy_fraction": a.energy_frac,
            "n_coefficients": n_coeffs,
            "note": ("symmetric 模式因邊界延拓使係數數大於 N²；論文表中的 '65,536' "
                     "指的是網格點數 N²，不是係數數"),
        },
        "fields": per_field,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"[wavelet] {a.wavelet} level={a.level} mode={a.mode}  "
          f"grid={N}²  coeffs={n_coeffs}  frames={len(idx)}")
    for name, r in per_field.items():
        sd_g = "" if r["gini_sd"] is None else f" ± {r['gini_sd']:.4f}"
        sd_s = "" if r["n_energy_sd"] is None else f" ± {r['n_energy_sd']:.0f}"
        print(f"  {name:6s} Gini = {r['gini_mean']:.4f}{sd_g}   "
              f"n_{int(a.energy_frac*100)} = {r['n_energy_mean']:.0f}{sd_s}"
              f"  [{r['n_energy_min']}, {r['n_energy_max']}]")
    print(f"[out] {op}")


if __name__ == "__main__":
    main()
