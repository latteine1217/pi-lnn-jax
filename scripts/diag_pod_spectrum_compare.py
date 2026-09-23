#!/usr/bin/env python3
"""重建場與 DNS 場的 POD 譜對照：跨時間有效秩。

What:
    對**同一組幀**的 DNS 場與模型重建場各做一次 economy SVD，
    比較有效秩 r_lv（累積能量達 lv 所需的模態數）。

Why:
    `diag_dns_pod_rank.py` 已經量出 DNS 場的有效秩（r_99=25、r_99.9=53）。
    那是**流場可用的自由度上限**——Gao et al. 2017 的 `D <= min(M, NTC)` 裡的 NTC。
    這支腳本補上另一半：**模型重建場實際用掉了幾個自由度**。

    判別觀測：
      r_pred ≈ r_dns  → 算子內化了那個低維結構，殘差來自別處
      r_pred ≪ r_dns  → 算子沒有內化（與 4D-Var 那條、POD-margin 那條方法上獨立）
      r_pred ≫ r_dns  → 重建帶進了 DNS 沒有的自由度（雜訊或偽結構）

⚠️ **r_99.99 是快照數敏感的，不要拿它當基準。** 同一份 DNS 在 201 幀下 r_99.99=92、
   101 幀下 =82，而 r_90/r_99/r_99.9 (9/25/53) 兩者完全相同。要比就比後三個。

## 試過而移除的兩個讀數（2026-09-20，不要再加回來）

  * **participation ratio `(Σλ)²/Σλ²`**：實測 pred/dns = 0.97 而同一批資料的
    r_99.9 差 25%。PR 被能量主導的少數模態綁死，而本問題丟失的自由度全在尾部，
    它結構性地看不見。
  * **冪律指數 α（Stringer et al. 2019 的可微性下界 1+2/d）**：擬合只跨 1.40 個
    數量級，低於該文穩健所需的約 2 個。且 POD 譜的模態數受快照數限制
    （r_99.99 已是 82/101），在本專案的資料規模下**結構性地**達不到穩健條件。
    另有兩個未解的範疇問題：d（刺激空間內在維度）未定，且該文量的是跨獨立刺激
    的共變異譜、這裡是跨時間的 POD 譜，時間軸有動力學相關性、不同構。

  兩者的實測數字記在 `knowledge/experiments/kolmogorov-pod-rank-latent-2026-09-20.md`。

Usage:
    uv run python scripts/diag_pod_spectrum_compare.py \
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \
        --pred artifacts/kolmogorov/main5_s42/final_eval/fields.npz \
        --time-stride 2 --out artifacts/diag/pod_compare_main5_s42.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import load_dns_from_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_dns_pod_rank import ENERGY_LEVELS, pod_energy_fractions  # noqa: E402

# 時間軸吻合的容差。DNS 的 t 由 solver 輸出、pred 的 t 由 eval 寫入，
# 兩者都是 float32/64 的整數倍時刻，1e-6 足夠寬又不會放過真正的錯位。
T_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dns", required=True, help="DNS .npy 路徑（相對路徑走 PILNJAX_DATA_ROOT）")
    p.add_argument("--pred", required=True, help="eval 產物的 fields.npz（含 u_pred/v_pred/t）")
    p.add_argument("--time-stride", type=int, required=True,
                   help="DNS 的時間下採樣，用來對齊 pred 的幀。**必填無預設**："
                        "POD 秩直接受快照數限制，吃預設等於讓取樣密度靜默決定結論")
    p.add_argument("--out", required=True, help="JSON 產物落點")
    p.add_argument("--subtract-mean", choices=("yes", "no"), default="yes",
                   help="POD 取脈動（減時間平均）或原場。預設 yes：Kolmogorov 有強迫出來的"
                        "平均剖面，不減會把它算成第一個模態")
    return p.parse_args()


def spectrum_of(u: np.ndarray, v: np.ndarray, subtract_mean: bool) -> np.ndarray:
    """回傳 POD 能量特徵值 λ_n（降序）。與 pod_energy_fractions 同一套前處理。"""
    x = np.concatenate([u.reshape(u.shape[0], -1), v.reshape(v.shape[0], -1)], axis=1)
    if subtract_mean:
        x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    return np.square(s.astype(np.float64))


def load_pred(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    d = np.load(path)
    missing = [k for k in ("u_pred", "v_pred", "t") if k not in d.files]
    if missing:
        raise SystemExit(f"[fail] {path} 缺少必要欄位 {missing}；不猜、不補預設")
    dns_path = str(d["dns_path"]) if "dns_path" in d.files else ""
    return np.asarray(d["u_pred"]), np.asarray(d["v_pred"]), np.asarray(d["t"]), dns_path


def assert_aligned(t_dns: np.ndarray, t_pred: np.ndarray,
                   shape_dns: tuple, shape_pred: tuple) -> None:
    """時間軸與網格必須逐點吻合，不符就停。

    eval 腳本最常見的靜默錯誤就是在這裡截短或內插。寧可要求呼叫端把 --time-stride
    調對，也不要自作主張對齊出一個看起來合理的數字。
    """
    if shape_dns[1:] != shape_pred[1:]:
        raise SystemExit(f"[fail] 網格不符：dns {shape_dns[1:]} vs pred {shape_pred[1:]}")
    if len(t_dns) != len(t_pred):
        raise SystemExit(
            f"[fail] 幀數不符：dns {len(t_dns)} vs pred {len(t_pred)}。"
            f"請調整 --time-stride（目前的 stride 讓 DNS 取出 {len(t_dns)} 幀）。"
            f"**本腳本不做截短或內插**——POD 秩受快照數限制，靜默對齊會直接改掉結論。")
    dt = np.abs(np.asarray(t_dns, dtype=np.float64) - np.asarray(t_pred, dtype=np.float64))
    if dt.max() > T_ATOL:
        bad = int(np.argmax(dt))
        raise SystemExit(
            f"[fail] 時間戳不吻合：最大偏差 {dt.max():.3e} 在第 {bad} 幀 "
            f"(dns {t_dns[bad]:.6f} vs pred {t_pred[bad]:.6f})，容差 {T_ATOL:.0e}")


def rank_at(lam: np.ndarray, levels) -> dict:
    cum = np.cumsum(lam) / lam.sum()
    return {str(lv): int(np.searchsorted(cum, lv) + 1) for lv in levels}


def main() -> None:
    a = parse_args()
    sub = a.subtract_mean == "yes"

    u_d, v_d, t_d = load_dns_from_path(a.dns, time_stride=a.time_stride)
    u_p, v_p, t_p, declared_dns = load_pred(a.pred)

    print(f"[dns ] {a.dns}  stride={a.time_stride}  frames={u_d.shape[0]}  "
          f"grid={u_d.shape[1]}x{u_d.shape[2]}  t=[{t_d[0]:.3f}, {t_d[-1]:.3f}]")
    print(f"[pred] {a.pred}  frames={u_p.shape[0]}  t=[{t_p[0]:.3f}, {t_p[-1]:.3f}]")
    if declared_dns and Path(declared_dns).name != Path(a.dns).name:
        print(f"[warn] pred 宣告的 DNS 是 {declared_dns}，與 --dns 不同檔名。"
              f"若非刻意，這代表在拿錯的參考場比對。")

    assert_aligned(t_d, t_p, u_d.shape, u_p.shape)
    print(f"[ok  ] 時間軸與網格逐點吻合（{len(t_d)} 幀）")

    out = {"provenance": {}, "dns": {}, "pred": {}, "comparison": {}}
    for tag, (u, v) in (("dns", (u_d, v_d)), ("pred", (u_p, v_p))):
        lam = spectrum_of(u, v, sub)
        frac = pod_energy_fractions(u, v, subtract_mean=sub)
        n_snap = int(u.shape[0])
        r = frac["rank_at"]
        out[tag] = {
            "n_snapshots": n_snap,
            "rank_at": r,
            # 快照數敏感旗標：逼近上限的 level 不得當成流場/重建的性質
            "rank_near_snapshot_limit": {k: bool(int(val) >= 0.8 * n_snap) for k, val in r.items()},
        }
        print(f"[{tag:4s}] " + "  ".join(f"r_{float(lv)*100:g}={r[str(lv)]}" for lv in ENERGY_LEVELS))

    rd, rp = out["dns"]["rank_at"], out["pred"]["rank_at"]
    out["comparison"] = {
        "rank_ratio": {k: (float(rp[k]) / float(rd[k]) if float(rd[k]) else None) for k in rd},
    }

    print("\n[cmp ] rank_pred / rank_dns: " +
          "  ".join(f"r_{float(k)*100:g}={out['comparison']['rank_ratio'][k]:.3f}"
                    for k in rd if out['comparison']['rank_ratio'][k] is not None))
    for tag in ("dns", "pred"):
        near = [k for k, vv in out[tag]["rank_near_snapshot_limit"].items() if vv]
        if near:
            print(f"[warn] {tag} 的 r_{'/'.join(f'{float(k)*100:g}' for k in near)} "
                  f"已達快照數的 80%，量到的可能是取樣不足而非結構秩，不要當基準。")

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance 缺失不該讓診斷失敗
        rev = ""
    out["provenance"] = {
        "dns": a.dns, "pred": a.pred, "time_stride": a.time_stride,
        "subtract_mean": a.subtract_mean, "git_head": rev,
        "pred_declared_dns": declared_dns,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
