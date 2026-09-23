#!/usr/bin/env python3
"""diag_subspace_transfer.py — LES-POD vs DNS-POD subspace transfer（baseline-controlled）。

What:
    量化「LES surrogate → DNS target」的 POD 子空間落差，但**對照 realization 基準線**。
    在共同窗 [t_spinup, t_max] 內把每個資料源切兩半，分別建 leading-r POD 基，計算：
      cross    = Θ(Φ_r^{LES,h1}, Φ_r^{DNS,h1})   跨源（matched 半窗）
      self_LES = Θ(Φ_r^{LES,h1}, Φ_r^{LES,h2})   同源基準（LES 兩半）
      self_DNS = Θ(Φ_r^{DNS,h1}, Φ_r^{DNS,h2})   同源基準（DNS 兩半）
    報 excess = cross_median − max(self_*_median)。

Why（這支為何不報裸角度）:
    本流（2D Kolmogorov）在 x 方向統計均勻，POD 模態是 (近)簡併 cos/sin Fourier 對 +
    反級串下一堆能量相近的大尺度模 ⇒ 有限時窗的 leading-r POD 子空間是 realization-specific
    的。實測：DNS 自己切兩半的 principal angle（~58°@r=12）與 LES-vs-DNS（~59°）幾乎相同。
    因此**裸的 cross-source 角度會把 realization variance 誤讀為 surrogate gap**。唯一有意義
    的量是 cross 相對同源基準的 excess：excess≈0 ⇒ LES 當替身不比「DNS 換個時窗」更差。
    定位：oracle transfer 旁證（placement 仍 DNS-free，見 diag_observability）。
    互補指標（realization-robust，主證據）：effective rank 與 κ(CΦ_r)，見 diag_observability。

Usage:
    PYTHONPATH=. uv run python scripts/diag_subspace_transfer.py \\
      --les /path/to/les_T50.npy --dns data/dns/..._T20_..._seed42.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path
from scripts.diag_observability import (
    effective_rank_thresholds,
    pod_basis,
    principal_angles,
    split_time_halves,
    transfer_verdict,
)

_DEFAULT_LES = _resolve_data_path("data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy")
_DEFAULT_DNS = "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_dt1p95e4_si128_seed42.npy"
_EXCESS_TOL_DEG = 5.0  # excess 小於此 → 視為與 realization 基準無法區分（無額外 surrogate gap）


def _load_split(path: Path, t_spinup: float, t_max: float | None, stride: int):
    """載 {time,u,v}，依共同窗切兩半，回傳 ((u1,v1),(u2,v2), N, T1, T2)。"""
    raw = np.load(path, allow_pickle=True).item()
    t = np.asarray(raw["time"], dtype=np.float64)
    m1, m2 = split_time_halves(t, t_spinup, t_max)
    u = np.asarray(raw["u"], dtype=np.float64)
    v = np.asarray(raw["v"], dtype=np.float64)
    h1 = (u[m1][::stride], v[m1][::stride])
    h2 = (u[m2][::stride], v[m2][::stride])
    return h1, h2, u.shape[-1], int(h1[0].shape[0]), int(h2[0].shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", default=str(_DEFAULT_LES), help="LES 全場 .npy（DNS-free surrogate）")
    ap.add_argument("--dns", default=_DEFAULT_DNS, help="DNS 全場 .npy（oracle target）")
    ap.add_argument("--t-spinup", type=float, default=0.0, help="共同窗下界（秒）")
    ap.add_argument("--t-max", type=float, default=20.0, help="共同窗上界（秒；對齊 deployed sensors t0-20）")
    ap.add_argument("--time-stride", type=int, default=1, help="時間抽樣間隔")
    ap.add_argument("--r-list", default="12,28,68", help="比較的 leading-r")
    ap.add_argument("--out", default="artifacts/diag/subspace_transfer.json")
    args = ap.parse_args()

    les_path, dns_path = Path(args.les), Path(args.dns)
    for tag, p in (("LES", les_path), ("DNS", dns_path)):
        if not p.exists():
            raise FileNotFoundError(f"{tag} 全場不存在: {p}（設 PILNJAX_DATA_ROOT 或 --{tag.lower()}）")
    r_list = [int(x) for x in args.r_list.split(",")]
    print(f"[window] t∈[{args.t_spinup}, {args.t_max}] split→halves  stride={args.time_stride}  r_list={r_list}")

    les_h1, les_h2, N_l, Tl1, Tl2 = _load_split(les_path, args.t_spinup, args.t_max, args.time_stride)
    dns_h1, dns_h2, N_d, Td1, Td2 = _load_split(dns_path, args.t_spinup, args.t_max, args.time_stride)
    if N_l != N_d:
        raise ValueError(f"grid 不一致：LES N={N_l} vs DNS N={N_d}（principal angles 需同 ambient grid）")
    N = N_l
    print(f"[LES] {les_path.name}  halves T={Tl1}/{Tl2}")
    print(f"[DNS] {dns_path.name}  halves T={Td1}/{Td2}")

    r_max = min(max(r_list), Tl1, Tl2, Td1, Td2, 2 * N * N)
    # 4 個半窗各自的 leading-r POD（matched 樣本量，公平對照）
    Pl1, svl1 = pod_basis(*les_h1, r_max)
    Pl2, _ = pod_basis(*les_h2, r_max)
    Pd1, svd1 = pod_basis(*dns_h1, r_max)
    Pd2, _ = pod_basis(*dns_h2, r_max)
    er_l = effective_rank_thresholds(svl1, (0.95, 0.99, 0.999))
    er_d = effective_rank_thresholds(svd1, (0.95, 0.99, 0.999))
    print(f"[eff rank @ half-window]  LES r_95/99/99.9={er_l[0.95]}/{er_l[0.99]}/{er_l[0.999]}  "
          f"DNS={er_d[0.95]}/{er_d[0.99]}/{er_d[0.999]}")
    # rank 護欄：r 超過兩源 leading energetic rank 較小者 → 比較進入 noise floor
    rank_floor = min(er_l[0.999] or r_max, er_d[0.999] or r_max)
    print(f"[rank floor] r≤{rank_floor} 才是乾淨比較（min 兩源 r_99.9）")

    print("\n=== principal angles (median°)  cross vs realization baseline ===")
    print(f"  {'r':>5} {'cross(LES-DNS)':>14} {'self_LES':>9} {'self_DNS':>9} {'excess':>8}  verdict")
    per_r = []
    for r in r_list:
        if r > r_max:
            print(f"  {r:>5}  [skip] r>r_max={r_max}")
            continue
        cross = principal_angles(Pl1[:, :r], Pd1[:, :r])["theta_median_deg"]
        self_les = principal_angles(Pl1[:, :r], Pl2[:, :r])["theta_median_deg"]
        self_dns = principal_angles(Pd1[:, :r], Pd2[:, :r])["theta_median_deg"]
        excess = cross - max(self_les, self_dns)  # cross 相對較保守的同源基準
        verdict = transfer_verdict(r, excess, rank_floor, _EXCESS_TOL_DEG)
        per_r.append({"r": r, "cross_median_deg": cross, "self_les_median_deg": self_les,
                      "self_dns_median_deg": self_dns, "excess_deg": excess,
                      "rank_floor": rank_floor, "verdict": verdict})
        print(f"  {r:>5} {cross:>13.2f}° {self_les:>8.2f}° {self_dns:>8.2f}° {excess:>+7.2f}°  {verdict}")

    report = {
        "les_file": str(les_path), "dns_file": str(dns_path),
        "window": [args.t_spinup, args.t_max], "time_stride": args.time_stride,
        "method": "baseline-controlled principal angles on matched half-windows",
        "excess_tol_deg": _EXCESS_TOL_DEG,
        "N": N, "T_les_halves": [Tl1, Tl2], "T_dns_halves": [Td1, Td2],
        "effective_rank_les_halfwindow": {str(k): v for k, v in er_l.items()},
        "effective_rank_dns_halfwindow": {str(k): v for k, v in er_d.items()},
        "per_r": per_r,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
