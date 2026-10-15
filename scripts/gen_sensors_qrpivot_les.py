"""gen_sensors_qrpivot_les.py — 部署用佈點：對 LES 快照的特徵做 column-pivoted QR，取值自 DNS。

What:
    從 LES 快照（陣列慣例 [T, x, y]，排除 t < t_spinup）建五個特徵：u、v、
    渦度 ω = ∂v/∂x − ∂u/∂y、|∇u|、|∇v|（導數皆為週期二階中央差分）。各特徵除以自身
    全域標準差後堆疊成 [5T, N²]，column-pivoted QR 給出一條網格點排名；每個預算 K 取
    前 K 個（各預算巢狀）。座標換到 DNS 網格的最近點，再從 DNS 抽出該處 u、v 時序，
    輸出與既有 sensor 檔同 schema 的 JSON + NPZ。

Why:
    工程現場沒有 DNS，佈點只能由 LES 大尺度估計決定（DNS-free）；DNS 只扮演「真實量測」。
    取代 pi-lnn 的 `generate_sensors_qrpivot_from_les.py`：那支假設陣列為 [T, y, x]，在
    [T, x, y] 資料上算出的 "omega" 實際是 ∂v/∂y − ∂u/∂x。本版的導數與 oracle 佈點、
    壓力評估共用 `_common.periodic_fields`，QR 的正規化與呼叫沿用
    `gen_sensors_qrpivot_dns.rank_grid_points`（兩支佈點只差特徵集與快照來源）。
    LES 場維持 float32 讀入：QR 的 pivot 在近似平手處對捨入敏感，與既有檔對拍時要同精度。

Usage:
    uv run python scripts/gen_sensors_qrpivot_les.py \\
        --les data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy \\
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --budgets 50 100 200 400 800 --t-spinup 20.0 \\
        --out-dir data/kolmogorov_sensors/re10000 --tag N256_t0-5_si100_les_n256_T50standalone
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from pi_lnn_jax.data import _resolve_data_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common.periodic_fields import periodic_gradient
from gen_sensors_qrpivot_dns import check_axis_convention, rank_grid_points

FEATURES = ("u", "v", "omega", "grad_u_mag", "grad_v_mag")
FEATURE_DEFINITIONS = {
    "u": "x 向速度",
    "v": "y 向速度",
    "omega": "渦度 ∂v/∂x − ∂u/∂y，週期二階中央差分",
    "grad_u_mag": "|∇u|，週期二階中央差分",
    "grad_v_mag": "|∇v|，週期二階中央差分",
}


def les_features(u: np.ndarray, v: np.ndarray, L: float) -> dict[str, np.ndarray]:
    """五個特徵，形狀皆為 [T, N, N]，順序即 FEATURES。"""
    gx_u, gy_u = periodic_gradient(u, L)
    gx_v, gy_v = periodic_gradient(v, L)
    return {
        "u": u,
        "v": v,
        "omega": gx_v - gy_u,
        "grad_u_mag": np.sqrt(gx_u ** 2 + gy_u ** 2),
        "grad_v_mag": np.sqrt(gx_v ** 2 + gy_v ** 2),
    }


def load_les(path: Path, t_spinup: float) -> dict:
    """讀 LES（dict：time/u/v/config），排除 t < t_spinup；u、v 以 float32 回傳。"""
    raw = np.load(path, allow_pickle=True).item()
    t = np.asarray(raw["time"], dtype=np.float64)
    keep = t >= t_spinup
    if keep.sum() < 5:
        raise ValueError(f"排除 spin-up 後只剩 {keep.sum()} 幀（t_spinup={t_spinup}, T_end={t[-1]:.2f}）")
    cfg = dict(raw.get("config", {}))
    u = np.asarray(raw["u"], dtype=np.float32)[keep]
    if u.shape[-1] != u.shape[-2]:
        raise ValueError(f"LES 須為方形網格 [T, N, N]，收到 {u.shape}")
    return {"u": u, "v": np.asarray(raw["v"], dtype=np.float32)[keep], "time": t[keep],
            "L": float(cfg.get("L", 1.0)), "N": u.shape[-1]}


def nearest_grid_index(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """每個座標在一維網格 axis 上的最近點索引。"""
    return np.argmin(np.abs(coords[:, None] - axis[None, :]), axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--les", required=True, help="LES .npy（決定佈點）")
    ap.add_argument("--dns", required=True, help="DNS .npy（提供感測值）")
    ap.add_argument("--budgets", type=int, nargs="+", required=True)
    ap.add_argument("--t-spinup", type=float, required=True,
                    help="排除 LES 的 t < t_spinup（既有 Re=10⁴ 佈點皆用 20.0）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="檔名 sensors_qrpivot_K{K}_{tag}.json")
    args = ap.parse_args()

    budgets = sorted(set(args.budgets))
    les_path, dns_path = _resolve_data_path(args.les), _resolve_data_path(args.dns)
    out_dir = Path(args.out_dir)
    paths = {K: (out_dir / f"sensors_qrpivot_K{K}_{args.tag}.json",
                 out_dir / f"sensors_qrpivot_K{K}_{args.tag}_dns_values.npz") for K in budgets}
    existing = [p for pair in paths.values() for p in pair if p.exists()]
    if existing:
        raise FileExistsError(f"拒絕覆寫既有檔：{existing}")

    les = load_les(les_path, args.t_spinup)
    axis_check = check_axis_convention(les["u"], les["v"], les["L"])
    print(f"[les] N={les['N']} L={les['L']} t∈[{les['time'][0]:.3f},{les['time'][-1]:.3f}] "
          f"frames={les['u'].shape[0]} axis ratio={axis_check['ratio']:.2e}")
    feats = les_features(les["u"], les["v"], les["L"])
    ranking = rank_grid_points([feats[k] for k in FEATURES], max(budgets))

    dns = np.load(dns_path, allow_pickle=True).item()
    dns_x = np.asarray(dns["x"], dtype=np.float64)
    dns_y = np.asarray(dns["y"], dtype=np.float64)
    dns_u = np.asarray(dns["u"], dtype=np.float32)
    dns_v = np.asarray(dns["v"], dtype=np.float32)
    dns_t = np.asarray(dns["time"], dtype=np.float32)
    dns_check = check_axis_convention(dns_u, dns_v, float(dns_x[1] - dns_x[0]) * len(dns_x))

    N_les, L = les["N"], les["L"]
    les_axis = np.arange(N_les) * (L / N_les)
    for K in budgets:
        ix_les, iy_les = np.unravel_index(ranking[:K], (N_les, N_les))
        ix = nearest_grid_index(les_axis[ix_les], dns_x)
        iy = nearest_grid_index(les_axis[iy_les], dns_y)
        coords = np.stack([dns_x[ix], dns_y[iy]], axis=1)
        if len({tuple(c) for c in coords}) < K:
            raise ValueError(f"K={K}：投影到 DNS 網格後有重疊的感測點")
        json_path, npz_path = paths[K]
        meta = {
            "K": K,
            "resolution": f"{len(dns_x)}x{len(dns_x)}",
            "spatial_downsample_res": f"{len(dns_x)}x{len(dns_x)}",
            "spatial_downsample_stride": 1,
            "method": "qr_pivoting_from_les",
            "features": list(FEATURES),
            "feature_definitions": FEATURE_DEFINITIONS,
            "time_stride": 1,
            "time_range": [float(dns_t[0]), float(dns_t[-1])],
            "time_steps": int(len(dns_t)),
            "selected_coordinates": coords.tolist(),
            # 記 CLI 給的相對路徑（同既有檔）：資料根因機器而異，_resolve_data_path 會再解析
            "source_file": args.dns,
            "les_source_file": args.les,
            "les_N": N_les,
            "les_t_spinup": args.t_spinup,
            "dns_values_npz": str(Path(args.out_dir) / npz_path.name),
            "sensor_dt": float(dns_t[1] - dns_t[0]),
            "sensor_time_points": int(len(dns_t)),
            "axis_check": {"les": axis_check, "dns": dns_check},
            "ranking_sha256": hashlib.sha256(np.ascontiguousarray(ranking[:K]).tobytes()).hexdigest(),
            "generator": "scripts/gen_sensors_qrpivot_les.py",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        np.savez(npz_path, time=dns_t, u=dns_u[:, ix, iy].T, v=dns_v[:, ix, iy].T)
        print(f"[out] K={K}: {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
