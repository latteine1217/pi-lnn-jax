"""gen_sensors_qrpivot_dns.py — DNS-pivot oracle 佈點：對 DNS 快照的物理特徵做 column-pivoted QR。

What:
    從 DNS 快照（陣列慣例 [T, x, y]）建六個特徵：u、v、由速度解 Poisson 的壓力 p_ref、
    渦度 ω = ∂v/∂x − ∂u/∂y、|∇u|、|∇v|（導數皆為週期二階中央差分）。各特徵除以自身
    全域標準差後堆疊成 [6T, N²]，column-pivoted QR 給出一條網格點排名；每個預算 K 取
    前 K 個，所以各預算的佈點是巢狀的。輸出與既有 sensor 檔同 schema 的 JSON + NPZ。

Why:
    取代 pi-lnn 的 `generate_sensors_qrpivot.py`，它有三個缺陷（2026-09-30 以原始碼逐點
    重現入庫的 K=200 oracle 確認）：
      1. 假設陣列為 [T, y, x]，資料實為 [T, x, y]：它的 "omega" 實際是 ∂v/∂y − ∂u/∂x，
         座標與取值也差一個轉置（K=200 的 `_axisfix` 檔就是修這個）。
      2. p 取 DNS 存檔欄，而該欄是 −(p_ref + |u|²) + C（生成器符號錯，TD-29），不是壓力。
      3. 不在本 repo，無法重現。
    本版的壓力與導數和壓力評估共用 `_common.periodic_fields`；軸向由散度檢查驗證
    （聲明的軸向下散度須遠小於對調軸向）；索引、座標、取值三者的一致性由
    `tests/test_gen_sensors_qrpivot_dns.py` 釘住。

Usage:
    uv run python scripts/gen_sensors_qrpivot_dns.py \\
        --dns data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy \\
        --domain-length 1.0 --budgets 100 200 \\
        --out-dir data/kolmogorov_sensors/re10000 --tag N256_t0-5_si100_dnsoracle_phys
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from pi_lnn_jax.data import _resolve_data_path, load_dns_from_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common.periodic_fields as _periodic_fields
from _common.periodic_fields import periodic_gradient, reference_pressure

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES = ("u", "v", "p_ref", "omega", "grad_u_mag", "grad_v_mag")
FEATURE_DEFINITIONS = {
    "u": "x 向速度",
    "v": "y 向速度",
    "p_ref": "由速度解 Poisson 的壓力 p̂ = −k_i k_j F(u_i u_j)/|k|²，空間均值 0（_common.periodic_fields）",
    "omega": "渦度 ∂v/∂x − ∂u/∂y，週期二階中央差分",
    "grad_u_mag": "|∇u|，週期二階中央差分",
    "grad_v_mag": "|∇v|，週期二階中央差分",
}
# 聲明軸向下的散度 rms 對「兩軸對調」讀法的比值上限。實測 DNS 為 ~5e-3（中央差分的截斷
# 底線），對調後的讀法是 O(1)；超過此值代表陣列不是 [T, x, y]。
AXIS_RATIO_MAX = 0.1


def check_axis_convention(u: np.ndarray, v: np.ndarray, L: float) -> dict:
    """驗證陣列為 [T, x, y]：無散度場在正確軸向下 ∂u/∂x + ∂v/∂y ≈ 0，對調讀法則否。

    擋得下單純的陣列轉置；擋不下「轉置且 u、v 標籤互換」（那仍是無散度場）。後者須另以
    求解器存檔的 ω 或強迫剖面確認——本專案的 Re=10⁴ DNS 已用存檔 ω 確認過軸向。
    """
    gx_u, gy_u = periodic_gradient(u, L)
    gx_v, gy_v = periodic_gradient(v, L)
    declared = float(np.sqrt(np.mean((gx_u + gy_v) ** 2)))
    swapped = float(np.sqrt(np.mean((gy_u + gx_v) ** 2)))
    ratio = declared / swapped
    if not ratio < AXIS_RATIO_MAX:
        raise ValueError(
            f"陣列軸向不是 [T, x, y]：散度 rms 聲明 {declared:.3e}、對調 {swapped:.3e}，"
            f"比值 {ratio:.3f} ≥ {AXIS_RATIO_MAX}")
    return {"div_rms_declared": declared, "div_rms_swapped": swapped, "ratio": ratio}


def physical_features(u: np.ndarray, v: np.ndarray, L: float) -> dict[str, np.ndarray]:
    """六個特徵，形狀皆為 [T, N, N]，順序即 FEATURES。"""
    gx_u, gy_u = periodic_gradient(u, L)
    gx_v, gy_v = periodic_gradient(v, L)
    return {
        "u": u,
        "v": v,
        "p_ref": reference_pressure(u, v, L),
        "omega": gx_v - gy_u,
        "grad_u_mag": np.sqrt(gx_u ** 2 + gy_u ** 2),
        "grad_v_mag": np.sqrt(gx_v ** 2 + gy_v ** 2),
    }


def rank_grid_points(fields: list[np.ndarray], k_max: int) -> np.ndarray:
    """各場轉 float32、除以自身全域標準差後堆疊為 [n_feat·T, N²]，回傳 QRCP 排名的前 k_max 個扁平索引。

    正規化與 QR 呼叫與 pi-lnn 生成器相同（float32、std + 1e-8、scipy geqp3），差別只在特徵的定義。
    """
    T = fields[0].shape[0]
    stacks = []
    for f in fields:
        flat = np.asarray(f, dtype=np.float32).reshape(T, -1)
        stacks.append(flat / (flat.std() + 1e-8))
    matrix = np.concatenate(stacks, axis=0).astype(np.float32)
    if k_max > matrix.shape[1]:
        raise ValueError(f"k_max={k_max} 超過網格點數 {matrix.shape[1]}")
    _, perm = qr(matrix, pivoting=True, mode="r")
    return np.asarray(perm[:k_max])


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_budgets(u: np.ndarray, v: np.ndarray, time: np.ndarray, L: float, ranking: np.ndarray,
                  budgets: list[int], out_dir: Path, tag: str, provenance: dict) -> list[Path]:
    """每個預算 K 寫出 sensors_qrpivot_K{K}_{tag}.json 與對應 _dns_values.npz。

    任一輸出檔已存在就在寫入任何檔之前拒絕，不留半套產出。
    """
    T, N, _ = u.shape
    paths = {K: (out_dir / f"sensors_qrpivot_K{K}_{tag}.json",
                 out_dir / f"sensors_qrpivot_K{K}_{tag}_dns_values.npz") for K in budgets}
    existing = [p for pair in paths.values() for p in pair if p.exists()]
    if existing:
        raise FileExistsError(f"拒絕覆寫既有檔：{existing}")
    written = []
    for K in budgets:
        idx = ranking[:K]
        ix, iy = np.unravel_index(idx, (N, N))
        coords = np.stack([ix * (L / N), iy * (L / N)], axis=1)
        values_u = np.asarray(u[:, ix, iy].T, dtype=np.float32)
        values_v = np.asarray(v[:, ix, iy].T, dtype=np.float32)
        json_path, npz_path = paths[K]
        meta = {
            "K": K,
            "resolution": f"{N}x{N}",
            "spatial_downsample_res": f"{N}x{N}",
            "spatial_downsample_stride": 1,
            "method": "qr_pivoting_dns_oracle",
            "features": list(FEATURES),
            "feature_definitions": FEATURE_DEFINITIONS,
            "axis_convention": "[T, x, y]; selected_coordinates[:, 0] = x",
            "time_stride": 1,
            "time_range": [float(time[0]), float(time[-1])],
            "time_steps": T,
            "selected_coordinates": coords.tolist(),
            "indices": [int(i) for i in idx],
            "dns_values_npz": _repo_relative(npz_path),
            "sensor_dt": float(time[1] - time[0]),
            "sensor_time_points": T,
            **provenance,
        }
        np.savez(npz_path, time=np.asarray(time, dtype=np.float32), u=values_u, v=values_v)
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        written += [json_path, npz_path]
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dns", required=True, help="DNS npy（相對路徑依 PILNJAX_ROOT / PILNJAX_DATA_ROOT 解析）")
    ap.add_argument("--domain-length", required=True, type=float, help="週期域邊長 L")
    ap.add_argument("--budgets", required=True, type=int, nargs="+", help="要輸出的 sensor 數，共用同一條排名")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="檔名中 K 之後的部分")
    args = ap.parse_args()

    budgets = sorted(set(args.budgets))
    dns_path = _resolve_data_path(args.dns)
    u, v, time = load_dns_from_path(args.dns)
    if u.shape != v.shape or u.ndim != 3 or u.shape[1] != u.shape[2]:
        raise ValueError(f"DNS 速度須為 [T, N, N]，收到 {u.shape} / {v.shape}")
    L = args.domain_length
    axis = check_axis_convention(u, v, L)
    print(f"[axis] 散度 rms 聲明/對調 = {axis['ratio']:.2e}", flush=True)
    feats = physical_features(u, v, L)
    ranking = rank_grid_points([feats[name] for name in FEATURES], budgets[-1])
    provenance = {
        "source_file": args.dns,
        "source_sha256": hashlib.sha256(dns_path.read_bytes()).hexdigest(),
        "generator": "scripts/gen_sensors_qrpivot_dns.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        # p_ref 與所有導數的定義在共用模組；它變了佈點就會變，故分開記
        "periodic_fields_sha256": hashlib.sha256(Path(_periodic_fields.__file__).read_bytes()).hexdigest(),
        "axis_check": axis,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in write_budgets(u, v, time, L, ranking, budgets, args.out_dir, args.tag, provenance):
        print(f"[write] {p}", flush=True)


if __name__ == "__main__":
    main()
