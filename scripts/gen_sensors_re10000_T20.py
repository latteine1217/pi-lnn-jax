"""從 Re=10000 T=20 DNS 檔案生成兩組 sensor 資料集。

Exp A — Random：K=100 感測器，位置均勻隨機抽自 N=256 網格，seed=42。
Exp B — LES QR：複用 pi-lnn EXP-245 T50standalone LES QR-pivot 的空間座標，
         對全 T=20 時間軸重新抽取 DNS 值。

輸出到 data/sensors/re10000/（相對本腳本位置的專案根目錄）：
  sensors_random_K100_N256_t0-20_si128_seed42.json
  sensors_random_K100_N256_t0-20_si128_seed42_values.npz
  sensors_les_qr_K100_N256_t0-20_si128.json
  sensors_les_qr_K100_N256_t0-20_si128_values.npz

NPZ 格式（與 pi-lnn convention 一致）：
  u:    shape (K, T)  float32
  v:    shape (K, T)  float32
  time: shape (T,)    float64

JSON 格式（load_sensors_from_path 需要的最小欄位）：
  K, resolution, method, selected_coordinates, dns_values_npz, source_file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ── 路徑設定 ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DNS_PATH = PROJECT_ROOT / "data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_dt1p95e4_si128_seed42.npy"
OUT_DIR = PROJECT_ROOT / "data/sensors/re10000"

# LES QR 座標來源
from pi_lnn_jax.data import _resolve_data_path

LES_QR_JSON = _resolve_data_path(
    "data/kolmogorov_sensors/re10000/sensors_qrpivot_K100_N256_t0-5_si100_les_n256_T50standalone.json")

K = 100
SEED = 42


def load_dns() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, u, v, p, time from DNS file."""
    if not DNS_PATH.exists():
        sys.exit(f"[ERR] DNS 檔不存在: {DNS_PATH}")
    obj = np.load(DNS_PATH, allow_pickle=True).item()
    x = np.asarray(obj["x"], dtype=np.float64)       # (N,)
    y = np.asarray(obj["y"], dtype=np.float64)       # (N,)
    u = np.asarray(obj["u"], dtype=np.float32)       # (T, N, N)
    v = np.asarray(obj["v"], dtype=np.float32)       # (T, N, N)
    p = np.asarray(obj["p"], dtype=np.float32)       # (T, N, N)
    time = np.asarray(obj["time"], dtype=np.float64) # (T,)
    print(f"[DNS] shape u={u.shape}  p={p.shape}  t=[{time[0]:.3f}, {time[-1]:.3f}]  N={len(x)}")
    return x, y, u, v, p, time


def extract_sensor_values(u, v, xi_idx: list[int], yj_idx: list[int]):
    """u, v: (T, N, N)  → sensor_u, sensor_v: (K, T)."""
    T = u.shape[0]
    K = len(xi_idx)
    su = np.empty((K, T), dtype=np.float32)
    sv = np.empty((K, T), dtype=np.float32)
    for k, (i, j) in enumerate(zip(xi_idx, yj_idx)):
        su[k] = u[:, i, j]
        sv[k] = v[:, i, j]
    return su, sv


def save_sensor(
    out_stem: str,
    coords: list[list[float]],
    xi_idx: list[int],
    yj_idx: list[int],
    u, v, time,
    meta_extra: dict,
):
    """存 JSON + NPZ，回傳兩個路徑。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_name = out_stem + "_values.npz"
    json_name = out_stem + ".json"

    su, sv = extract_sensor_values(u, v, xi_idx, yj_idx)
    np.savez(OUT_DIR / npz_name, u=su, v=sv, time=time)

    meta = {
        "K": len(coords),
        "resolution": u.shape[1],
        "selected_coordinates": coords,
        "dns_values_npz": npz_name,
        "source_file": str(DNS_PATH.name),
        **meta_extra,
    }
    with open(OUT_DIR / json_name, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OUT] {json_name}  K={len(coords)}  T={len(time)}")
    print(f"      {npz_name}  u {su.shape}")
    return OUT_DIR / json_name, OUT_DIR / npz_name


def extract_sensor_p(p, xi_idx: list[int], yj_idx: list[int]):
    """p: (T, N, N) → sensor_p: (K, T)."""
    T = p.shape[0]
    K = len(xi_idx)
    sp = np.empty((K, T), dtype=np.float32)
    for k, (i, j) in enumerate(zip(xi_idx, yj_idx)):
        sp[k] = p[:, i, j]
    return sp


def save_sensor_uvp(out_stem, coords, xi_idx, yj_idx, u, v, p, time, meta_extra):
    """存含 p 的 JSON + NPZ（channels=[u,v,p]），回傳兩路徑。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_name = out_stem + "_values.npz"
    json_name = out_stem + ".json"

    su, sv = extract_sensor_values(u, v, xi_idx, yj_idx)
    sp = extract_sensor_p(p, xi_idx, yj_idx)
    np.savez(OUT_DIR / npz_name, u=su, v=sv, p=sp, time=time)

    meta = {
        "K": len(coords),
        "resolution": u.shape[1],
        "selected_coordinates": coords,
        "dns_values_npz": npz_name,
        "source_file": str(DNS_PATH.name),
        "channels": ["u", "v", "p"],
        **meta_extra,
    }
    with open(OUT_DIR / json_name, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OUT] {json_name}  K={len(coords)}  channels=[u,v,p]")
    print(f"      {npz_name}  u {su.shape}  p {sp.shape}")
    return OUT_DIR / json_name, OUT_DIR / npz_name


def gen_random(x, y, u, v, time):
    rng = np.random.RandomState(SEED)
    N = len(x)
    # 隨機抽 K 個不重複 grid index
    flat_idx = rng.choice(N * N, size=K, replace=False)
    xi_idx = (flat_idx // N).tolist()
    yj_idx = (flat_idx % N).tolist()
    coords = [[float(x[i]), float(y[j])] for i, j in zip(xi_idx, yj_idx)]

    save_sensor(
        "sensors_random_K100_N256_t0-20_si128_seed42",
        coords, xi_idx, yj_idx, u, v, time,
        meta_extra={
            "method": "random",
            "placement_seed": SEED,
            "time_range": [float(time[0]), float(time[-1])],
            "time_steps": len(time),
        },
    )


def gen_les_qr(x, y, u, v, time):
    if not LES_QR_JSON.exists():
        sys.exit(f"[ERR] LES QR JSON 不存在: {LES_QR_JSON}")
    with open(LES_QR_JSON) as f:
        meta_src = json.load(f)
    src_coords = meta_src["selected_coordinates"]  # list of [x, y]
    assert len(src_coords) == K, f"期望 K={K}，LES QR 有 {len(src_coords)} 個座標"

    # 找最近網格點（x, y 分別 argmin）
    xa = np.asarray(x)
    ya = np.asarray(y)
    xi_idx = [int(np.argmin(np.abs(xa - c[0]))) for c in src_coords]
    yj_idx = [int(np.argmin(np.abs(ya - c[1]))) for c in src_coords]
    # 使用實際網格座標（非原始浮點座標，確保與 DNS 一致）
    coords = [[float(x[i]), float(y[j])] for i, j in zip(xi_idx, yj_idx)]

    save_sensor(
        "sensors_les_qr_K100_N256_t0-20_si128",
        coords, xi_idx, yj_idx, u, v, time,
        meta_extra={
            "method": "qrpivot",
            "les_source": str(LES_QR_JSON.name),
            "time_range": [float(time[0]), float(time[-1])],
            "time_steps": len(time),
        },
    )


def gen_les_qr_uvp(x, y, u, v, p, time):
    if not LES_QR_JSON.exists():
        sys.exit(f"[ERR] LES QR JSON 不存在: {LES_QR_JSON}")
    with open(LES_QR_JSON) as f:
        meta_src = json.load(f)
    src_coords = meta_src["selected_coordinates"]
    assert len(src_coords) == K, f"期望 K={K}，LES QR 有 {len(src_coords)} 個座標"
    xa = np.asarray(x); ya = np.asarray(y)
    xi_idx = [int(np.argmin(np.abs(xa - c[0]))) for c in src_coords]
    yj_idx = [int(np.argmin(np.abs(ya - c[1]))) for c in src_coords]
    coords = [[float(x[i]), float(y[j])] for i, j in zip(xi_idx, yj_idx)]
    save_sensor_uvp(
        "sensors_les_qr_K100_N256_t0-20_si128_uvp",
        coords, xi_idx, yj_idx, u, v, p, time,
        meta_extra={
            "method": "qrpivot",
            "les_source": str(LES_QR_JSON.name),
            "time_range": [float(time[0]), float(time[-1])],
            "time_steps": len(time),
        },
    )


if __name__ == "__main__":
    x, y, u, v, p, time = load_dns()
    print("\n--- Exp A: Random (u,v) ---")
    gen_random(x, y, u, v, time)
    print("\n--- Exp B: LES QR (u,v) ---")
    gen_les_qr(x, y, u, v, time)
    print("\n--- Exp B-pin: LES QR (u,v,p) ---")
    gen_les_qr_uvp(x, y, u, v, p, time)
    print("\n[DONE] sensor 檔已存至:", OUT_DIR)
