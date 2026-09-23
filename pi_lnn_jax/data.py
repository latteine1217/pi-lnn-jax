"""Load pi-lnn Re=1000 sensor + DNS data into JAX arrays.

Axis convention（極關鍵，EXP-101 災難根因）:
  - JSON `indices` 是 flat index on N=128 grid, row-major: `flat = x_idx * N + y_idx`
  - `selected_coordinates` 對應 `(x_arr[x_idx], y_arr[y_idx])`，x,y ∈ [0,1]
  - NPZ `u`, `v` shape [K, T]，由 pi-lnn unversioned correct script 從 DNS 抽取，已驗證 axis 正確
  - 本檔僅消費 NPZ + JSON 既有 coords，不重抽 → 完全 bypass axis bug

What: 把 pi-lnn 既有 EXP-030 資料 (sensors_qrpivot_K100_N128_t0-5*.{json,npz}) 載入為
      LiquidOperator 期望的 (sensor_vals [T, K, 2], sensor_pos [K, 2], sensor_time [T])。
Why: POC 必須與 pi-lnn baseline 用同一份 sensor file，否則「JAX vs PyTorch 數值對比」
     會混入 axis convention 噪音。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


# PILNJAX_ROOT: 本專案的 data root，預設為 repo 目錄。
# Why: 避免在 data.py / config / TOML 三處同步維護絕對路徑；單一覆寫點。
PILNJAX_ROOT = Path(
    os.environ.get("PILNJAX_ROOT", Path(__file__).parent.parent)
)

# PILNJAX_DATA_ROOT: 大檔（DNS / LES 全場）的集中存放根。
# Why: >100MB 的場檔不進 git（見 .gitignore），lab-server 上多個 worktree 共用同一份，
#      由此 env var 指向集中的 checkout。預設等同 PILNJAX_ROOT——單一 checkout 時
#      兩者重合，解析退化為只查本專案 data/。
PILNJAX_DATA_ROOT = Path(
    os.environ.get("PILNJAX_DATA_ROOT", PILNJAX_ROOT)
)


def _resolve_data_path(path: str | Path) -> Path:
    """將相對路徑解析為存在的絕對路徑。
    搜尋順序：PILNJAX_ROOT（本專案 data/）→ PILNJAX_DATA_ROOT（集中大檔池）。
    絕對路徑直接返回（不做 fallback）。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (PILNJAX_ROOT, PILNJAX_DATA_ROOT):
        candidate = root / p
        if candidate.exists():
            return candidate
    # 都找不到，返回 PILNJAX_ROOT 下的路徑讓呼叫方產出清楚的錯誤訊息
    return PILNJAX_ROOT / p


SENSOR_JSON = _resolve_data_path("data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5.json")
SENSOR_NPZ = _resolve_data_path("data/kolmogorov_sensors/re1000/sensors_qrpivot_K100_N128_t0-5_dns_values.npz")
DNS_NPY = _resolve_data_path("data/dns/kolmogorov_dns_fp64_etdrk4_Re1000_N128_T5_ds4.npy")


def load_dns(time_stride: int = 2):
    """載入 DNS 全場做 offline benchmark。
    Returns:
      dns_u: [T, N, N] (axis=1 是 x, axis=2 是 y, per pi-lnn convention)
      dns_v: [T, N, N]
      dns_t: [T]
    """
    obj = np.load(DNS_NPY, allow_pickle=True).item()
    # pi-lnn DNS convention: u[t, axis1=x, axis2=y]，shape (T_all, N, N)
    u_full = np.asarray(obj["u"], dtype=np.float32)
    v_full = np.asarray(obj["v"], dtype=np.float32)
    t_full = np.asarray(obj["time"], dtype=np.float32)
    u_s = u_full[::time_stride]
    v_s = v_full[::time_stride]
    t_s = t_full[::time_stride]
    return u_s, v_s, t_s


def load_dns_from_path(dns_path: str | Path, time_stride: int = 1, return_p: bool = False):
    """從任意路徑讀取 DNS npy（pi-lnn convention: dict {u, v, [p], time}）。
    支援相對路徑（優先 PILNJAX_ROOT，次 PILNJAX_DATA_ROOT）。
    return_p=True 時額外回 p（需 npy 含 'p'）：回 (u, v, p, time)；否則 (u, v, time)。
    """
    resolved = _resolve_data_path(dns_path)
    if not resolved.exists():
        raise FileNotFoundError(f"DNS .npy 不存在: {resolved}")
    obj = np.load(resolved, allow_pickle=True).item()
    u_full = np.asarray(obj["u"], dtype=np.float32)
    v_full = np.asarray(obj["v"], dtype=np.float32)
    t_full = np.asarray(obj["time"], dtype=np.float32)
    if return_p:
        if "p" not in obj:
            raise KeyError(f"DNS npy 無 'p' 場：{resolved}")
        p_full = np.asarray(obj["p"], dtype=np.float32)
        return u_full[::time_stride], v_full[::time_stride], p_full[::time_stride], t_full[::time_stride]
    return u_full[::time_stride], v_full[::time_stride], t_full[::time_stride]


def load_sensors_from_path(json_path: str | Path, time_stride: int = 2) -> dict:
    """Like load_sensors() 但接受任意 json path（multi-Re 用）。
    NPZ 路徑從 JSON meta 內 'dns_values_npz' 推（pi-lnn convention）；
    若失敗則嘗試把 json filename 結尾 .json → _dns_values.npz。
    相對路徑優先查 PILNJAX_ROOT，再 fallback 到 PILNJAX_DATA_ROOT。
    """
    json_path = _resolve_data_path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"sensor JSON not found: {json_path}")

    with open(json_path) as f:
        meta = json.load(f)
    K = int(meta["K"])
    coords = np.asarray(meta["selected_coordinates"], dtype=np.float32)
    if coords.shape != (K, 2):
        raise ValueError(f"sensor coordinates shape {coords.shape} does not match K={K}")

    # NPZ 對應檔：優先用 meta['dns_values_npz']；依序搜尋 PILNJAX_ROOT / PILNJAX_DATA_ROOT / 同目錄
    npz_rel = meta.get("dns_values_npz")
    if npz_rel:
        candidates = [
            PILNJAX_ROOT / npz_rel,
            PILNJAX_ROOT / Path(npz_rel).name,
            PILNJAX_DATA_ROOT / npz_rel,
            PILNJAX_DATA_ROOT / Path(npz_rel).name,
            json_path.parent / Path(npz_rel).name,
        ]
        npz_path = next((p for p in candidates if p.exists()), None)
    else:
        # fallback: 推從 json filename
        guess = json_path.with_name(json_path.stem + "_dns_values.npz")
        npz_path = guess if guess.exists() else None

    if npz_path is None:
        tried = candidates if npz_rel else [guess]
        raise FileNotFoundError(
            f"無法定位 NPZ for {json_path.name}；tried: {tried}"
        )

    npz = np.load(npz_path)
    channels = meta.get("channels", ["u", "v"])
    if channels[:2] != ["u", "v"]:
        raise ValueError(f"channels 必須以 ['u','v'] 開頭（decoder 輸出順序）：{channels}")
    time_all = np.asarray(npz["time"], dtype=np.float32)
    t_s = time_all[::time_stride]

    norm_stats: dict[str, float] = {}
    cols = []
    for c in channels:
        if c not in npz.files:
            raise FileNotFoundError(f"NPZ {npz_path.name} 缺 channel '{c}'；有 {list(npz.files)}")
        raw = np.asarray(npz[c], dtype=np.float32)
        # 形狀契約在此強制。原本只驗了 coords 是 (K,2)，channel 陣列的 K 與 T 都沒驗：
        # 手工組裝或半途中斷的 NPZ 會被載進來，直到下游 `broadcast_to((T,K,2))` 才炸，
        # 而那個 JAX broadcast 錯誤指向的是模型不是資料。
        if raw.ndim != 2:
            raise ValueError(
                f"NPZ {npz_path.name} 的 channel '{c}' 應為 2D (K, T)，得到 {raw.shape}")
        if raw.shape[0] != K:
            raise ValueError(
                f"NPZ {npz_path.name} 的 channel '{c}' 有 {raw.shape[0]} 個 sensor，"
                f"但 JSON 宣告 K={K}（selected_coordinates 有 {coords.shape[0]} 列）")
        if raw.shape[1] != time_all.shape[0]:
            raise ValueError(
                f"NPZ {npz_path.name} 的 channel '{c}' 有 {raw.shape[1]} 幀，"
                f"但同檔的 'time' 有 {time_all.shape[0]} 個時刻")
        arr = raw[:, ::time_stride]  # (K, T_s)
        m, s = float(arr.mean()), float(arr.std())
        if s < 1e-8:
            raise ValueError(f"channel '{c}' std≈0 ({s:.2e})，無法 normalize；資料可能為常數場")
        norm_stats[f"{c}_mean"] = m
        norm_stats[f"{c}_std"] = s
        cols.append(((arr - m) / s).T)  # (T_s, K)
    sensor_vals = np.stack(cols, axis=-1)  # (T_s, K, C)
    return {
        "sensor_vals": sensor_vals.astype(np.float32),
        "sensor_pos":  coords,
        "sensor_time": t_s,
        "norm_stats": norm_stats,
        "meta": meta,
        "json_path": str(json_path),
        "npz_path": str(npz_path),
    }


def apply_sensor_noise(
    sensor_vals: np.ndarray, noise_frac: float, seed: int
) -> np.ndarray:
    """對 sensor 觀測加 per-channel 高斯量測噪音，σ_c = noise_frac × std_c。

    語意是「量測本身帶噪」：噪音在載入階段注入一次，訓練全程看到同一份 noisy
    觀測。這與 train-time dropout augmentation 不同——後者每步重抽，目的是讓
    模型學會容忍缺值，前者是觀測品質本身的下限。

    RNG 用 host-side numpy 且與 training seed 分離：`run.py` 的主 RNG 消費順序
    是 bit-identical 契約的承重面，噪音若從主 RNG 取子鍵，clean 與 noisy 兩種
    跑法的後續 RNG 流會錯位，連 clean baseline 都不再可重現。

    noise_frac=0 走 identity，確保這條路徑的存在不改變既有結果。
    """
    if noise_frac < 0:
        raise ValueError(f"sensor_noise_std 不可為負: {noise_frac}")
    if noise_frac == 0:
        return sensor_vals
    vals = np.asarray(sensor_vals)
    # 每個 channel 用自身 std 定尺度；跨 (T, K) 聚合，保留 channel 軸
    sigma = vals.std(axis=tuple(range(vals.ndim - 1)), keepdims=True) * noise_frac
    rng = np.random.default_rng(seed)
    return vals + rng.normal(0.0, 1.0, vals.shape) * sigma


def load_multi_re_sensors(
    json_paths: list[str],
    re_values: list[float],
    time_stride: int | list[int] = 2,
    re_norm_scale: float = 10000.0,
) -> list[dict]:
    """載入多個 (json, re) 對，返回 list of dataset dicts。

    每個 dataset 增加 're_value' 欄位（傳給 LiquidOperator.forward 用 normalize）。
    Why: multi-Re training 讓單一 model 在多 Re 上 generalize；config 透過 sensor_jsons
         + re_values 平行 list 指定。

    time_stride 可為 int（所有 dataset 共用）或 list[int]（per-dataset）。
    per-dataset 用於對齊不同 Re 的原始 T：例如 Re=1000 stride=1 (T=51) 配
    Re=10000 stride=4 (T=51)，讓 jit cache 共用同一個 shape。

    re_norm_scale: re_norm = log(Re)/log(re_norm_scale) 的上界 Re_max。預設 1e4
    保持單-Re 既有行為；訓練集涵蓋更高 Re（如 1e6）時設成該上界，讓 FiLM 條件
    落在 [0, 1]、不外推。

    Returns: list of dicts；K 跨 dataset 必須一致（raise）；T 容差 2 內自動截到共同
    最小值（log），超過視為 time_stride 設定錯誤（raise）。
    """
    if len(json_paths) != len(re_values):
        raise ValueError(
            f"json_paths ({len(json_paths)}) 與 re_values ({len(re_values)}) 長度不符"
        )
    if isinstance(time_stride, list):
        if len(time_stride) != len(json_paths):
            raise ValueError(
                f"time_stride list ({len(time_stride)}) 與 json_paths "
                f"({len(json_paths)}) 長度不符"
            )
        strides = list(time_stride)
    else:
        strides = [int(time_stride)] * len(json_paths)

    datasets = []
    for i, (jp, re, stride) in enumerate(zip(json_paths, re_values, strides)):
        d = load_sensors_from_path(jp, time_stride=int(stride))
        d["re_value"] = float(re)
        d["time_stride"] = int(stride)
        # re_norm: log(Re) / log(Re_max)；Re_max 可配置（多-Re 訓練到 1e6 時設成 1e6）
        d["re_norm"] = float(np.log(re) / np.log(re_norm_scale))
        datasets.append(d)

    shapes = [d["sensor_vals"].shape for d in datasets]
    Ts = [s[0] for s in shapes]
    Ks = [s[1] for s in shapes]
    # K 跨 dataset 必須相同（jit cache 假設，不可截）
    if len(set(Ks)) > 1:
        raise ValueError(
            f"Multi-Re dataset K (sensors) 不一致：{list(zip(re_values, Ks))}; "
            f"jit cache 需要 K 跨 Re 相同。"
        )
    # T 對齊：不同 Re 的 DNS snapshot 數可能差 1（如 201 vs 200），stride 後仍差 1。
    # 容差 2（off-by-one + stride rounding 上限）內 → 截到共同最小 T 並 log；
    # 超過則視為 time_stride 設定錯誤 → fail-fast（不靜默截斷）。
    T_min = min(Ts)
    if max(Ts) - T_min > 2:
        raise ValueError(
            f"Multi-Re dataset T (time_steps) 差距過大：{list(zip(re_values, Ts))}; "
            f"應為 time_stride 設定錯誤，請對齊各 Re 的採樣率。"
        )
    if len(set(Ts)) > 1:
        print(
            f"[load_multi_re_sensors] T 不齊 {list(zip(re_values, Ts))} → "
            f"截到共同 T={T_min}（drop 各 dataset 末端 {[t - T_min for t in Ts]} snapshot）",
            flush=True,
        )
        for d in datasets:
            d["sensor_vals"] = d["sensor_vals"][:T_min]
            d["sensor_time"] = d["sensor_time"][:T_min]
    return datasets


def parse_re_set(s: str) -> set[float]:
    """把 `--held-out` / `--re-subset` 這類 comma-sep Re 字串解析成集合；空字串回空集。"""
    return {float(x) for x in s.split(",") if x.strip()} if s and s.strip() else set()


def resolve_re_inputs(
    config: str | Path | None = None,
    re_values: str = "",
    sensor_jsons: str = "",
    dns_paths: str = "",
) -> list[tuple[float, str, str]]:
    """解析 multi-Re 腳本的輸入來源，回傳 `[(Re, sensor_json, dns_path), ...]`。

    兩條路徑擇一：`config`（讀 TOML 的 `data_kwargs`）或 direct 的三個 comma-sep 字串。
    `config` 優先。

    Why 集中：這段解析原本在 cost_accuracy / evaluate_baselines / train_baseline_shred
    各抄一份，而 eval_gappy_cross_re 的第四份省略了長度驗證——三個 list 不等長時
    `zip` 會靜默截短，少評的 Re 一聲不吭。三者長度不等一律 raise。
    """
    if config is not None:
        from .config import load_config

        dk = load_config(config)["data_kwargs"]
        re_list = [float(r) for r in dk.get("re_values", [])]
        sensor_list = list(dk.get("sensor_jsons", []))
        dns_list = list(dk.get("dns_paths", []))
        source = f"config {config}"
    else:
        re_list = [float(x) for x in re_values.split(",") if x.strip()]
        sensor_list = [x.strip() for x in sensor_jsons.split(",") if x.strip()]
        dns_list = [x.strip() for x in dns_paths.split(",") if x.strip()]
        source = "direct 三件組"

    if not re_list:
        raise ValueError(
            "未提供輸入：請給 config，或 direct 三件組"
            "（--re-values / --sensor-jsons / --dns-paths）"
        )
    if not (len(re_list) == len(sensor_list) == len(dns_list)):
        raise ValueError(
            f"{source}: re_values({len(re_list)})/sensor_jsons({len(sensor_list)})/"
            f"dns_paths({len(dns_list)}) 長度需相等"
        )
    return list(zip(re_list, sensor_list, dns_list))


if __name__ == "__main__":
    d = load_sensors_from_path(SENSOR_JSON)
    print(f"sensor_vals: {d['sensor_vals'].shape} dtype={d['sensor_vals'].dtype}")
    print(f"sensor_pos:  {d['sensor_pos'].shape}")
    print(f"sensor_time: {d['sensor_time'].shape} from {d['sensor_time'][0]} to {d['sensor_time'][-1]}")
    print(f"norm_stats:  {d['norm_stats']}")
    u, v, t = load_dns()
    print(f"DNS u: {u.shape} dtype={u.dtype}, mean={float(u.mean()):.3f}, std={float(u.std()):.3f}")
    print(f"DNS v: {v.shape} dtype={v.dtype}, mean={float(v.mean()):.3f}, std={float(v.std()):.3f}")
    print(f"DNS t: {t.shape} from {t[0]} to {t[-1]}")
