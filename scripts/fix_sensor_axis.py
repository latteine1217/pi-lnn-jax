"""fix_sensor_axis.py — 修復 coords 與 values 軸向不一致的 sensor 檔。

What:
    某些 pi-lnn 生成器（`generate_sensors_qrpivot.py`、`generate_sensors_random.py`）
    用影像慣例寫 coords（`stack([x[col], y[row]])`）卻用 `u_full[:, row, col]` 取值，
    使 coords 綁第二軸、值取自第一軸——與本專案慣例（coords[:,0] ↔ 第一軸，見
    `generate_sensors_spacefill.py` 與 knowledge 的 sensor axis invariant）差一個轉置。

    後果是**靜默**的：訓練 loss 正常收斂（模型照樣擬合那些數值），但重建場對真值
    爆掉（實測 uv_rel_err > 100%）。

Why 對調兩欄就夠:
    值取自 `u[:, row, col]`，在本專案慣例下那就是 (x=row, y=col)。檔案寫的是
    `[x[col], y[row]]`；x_arr 與 y_arr 是同一組 [0,1) 格點，故交換兩欄後
    等同 `[x[row], y[col]]` —— 與值一致，且**保留生成器真正選到的點位**
    （不是把佈點鏡射，是把標籤修正）。

    修完必過不變量：`u_full[:, xi, yi].T == npz["u"]`。

Usage:
    uv run python scripts/fix_sensor_axis.py --json <壞檔.json> --dns <DNS.npy> \
        --out-tag <新檔名（不含 sensors_ 前綴與副檔名）>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_u(dns_path: Path) -> np.ndarray:
    raw = np.load(dns_path, allow_pickle=True)
    return np.asarray(raw.item()["u"]) if raw.dtype == object else (
        raw[..., 0] if raw.ndim == 4 else raw)


def invariant_error(u_full, coords, vals):
    """回傳 (direct_err, swapped_err)；direct 小 = 符合本專案慣例。"""
    N = u_full.shape[-1]
    xi = np.clip((coords[:, 0] * N).astype(int), 0, N - 1)
    yi = np.clip((coords[:, 1] * N).astype(int), 0, N - 1)
    T = min(vals.shape[1], u_full.shape[0])
    d = np.abs(vals[:, :T] - u_full[:, xi, yi].T[:, :T]).max()
    s = np.abs(vals[:, :T] - u_full[:, yi, xi].T[:, :T]).max()
    return float(d), float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--dns", required=True, type=Path)
    ap.add_argument("--out-tag", required=True, help="輸出檔名（不含 sensors_ 與副檔名）")
    args = ap.parse_args()

    src = args.json
    meta = json.loads(src.read_text())
    base = src.parent
    npz_name = Path(meta.get("dns_values_npz", src.stem + "_dns_values.npz")).name
    vals_npz = base / npz_name
    if not vals_npz.exists():
        raise SystemExit(f"[ERR] 找不到值檔：{vals_npz}")

    u_full = load_u(args.dns)
    coords = np.asarray(meta["selected_coordinates"], dtype=float)
    vals_u = np.load(vals_npz)["u"]
    if vals_u.shape[0] != coords.shape[0]:
        vals_u = vals_u.T

    d0, s0 = invariant_error(u_full, coords, vals_u)
    print(f"[before] direct={d0:.3e}  swapped={s0:.3e}")
    if d0 < 1e-6:
        raise SystemExit("[ERR] 這個檔本來就是對的（direct 已符合），不需修復——中止避免誤改。")
    if s0 >= 1e-6:
        raise SystemExit("[ERR] direct 與 swapped 皆不符：不是單純的軸向對調，需人工判讀。")

    fixed = coords[:, ::-1].copy()          # 兩欄對調
    d1, s1 = invariant_error(u_full, fixed, vals_u)
    print(f"[after ] direct={d1:.3e}  swapped={s1:.3e}")
    if d1 >= 1e-6:
        raise SystemExit("[ERR] 修復後仍不滿足不變量，未輸出。")

    out_json = base / f"sensors_{args.out_tag}.json"
    out_npz = base / f"sensors_{args.out_tag}_dns_values.npz"
    if out_json.exists() or out_npz.exists():
        raise SystemExit(f"[ERR] 輸出已存在，不覆寫：{out_json.name}")

    meta["selected_coordinates"] = fixed.tolist()
    meta["dns_values_npz"] = str(Path(meta.get("dns_values_npz", "")).parent / out_npz.name) \
        if meta.get("dns_values_npz") else out_npz.name
    meta["axis_fix_note"] = (
        f"由 {src.name} 修復：原檔 coords 用影像慣例（x←col, y←row）而值取自 "
        f"u[:, row, col]，兩者差一個轉置。修法為對調 coords 兩欄，保留原生成器選到的點位。"
        f"修復後通過不變量 u[:, xi, yi].T == values（max err {d1:.2e}）。"
        f"工具：scripts/fix_sensor_axis.py")
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    shutil.copy2(vals_npz, out_npz)
    print(f"[save] {out_json}\n[save] {out_npz}")


if __name__ == "__main__":
    main()
