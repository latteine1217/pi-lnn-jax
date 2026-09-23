# Controlled Cylinder 實作 Continuation（新對話執行用）

> 這份文件自包含——新對話不需要上一個 session 的記憶就能接續。
> 上一個 session 的 tool 通道遭 prompt-injection 污染（Read/grep/Bash stdout/部分 Write 會造假），
> 所以：**已完成的東西以 `git status` + `pytest` 為準**；本文附上「沒落地」的內容供你重建。

---

## 0. 現狀速覽（先讀這段）

**目標**：把 RealPDEBench 的 controlled cylinder（受迫振動 bluff body）做成 sparse-sensor + PDE-residual
重建，作為論文方向 B 的「光譜中間點」。

**已完成且驗證（在磁碟，`git status` 可見，`pytest` 10 passed）**：
- `pi_lnn_jax/boundary.py` — 動邊界運動學（time-varying mask + moving no-slip BC）
- `pi_lnn_jax/controlled.py` — 從場反推振幅 A（z 是 static、sim_params 沒存振幅）
- `tests/test_boundary_moving.py`（6）+ `tests/test_controlled_traj.py`（4）= **10 test 全過**

**未落地（上個 session Write 沒生效，本文 §5/§6 供重建）**：
- `scripts/dump_controlled_cylinder_v2.py`（完整內容見 §6，存檔即可）
- `config.py` 的 `controlled_cylinder` case（§5 Step 2）
- `train_cylinder.py` 的 controlled 分支（§5 Step 3）

**開新對話第一件事**：
```bash
cd ~/Documents/coding/pi-lnn-jax
git status
PYTHONPATH=. uv run python -m pytest tests/test_boundary_moving.py tests/test_controlled_traj.py -q  # 應 10 passed
```

---

## 1. 研究背景與方向（新對話需要的 context）

**2D 相容性 gate 結果**（用 `scripts/check_2d_divergence.py` 測，相對散度 r = RMS(∂u/∂x+∂v/∂y)/RMS(|∂u/∂x|+|∂v/∂y|)）：

| case | r | 判定 | 意義 |
|---|---|---|---|
| cylinder | 0.020 | 乾淨 2D | 2D 生成，physics-residual 強 |
| **controlled** | **0.148** | **marginal** | spanwise leakage ~15%，physics **打折** |
| foil | 0.38 | 3D | 3D 切片，physics 崩，出局 |
| fsi | 0.42 | 3D | 同上，出局 |

**方向 B（已定案）**：把「RealPDEBench 哪些 case 真滿足 2D incompressibility」當 audit 貢獻 +
cylinder 深化（physics-informed）+ **controlled 當光譜中間點**（physics 效用隨 2D 相容衰減：
cylinder 強→controlled 打折→foil/fsi 崩）。controlled 的 marginal 是**證據**，不是缺陷。落點 JCP/PRF。

**為什麼 controlled 這樣做**：body 受迫振動、軌跡由 control_freq 決定（已知運動）；因 r=0.148 marginal，
2D continuity residual 有 ~15% 系統誤差 → physics 是**弱/有偏約束**，重建主力是 sensor data，
physics weight 要**調低（×0.3）**。舊 `feat/controlled-cylinder` 分支用「靜態平均 mask + 關 body BC +
control FiLM」迴避動邊界 → 尾流爛（KE 1.39）；正確做法是真正的 moving no-slip（下面的 code）。

---

## 2. 已完成 code 的 API（新對話直接用，勿重寫）

### `pi_lnn_jax/boundary.py`（新增區塊，靜態 cylinder 路徑未動；amp=0 退化回靜態）
```python
class OscillatingGeometry(NamedTuple):
    base_center: tuple; body_radius: float; Lx: float; Ly: float; u_inf: float
    amp: float; freq: float; phase: float; axis: tuple   # center(t)=base+amp*sin(2πf t+phase)*axis

body_center_at(geom, t)      -> (cx, cy)                  # 時變中心
body_velocity_at(geom, t)    -> (vx, vy)                  # 剛體壁速 = d center/dt
body_sdf_moving(x, y, t, geom)   -> sdf                   # >0 流體, <0 body
fluid_mask_moving(x, y, t, geom) -> mask                  # 1 流體, 0 body
sample_wall_bc_moving(key, geom, n, t_lo, t_hi)
    -> (inflow[n,3], body[n,3], slip[n,3], body_vel[n,2]) # body 點用物理時間, 附剛體壁速
wall_bc_loss_moving(decode_fn, inflow, body, slip, body_vel_n, u_inf_n, v_zero_n, bc_body_w)
    # body target = 移動壁速(normalized), 非零
```

### `pi_lnn_jax/controlled.py`（純 numpy，反推振幅）
```python
body_center_from_field(u, v, base_center, x_axis, y_axis, search_frac=0.15) -> (cx,cy)
    # 低速加權質心追 body 中心（robust to 小 body / 漸變邊界層）
fit_oscillation(disp, t, freq) -> (amp, phase)
    # 線性最小二乘 disp = A*sin(2πf t + phase)
infer_trajectory(u, v, base_center, freq, t, x_axis, y_axis, axis=(0,1), search_frac=0.15)
    -> dict(amp, phase, centers[T,2], disp[T])            # 完整 pipeline
```

---

## 3. 假設卡（跑訓練前記得）

- **H**：正確 moving-boundary（時變 mask + moving no-slip）+ physics weight ×0.3 下，controlled 重建把
  KE 從舊分支的 1.39 壓回 O(0.1)。
- **Expected**：KE rel-err 大幅改善 vs 舊分支；**physics on/off 差異 < cylinder**（marginal 佐證，是論文光譜證據）。
- **Falsify**：正確動邊界後尾流仍爛 → 問題在 sensor/regime 非 mask；physics on/off 完全無差 → controlled physics 無效（非只打折）。

---

## 4. 驗證階梯（不跳級）
1. gate: r=0.148（已知）✓
2. **body 軌跡反推 sanity**（乾淨環境先做）：對一個 controlled shard 跑 `infer_trajectory`，畫 `centers[:,1]` vs
   `disp`，確認 fit 的 sin 對得上 body 實際擺動、amp 量級合理（~0.05-0.1 normalized）。
3. smoke（lab-server sbatch r740）：能訓、loss 非 NaN、有 ckpt
4. sanity：KE 量級 vs 舊分支 1.39
5. metric：重建 vs DNS + **physics on/off ablation**（光譜證據）
6. multi-seed n≥3 才宣稱

---

## 5. 剩餘整合步驟（新對話執行，那裡通道乾淨）

### Step 1 — 存 dump 腳本
把 §6 的內容存成 `scripts/dump_controlled_cylinder_v2.py`。

### Step 2 — `config.py` 加 controlled_cylinder case
- 找 case 白名單 `_one_of("case", ("kolmogorov", "cylinder"))` → 加 `"controlled_cylinder"`。
- 加欄位（對齊現有 cylinder 欄位風格）：`controlled_data_npz`(str)、`physics_weight_scale`(float, 預設 0.3)。
  amp/phase/freq/axis 由 dump 反推寫進 npz，**不進 config**（訓練時從 npz 讀）。
- `pytest tests/test_config_cylinder.py` 確認 schema 沒壞。

### Step 3 — `train_cylinder.py` 加 controlled 分支
`case=="controlled_cylinder"` 時：
```python
from pi_lnn_jax.boundary import (OscillatingGeometry, fluid_mask_moving,
                                 sample_wall_bc_moving, wall_bc_loss_moving)
geom = OscillatingGeometry(base_center=tuple(d["base_center"]), body_radius=float(d["body_radius"]),
    Lx=Lx, Ly=Ly, u_inf=u_inf, amp=float(d["osc_amp"]), freq=float(d["osc_freq"]),
    phase=float(d["osc_phase"]), axis=tuple(d["osc_axis"]))
# loss 內 body mask（t 用物理時間，對齊 sensor_time 映射，見 boundary docstring）:
m = fluid_mask_moving(cx, cy, t_phys, geom)          # 取代靜態 fluid_mask
# wall BC:
inflow, body, slip, body_vel = sample_wall_bc_moving(k, geom, bc_n, t_lo, t_hi)
body_vel_n = body_vel / u_inf                         # normalize 到預測空間
loss_bc = wall_bc_loss_moving(decode_fn, inflow, body, slip, body_vel_n, u_inf_n, v_zero_n, bc_body_w)
# physics weight 打折:
physics_weight *= cfg.physics_weight_scale            # 預設 0.3（marginal）
```
參考長度沿用 cylinder `D=0.03`（同 rig）。

### Step 4 — eval 時變 body_mask
eval 排除 body 格點改用 `body_center_at(geom, t)` 逐時刻算圓判斷（現在是靜態圓）。
主 stop-loss：KE 與 `ke_pred/ke_ref`（對比舊分支爛尾流 1.39）。

### Step 5 — lab-server 訓練
- head node：`git pull` + `uv sync --python 3.12`
- dump（pi-lnn env）：`python scripts/dump_controlled_cylinder_v2.py --arrow <controlled shard> --out data/controlled_v2.npz --sensor-stem sensors_qrpivot_K100_cylinder_Re10031`
- sbatch r740 訓練（用 `scripts/slurm/submit_exp.sh`，config 指向 controlled_v2.npz + case=controlled_cylinder）

---

## 6. dump 腳本完整內容（存成 `scripts/dump_controlled_cylinder_v2.py`）

```python
"""Dump RealPDEBench controlled-cylinder to npz (run in pi-lnn env).
z geometry is static + sim_params=[Re,control_freq] has no amplitude, so the
oscillation amplitude is reconstructed via pi_lnn_jax.controlled.infer_trajectory.
Usage (lab-server pi-lnn env, with ~/pi-lnn-jax on PYTHONPATH):
  cd ~/pi-lnn && PYTHONPATH=src:scripts:~/pi-lnn-jax .venv/bin/python \
    ~/pi-lnn-jax/scripts/dump_controlled_cylinder_v2.py \
      --arrow <controlled shard.arrow> --out ~/pi-lnn-jax/data/controlled_v2.npz \
      --sensor-stem sensors_qrpivot_K100_cylinder_Re10031
"""
import argparse, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor-stem", required=True, help="borrow cylinder QR placement")
    ap.add_argument("--osc-axis", default="0,1")
    ap.add_argument("--search-frac", type=float, default=0.15)
    ap.add_argument("--n-eval-times", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
    from cylinder_dataset import CylinderDataset
    from evaluate_cylinder import load_arrow_fields
    from pi_lnn_jax.controlled import infer_trajectory

    ds = CylinderDataset(f"data/cylinder_sensors/{a.sensor_stem}.json",
                         f"data/cylinder_sensors/{a.sensor_stem}_values.npz",
                         a.arrow, sensor_subsample=20)
    body = ds.body_xy; base = body.mean(0)
    radius = float(np.sqrt(2.0 * np.sum((body-base)**2, 1).mean()))
    f = load_arrow_fields(a.arrow); T,H,W = f["T"],f["H"],f["W"]
    xlo,xhi,ylo,yhi = ds.x_lo,ds.x_hi,ds.y_lo,ds.y_hi
    dns_x = ((f["x"][0,:]-xlo)/(xhi-xlo)).astype(np.float32)
    dns_y = ((f["y"][:,0]-ylo)/(yhi-ylo)).astype(np.float32)
    cf = float(getattr(ds,"control_freq",0.0)) or float(f.get("control_freq",0.0))
    axis = tuple(float(z) for z in a.osc_axis.split(","))
    t = np.asarray(f["t"], np.float64)
    sub = max(1, T//400); idx = np.arange(T//5, T, sub)
    tr = infer_trajectory(f["u"][idx], f["v"][idx], tuple(base), cf, t[idx],
                          dns_x, dns_y, axis=axis, search_frac=a.search_frac)
    print(f"control_freq={cf:.4f} reconstructed amp={tr['amp']:.5f} phase={tr['phase']:.4f}")
    ti = np.arange(0,T,20); us,vs = f["u"][ti],f["v"][ti]
    vi = np.sort(ds.val_t_idx); et = vi[::max(1,len(vi)//a.n_eval_times)][:a.n_eval_times]
    np.savez_compressed(a.out,
        sensor_vals=ds.sensor_vals.astype(np.float32), sensor_pos=ds.sensor_pos.astype(np.float32),
        sensor_time=ds.sensor_time.astype(np.float32), obs_mean=ds.observed_channel_mean.astype(np.float32),
        obs_std=ds.observed_channel_std.astype(np.float32), train_t_idx=ds.train_t_idx.astype(np.int32),
        val_t_idx=ds.val_t_idx.astype(np.int32), base_center=base.astype(np.float32),
        body_radius=np.float32(radius), osc_amp=np.float32(tr["amp"]), osc_phase=np.float32(tr["phase"]),
        osc_freq=np.float32(cf), osc_axis=np.asarray(axis,np.float32),
        Lx=np.float32(ds.Lx), Ly=np.float32(ds.Ly), re_value=np.float32(ds.re_value),
        control_freq=np.float32(cf), dns_u=us[et].astype(np.float32), dns_v=vs[et].astype(np.float32),
        dns_x=dns_x, dns_y=dns_y, dns_t_idx=et.astype(np.int32))
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
```

---

## 7. 資料位置 / 前置 / 風險

- **資料**：`~/Documents/coding/RealPDEBench/data/realpdebench/controlled_cylinder/hf_dataset/numerical/*.arrow`
  （numerical 有 p；96 shards；sim_params=[Re, control_freq]，例 [1781, 0.7]）。
  ⚠️ 上個 session 的多次 `hf_hub_download` 到同一 local_dir 可能弄壞了目錄裡的 arrow；若讀到
  `ArrowInvalid`，重新下載乾淨的 shard 到獨立路徑。
- **振幅反推已解**：不需 metadata，`controlled.infer_trajectory` 從場反推（§2）。已用合成資料驗證（test 4 過）。
- **sensor placement**：controlled 無現成 QR JSON，先借 cylinder 的（`--sensor-stem sensors_qrpivot_K100_cylinder_Re10031`）。
- **marginal physics**：physics 是弱約束（r=0.148），重建主靠 sensor data；physics_weight ×0.3。別強推。
- **相關記錄**：`memory/project_realpdebench_multicase_upgrade.md`（gate 結果 + 方向 B）；
  gate 工具 `scripts/check_2d_divergence.py`。
