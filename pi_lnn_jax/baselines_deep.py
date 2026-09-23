"""Phase B 深度監督式 baseline（需 GPU 訓練；本機僅 isolated unit 驗證）。

目前含：
  - SHRED（Williams et al. 2024）：LSTM over sensor 時間窗 → 末端 hidden → shallow MLP decoder
    → 全場（flattened）。full-field 監督、離散時間 RNN，作為 PI-CON（CfC, continuous-time,
    sensor-only）的外部對照（Tier 0.1）；其 LSTM 亦對照「continuous-time dynamical」賣點。

注意（與 PI-CON 的差異，論文須說明）:
  - SHRED 需 dense full-field 監督訓練（吃比 PI-CON 多的資訊）。
  - SHRED 輸出固定格（非 query-anywhere）；離散時間 RNN（非 CfC 連續時間狀態）。
  - SHRED 需長度 L 的時間窗，故只能重建 t≥L-1 的快照。

完整訓練在 scripts/train_baseline_shred.py（lab-server GPU）。本檔只放可單測的 model + 純函式。
"""
from __future__ import annotations

import numpy as np
from flax import linen as nn


class SHRED(nn.Module):
    """Shallow REcurrent Decoder：LSTM 編碼 sensor 時間窗 → shallow MLP 解出全場。

    輸入 x: [B, L, K*C]（L 個時間步、K 感測點、C 通道攤平）。
    輸出  : [B, out_dim]（全場 flattened，out_dim = Nx*Ny*C）。
    """

    out_dim: int
    hidden: int = 64
    decoder_hidden: tuple = (350, 400)

    @nn.compact
    def __call__(self, x):
        seq = nn.RNN(nn.LSTMCell(features=self.hidden))(x)  # [B, L, hidden]
        z = seq[:, -1, :]                                   # 末端 hidden state
        for d in self.decoder_hidden:
            z = nn.relu(nn.Dense(d)(z))
        return nn.Dense(self.out_dim)(z)


def build_windows(sensor_vals, fields, L: int):
    """把 (sensor 時間序列, 全場序列) 切成 SHRED 的 (window, target) 對。

    sensor_vals [T,K,C] → X [T-L+1, L, K*C]；fields [T, n_field] → Y [T-L+1, n_field]。
    第 i 個 window 對應時間 [i, i+L)，target = 末端時間 (i+L-1) 的 field（causal，對齊真實部署）。
    """
    sensor_vals = np.asarray(sensor_vals)
    fields = np.asarray(fields)
    T, K = sensor_vals.shape[0], sensor_vals.shape[1]
    if T < L:
        raise ValueError(f"序列長度 T={T} < 窗長 L={L}，無法切窗")
    xs, ys = [], []
    for t in range(L - 1, T):
        xs.append(sensor_vals[t - L + 1:t + 1].reshape(L, K * sensor_vals.shape[2]))
        ys.append(fields[t])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
