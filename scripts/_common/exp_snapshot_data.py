"""訓練快照數 ablation（2026-07 實驗1）的 verified 數據——兩份論文的單一來源。

TMLR 圖（`plot_exp_snap_drop.py`）與 thesis 圖（`plot_temporal_density_ablation.py`）
畫的是同一組跑，只是版面與 venue 樣式不同。數字若各抄一份，改一邊忘另一邊就會
讓兩份論文對同一實驗報不同的值，且不會有任何測試抓到，故集中在此。

Provenance：lab-server worktree `~/pi-lnn-jax-expdrop`，jobs 4576-4619（fixed
eval stride）與 4669-4670, 4682-4688（matched Δt），3 seeds = 42/1/2，
Re=10000、K=100、B3、LES_T50 sensor、20k iterations。
判讀見 `knowledge/experiments/kolmogorov-snapshot-dropout-2026-07-26.md`。

兩種評估協定回答不同問題，因此並存而非擇一（CfC 用實際 Δt 前進狀態，
評估時餵入的 Δt 本身就是變因）：
  - EXP1_KE          固定 eval stride=2：所有模型在同一 DNS 格點評估，但
                     T≠101 的模型面臨 train/eval Δt 不匹配。
  - EXP1_KE_MATCHED  matched Δt：每個模型在自身訓練 stride 下評估，分離出
                     「訓練資料量」與「Δt 分佈位移」——後者佔固定-stride
                     曲線退化的 82–83%。
"""
from __future__ import annotations

# 訓練快照數 T（time_stride 1/2/4/8 對應 201/101/51/26 frames over T=5 s）
EXP1_T: list[int] = [201, 101, 51, 26]

EXP1_KE: dict[int, list[float]] = {
    201: [0.18414, 0.18672, 0.18567],
    101: [0.18263, 0.18423, 0.18310],
    51:  [0.21811, 0.22469, 0.22340],
    26:  [0.35117, 0.37227, 0.37355],
}

EXP1_KE_MATCHED: dict[int, list[float]] = {
    201: [0.1824, 0.1844, 0.1833],
    101: [0.1826, 0.1842, 0.1831],
    51:  [0.1893, 0.1896, 0.1903],
    26:  [0.2168, 0.2169, 0.2147],
}

# 儲存軌跡的總時窗與 frame 間隔（Δt 上軸用）
WINDOW_SECONDS: float = 5.0
N_FRAMES_STORED: int = 201
