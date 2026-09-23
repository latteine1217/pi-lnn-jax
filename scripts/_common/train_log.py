"""從訓練 stdout（Slurm .out）還原收斂診斷序列。

訓練迴圈每 `log_every` 印一行定寬表格，欄位為
`step total sensor mom_u mom_v cont poisson C_AL w_d/u/v/c λ_AL w_ph n_co wall`
（見 `pi_lnn_jax/pipeline/kolmogorov/run.py` 的 header）。這是 loss 分項、
GradNorm 任務權重與 Augmented-Lagrangian dual λ 的**唯一**持久化來源——訓練
不另寫結構化 metrics 檔，`artifacts/ledger/` 記的是 RNG 重播欄位而非 loss。

解析採白名單而非黑名單：只認「首欄為整數且恰好 13 欄」的行，其餘（ckpt、
mid-eval、dropout 宣告、分隔線）一律跳過。若整份檔案認不出任何一行，硬失敗
而不是回空表——空表在下游只會畫出一張空圖，錯誤要在這裡就爆。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 定寬表格的欄位數；w_d/u/v/c 佔一欄（內含 "/"）
_N_COLUMNS = 13

# 欄位 index → 輸出鍵。task_weights（index 8）另外攤開成矩陣。
_SCALAR_COLUMNS: dict[int, str] = {
    0: "step",
    1: "total",
    2: "sensor",
    3: "mom_u",
    4: "mom_v",
    5: "cont",
    6: "poisson",
    7: "c_al",
    9: "lambda_al",
    10: "w_phys",
    11: "n_collo",
    12: "wall",
}


def _is_data_row(tokens: list[str]) -> bool:
    if len(tokens) != _N_COLUMNS:
        return False
    try:
        int(tokens[0])
    except ValueError:
        return False
    return "/" in tokens[8]


def parse_training_log(lines) -> dict:
    """訓練 stdout 的行序列 → {欄位名: np.ndarray}。

    回傳的 `task_weights` 形狀為 [n_step, n_task]（GradNorm 是三 task 或四
    task 隨 config 而異，故不寫死）；任務數在中途改變即 fail-fast。
    `step` 保持原順序不排序：resume 會讓 step 重來一段，那是真實情況。
    """
    rows: list[list[str]] = []
    for line in lines:
        tokens = line.split()
        if _is_data_row(tokens):
            rows.append(tokens)
    if not rows:
        raise ValueError(
            "log 中找不到任何訓練資料行（期望首欄整數、共 13 欄）；"
            "可能抓錯檔案，或訓練 log 格式已與 run.py 的 header 漂移"
        )

    weight_counts = {len(r[8].split("/")) for r in rows}
    if len(weight_counts) != 1:
        raise ValueError(
            f"GradNorm task 數在 log 中途改變: {sorted(weight_counts)}；"
            "同一份 log 混了不同 config 的輸出"
        )

    out: dict = {
        key: np.array([float(r[idx]) for r in rows])
        for idx, key in _SCALAR_COLUMNS.items()
    }
    out["step"] = out["step"].astype(int)
    out["n_collo"] = out["n_collo"].astype(int)
    out["task_weights"] = np.array(
        [[float(w) for w in r[8].split("/")] for r in rows]
    )
    return out


def parse_training_log_file(path: str | Path) -> dict:
    """讀檔版本。編碼固定 utf-8（log 內含 λ、ω 等非 ASCII 字元）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"訓練 log 不存在: {p}")
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return parse_training_log(f.readlines())
