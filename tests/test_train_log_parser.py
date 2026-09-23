"""parse_training_log 的 isolated 單元測試（純 stdlib + numpy）。

訓練的收斂診斷（loss 分項 / GradNorm 權重 / AL dual λ）只存在於 stdout，
沒有結構化輸出，所以繪圖端唯一的資料來源是 Slurm 的 .out。解析器因此要能
在混著 ckpt、mid-eval、dropout 等雜訊 print 的檔案裡只認訓練資料行，
而遇到「一行都認不出來」時硬失敗——那代表 log 格式已經漂移，靜默回空表
會讓下游畫出一張空圖。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common.train_log import parse_training_log  # noqa: E402


HEADER = ("  step       total      sensor       mom_u       mom_v        cont"
          "    poisson        C_AL              w_d/u/v/c    λ_AL    w_ph  n_co    wall")


def _row(step: int, total: float, lam: float, w: str = "1.00/0.50/0.50/0.30") -> str:
    return (f"{step:>6d} {total:>11.4e} {3.4e-2:>11.4e} {1.0e-3:>11.4e} "
            f"{2.0e-3:>11.4e} {5.0e-4:>11.4e} {1.0e-5:>10.3e} {2.3e-3:>11.4e} "
            f"{w:>22s} {lam:>7.3f} {0.05:>7.4f} {1024:>5d} {3.21:>7.2f}")


def test_parses_scalar_columns():
    lines = [HEADER, "-" * 80, _row(100, 1.2e-1, 0.123), _row(200, 1.0e-1, 0.456)]

    out = parse_training_log(lines)

    np.testing.assert_array_equal(out["step"], [100, 200])
    np.testing.assert_allclose(out["total"], [1.2e-1, 1.0e-1])
    np.testing.assert_allclose(out["lambda_al"], [0.123, 0.456])
    np.testing.assert_allclose(out["n_collo"], [1024, 1024])


def test_task_weights_split_into_matrix():
    """w_d/u/v/c 是一個 token，攤成 [n_step, n_task] 才畫得出權重軌跡。"""
    lines = [HEADER,
             _row(100, 1.0e-1, 0.1, w="1.00/0.50/0.25/0.10"),
             _row(200, 1.0e-1, 0.1, w="1.00/0.60/0.30/0.20")]

    out = parse_training_log(lines)

    assert out["task_weights"].shape == (2, 4)
    np.testing.assert_allclose(out["task_weights"][1], [1.00, 0.60, 0.30, 0.20])


def test_three_task_weights_also_supported():
    """GradNorm 是三 task 還是四 task 隨 config 而異（continuity 可能 AL-only）。"""
    lines = [HEADER, _row(100, 1.0e-1, 0.1, w="1.00/0.50/0.25")]

    out = parse_training_log(lines)

    assert out["task_weights"].shape == (1, 3)


def test_noise_lines_are_ignored():
    """ckpt / mid-eval / dropout 等 print 混在同一份 log，必須跳過而非炸掉。"""
    lines = [
        "=" * 80,
        "=== Training: step 1 → 20000 ===",
        HEADER,
        "-" * 80,
        "[dropout] train-time sensor dropout rate=0.0 (denoising：mask 輸入)",
        _row(100, 1.2e-1, 0.123),
        "  [ckpt] saved step=100 → artifacts/...",
        "  --- mid-eval at step 100 ---",
        "  t=0.00: KE rel-err=0.183  low_band=0.043  ω=0.421",
        _row(200, 1.0e-1, 0.456),
        "Total training wall: 1234.56s",
    ]

    out = parse_training_log(lines)

    np.testing.assert_array_equal(out["step"], [100, 200])


def test_inconsistent_task_count_raises():
    """權重欄位數在中途改變 → 硬失敗（拼成 ragged 會在繪圖時才炸）。"""
    lines = [HEADER,
             _row(100, 1.0e-1, 0.1, w="1.00/0.50/0.25/0.10"),
             _row(200, 1.0e-1, 0.1, w="1.00/0.50/0.25")]

    with pytest.raises(ValueError, match="task"):
        parse_training_log(lines)


def test_no_data_rows_raises():
    """認不出任何訓練行 = 格式漂移或抓錯檔，不可回空表。"""
    with pytest.raises(ValueError, match="訓練資料行"):
        parse_training_log([HEADER, "-" * 80, "Total training wall: 1.0s"])


def test_monotonic_step_not_required_but_duplicates_flagged():
    """resume 會讓 step 重來一段；重複的 step 是真實情況，保留但不排序。"""
    lines = [HEADER, _row(200, 1.0e-1, 0.1), _row(100, 1.2e-1, 0.2)]

    out = parse_training_log(lines)

    np.testing.assert_array_equal(out["step"], [200, 100])
