"""`scripts/diag_steady_state.py` 的判定閘門。

這道閘門決定要不要投入 N=1024 DNS + 四臂重訓（半天）。它必須在兩個方向都會擋：
仍在衰減（沒到穩態）要擋，波動太小（靜止/近層流，外推平庸地容易）也要擋。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diag_steady_state import steady_verdict  # noqa: E402


def _series(t, level, drift=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return level + drift * (t - t[0]) + noise * rng.standard_normal(t.size)


T = np.linspace(20.0, 30.0, 101)


def test_plateau_with_fluctuation_is_steady():
    q = _series(T, level=8.0, drift=0.0, noise=0.8)
    v = steady_verdict(T, q, min_cv=0.05)
    assert v["steady"] and v["plateau"] and v["fluctuating"], v
    print("✓ plateau_with_fluctuation_is_steady")


def test_still_decaying_is_rejected():
    """仍在衰減 → 穩態窗還沒到，必須擋。"""
    q = _series(T, level=8.0, drift=-0.4, noise=0.2)   # 窗內掉 4，遠大於雜訊
    v = steady_verdict(T, q, min_cv=0.05)
    assert not v["steady"] and not v["plateau"], v
    print("✓ still_decaying_is_rejected")


def test_frozen_solution_is_rejected():
    """幾乎沒波動 → 靜止/近層流解，外推平庸地容易，同樣必須擋。"""
    q = _series(T, level=8.0, drift=0.0, noise=1e-4)
    v = steady_verdict(T, q, min_cv=0.05)
    assert not v["steady"] and v["plateau"] and not v["fluctuating"], v
    print("✓ frozen_solution_is_rejected")


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="不足以判斷趨勢"):
        steady_verdict(np.linspace(0, 1, 4), np.ones(4), min_cv=0.05)
    print("✓ too_few_points_raises")
