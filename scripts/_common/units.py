"""軸標籤的單位括號：thesis 標 SI 單位，PoF（各繪圖腳本的 `--nondim`）不標。

Why：Kolmogorov 案本身已無因次（L*=1、U*=1、Re=1/ν），數值不因標示而變；
thesis 已驗收維持 SI 標示，PoF 稿改為無因次標示。只換標籤文字，不動任何數值。
"""
from __future__ import annotations


def unit(u: str, nondim: bool) -> str:
    """回傳接在量名後的 `" (u)"`；`nondim` 時回傳空字串。"""
    return "" if nondim else f" ({u})"
