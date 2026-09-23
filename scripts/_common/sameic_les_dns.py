"""同一初始條件（same-IC）的 LES／DNS 場對：載入與一致性檢查。

What:
    `load_sameic_pair()` 取回 LES 與 DNS 的 `(u, v, t)`，並在回傳前檢查三件
    必須成立的事：兩邊時間軸逐點相同、網格同形、`t=0` 的場一致（相關係數
    ≈ 1）。任何一項不成立就 raise 並點名路徑。

Why:
    這對資料餵給 thesis 附錄 A.2 的兩張圖（`fig:les_predictability`、
    `fig:les_vort_compare`）。兩張圖整條敘事都建立在「LES 從 DNS 的場起步」
    這個前提上——前提若不成立，畫出來的「可預報時長」就不是可預報時長，而
    圖上看不出任何異狀。故前提用 assert 表達，不用註解表達。

    `t=0` 檢查另外擋掉一個具體的坑：LES 與 DNS 兩支求解器的速度**正負號約定**
    曾經相反（`home-gpu:~/les-gen/sameic_compare.py` 為此寫了自動翻號）。自動
    翻號會把「讀錯檔」也一起修掉，所以這裡只檢查、不修正——真的需要翻號時，
    正確的作法是回去修生成端的約定。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pi_lnn_jax.data import load_dns_from_path


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相關係數（攤平後去均值）。"""
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def load_sameic_pair(les_path: str | Path, dns_path: str | Path):
    """載入 same-IC 的 LES／DNS 對。

    Returns:
        (lu, lv, du, dv, t)，皆 float64；速度場形狀 [T, N, N]。

    Raises:
        FileNotFoundError: 任一檔不存在（`load_dns_from_path` 負責點名路徑）。
        ValueError: 時間軸、網格或初始場不一致。
    """
    lu, lv, lt = load_dns_from_path(les_path)   # LES 與 DNS 同一 dict 慣例
    du, dv, dt = load_dns_from_path(dns_path)

    if lu.shape != du.shape:
        raise ValueError(
            f"LES 與 DNS 形狀不符: {lu.shape} vs {du.shape}\n"
            f"  LES: {les_path}\n  DNS: {dns_path}"
        )
    if lt.shape != dt.shape or not np.allclose(lt, dt, atol=1e-9):
        raise ValueError(
            f"LES 與 DNS 的時間軸不同（{lt.shape} vs {dt.shape}）——"
            "本圖逐格比對同一時刻，不得靠最近鄰對齊。\n"
            f"  LES: {les_path}\n  DNS: {dns_path}"
        )

    lu = lu.astype(np.float64)
    lv = lv.astype(np.float64)
    du = du.astype(np.float64)
    dv = dv.astype(np.float64)

    c0 = min(_corr(lu[0], du[0]), _corr(lv[0], dv[0]))
    if c0 < 0.999:
        raise ValueError(
            f"t=0 的 LES 與 DNS 場不一致（corr={c0:.4f}）：這不是 same-IC 的一對。\n"
            "  若 corr 接近 −1，是兩支求解器的速度正負號約定相反——請修生成端，"
            "不要在繪圖端翻號。\n"
            f"  LES: {les_path}\n  DNS: {dns_path}"
        )
    return lu, lv, du, dv, np.asarray(lt, dtype=np.float64)
