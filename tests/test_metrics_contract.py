"""`compute_metrics` 回傳哪些 key，由參數決定——把那個契約釘成表。

問題（架構深化候選 3）：回傳是條件式的字串 dict。三個正交的閘（`periodic`、
`p_pred`、`nu`）產出 13–33 個 key，而呼叫端只能字串索引，從簽名看不出這次
會拿到什麼。下游 16 處索引都是「知道就知道」。

友善提醒不是重點，**重點是它會反覆咬人**：本 repo 在兩週內兩次擴充這個 dict
（forcing-mode 診斷 +4、metric suite 重設計 +5），兩次都得有人**手寫一次性探針**
——用同一組合成輸入實跑新舊兩版、diff key 集——才能回答「這會不會讓
bit-identical 對拍假紅」。同一支探針寫了兩遍，第三次一定會再來。

本檔把那支探針變成常設測試：每個參數組合的 key 集被釘住，加了 metric 就紅，
且失敗訊息直接說出「哪個組合多了／少了哪些 key」——不必再手寫探針。

**這不是要凍結介面**，是要讓擴充變成一件需要明說的事。加 metric 時更新下表
即可，而更新的動作本身就會逼你回答「它屬於哪個閘、誰會拿到」。

⚠️ `evaluate.py` 是兩案共用的 leaf module。改它必須同步兩個基準分支，
否則下次對拍會因兩側 key 數不同而報**假紅**（見 `CLAUDE.md` §7.1 第 4 點）。
"""
from __future__ import annotations

import numpy as np
import pytest

from pi_lnn_jax.evaluate import compute_metrics

_H = _W = 32


def _fields():
    r = np.random.RandomState(0)
    return tuple(r.randn(_H, _W) for _ in range(4))


def _nonperiodic_kwargs():
    return dict(dns_x=np.linspace(0, 1, _W), dns_y=np.linspace(0, 1, _H),
                Lx=0.32, Ly=0.17, mask=np.ones((_H, _W), bool))


#: 恆有的核心量（13）——下游 16 處索引讀的都在這裡。任何組合都不得缺。
CORE = frozenset({
    "div_dns_l2", "div_pred_l2", "enstrophy_rel_err", "ke_dns_mean",
    "ke_pred_mean", "ke_pred_over_ref", "ke_pw_mape", "ke_pw_nmae",
    "ke_rel_err", "omega_rel_err", "u_rel_err", "uv_rel_err", "v_rel_err",
})

#: 各閘各自帶進來的 key。分開列而非列總集，讀者才看得出「哪個閘負責什麼」。
#: 這四張表由實測填入（手列會漏——初版就漏了 CORE 的四個）。

SPECTRAL = frozenset({          # periodic=True 才有（需週期性 FFT）
    "E_dns_k", "E_pred_k", "div_ratio", "gamma_k", "k_cut",
    "kf_amp_ratio", "kf_mode_amp_dns", "kf_mode_amp_pred",
    "kf_mode_phase_dns", "kf_mode_phase_pred", "low_band_rel_err",
    "spectrum_rel_err",
})
PRESSURE = frozenset({"p_rel_err"})   # p_pred 與 p_dns 都給才有
DISSIPATION = frozenset({       # nu 有給才有（需黏滯係數才定義得出 k_eta）
    "band_rel_err_high", "band_rel_err_low", "band_rel_err_mid",
    "gamma_high", "gamma_low", "gamma_mid", "k_eta", "kcut_over_keta",
    # k_d 是 2D 的 Kraichnan 尺度，僅診斷；band 邊界仍由 k_eta 定義。
    "k_d_kraichnan", "kd_over_keta",
})

#: 組合 → 額外閘。核心 CORE 一律隱含。
COMBOS = {
    "periodic": ({}, frozenset()),
    "periodic+pressure": ({"pressure": True}, PRESSURE),
    "periodic+nu": ({"nu": True}, DISSIPATION),
    "nonperiodic": ({"nonperiodic": True}, frozenset()),
    "nonperiodic+pressure": ({"nonperiodic": True, "pressure": True}, PRESSURE),
}


def _call(flags: dict) -> frozenset:
    u_p, v_p, u_d, v_d = _fields()
    kw: dict = {}
    if flags.get("nonperiodic"):
        kw.update(periodic=False, **_nonperiodic_kwargs())
    else:
        kw.update(periodic=True)
    if flags.get("pressure"):
        r = np.random.RandomState(1)
        kw.update(p_pred=r.randn(_H, _W), p_dns=r.randn(_H, _W))
    if flags.get("nu"):
        kw.update(nu=1e-3)
    return frozenset(compute_metrics(u_p, v_p, u_d, v_d, **kw))


@pytest.mark.parametrize("name", list(COMBOS), ids=list(COMBOS))
def test_key_set_matches_the_declared_contract(name):
    flags, extra = COMBOS[name]
    want = CORE | extra | (frozenset() if flags.get("nonperiodic") else SPECTRAL)

    got = _call(flags)

    assert got == want, (
        f"組合 {name!r} 的 key 集與宣告不符——\n"
        f"    多出：{sorted(got - want) or '—'}\n"
        f"    缺少：{sorted(want - got) or '—'}\n"
        "  這正是每次擴充 metric 時要回答的問題：新 key 屬於哪個閘、誰會拿到。\n"
        "  更新本檔的 CORE / SPECTRAL / PRESSURE / DISSIPATION 表即可；\n"
        "  ⚠️ evaluate.py 是共用 leaf——改完要同步兩個基準分支，否則對拍假紅"
        "（CLAUDE.md §7.1 第 4 點）。")


def test_core_keys_are_present_in_every_combination():
    """下游 16 處索引讀的都是核心量。任何組合缺了它們，那些呼叫端就會 KeyError。"""
    for name, (flags, _) in COMBOS.items():
        missing = CORE - _call(flags)
        assert not missing, f"組合 {name!r} 缺核心量 {sorted(missing)}"


def test_each_gate_actually_gates_something():
    """自證：三張閘表若有一張其實是空的（例如某個閘被拿掉了），
    上面的比對仍會全綠——那時契約表就變成一份沒在描述任何東西的裝飾。"""
    periodic_only = _call({}) - _call({"nonperiodic": True})
    assert periodic_only == SPECTRAL, (
        f"SPECTRAL 表與實際不符：實際只在 periodic 出現的是 {sorted(periodic_only)}")

    assert _call({"pressure": True}) - _call({}) == PRESSURE
    assert _call({"nu": True}) - _call({}) == DISSIPATION
