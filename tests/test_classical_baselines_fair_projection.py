"""tab:fair 的每-method 列在遷到 metric artifact seam 前後必須逐鍵逐值相同。

這張表進論文。遷移把「per-t 迴圈 + np.mean + 手寫 ke_mape_def」換成
`metric_artifact.evaluate_field_series`，本檔把**舊公式原文**留作參考實作，
證明換掉的是路徑不是數字。舊實作抄在這裡是刻意的——它是斷言的依據，
從 production 讀回來就不是獨立參考了。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pi_lnn_jax.metric_artifact import (
    EvaluationContext,
    SourceProvenance,
    compute_metrics,
    energy_timeseries_errors,
    evaluate_field_series,
)
from scripts.classical_baselines_fair import method_agg


def _reference_agg(u_rec, v_rec, dns_u, dns_v, *, nu):
    """遷移前 classical_baselines_fair.main 的聚合，逐行照抄。"""
    T = u_rec.shape[0]
    agg_keys = ["uv_rel_err", "u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err",
                "ke_pw_mape", "ke_pw_nmae"]
    if nu is not None:
        agg_keys += ["band_rel_err_low", "band_rel_err_mid", "gamma_low", "gamma_mid"]

    ke_p = np.zeros(T); ke_d = np.zeros(T); per_t = []
    for t in range(T):
        m = compute_metrics(u_rec[t], v_rec[t], dns_u[t], dns_v[t], nu=nu)
        per_t.append(m)
        ke_p[t] = 0.5 * (u_rec[t] ** 2 + v_rec[t] ** 2).mean()
        ke_d[t] = 0.5 * (dns_u[t] ** 2 + dns_v[t] ** 2).mean()
    agg = {k: float(np.mean([mm[k] for mm in per_t])) for k in agg_keys}
    ket = energy_timeseries_errors(ke_p, ke_d)
    agg["ke_t_mape"] = agg["ke_pw_mape"]
    agg["ke_t_mape_spatialmean"] = ket["ke_t_mape_spatialmean"]
    agg["ke_t_rel_l2"] = ket["ke_t_rel_l2"]
    agg["ke_mape_def"] = "pointwise_v2"
    return agg


def _fields(seed=3, T=4, N=32):
    # N=32 是下界：N=16 配 Re=1e4 時 k_cut=11 遠低於 k_eta=116 導出的 mid band
    # 下界，band/γ 欄位整欄 NaN，聚合層會直接拒收（見 test_coarse_grid_… ）。
    rng = np.random.default_rng(seed)
    dns_u = rng.normal(size=(T, N, N))
    dns_v = rng.normal(size=(T, N, N))
    # 重建場：帶偏差但同階，讓每個 metric 都落在非退化區。
    u_rec = dns_u * 0.88 + 0.03 * rng.normal(size=(T, N, N))
    v_rec = dns_v * 1.07 + 0.03 * rng.normal(size=(T, N, N))
    return u_rec, v_rec, dns_u, dns_v


def _through_seam(u_rec, v_rec, dns_u, dns_v, *, nu):
    T, N = u_rec.shape[0], u_rec.shape[1]
    context = EvaluationContext(
        case="kolmogorov", times=tuple(float(t) for t in range(T)),
        periodic=True, domain_length=1.0, viscosity=nu,
        pressure_evaluated=False, grid_shape=(N, N),
    )
    _, projection = evaluate_field_series(
        u_rec, v_rec, dns_u, dns_v, context=context,
        provenance=SourceProvenance(
            producer="tests", code_revision="deadbeef", code_dirty=False,
        ),
    )
    return method_agg(projection, banded=nu is not None)


@pytest.mark.parametrize("nu", [None, 1.0 / 10000.0])
def test_projection_reproduces_the_previous_row(nu):
    u_rec, v_rec, dns_u, dns_v = _fields()

    reference = _reference_agg(u_rec, v_rec, dns_u, dns_v, nu=nu)
    through = _through_seam(u_rec, v_rec, dns_u, dns_v, nu=nu)

    # 欄位與順序：tab:fair 逐欄讀，順序漂移比數值漂移更難察覺。
    assert list(through) == list(reference)
    assert through["ke_mape_def"] == reference["ke_mape_def"]
    for key in reference:
        if key == "ke_mape_def":
            continue
        assert through[key] == pytest.approx(reference[key], rel=0.0, abs=1e-12), key


def test_band_columns_are_absent_without_a_reynolds_number():
    """未給 --re 時 band/γ 欄位必須缺席——不猜、不填 0、不套預設。"""
    u_rec, v_rec, dns_u, dns_v = _fields()

    through = _through_seam(u_rec, v_rec, dns_u, dns_v, nu=None)

    for key in ("band_rel_err_low", "band_rel_err_mid", "gamma_low", "gamma_mid"):
        assert key not in through


def test_coarse_grid_keeps_the_nan_band_column_out_of_the_published_row():
    """**行為變更（刻意）**：grid 相對 k_eta 太粗時 band/γ 逐幀無定義。

    舊路徑把 NaN 直接寫進 tab_fair_metrics.json 的欄位。`json.dump` 會把它寫成
    字面 `NaN`——不是合法 JSON，而且下游印出來像個數字。新路徑在 artifact 裡
    記為 NOT_APPLICABLE，投影落 `null`，消費端索引時會大聲失敗。

    `--max-grid` 可粗化格點，所以這條路徑在 production 可達，不是理論風險。
    """
    u_rec, v_rec, dns_u, dns_v = _fields(N=16)

    through = _through_seam(u_rec, v_rec, dns_u, dns_v, nu=1.0 / 10000.0)

    assert through["gamma_mid"] is None
    # 有定義的欄位照常產出——不是整組 band 一起消失。
    assert np.isfinite(through["uv_rel_err"])
    # 投影必須是合法 JSON（舊路徑在此會寫出字面 NaN）。
    json.dumps(through, allow_nan=False)


def test_ke_semantics_marker_comes_from_the_package_not_the_script():
    """ke_mape_def 由 aggregate_metric_rows 寫入；腳本不得再自己寫這個字串。"""
    import inspect

    import scripts.classical_baselines_fair as mod

    src = inspect.getsource(mod)
    assert '"pointwise_v2"' not in src, "腳本又手寫了語意標記"
    assert "ke_mape_def" in src, "投影仍須帶出語意標記"
