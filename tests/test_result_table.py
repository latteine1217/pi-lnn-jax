"""Result table —— 透過介面驗行為，不驗實作細節。

問題（ADR-0004 讀側對稱面）：寫側已由 `EvaluationRunRecorder` 收斂成一個 deep module，
但**讀側**仍散在各 consumer 腳本裡：`compare_baselines` 手刻 `_rows()`（硬挑
`d["rows"]` + 固定鍵）、硬編 method taxonomy、「無 method 欄 → pi-con」啟發式；
`aggregate_pv_campaign` 則 `d["metrics_mean"].get(k, nan)`（漏鍵靜默變 NaN）。
`ResultTable` 把讀取收成一個 seam，metric 存取一律走 metric_artifact 的語意層。

本檔驗的是這個 module 的**介面**（測試面就是介面本身）：

  1. 表能服務 (method, Re, split) → value、能依 method/split 疊代、能盤點在場的
     method/Re/split。
  2. taxonomy 與「無 method 欄 → producer 宣告的 method id（pi-con）」身分規則。
  3. `read_metric_value` 服務 aggregate_pv 的 per-run typed 存取（含 ke_t_mape 雙語意
     一律經 read_metric_summary）。

汙染探針（承 `test_evaluation_run` / `test_ns_residual_parity` 的作法）：

  - 投影鍵被改名／漏掉時，讀該 metric 必須**大聲失敗**（raise），不得靜默回 None
     —— 那正是讀側要拿到的「沒有語意就沒有值」保證。
  - 一份 row 沒有 `method` 欄，必須對到 producer 宣告的 method id（pi-con），而不是
     被靜默標成占位符混進表裡。
"""
from __future__ import annotations

import pytest

from pi_lnn_jax.metric_artifact import (
    KE_T_MAPE_POINTWISE_V2,
    KE_T_MAPE_SPATIALMEAN_V1,
    AmbiguousMetricSemantics,
    LegacyInterpretation,
    MetricUnavailable,
)
from pi_lnn_jax.result_table import (
    ResultTable,
    Split,
    metric_definition_id,
    method_label,
    read_metric_value,
)

_METRIC_KEYS = ["u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err",
                "low_band_rel_err", "div_pred_l2"]


def _metrics(base: float) -> dict:
    """一組可預測的 metric 欄，值隨 base 偏移，方便斷言取到的是「這一格」的值。"""
    return {k: round(base + 0.01 * i, 4) for i, k in enumerate(_METRIC_KEYS)}


def _row(re_value, held_out, base, *, method=None, extra=None):
    """一個 Shape A（多列 projection）的 row：metric 攤平 + identity/aux 欄。"""
    row = {"Re": re_value, "held_out": held_out, "eval_grid": [64, 64], "K": 100,
           "wall_s": 1.5, **_metrics(base)}
    if method is not None:
        row["method"] = method
    if extra:
        row.update(extra)
    return row


@pytest.fixture
def baselines_projection():
    """含 `method` 欄的多列 projection（interp / gappy，兩 Re × split）。"""
    return {"rows": [
        _row(1000, False, 0.10, method="interp_linear"),
        _row(4000, True, 0.20, method="interp_linear"),
        _row(1000, False, 0.30, method="gappy_pod"),
        _row(4000, True, 0.40, method="gappy_pod"),
    ]}


@pytest.fixture
def picon_projection():
    """**無 `method` 欄**的多列 projection（evaluate_multi_re 形；含攤平 + 內嵌
    `metrics_mean` 子塊，兩者都不該弄亂正規化）。"""
    rows = []
    for re_value, held, base in ((1000, False, 0.01), (4000, True, 0.02)):
        m = _metrics(base)
        rows.append({"Re": re_value, "held_out": held, "re_norm": 0.5,
                     "eval_grid": [64, 64], "metrics_mean": m, "wall_s": 2.0, **m})
    return {"rows": rows}


# ── (method, Re, split) 服務 ───────────────────────────────────────────────
def test_table_serves_method_re_split_value(baselines_projection):
    table = ResultTable.from_projection(baselines_projection)
    du = metric_definition_id("u_rel_err")
    # (interp, Re=1000) 的 u_rel_err = base 0.10 的第一個 metric。
    assert table.value("interp_linear", 1000, du) == pytest.approx(0.10)
    assert table.value("gappy_pod", 4000, du) == pytest.approx(0.40)
    # split 由 held_out 導出。
    assert table.cell("interp_linear", 1000).split is Split.IN_TRAIN
    assert table.cell("interp_linear", 4000).split is Split.HELD_OUT
    # 不在場的格回 None（不是 raise）。
    assert table.value("shred", 1000, du) is None


def test_inventory_methods_res_splits(baselines_projection):
    table = ResultTable.from_projection(baselines_projection)
    # method 依 taxonomy 規範順序（interp 在 gappy 前）。
    assert table.methods() == ("interp_linear", "gappy_pod")
    assert table.reynolds_numbers() == (1000.0, 4000.0)
    assert set(table.splits()) == {Split.IN_TRAIN, Split.HELD_OUT}
    assert table.reynolds_split(1000) is Split.IN_TRAIN
    assert table.reynolds_split(4000) is Split.HELD_OUT


def test_iteration_by_method_and_split(baselines_projection):
    table = ResultTable.from_projection(baselines_projection)
    du = metric_definition_id("u_rel_err")
    held = table.units(method="interp_linear", split=Split.HELD_OUT)
    assert [u.reynolds for u in held] == [4000.0]
    # per-(method,split) 均值分組（compare_baselines.mean 的疊代面）。
    in_train = [u.value(du) for u in table.units(method="gappy_pod", split=Split.IN_TRAIN)]
    assert in_train == [pytest.approx(0.30)]


# ── taxonomy + pi-con 身分規則 ─────────────────────────────────────────────
def test_no_method_column_maps_to_producer_method_id(picon_projection):
    # 「無 method 欄 → 此 producer 的 method id」＝ pi-con。
    table = ResultTable.from_projection(picon_projection, default_method="pi-con")
    assert table.methods() == ("pi-con",)
    unit = table.cell("pi-con", 1000)
    assert unit is not None and unit.method == "pi-con"
    assert unit.value(metric_definition_id("u_rel_err")) == pytest.approx(0.01)


def test_missing_method_and_no_default_raises(picon_projection):
    # 沒 method 欄又沒宣告 default → 大聲失敗，不靜默標占位符。
    with pytest.raises(ValueError, match="no 'method'"):
        ResultTable.from_projection(picon_projection)


def test_method_taxonomy_labels():
    assert method_label("pi-con") == "PI-CON"
    assert method_label("gappy_pod") == "gappy-POD (per-Re)"
    # 未登錄的 method 退回其 id（不猜、不失敗）。
    assert method_label("mystery_method") == "mystery_method"


def test_merge_projections_builds_four_way_table(baselines_projection, picon_projection):
    # compare_baselines 的四方合成：三份 projection 併成一張表。
    table = ResultTable.from_projections([
        baselines_projection,
        (picon_projection, "pi-con"),
    ])
    assert table.methods() == ("interp_linear", "gappy_pod", "pi-con")
    assert table.reynolds_numbers() == (1000.0, 4000.0)


def test_duplicate_unit_raises(baselines_projection):
    # 同一 (method, Re, split) 載入兩次 → raise，不靜默覆蓋。
    with pytest.raises(ValueError, match="duplicate"):
        ResultTable.from_projections([baselines_projection, baselines_projection])


# ── 汙染探針：投影鍵漂移必須大聲失敗 ────────────────────────────────────────
def test_renamed_metric_key_raises_not_silent_none(baselines_projection):
    # 把 u_rel_err 改名成 u_relerr（模擬投影鍵漂移）：該 metric 被讀時必須 raise，
    # 而不是靜默回 None（那會讓錯的空值混進論文表）。
    rows = baselines_projection["rows"]
    rows[0]["u_relerr"] = rows[0].pop("u_rel_err")
    table = ResultTable.from_projection(baselines_projection)
    with pytest.raises(MetricUnavailable, match="u_rel_err"):
        table.value("interp_linear", 1000, metric_definition_id("u_rel_err"))
    # 對照：同表其他格照常讀得到（漂移不是全表壞掉）。
    assert table.value("gappy_pod", 1000, metric_definition_id("u_rel_err")) == pytest.approx(0.30)


def test_null_metric_value_is_none_not_raise(baselines_projection):
    # 鍵在、值為 null（合法的 not-applicable）→ 回 None，不 raise。這是「空格」與
    # 「漂移」的分野。
    baselines_projection["rows"][0]["low_band_rel_err"] = None
    table = ResultTable.from_projection(baselines_projection)
    assert table.value("interp_linear", 1000, metric_definition_id("low_band_rel_err")) is None


def test_unregistered_projection_key_raises():
    with pytest.raises(KeyError, match="unregistered projection key"):
        metric_definition_id("not_a_metric")


# ── read_metric_value：aggregate_pv 的 per-run typed 存取 ────────────────────
def _metrics_json(base: float) -> dict:
    """一份單筆 metrics.json（Shape B）：metrics_mean 子塊 + ke_t_errors。"""
    mm = _metrics(base)
    return {
        "metrics_mean": mm,
        "metrics_per_t": [{"t": 0.0, **mm}],
        "ke_t_errors": {"ke_t_mape": 6.81, "ke_t_mape_spatialmean": 5.73,
                        "ke_mape_def": "pointwise_v2"},
    }


def test_read_metric_value_strict_field_access():
    d = _metrics_json(0.5)
    # 逐欄嚴格讀（aggregate_pv 的 FIELDS 面）。
    assert read_metric_value(d, metric_definition_id("div_pred_l2")) == pytest.approx(0.55)
    assert read_metric_value(d, metric_definition_id("v_rel_err")) == pytest.approx(0.51)
    # 漏鍵 → raise（不像原本 .get(k, nan) 靜默變 NaN）。
    del d["metrics_mean"]["div_pred_l2"]
    with pytest.raises(MetricUnavailable, match="div_pred_l2"):
        read_metric_value(d, metric_definition_id("div_pred_l2"))


def test_read_metric_value_ke_mape_dual_semantics_via_summary():
    d = _metrics_json(0.5)
    # ke_t_mape 雙語意一律經 read_metric_summary：pointwise 與 spatialmean 取到不同的量。
    assert read_metric_value(d, KE_T_MAPE_POINTWISE_V2) == pytest.approx(6.81)
    assert read_metric_value(d, KE_T_MAPE_SPATIALMEAN_V1) == pytest.approx(5.73)


def test_ke_mape_ambiguous_without_marker_raises():
    # 未標記語意的 legacy artifact（有 ke_t_mape、無 ke_mape_def）讀 spatialmean：
    # 沒給 legacy interpretation 就 raise（read_metric_summary 拒絕猜）。
    d = {"metrics_mean": _metrics(0.5), "ke_t_errors": {"ke_t_mape": 5.7}}
    with pytest.raises(AmbiguousMetricSemantics):
        read_metric_value(d, KE_T_MAPE_SPATIALMEAN_V1)
    # 給了明示 interpretation 就讀得到（aggregate_pv 正是這樣做）。
    got = read_metric_value(
        d, KE_T_MAPE_SPATIALMEAN_V1,
        legacy_interpretation=LegacyInterpretation(
            definition_id=KE_T_MAPE_SPATIALMEAN_V1,
            basis="placement-variance campaign used spatial-mean E(t)"),
    )
    assert got == pytest.approx(5.7)
