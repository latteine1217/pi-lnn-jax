"""Metric artifact public seam: semantic summaries, not storage-key lookups."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pi_lnn_jax.metric_artifact import (
    AmbiguousMetricSemantics,
    EvaluationContext,
    KE_T_MAPE_SPATIALMEAN_V1,
    LegacyInterpretation,
    MetricArtifact,
    MetricAvailability,
    MetricMeasurement,
    MetricUnavailable,
    MetricSummary,
    SourceProvenance,
    build_metric_artifact,
    evaluate_field_series,
    read_metric_summary,
    VORTICITY_REL_ERR_TIME_MEAN_V1,
)


def _evaluation_result():
    return {
        "ke_t_pred": [1.0, 2.0],
        "ke_t_dns": [1.25, 2.5],
        "metrics_per_t": [
            {
                "uv_rel_err": 0.1,
                "omega_rel_err": 0.2,
                "ke_pred_mean": 1.0,
                "ke_dns_mean": 1.25,
                "gamma_high": float("nan"),
                "E_pred_k": [1.0, 0.25],
                "E_dns_k": [1.0, 0.5],
                "gamma_k": [1.0, float("nan")],
                "t": 0.0,
            },
            {
                "uv_rel_err": 0.3,
                "omega_rel_err": 0.4,
                "ke_pred_mean": 2.0,
                "ke_dns_mean": 2.5,
                "gamma_high": 0.75,
                "E_pred_k": [0.8, 0.2],
                "E_dns_k": [1.0, 0.4],
                "gamma_k": [0.9, 0.5],
                "t": 1.0,
            },
        ],
        "metrics_mean": {"uv_rel_err": 0.2, "omega_rel_err": 0.3},
        "metrics_p90": {"uv_rel_err": 0.28, "omega_rel_err": 0.38},
        "ke_t_errors": {
            "ke_t_mape": 0.2,
            "ke_t_mape_spatialmean": 0.2,
            "ke_t_rel_l2": 0.2,
            "ke_t_rel_linf": 0.2,
            "ke_t_final_rel": 0.2,
            "ke_t_nmae": 0.2,
            "ke_mape_def": "pointwise_v2",
        },
        "band_high_diag": {"n_valid_frames": 1, "n_total_frames": 2},
    }


def _artifact_with_summary(metrics_mean_extra, *, per_t_extra):
    """把額外的 metric 塞進 per-t 與 metrics_mean，建出 artifact。"""
    evaluation = _evaluation_result()
    for row, extra in zip(evaluation["metrics_per_t"], per_t_extra, strict=True):
        row.update(extra)
    evaluation["metrics_mean"].update(metrics_mean_extra)
    return build_metric_artifact(
        evaluation,
        context=EvaluationContext(
            case="kolmogorov", times=(0.0, 1.0), periodic=True,
            domain_length=1.0, viscosity=1e-4, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
    )


def _summary_for(artifact, definition_id):
    return next(s for s in artifact.summaries if s.definition_id == definition_id)


_BAND_MID_V1 = "spectrum.energy.rel_l2.mid.v1"


def test_summary_of_an_always_inapplicable_band_is_not_applicable_not_an_error():
    """grid 相對 k_eta 太粗 → band 逐幀無定義。那是 evaluation condition，不是錯誤。

    per-time 層本來就把這些鍵的 NaN 記為 NOT_APPLICABLE；summary 層先前無條件
    raise，與模組自己的 availability 詞彙矛盾。
    """
    artifact = _artifact_with_summary(
        {"band_rel_err_mid": float("nan")},
        per_t_extra=[{"band_rel_err_mid": float("nan")},
                     {"band_rel_err_mid": float("nan")}],
    )

    summary = _summary_for(artifact, _BAND_MID_V1)
    assert summary.availability is MetricAvailability.NOT_APPLICABLE
    assert summary.value is None
    assert summary.sample_count == 0


def test_summary_of_a_partially_measured_band_is_unavailable():
    """部分幀有定義 → 聚合值取不出來，但不是「條件上不適用」。

    兩者分開才看得出「這個設定根本測不到」與「這批證據不夠算出聚合」的差別；
    sample_count 記下實際有定義的幀數。
    """
    artifact = _artifact_with_summary(
        {"band_rel_err_mid": float("nan")},
        per_t_extra=[{"band_rel_err_mid": 0.05},
                     {"band_rel_err_mid": float("nan")}],
    )

    summary = _summary_for(artifact, _BAND_MID_V1)
    assert summary.availability is MetricAvailability.UNAVAILABLE
    assert summary.value is None
    assert summary.sample_count == 1


def test_non_finite_summary_outside_the_eligibility_set_still_fails():
    """寬容只給 band/γ 那組。速度誤差是 NaN 就是壞了，必須大聲失敗。"""
    evaluation = _evaluation_result()
    evaluation["metrics_mean"]["uv_rel_err"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        build_metric_artifact(
            evaluation,
            context=EvaluationContext(
                case="kolmogorov", times=(0.0, 1.0), periodic=True,
                domain_length=1.0, viscosity=None, pressure_evaluated=False,
            ),
            provenance=SourceProvenance(
                producer="tests", code_revision="abc123", code_dirty=False,
            ),
        )


def test_non_measured_summary_must_not_carry_a_number():
    with pytest.raises(ValueError, match="must not carry"):
        MetricSummary(
            definition_id=_BAND_MID_V1, value=0.1, provenance="embedded",
            availability=MetricAvailability.NOT_APPLICABLE,
        )


_RECON_SECONDS_V1 = "cost.reconstruction.seconds_per_field.v1"


def _cost_artifact(run_measurements):
    return build_metric_artifact(
        _evaluation_result(),
        context=EvaluationContext(
            case="kolmogorov", times=(0.0, 1.0), periodic=True,
            domain_length=1.0, viscosity=None, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
        run_measurements=run_measurements,
    )


def test_run_level_cost_is_a_defined_measurement_not_a_provenance_note():
    """成本量進圖（cost-accuracy 曲線的 x 軸），所以它需要語意契約。

    塞進 provenance 就是在新地方重蹈 TD-17：一個沒有定義、沒有單位的數字
    被下游直接消費。
    """
    artifact = _cost_artifact({"recon_s_per_field": 0.25})

    summary = _summary_for(artifact, _RECON_SECONDS_V1)
    assert summary.value == 0.25
    assert summary.availability is MetricAvailability.MEASURED
    assert summary.sample_count == 1

    definition = next(d for d in artifact.metric_definitions
                      if d.definition_id == _RECON_SECONDS_V1)
    assert definition.unit == "s"

    # 聚合規則本身也要有名字：它不是對時間平均，是單次 run 級觀測。
    summary_def = next(s for s in artifact.summary_definitions
                       if s.definition_id == summary.summary_definition_id)
    assert summary_def.aggregation == "run_scalar"


def test_unregistered_run_measurement_is_refused():
    """打錯字不該變成一個新的成本量。"""
    with pytest.raises(KeyError, match="unregistered run measurement"):
        _cost_artifact({"seconds": 0.25})


def test_non_finite_run_measurement_is_refused():
    """成本要嘛量到了要嘛沒有；NaN 秒數沒有意義，不給 availability 寬容。"""
    with pytest.raises(ValueError, match="non-finite"):
        _cost_artifact({"recon_s_per_field": float("nan")})


def test_run_measurements_do_not_change_the_schema_version():
    """既有 artifact 已散落在 lab-server；schema 一動，讀取端就全數失效。"""
    assert _cost_artifact({"recon_s_per_field": 0.25}).schema_version == "metric-artifact.v1"
    assert MetricArtifact.from_dict(
        _cost_artifact({"recon_s_per_field": 0.25}).to_dict()
    ).schema_version == "metric-artifact.v1"


def test_every_definition_is_either_a_field_metric_or_a_declared_cost():
    """定義表的邊界要明說。

    鬆綁 formula 模板之後，若沒有這條，任何東西都能被註冊成 metric。
    """
    artifact = _cost_artifact({"recon_s_per_field": 0.25, "wall_s": 3.5})

    for definition in artifact.metric_definitions:
        assert (definition.definition_id.startswith("cost.")
                or not definition.formula.startswith("cost")), definition.definition_id
        if definition.definition_id.startswith("cost."):
            assert definition.unit == "s", definition.definition_id


def test_build_metric_artifact_is_complete_typed_and_json_round_trips():
    artifact = build_metric_artifact(
        _evaluation_result(),
        context=EvaluationContext(
            case="kolmogorov",
            times=(0.0, 1.0),
            periodic=True,
            domain_length=1.0,
            viscosity=None,
            pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests",
            code_revision="abc123",
            code_dirty=False,
            inputs=(("config", "fixture.toml"),),
        ),
    )

    assert artifact.schema_version == "metric-artifact.v1"
    assert artifact.provenance.code_revision == "abc123"
    assert {d.definition_id for d in artifact.metric_definitions} >= {
        "velocity.vector.rel_l2.v1",
        "vorticity.rel_l2.v1",
        "pressure.rel_l2.gauge_corrected.v1",
    }
    pressure = [
        m for m in artifact.measurements
        if m.definition_id == "pressure.rel_l2.gauge_corrected.v1"
    ]
    assert {m.availability for m in pressure} == {MetricAvailability.NOT_APPLICABLE}
    gamma_high = [
        m for m in artifact.measurements
        if m.definition_id == "spectrum.coherence.high.v1"
    ]
    assert [m.availability for m in gamma_high] == [
        MetricAvailability.NOT_APPLICABLE,
        MetricAvailability.MEASURED,
    ]
    assert next(
        s for s in artifact.summaries
        if s.summary_definition_id == "velocity.vector.rel_l2.time_mean.v1"
    ).sample_count == 2
    gamma = next(
        d for d in artifact.diagnostic_series
        if d.definition_id == "spectrum.coherence.shell.v1" and d.time_index == 0
    )
    assert gamma.values == (1.0, None)
    assert gamma.validity == (True, False)
    assert dict(artifact.diagnostic_metadata)["n_valid_frames"] == 1

    payload = artifact.to_dict()
    assert payload["definitions"]["metrics"][0]["meaning"]
    assert payload["definitions"]["metrics"][0]["formula"]
    assert MetricArtifact.from_dict(payload) == artifact


def test_metric_artifact_rejects_missing_or_nonfinite_evidence():
    evaluation = _evaluation_result()
    evaluation["metrics_per_t"][0]["uv_rel_err"] = float("nan")

    with pytest.raises(ValueError, match="uv_rel_err.*non-finite"):
        build_metric_artifact(
            evaluation,
            context=EvaluationContext(
                case="kolmogorov", times=(0.0, 1.0), periodic=True,
                domain_length=1.0, viscosity=1e-3, pressure_evaluated=False,
            ),
            provenance=SourceProvenance(
                producer="tests", code_revision="abc123", code_dirty=False,
            ),
        )


def test_evaluate_module_is_a_compatibility_adapter_for_metric_formulas():
    from pi_lnn_jax import evaluate, metric_artifact

    assert evaluate.compute_metrics is metric_artifact.compute_metrics
    assert evaluate.energy_timeseries_errors is metric_artifact.energy_timeseries_errors
    assert evaluate.compute_vorticity is metric_artifact.compute_vorticity


def test_semantic_reader_accepts_the_canonical_projection():
    artifact = build_metric_artifact(
        _evaluation_result(),
        context=EvaluationContext(
            case="kolmogorov", times=(0.0, 1.0), periodic=True,
            domain_length=1.0, viscosity=None, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
    )

    summary = read_metric_summary(
        artifact.to_dict(), VORTICITY_REL_ERR_TIME_MEAN_V1,
    )

    assert summary.value == 0.3
    assert summary.sample_count == 2


def test_measurement_availability_and_value_cannot_contradict_each_other():
    with pytest.raises(ValueError, match="finite value"):
        MetricMeasurement(
            definition_id="x.v1", time_index=0, time=0.0,
            availability=MetricAvailability.MEASURED, value=None,
        )
    with pytest.raises(ValueError, match="must not carry"):
        MetricMeasurement(
            definition_id="x.v1", time_index=0, time=0.0,
            availability=MetricAvailability.UNAVAILABLE, value=1.0,
        )


def test_legacy_interpretation_survives_canonical_round_trip():
    artifact = build_metric_artifact(
        _evaluation_result(),
        context=EvaluationContext(
            case="kolmogorov", times=(0.0, 1.0), periodic=True,
            domain_length=1.0, viscosity=None, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
    )
    interpretation = LegacyInterpretation(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        basis="reviewed campaign manifest",
    )
    index = next(i for i, summary in enumerate(artifact.summaries)
                 if summary.definition_id == KE_T_MAPE_SPATIALMEAN_V1)
    summaries = list(artifact.summaries)
    summaries[index] = replace(
        summaries[index], provenance="legacy_interpretation",
        legacy_interpretation=interpretation,
    )
    artifact = replace(artifact, summaries=tuple(summaries))

    restored = MetricArtifact.from_dict(artifact.to_dict())

    assert restored.summaries[index].legacy_interpretation == interpretation


def test_field_series_is_the_deep_artifact_seam():
    rng = np.random.default_rng(7)
    reference_u = rng.normal(size=(2, 16, 16))
    reference_v = rng.normal(size=(2, 16, 16))
    predicted_u = reference_u * 0.9
    predicted_v = reference_v * 1.1
    context = EvaluationContext(
        case="kolmogorov", times=(0.0, 1.0), periodic=True,
        domain_length=1.0, viscosity=None, pressure_evaluated=False,
        grid_shape=(16, 16),
    )

    artifact, projection = evaluate_field_series(
        predicted_u, predicted_v, reference_u, reference_v,
        context=context,
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
    )

    summary = read_metric_summary(
        artifact.to_dict(), VORTICITY_REL_ERR_TIME_MEAN_V1,
    )
    assert summary.availability is MetricAvailability.MEASURED
    assert summary.sample_count == 2

    # 投影與 artifact 同源：canonical summary 的值就是投影裡那個鍵，不是另算一次。
    assert projection["metrics_mean"]["omega_rel_err"] == summary.value
    assert len(projection["metrics_per_t"]) == len(context.times)


def test_field_series_matches_per_frame_calls_and_a_channel_swap_is_visible():
    """遷移的承重假設：整段 [T,Nx,Ny] 傳入 ≡ 逐幀呼叫 compute_metrics。

    五支評估器原本逐幀呼叫，遷移後改成把 pred/ref 切成四個 [T,Nx,Ny] 陣列交給
    seam（SHRED 那支是 `pred[:, :, :, 0]` 這種切片）。切錯或把 u/v 對調不會 crash，
    只會安靜給出另一組數字——這正是 `summary_diff` docstring 警告的失效形狀。

    汙染探針（下半段）確認這條測試抓得到對調；沒有它，上半段的相等可能只是
    「兩邊都算了同樣的錯東西」。
    """
    from pi_lnn_jax.metric_artifact import aggregate_metric_rows, compute_metrics

    rng = np.random.default_rng(19)
    T, N = 3, 16
    # u/v 必須不對稱，否則對調偵測不到。
    ref_u = rng.normal(size=(T, N, N))
    ref_v = rng.normal(size=(T, N, N)) * 3.0 + 1.0
    pred = np.stack([ref_u * 0.9, ref_v * 1.2], axis=-1)   # [T,N,N,2]，模仿 SHRED 的形狀

    per_frame = [compute_metrics(pred[i, :, :, 0], pred[i, :, :, 1],
                                 ref_u[i], ref_v[i], periodic=True) for i in range(T)]
    ke_p = np.array([r["ke_pred_mean"] for r in per_frame])
    ke_d = np.array([r["ke_dns_mean"] for r in per_frame])
    reference = aggregate_metric_rows(per_frame, ke_p, ke_d)["metrics_mean"]

    context = EvaluationContext(
        case="kolmogorov", times=tuple(float(t) for t in range(T)), periodic=True,
        domain_length=1.0, viscosity=None, pressure_evaluated=False,
        grid_shape=(N, N),
    )
    provenance = SourceProvenance(
        producer="tests", code_revision="abc123", code_dirty=False)

    _, projection = evaluate_field_series(
        pred[:, :, :, 0], pred[:, :, :, 1], ref_u, ref_v,
        context=context, provenance=provenance)

    for key, value in reference.items():
        assert projection["metrics_mean"][key] == pytest.approx(value, rel=0.0, abs=1e-12), key

    # 汙染探針：把預測的兩個通道對調，同一組斷言必須失敗。
    _, swapped = evaluate_field_series(
        pred[:, :, :, 1], pred[:, :, :, 0], ref_u, ref_v,
        context=context, provenance=provenance)
    assert swapped["metrics_mean"]["u_rel_err"] != pytest.approx(
        reference["u_rel_err"], rel=0.0, abs=1e-12), "通道對調卻讀不出差別——本測試無鑑別力"


def test_embedded_spatialmean_ke_returns_an_immutable_semantic_summary():
    payload = {
        "ke_t_errors": {
            "ke_t_mape": 0.91,
            "ke_t_mape_spatialmean": 0.125,
            "ke_mape_def": "pointwise_v2",
        }
    }

    summary = read_metric_summary(payload, KE_T_MAPE_SPATIALMEAN_V1)

    assert summary == MetricSummary(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        value=0.125,
        provenance="embedded",
    )
    with pytest.raises(FrozenInstanceError):
        summary.value = 99.0


def test_unmarked_legacy_ke_refuses_to_guess():
    payload = {"ke_t_errors": {"ke_t_mape": 0.125}}

    with pytest.raises(AmbiguousMetricSemantics, match="legacy interpretation"):
        read_metric_summary(payload, KE_T_MAPE_SPATIALMEAN_V1)


def test_explicit_legacy_interpretation_is_preserved_as_provenance():
    payload = {"ke_t_errors": {"ke_t_mape": 0.125}}
    interpretation = LegacyInterpretation(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        basis="campaign predates the pointwise_v2 definition",
    )

    summary = read_metric_summary(
        payload,
        KE_T_MAPE_SPATIALMEAN_V1,
        legacy_interpretation=interpretation,
    )

    assert summary.value == 0.125
    assert summary.provenance == "legacy_interpretation"
    assert summary.legacy_interpretation == interpretation


def test_embedded_pointwise_marker_cannot_be_overridden_as_legacy_spatialmean():
    payload = {
        "ke_t_errors": {
            "ke_t_mape": 0.91,
            "ke_mape_def": "pointwise_v2",
        }
    }
    interpretation = LegacyInterpretation(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        basis="caller thinks this artifact is old",
    )

    with pytest.raises(MetricUnavailable, match="spatialmean"):
        read_metric_summary(
            payload,
            KE_T_MAPE_SPATIALMEAN_V1,
            legacy_interpretation=interpretation,
        )


def test_vorticity_time_mean_is_read_by_definition():
    payload = {"metrics_mean": {"omega_rel_err": 0.42}}

    summary = read_metric_summary(payload, VORTICITY_REL_ERR_TIME_MEAN_V1)

    assert summary == MetricSummary(
        definition_id=VORTICITY_REL_ERR_TIME_MEAN_V1,
        value=0.42,
        provenance="embedded",
    )


def test_legacy_interpretation_requires_a_nonempty_basis():
    with pytest.raises(ValueError, match="basis"):
        LegacyInterpretation(
            definition_id=KE_T_MAPE_SPATIALMEAN_V1,
            basis="   ",
        )
