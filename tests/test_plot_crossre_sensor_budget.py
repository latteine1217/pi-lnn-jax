"""The cross-Re plot consumes semantic summaries, not artifact storage keys."""

import json
from pathlib import Path
import sys


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_crossre_sensor_budget import collect  # noqa: E402
from pi_lnn_jax.metric_artifact import (  # noqa: E402
    EvaluationContext,
    KE_T_MAPE_SPATIALMEAN_V1,
    LegacyInterpretation,
    SourceProvenance,
    VORTICITY_REL_ERR_TIME_MEAN_V1,
    build_metric_artifact,
    write_metric_artifact,
)


def test_collect_returns_values_keyed_by_metric_definition(tmp_path):
    result_dir = tmp_path / "crossre_re100_k10" / "final_eval"
    result_dir.mkdir(parents=True)
    (result_dir / "metrics.json").write_text(json.dumps({
        "metrics_mean": {"omega_rel_err": 0.42},
        "ke_t_errors": {
            "ke_t_mape": 0.91,
            "ke_t_mape_spatialmean": 0.125,
            "ke_mape_def": "pointwise_v2",
        },
    }))

    found, missing = collect(tmp_path)

    assert found[("100", 10)] == {
        KE_T_MAPE_SPATIALMEAN_V1: 0.125,
        VORTICITY_REL_ERR_TIME_MEAN_V1: 0.42,
    }
    assert len(missing) == 14


def test_collect_accepts_an_explicit_legacy_interpretation(tmp_path):
    result_dir = tmp_path / "crossre_re100_k10" / "final_eval"
    result_dir.mkdir(parents=True)
    (result_dir / "metrics.json").write_text(json.dumps({
        "metrics_mean": {"omega_rel_err": 0.42},
        "ke_t_errors": {"ke_t_mape": 0.125},
    }))
    interpretation = LegacyInterpretation(
        definition_id=KE_T_MAPE_SPATIALMEAN_V1,
        basis="cross-Re campaign predates the pointwise_v2 definition",
    )

    found, _ = collect(tmp_path, legacy_interpretation=interpretation)

    assert found[("100", 10)][KE_T_MAPE_SPATIALMEAN_V1] == 0.125


def test_collect_prefers_canonical_artifact_and_keeps_requested_summary_ids(tmp_path):
    result_dir = tmp_path / "crossre_re100_k10" / "final_eval"
    result_dir.mkdir(parents=True)
    evaluation = {
        "ke_t_pred": [1.0], "ke_t_dns": [2.0],
        "metrics_per_t": [{
            "omega_rel_err": 0.42, "ke_pred_mean": 1.0,
            "ke_dns_mean": 2.0, "ke_pw_mape": 0.9, "ke_pw_nmae": 0.9,
            "t": 0.0,
        }],
        "metrics_mean": {"omega_rel_err": 0.42, "ke_pw_mape": 0.9,
                         "ke_pw_nmae": 0.9},
        "metrics_p90": {"omega_rel_err": 0.42, "ke_pw_mape": 0.9,
                        "ke_pw_nmae": 0.9},
        "ke_t_errors": {
            "ke_t_mape": 0.9, "ke_t_nmae": 0.9,
            "ke_t_mape_spatialmean": 0.5, "ke_t_rel_l2": 0.5,
            "ke_t_rel_linf": 0.5, "ke_t_final_rel": 0.5,
            "ke_mape_def": "pointwise_v2",
        },
        "band_high_diag": None,
    }
    write_metric_artifact(
        result_dir / "metric_artifact.json",
        build_metric_artifact(
            evaluation,
            context=EvaluationContext(
                case="kolmogorov", times=(0.0,), periodic=True,
                domain_length=1.0, viscosity=None, pressure_evaluated=False,
            ),
            provenance=SourceProvenance(
                producer="tests", code_revision="abc123", code_dirty=False,
            ),
        ),
    )

    found, _ = collect(tmp_path)

    assert found[("100", 10)] == {
        KE_T_MAPE_SPATIALMEAN_V1: 0.5,
        VORTICITY_REL_ERR_TIME_MEAN_V1: 0.42,
    }
