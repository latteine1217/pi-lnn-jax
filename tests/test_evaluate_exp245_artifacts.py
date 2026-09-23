"""The EXP-245 producer writes a canonical artifact beside its unchanged projection."""

import json

from pi_lnn_jax.metric_artifact import EvaluationContext, SourceProvenance
from scripts.evaluate_exp245 import _write_metric_json_artifacts


def test_writer_preserves_metrics_projection_and_adds_canonical_artifact(tmp_path):
    evaluation = {
        "ke_t_pred": [1.0], "ke_t_dns": [1.0],
        "metrics_per_t": [{
            "uv_rel_err": 0.0, "omega_rel_err": 0.0,
            "ke_pred_mean": 1.0, "ke_dns_mean": 1.0, "t": 0.0,
        }],
        "metrics_mean": {"uv_rel_err": 0.0, "omega_rel_err": 0.0},
        "metrics_p90": {"uv_rel_err": 0.0, "omega_rel_err": 0.0},
        "ke_t_errors": {
            "ke_t_mape": 0.0, "ke_t_nmae": 0.0,
            "ke_t_mape_spatialmean": 0.0, "ke_t_rel_l2": 0.0,
            "ke_t_rel_linf": 0.0, "ke_t_final_rel": 0.0,
            "ke_mape_def": "pointwise_v2",
        },
        "band_high_diag": None,
    }
    projection = {
        "config": "/tmp/config.toml", "ckpt_step": 7,
        "metrics_mean": evaluation["metrics_mean"],
        "ke_t_errors": evaluation["ke_t_errors"],
        "metrics_per_t": evaluation["metrics_per_t"],
        "ckpt_provenance": {"fingerprint_verified": True},
    }

    _write_metric_json_artifacts(
        tmp_path, projection=projection, evaluation=evaluation,
        context=EvaluationContext(
            case="kolmogorov", times=(0.0,), periodic=True,
            domain_length=1.0, viscosity=1e-3, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="scripts/evaluate_exp245.py", code_revision="abc123",
            code_dirty=True,
        ),
    )

    assert json.loads((tmp_path / "metrics.json").read_text()) == projection
    canonical = json.loads((tmp_path / "metric_artifact.json").read_text())
    assert canonical["schema_version"] == "metric-artifact.v1"
    assert canonical["provenance"]["code_dirty"] is True
    assert canonical["summaries"]
