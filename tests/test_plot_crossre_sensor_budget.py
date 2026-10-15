"""The cross-Re plot consumes semantic summaries, not artifact storage keys."""

import json
from pathlib import Path
import sys


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_crossre_sensor_budget import (  # noqa: E402
    VELOCITY_REL_ERR_TIME_MEAN_V1,
    collect,
)
from pi_lnn_jax.metric_artifact import (  # noqa: E402
    EvaluationContext,
    SourceProvenance,
    VORTICITY_REL_ERR_TIME_MEAN_V1,
    build_metric_artifact,
    write_metric_artifact,
)


def _write_canonical(path: Path, *, uv: float, omega: float) -> None:
    # time_mean 與 time_p90 並存：這正是只給 definition id 會歧義的情形
    frame = {"uv_rel_err": uv, "omega_rel_err": omega}
    evaluation = {
        "ke_t_pred": [1.0], "ke_t_dns": [2.0],
        "metrics_per_t": [{**frame, "ke_pred_mean": 1.0, "ke_dns_mean": 2.0,
                           "t": 0.0}],
        "metrics_mean": dict(frame),
        "metrics_p90": {"uv_rel_err": 9.0, "omega_rel_err": 9.0},
        "ke_t_errors": {},
        "band_high_diag": None,
    }
    path.parent.mkdir(parents=True)
    write_metric_artifact(path, build_metric_artifact(
        evaluation,
        context=EvaluationContext(
            case="kolmogorov", times=(0.0,), periodic=True,
            domain_length=1.0, viscosity=None, pressure_evaluated=False,
        ),
        provenance=SourceProvenance(
            producer="tests", code_revision="abc123", code_dirty=False,
        ),
    ))


def test_collect_reads_time_mean_velocity_and_vorticity(tmp_path):
    _write_canonical(
        tmp_path / "crossre_re100_k10" / "eval_uv" / "metric_artifact.json",
        uv=0.2, omega=0.42,
    )

    found, missing = collect(tmp_path)

    assert found[("100", 10)] == {
        VELOCITY_REL_ERR_TIME_MEAN_V1: 0.2,
        VORTICITY_REL_ERR_TIME_MEAN_V1: 0.42,
    }
    assert len(missing) == 14


def test_collect_reads_the_re1000_k100_cell_from_eval_uv_matched(tmp_path):
    _write_canonical(
        tmp_path / "crossre_re1000_k100" / "eval_uv_matched" / "metric_artifact.json",
        uv=0.045, omega=0.097,
    )

    found, _ = collect(tmp_path)

    assert found[("1000", 100)][VELOCITY_REL_ERR_TIME_MEAN_V1] == 0.045


def test_collect_does_not_fall_back_to_the_legacy_final_eval_bundle(tmp_path):
    # 舊 bundle 沒有向量速度誤差；讀到它只會讓格子靜默換成另一個量
    result_dir = tmp_path / "crossre_re100_k10" / "final_eval"
    result_dir.mkdir(parents=True)
    (result_dir / "metrics.json").write_text(json.dumps({
        "metrics_mean": {"omega_rel_err": 0.42, "u_rel_err": 0.1},
    }))

    found, missing = collect(tmp_path)

    assert ("100", 10) not in found
    assert len(missing) == 15
