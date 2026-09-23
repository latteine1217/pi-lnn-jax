"""Semantic access to metric artifacts.

Callers name the metric definition they need; storage keys remain an
implementation detail of this module and its compatibility adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np


KE_T_MAPE_SPATIALMEAN_V1 = "ke_t_mape.spatialmean.v1"
KE_T_MAPE_POINTWISE_V2 = "kinetic_energy.pointwise.mape.v2"
VORTICITY_REL_ERR_TIME_MEAN_V1 = "vorticity.rel_l2.time_mean.v1"
METRIC_ARTIFACT_SCHEMA_VERSION = "metric-artifact.v1"


class AmbiguousMetricSemantics(ValueError):
    """A legacy projection has a value but no authoritative definition."""


class MetricUnavailable(ValueError):
    """The requested metric definition is absent from this artifact."""


class MetricAvailability(str, Enum):
    """Whether one metric measurement has valid numerical evidence."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MetricDefinition:
    definition_id: str
    version: int
    meaning: str
    formula: str
    unit: str
    applicability: str


@dataclass(frozen=True)
class SummaryDefinition:
    definition_id: str
    metric_definition_id: str
    aggregation: str
    eligibility: str


@dataclass(frozen=True)
class MetricMeasurement:
    definition_id: str
    time_index: int
    time: float
    availability: MetricAvailability
    value: float | None = None

    def __post_init__(self) -> None:
        if self.availability is MetricAvailability.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("measured metric requires a finite value")
        elif self.value is not None:
            raise ValueError("non-measured metric must not carry a numerical value")


@dataclass(frozen=True)
class DiagnosticSeries:
    definition_id: str
    time_index: int
    time: float
    values: tuple[float | None, ...]
    validity: tuple[bool, ...]


@dataclass(frozen=True)
class EvaluationContext:
    case: str
    times: tuple[float, ...]
    periodic: bool
    domain_length: float
    viscosity: float | None
    pressure_evaluated: bool
    Lx: float = 1.0
    Ly: float = 1.0
    grid_shape: tuple[int, int] | None = None
    coordinate_system: str = "normalized_cartesian"
    mask_applied: bool = False


@dataclass(frozen=True)
class SourceProvenance:
    producer: str
    code_revision: str
    code_dirty: bool
    inputs: tuple[tuple[str, str], ...] = ()
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.producer.strip():
            raise ValueError("source provenance producer must be nonempty")
        if not self.code_revision.strip():
            raise ValueError("source provenance code_revision must be nonempty")


@dataclass(frozen=True)
class LegacyInterpretation:
    """An explicit, reviewable meaning assigned to an unmarked artifact."""

    definition_id: str
    basis: str

    def __post_init__(self) -> None:
        if not self.basis.strip():
            raise ValueError("legacy interpretation basis must be nonempty")


@dataclass(frozen=True)
class MetricSummary:
    """One scalar summary with an explicit scientific definition."""

    definition_id: str
    value: float | None
    provenance: Literal["embedded", "legacy_interpretation"]
    legacy_interpretation: LegacyInterpretation | None = None
    summary_definition_id: str | None = None
    availability: MetricAvailability = MetricAvailability.MEASURED
    sample_count: int | None = None

    def __post_init__(self) -> None:
        # 與 MetricMeasurement 同一條不變量：availability 說了算，值不得與它矛盾。
        if self.availability is MetricAvailability.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("measured summary requires a finite value")
        elif self.value is not None:
            raise ValueError("non-measured summary must not carry a numerical value")


@dataclass(frozen=True)
class MetricArtifact:
    """Immutable canonical evaluation record at the metric-artifact seam."""

    context: EvaluationContext
    provenance: SourceProvenance
    metric_definitions: tuple[MetricDefinition, ...]
    summary_definitions: tuple[SummaryDefinition, ...]
    measurements: tuple[MetricMeasurement, ...]
    summaries: tuple[MetricSummary, ...]
    diagnostic_series: tuple[DiagnosticSeries, ...]
    diagnostic_metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: str = METRIC_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context": {
                "case": self.context.case,
                "times": list(self.context.times),
                "periodic": self.context.periodic,
                "domain_length": self.context.domain_length,
                "viscosity": self.context.viscosity,
                "pressure_evaluated": self.context.pressure_evaluated,
                "Lx": self.context.Lx,
                "Ly": self.context.Ly,
                "grid_shape": (list(self.context.grid_shape)
                               if self.context.grid_shape is not None else None),
                "coordinate_system": self.context.coordinate_system,
                "mask_applied": self.context.mask_applied,
            },
            "provenance": {
                "producer": self.provenance.producer,
                "code_revision": self.provenance.code_revision,
                "code_dirty": self.provenance.code_dirty,
                "inputs": dict(self.provenance.inputs),
                "details": _thaw_pairs(self.provenance.details),
            },
            "definitions": {
                "metrics": [_metric_definition_dict(d) for d in self.metric_definitions],
                "summaries": [_summary_definition_dict(d) for d in self.summary_definitions],
            },
            "measurements": [_measurement_dict(m) for m in self.measurements],
            "summaries": [_summary_dict(s) for s in self.summaries],
            "diagnostic_series": [_diagnostic_dict(d) for d in self.diagnostic_series],
            "diagnostic_metadata": _thaw_pairs(self.diagnostic_metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricArtifact":
        if payload["schema_version"] != METRIC_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported metric artifact schema: {payload['schema_version']!r}")
        c = payload["context"]
        p = payload["provenance"]
        return cls(
            schema_version=payload["schema_version"],
            context=EvaluationContext(
                case=c["case"], times=tuple(c["times"]), periodic=bool(c["periodic"]),
                domain_length=float(c["domain_length"]), viscosity=c["viscosity"],
                pressure_evaluated=bool(c["pressure_evaluated"]),
                Lx=float(c["Lx"]), Ly=float(c["Ly"]),
                grid_shape=(tuple(c["grid_shape"]) if c.get("grid_shape") is not None else None),
                coordinate_system=c.get("coordinate_system", "normalized_cartesian"),
                mask_applied=bool(c.get("mask_applied", False)),
            ),
            provenance=SourceProvenance(
                producer=p["producer"], code_revision=p["code_revision"],
                code_dirty=bool(p["code_dirty"]), inputs=tuple(p["inputs"].items()),
                details=_freeze_mapping(p["details"]),
            ),
            metric_definitions=tuple(MetricDefinition(**d) for d in payload["definitions"]["metrics"]),
            summary_definitions=tuple(SummaryDefinition(**d) for d in payload["definitions"]["summaries"]),
            measurements=tuple(MetricMeasurement(
                definition_id=m["definition_id"], time_index=int(m["time_index"]),
                time=float(m["time"]), availability=MetricAvailability(m["availability"]),
                value=m["value"],
            ) for m in payload["measurements"]),
            summaries=tuple(MetricSummary(
                definition_id=s["definition_id"], value=s["value"],
                provenance=s["provenance"], summary_definition_id=s["summary_definition_id"],
                availability=MetricAvailability(s["availability"]),
                sample_count=s["sample_count"],
                legacy_interpretation=(LegacyInterpretation(**s["legacy_interpretation"])
                                       if s.get("legacy_interpretation") else None),
            ) for s in payload["summaries"]),
            diagnostic_series=tuple(DiagnosticSeries(
                definition_id=d["definition_id"], time_index=int(d["time_index"]),
                time=float(d["time"]), values=tuple(d["values"]),
                validity=tuple(d["validity"]),
            ) for d in payload["diagnostic_series"]),
            diagnostic_metadata=_freeze_mapping(payload.get("diagnostic_metadata", {})),
        )


def _metric_definition_dict(d: MetricDefinition) -> dict[str, Any]:
    return {
        "definition_id": d.definition_id, "version": d.version,
        "meaning": d.meaning, "formula": d.formula,
        "unit": d.unit, "applicability": d.applicability,
    }


def _summary_definition_dict(d: SummaryDefinition) -> dict[str, Any]:
    return {
        "definition_id": d.definition_id,
        "metric_definition_id": d.metric_definition_id,
        "aggregation": d.aggregation,
        "eligibility": d.eligibility,
    }


def _measurement_dict(m: MetricMeasurement) -> dict[str, Any]:
    return {
        "definition_id": m.definition_id, "time_index": m.time_index, "time": m.time,
        "availability": m.availability.value, "value": m.value,
    }


def _summary_dict(s: MetricSummary) -> dict[str, Any]:
    return {
        "definition_id": s.definition_id,
        "summary_definition_id": s.summary_definition_id,
        "availability": s.availability.value,
        "value": s.value,
        "sample_count": s.sample_count,
        "provenance": s.provenance,
        "legacy_interpretation": (
            {"definition_id": s.legacy_interpretation.definition_id,
             "basis": s.legacy_interpretation.basis}
            if s.legacy_interpretation is not None else None
        ),
    }


def _diagnostic_dict(d: DiagnosticSeries) -> dict[str, Any]:
    return {
        "definition_id": d.definition_id, "time_index": d.time_index, "time": d.time,
        "values": list(d.values), "validity": list(d.validity),
    }


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(k), _freeze_value(v)) for k, v in sorted(value.items()))


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
               for item in value):
            return {k: _thaw_value(v) for k, v in value}
        return [_thaw_value(v) for v in value]
    return value


def _thaw_pairs(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {k: _thaw_value(v) for k, v in value}


def freeze_provenance_details(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Freeze JSON-like producer metadata for immutable source provenance."""
    return _freeze_mapping(value)


def _definition(key: str, definition_id: str, meaning: str, unit: str = "1",
                applicability: str = "all evaluations") -> tuple[str, MetricDefinition]:
    return key, MetricDefinition(
        definition_id, 1, meaning, f"compute_metrics[{key!r}]: {meaning}",
        unit, applicability,
    )


_METRIC_DEFINITION_ITEMS = (
    _definition("uv_rel_err", "velocity.vector.rel_l2.v1", "Relative L2 error of the vector velocity field"),
    _definition("u_rel_err", "velocity.u.rel_l2.v1", "Relative L2 error of the u velocity component"),
    _definition("v_rel_err", "velocity.v.rel_l2.v1", "Relative L2 error of the v velocity component"),
    _definition("ke_rel_err", "kinetic_energy.field.rel_l2.v1", "Relative L2 error of the kinetic-energy field"),
    _definition("ke_pw_mape", "kinetic_energy.pointwise.mape.v2", "Spatial mean of pointwise kinetic-energy absolute percentage error"),
    _definition("ke_pw_nmae", "kinetic_energy.pointwise.nmae.v1", "Spatial mean absolute kinetic-energy error normalized by reference mean"),
    _definition("ke_pred_over_ref", "kinetic_energy.integrated.ratio.v1", "Ratio of predicted to reference integrated kinetic energy"),
    _definition("ke_pred_mean", "kinetic_energy.pred.spatial_mean.v1", "Predicted spatial-mean kinetic energy"),
    _definition("ke_dns_mean", "kinetic_energy.reference.spatial_mean.v1", "Reference spatial-mean kinetic energy"),
    _definition("omega_rel_err", "vorticity.rel_l2.v1", "Relative L2 error of the vorticity field"),
    _definition("enstrophy_rel_err", "enstrophy.integrated.rel_error.v1", "Relative error of spatial-mean enstrophy"),
    _definition("div_pred_l2", "divergence.pred.rms.v1", "Root-mean-square predicted velocity divergence"),
    _definition("div_dns_l2", "divergence.reference.rms.v1", "Root-mean-square reference velocity divergence"),
    _definition("low_band_rel_err", "spectrum.energy.low.integrated_rel_error.v1", "Relative error of integrated low-band energy", applicability="periodic square domains"),
    _definition("spectrum_rel_err", "spectrum.energy.rel_l2.v1", "Relative L2 error of the shell energy spectrum", applicability="periodic square domains"),
    _definition("k_cut", "spectrum.reference.cutoff.v1", "Reference spectral cutoff at 1e-6 of peak shell energy", "integer wavenumber", "periodic square domains"),
    _definition("k_eta", "spectrum.dissipation_wavenumber.v1", "Dissipation wavenumber inferred from the reference spectrum", "integer wavenumber", "periodic square domains with viscosity"),
    _definition("kcut_over_keta", "spectrum.cutoff_over_dissipation.v1", "Ratio of spectral cutoff to dissipation wavenumber", applicability="periodic square domains with viscosity"),
    _definition("k_d_kraichnan", "spectrum.enstrophy_dissipation_wavenumber.v1", "Kraichnan enstrophy-cascade dissipation wavenumber; diagnostic only, does not define band boundaries", "integer wavenumber", "periodic square domains with viscosity"),
    _definition("kd_over_keta", "spectrum.kraichnan_over_viscous.v1", "Ratio of the Kraichnan wavenumber to the 3D-form viscous wavenumber used for band boundaries", applicability="periodic square domains with viscosity"),
    _definition("band_rel_err_low", "spectrum.energy.rel_l2.low.v1", "Shell-wise relative L2 error below 0.1 k_eta", applicability="periodic square domains with viscosity"),
    _definition("band_rel_err_mid", "spectrum.energy.rel_l2.mid.v1", "Shell-wise relative L2 error from 0.1 to 0.4 k_eta", applicability="periodic square domains with viscosity"),
    _definition("band_rel_err_high", "spectrum.energy.rel_l2.high.v1", "Shell-wise relative L2 error from 0.4 to 1.0 k_eta", applicability="periodic square domains with viscosity and eligible shells"),
    _definition("gamma_low", "spectrum.coherence.low.v1", "Reference-energy-weighted low-band spectral coherence", applicability="periodic square domains with viscosity"),
    _definition("gamma_mid", "spectrum.coherence.mid.v1", "Reference-energy-weighted mid-band spectral coherence", applicability="periodic square domains with viscosity"),
    _definition("gamma_high", "spectrum.coherence.high.v1", "Reference-energy-weighted high-band spectral coherence", applicability="periodic square domains with viscosity and eligible shells"),
    _definition("div_ratio", "divergence.over_reference_gradient.v1", "Predicted divergence norm divided by reference velocity-gradient Frobenius norm", applicability="periodic square domains"),
    _definition("kf_amp_ratio", "forcing_shell.energy_ratio.v1", "Predicted to reference energy ratio at the forcing shell", applicability="periodic square domains"),
    _definition("kf_mode_amp_pred", "forcing_mode.pred.amplitude.v1", "Predicted complex forcing-mode amplitude", applicability="periodic square domains"),
    _definition("kf_mode_amp_dns", "forcing_mode.reference.amplitude.v1", "Reference complex forcing-mode amplitude", applicability="periodic square domains"),
    _definition("kf_mode_phase_pred", "forcing_mode.pred.phase.v1", "Predicted complex forcing-mode phase", "radian", "periodic square domains"),
    _definition("kf_mode_phase_dns", "forcing_mode.reference.phase.v1", "Reference complex forcing-mode phase", "radian", "periodic square domains"),
    _definition("p_rel_err", "pressure.rel_l2.gauge_corrected.v1", "Relative L2 pressure error after independent spatial de-meaning", applicability="evaluations with pressure evidence"),
)
_METRIC_DEFINITIONS = dict(_METRIC_DEFINITION_ITEMS)

_DIAGNOSTIC_DEFINITIONS = {
    "E_pred_k": MetricDefinition("spectrum.energy.pred.shell.v1", 1, "Predicted kinetic energy by integer wavenumber shell", "azimuthal shell sum of 0.5*(abs(FFT(u))^2+abs(FFT(v))^2)", "energy", "periodic square domains"),
    "E_dns_k": MetricDefinition("spectrum.energy.reference.shell.v1", 1, "Reference kinetic energy by integer wavenumber shell", "azimuthal shell sum of 0.5*(abs(FFT(u))^2+abs(FFT(v))^2)", "energy", "periodic square domains"),
    "gamma_k": MetricDefinition("spectrum.coherence.shell.v1", 1, "Velocity spectral coherence by integer wavenumber shell", "Re(sum(pred*conj(ref)))/sqrt(sum(abs(pred)^2)*sum(abs(ref)^2))", "1", "periodic square domains"),
}

#: Run-level cost measurements: one observation per evaluated run, not per frame.
#:
#: These are not produced by `compute_metrics` and have no per-time counterpart, so
#: they carry their own truthful formula rather than the projection template. They
#: live here rather than in provenance because they are consumed as data — the
#: cost-accuracy curve plots one of them on its x axis — and a number that reaches a
#: figure needs a definition and a unit like any other.
_RUN_MEASUREMENT_DEFINITIONS = {
    "recon_s_per_field": MetricDefinition(
        "cost.reconstruction.seconds_per_field.v1", 1,
        "Wall-clock seconds to reconstruct one field",
        "measured reconstruction wall time divided by the number of fields",
        "s", "all evaluations",
    ),
    "wall_s": MetricDefinition(
        "cost.reconstruction.wall_seconds.v1", 1,
        "Wall-clock seconds for the evaluated reconstruction",
        "measured wall time around the reconstruction call",
        "s", "all evaluations",
    ),
}

_DISSIPATION_KEYS = frozenset({
    "k_eta", "kcut_over_keta", "k_d_kraichnan", "kd_over_keta",
    "band_rel_err_low", "band_rel_err_mid",
    "band_rel_err_high", "gamma_low", "gamma_mid", "gamma_high",
})
_SPECTRAL_KEYS = frozenset({
    "low_band_rel_err", "spectrum_rel_err", "k_cut", "div_ratio", "kf_amp_ratio",
    "kf_mode_amp_pred", "kf_mode_amp_dns", "kf_mode_phase_pred", "kf_mode_phase_dns",
}) | _DISSIPATION_KEYS
_ELIGIBILITY_NAN_KEYS = frozenset({
    "band_rel_err_low", "band_rel_err_mid", "band_rel_err_high",
    "gamma_low", "gamma_mid", "gamma_high",
})

_KE_SUMMARY_SPECS = {
    "ke_t_mape": ("kinetic_energy.pointwise.mape.v2", "kinetic_energy.pointwise.mape.time_mean.v2", "mean over evaluation times"),
    "ke_t_nmae": ("kinetic_energy.pointwise.nmae.v1", "kinetic_energy.pointwise.nmae.time_mean.v1", "mean over evaluation times"),
    "ke_t_mape_spatialmean": (KE_T_MAPE_SPATIALMEAN_V1, "ke_t_mape.spatialmean.time_mean.v1", "mean absolute percentage error of spatial-mean E(t)"),
    "ke_t_rel_l2": ("ke_time_series.rel_l2.v1", "ke_time_series.rel_l2.summary.v1", "relative L2 over evaluation times"),
    "ke_t_rel_linf": ("ke_time_series.rel_linf.v1", "ke_time_series.rel_linf.summary.v1", "relative Linf over evaluation times"),
    "ke_t_final_rel": ("ke_time_series.final_rel.v1", "ke_time_series.final_rel.summary.v1", "relative error at the final evaluation time"),
}


def _availability_for_missing(key: str, context: EvaluationContext) -> MetricAvailability:
    if key == "p_rel_err":
        return (MetricAvailability.UNAVAILABLE if context.pressure_evaluated
                else MetricAvailability.NOT_APPLICABLE)
    if key in _SPECTRAL_KEYS and not context.periodic:
        return MetricAvailability.NOT_APPLICABLE
    if key in _DISSIPATION_KEYS and context.viscosity is None:
        return MetricAvailability.UNAVAILABLE
    return MetricAvailability.UNAVAILABLE


def _measured_frame_count(per_t: Sequence[Mapping[str, Any]], key: str) -> int:
    """有定義的幀數。缺鍵與 NaN 都不算——兩者都不是一次量到的值。"""
    return sum(
        1 for row in per_t
        if key in row and math.isfinite(float(row[key]))
    )


def build_metric_artifact(
    evaluation: Mapping[str, Any], *, context: EvaluationContext,
    provenance: SourceProvenance,
    run_measurements: Mapping[str, float] | None = None,
) -> MetricArtifact:
    """Build the canonical record from the established evaluation projection.

    `run_measurements` carries run-level observations that have no per-frame
    counterpart, keyed by the names in `_RUN_MEASUREMENT_DEFINITIONS`. Unknown keys
    are refused so that a typo cannot become a new quantity.
    """
    per_t = evaluation["metrics_per_t"]
    if len(per_t) != len(context.times):
        raise ValueError(
            f"evaluation has {len(per_t)} metric rows but context has {len(context.times)} times"
        )
    ke_t_pred = tuple(float(v) for v in evaluation["ke_t_pred"])
    ke_t_dns = tuple(float(v) for v in evaluation["ke_t_dns"])
    if len(ke_t_pred) != len(per_t) or len(ke_t_dns) != len(per_t):
        raise ValueError("kinetic-energy projection length does not match metric rows")
    for time_index, row in enumerate(per_t):
        for key, projected in (
            ("ke_pred_mean", ke_t_pred[time_index]),
            ("ke_dns_mean", ke_t_dns[time_index]),
        ):
            if key in row and float(row[key]) != projected:
                raise ValueError(
                    f"{key} at time index {time_index} disagrees with kinetic-energy projection"
                )
    measurements: list[MetricMeasurement] = []
    diagnostics: list[DiagnosticSeries] = []
    for time_index, (time, row) in enumerate(zip(context.times, per_t, strict=True)):
        for key, definition in _METRIC_DEFINITION_ITEMS:
            if key not in row:
                measurements.append(MetricMeasurement(
                    definition.definition_id, time_index, float(time),
                    _availability_for_missing(key, context), None,
                ))
                continue
            value = float(row[key])
            if not math.isfinite(value):
                if key in _ELIGIBILITY_NAN_KEYS:
                    measurements.append(MetricMeasurement(
                        definition.definition_id, time_index, float(time),
                        MetricAvailability.NOT_APPLICABLE, None,
                    ))
                    continue
                raise ValueError(f"metric {key} at time index {time_index} is non-finite")
            measurements.append(MetricMeasurement(
                definition.definition_id, time_index, float(time),
                MetricAvailability.MEASURED, value,
            ))
        for key, definition in _DIAGNOSTIC_DEFINITIONS.items():
            if key not in row:
                continue
            raw = tuple(float(v) for v in row[key])
            validity = tuple(math.isfinite(v) for v in raw)
            diagnostics.append(DiagnosticSeries(
                definition.definition_id, time_index, float(time),
                tuple(v if ok else None for v, ok in zip(raw, validity, strict=True)),
                validity,
            ))

    summaries: list[MetricSummary] = []
    summary_definitions: list[SummaryDefinition] = []
    for projection_key, aggregation in (("metrics_mean", "time_mean"), ("metrics_p90", "time_p90")):
        for key, raw_value in evaluation.get(projection_key, {}).items():
            if key not in _METRIC_DEFINITIONS:
                raise KeyError(f"unregistered scalar metric: {key}")
            value = float(raw_value)
            metric_id = _METRIC_DEFINITIONS[key].definition_id
            summary_id = f"{metric_id.rsplit('.v', 1)[0]}.{aggregation}.v1"
            measured = _measured_frame_count(per_t, key)
            if not math.isfinite(value):
                # 這組鍵的 NaN 在 per-time 層已是 availability 而非錯誤；聚合層若
                # 反過來 raise，同一件事就有兩套語意。分成兩態才說得清楚：
                # 全部無定義 = 這個 evaluation condition 測不到；部分無定義 =
                # 聚合值取不出來（證據不足），兩者都不是「壞掉」。
                if key not in _ELIGIBILITY_NAN_KEYS:
                    raise ValueError(f"summary {key} in {projection_key} is non-finite")
                summary_definitions.append(SummaryDefinition(
                    summary_id, metric_id, aggregation, "measured values only",
                ))
                summaries.append(MetricSummary(
                    definition_id=metric_id, value=None, provenance="embedded",
                    summary_definition_id=summary_id, sample_count=measured,
                    availability=(MetricAvailability.NOT_APPLICABLE if measured == 0
                                  else MetricAvailability.UNAVAILABLE),
                ))
                continue
            summary_definitions.append(SummaryDefinition(
                summary_id, metric_id, aggregation, "measured values only",
            ))
            summaries.append(MetricSummary(
                definition_id=metric_id, value=value, provenance="embedded",
                summary_definition_id=summary_id, sample_count=measured,
            ))
    for key, spec in _KE_SUMMARY_SPECS.items():
        if key not in evaluation.get("ke_t_errors", {}):
            continue
        metric_id, summary_id, aggregation = spec
        value = float(evaluation["ke_t_errors"][key])
        if not math.isfinite(value):
            raise ValueError(f"summary {key} in ke_t_errors is non-finite")
        summary_definitions.append(SummaryDefinition(
            summary_id, metric_id, aggregation, "all paired evaluation times",
        ))
        summaries.append(MetricSummary(
            definition_id=metric_id, value=value, provenance="embedded",
            summary_definition_id=summary_id, sample_count=len(per_t),
        ))

    run_definitions: list[MetricDefinition] = []
    for key, raw_value in (run_measurements or {}).items():
        if key not in _RUN_MEASUREMENT_DEFINITIONS:
            raise KeyError(f"unregistered run measurement: {key}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"run measurement {key} is non-finite")
        definition = _RUN_MEASUREMENT_DEFINITIONS[key]
        summary_id = f"{definition.definition_id.rsplit('.v', 1)[0]}.run_scalar.v1"
        run_definitions.append(definition)
        summary_definitions.append(SummaryDefinition(
            summary_id, definition.definition_id, "run_scalar",
            "single run-level observation",
        ))
        summaries.append(MetricSummary(
            definition_id=definition.definition_id, value=value,
            provenance="embedded", summary_definition_id=summary_id,
            sample_count=1,
        ))

    extra_metric_definitions = tuple(MetricDefinition(
        metric_id, int(metric_id.rsplit(".v", 1)[-1]) if ".v" in metric_id else 1,
        aggregation, aggregation, "1", "paired kinetic-energy time series",
    ) for metric_id, _, aggregation in _KE_SUMMARY_SPECS.values()
      if metric_id not in {d.definition_id for d in _METRIC_DEFINITIONS.values()})
    return MetricArtifact(
        context=context,
        provenance=provenance,
        metric_definitions=(tuple(_METRIC_DEFINITIONS.values())
                            + tuple(_DIAGNOSTIC_DEFINITIONS.values())
                            + extra_metric_definitions
                            + tuple(run_definitions)),
        summary_definitions=tuple(summary_definitions),
        measurements=tuple(measurements),
        summaries=tuple(summaries),
        diagnostic_series=tuple(diagnostics),
        diagnostic_metadata=(
            _freeze_mapping(evaluation["band_high_diag"])
            if evaluation.get("band_high_diag") is not None else ()
        ),
    )


def write_metric_artifact(path: str | Path, artifact: MetricArtifact) -> None:
    """Persist canonical JSON; non-finite values are always a hard failure."""
    target = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            json.dump(artifact.to_dict(), f, indent=2, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_compatibility_projection(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist the established JSON projection without changing its schema."""
    target = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as f:
            temporary = Path(f.name)
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def repository_revision(repo_root: str | Path) -> tuple[str, bool]:
    """Return the exact source revision and whether local source is dirty."""
    root = Path(repo_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return revision, dirty


def compute_divergence_l2(u: np.ndarray, v: np.ndarray, domain_length: float = 1.0,
                          periodic: bool = True, dns_x=None, dns_y=None,
                          Lx: float = 1.0, Ly: float = 1.0, mask=None) -> float:
    """∇·u 的 L2 norm。periodic=True → np.roll 中央差分（週期，預設，行為不變）；
    periodic=False → np.gradient（非週期，需 dns_x/dns_y normalized 軸 + Lx/Ly 物理縮放）。
    非週期約定：u,v 形狀 [H,W]，axis=1 為 x，axis=0 為 y（對齊 meshgrid(dns_x,dns_y)）。
    mask（bool, 同形）若給定則只算遮罩內。返回 scalar。
    """
    if periodic:
        Nx, Ny = u.shape
        dx = domain_length / Nx
        dy = domain_length / Ny
        # 週期 wrap: u[i+1] - u[i-1] / (2·dx) 用 np.roll
        du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
        dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)
        div = du_dx + dv_dy
        return float(np.linalg.norm(div) / np.sqrt(div.size))  # L2 normalized by grid size
    if dns_x is None or dns_y is None:
        raise ValueError("periodic=False 需要 dns_x/dns_y (normalized 軸)")
    du_dx = np.gradient(u, dns_x, axis=1) / Lx
    dv_dy = np.gradient(v, dns_y, axis=0) / Ly
    div = du_dx + dv_dy
    if mask is not None:
        div = div * mask
        return float(np.linalg.norm(div) / np.sqrt(max(int(np.asarray(mask).sum()), 1)))
    return float(np.linalg.norm(div) / np.sqrt(div.size))


def compute_vorticity(u: np.ndarray, v: np.ndarray, domain_length: float = 1.0,
                      periodic: bool = True, dns_x=None, dns_y=None,
                      Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    """ω = ∂v/∂x - ∂u/∂y。periodic=True → np.roll（週期，預設）；False → np.gradient。
    非週期約定：u,v 形狀 [H,W]，axis=1 為 x，axis=0 為 y。
    """
    if periodic:
        Nx, Ny = u.shape
        dx = domain_length / Nx
        dy = domain_length / Ny
        dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
        du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
        return dv_dx - du_dy
    if dns_x is None or dns_y is None:
        raise ValueError("periodic=False 需要 dns_x/dns_y (normalized 軸)")
    dv_dx = np.gradient(v, dns_x, axis=1) / Lx
    du_dy = np.gradient(u, dns_y, axis=0) / Ly
    return dv_dx - du_dy


def compute_grad_frobenius(u: np.ndarray, v: np.ndarray, domain_length: float = 1.0) -> float:
    """速度梯度張量 Frobenius 範數 ‖∇u‖_F = sqrt(Σ_grid u_x²+u_y²+v_x²+v_y²)。
    週期 np.roll 中央差分；axis 慣例對齊 compute_divergence_l2 / compute_vorticity
    （axis=0 為 x, axis=1 為 y）。作為論文 divergence-ratio 的分母 ‖∇u‖_F^DNS，僅週期方域用。
    """
    Nx, Ny = u.shape
    dx = domain_length / Nx
    dy = domain_length / Ny
    u_x = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
    u_y = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
    v_x = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
    v_y = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)
    return float(np.sqrt(np.sum(u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2)))


def compute_energy_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D energy spectrum E(k) via azimuthal average of |FFT|².

    bin 涵蓋到對角 corner 波數 sqrt(2)*N//2（非僅內接圓 N//2），保 Parseval 完整
    （sum E_k == 0.5*mean(u²+v²)）；否則對角高頻 modes 會被丟（如 checkerboard 場全失）。
    Returns: (k_bins, E_k)，長度 = ceil(K.max())+1。
    """
    Nx, Ny = u.shape
    if Nx != Ny:
        raise ValueError(f"非方域 ({Nx}, {Ny}) 暫不支援；POC 假設 Nx=Ny")
    N = Nx
    u_hat = np.fft.fft2(u) / (N * N)
    v_hat = np.fft.fft2(v) / (N * N)
    energy_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
    # wavenumber grid
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    # bin into integer k（涵蓋對角 corner modes 到 sqrt(2)*N//2，保 Parseval 完整）
    n_bins = int(np.ceil(K.max())) + 1
    E_k = np.zeros(n_bins)
    for ki in range(n_bins):
        mask = (K >= ki - 0.5) & (K < ki + 0.5)
        E_k[ki] = energy_2d[mask].sum()
    k_arr = np.arange(n_bins)
    return k_arr, E_k


def spectral_coherence(u_pred, v_pred, u_dns, v_dns, eps: float = 1e-30):
    """shell-wise spectral coherence γ(k)：速度 Fourier 係數的正規化互譜。

        γ(k) = Re Σ_{|k'|∈shell k} û_p·û_d* / sqrt( Σ|û_p|² · Σ|û_d|² )

    補的是能譜看不到的那個洞：E(k) 只有振幅，「shell 能量對但 Fourier realization
    已去相關」（結構長在錯的地方）在 E(k) 上完全看不出來，γ(k) 會掉到 0。

    關鍵性質（見 tests/test_eval_structure_metrics.py）：γ 對**純幅值縮放不敏感**
    ——把能量全砍半的低通預測仍可 γ≈1。因此 γ 必須與 band 譜誤差配對解讀，
    單獨用 γ 宣稱「結構正確」是錯的。

    Returns: (k_arr, gamma_k)。DNS 或 pred 能量為零的 shell 給 nan（未定義，不是 0）。
    """
    Nx, Ny = u_pred.shape
    if Nx != Ny:
        raise ValueError(f"非方域 ({Nx}, {Ny}) 不支援 γ(k)")
    N = Nx
    up, vp = np.fft.fft2(u_pred) / (N * N), np.fft.fft2(v_pred) / (N * N)
    ud, vd = np.fft.fft2(u_dns) / (N * N), np.fft.fft2(v_dns) / (N * N)
    cross = np.real(up * np.conj(ud) + vp * np.conj(vd))   # 向量內積的實部
    pow_p = np.abs(up) ** 2 + np.abs(vp) ** 2
    pow_d = np.abs(ud) ** 2 + np.abs(vd) ** 2

    kx = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    n_bins = int(np.ceil(K.max())) + 1
    gamma = np.full(n_bins, np.nan)
    for ki in range(n_bins):
        m = (K >= ki - 0.5) & (K < ki + 0.5)
        den = np.sqrt(pow_p[m].sum() * pow_d[m].sum())
        if den > eps:
            gamma[ki] = cross[m].sum() / den
    return np.arange(n_bins), gamma


def dissipation_wavenumber(k_arr, E_dns, nu: float, domain_length: float = 1.0) -> float:
    """黏性波數 k_η（**3D Kolmogorov 形式**），以整數波數表示（供 band 邊界使用）。

        ε = 2ν Σ_k (2πk/L)² E(k)      k_η = (ε/ν³)^(1/4)      k_η,int = k_η·L/(2π)

    用途：固定整數 band 邊界（k≤5 / 5<k≤16 / k>16）與時變流動不匹配——本 case 的
    DNS 譜動態範圍隨時間變化 5 倍，同一個「mid band」在 t=0 佔 23% 能量、t=20 佔
    0.014%（1680 倍差距），跨時間量的不是同一件事。改用 k/k_η 定義後降到 13 倍。

    **這裡不主張 k_η 是 2D 流的耗散尺度。** ε 那一步在 2D 正確（就是 KE 的黏性
    耗散率），但 (ε/ν³)^{1/4} 是 3D 正向能量串級的微尺度；2D 的小尺度由 enstrophy
    串級設定，對應 Kraichnan 尺度 k_d（見 `enstrophy_dissipation_wavenumber`）。
    本函式的角色是**一個隨流動演化、pred 與 DNS 共用的一致標尺**——band 比較的有效性
    只要求兩側用同一條界線，不要求那條界線是耗散尺度。k_d 併行輸出為診斷欄位，
    供日後決定是否改用它定義 band（改了所有 band 數字都會變）。

    注意 k_η 假設慣性區存在：t=0 這類尚未發展的初始場會被高估（實測 k_η=65 但譜
    實際只到 k=9）。這是 high band 不作為 headline 的原因之一。
    """
    k_phys = 2.0 * np.pi * np.asarray(k_arr) / domain_length
    eps_diss = 2.0 * nu * float(np.sum(k_phys ** 2 * np.asarray(E_dns)))
    return float((eps_diss / nu ** 3) ** 0.25 * domain_length / (2.0 * np.pi))


def enstrophy_dissipation_wavenumber(k_arr, E_dns, nu: float,
                                     domain_length: float = 1.0) -> float:
    """Kraichnan 波數 k_d，2D 亂流 enstrophy 串級的耗散尺度，以整數波數表示。

        η = 2ν Σ_k (2πk/L)⁴ E(k)      k_d = (η/ν³)^(1/6)      k_d,int = k_d·L/(2π)

    η 是 enstrophy 耗散率（enstrophy 譜為 k²E(k)，其黏性耗散多兩個 k 的權重）。
    2D 沒有 3D 那樣的正向能量串級，小尺度由 enstrophy 串級設定，所以在 2D 文獻裡
    「耗散波數」指的是這個量，不是 `dissipation_wavenumber` 回傳的 (ε/ν³)^{1/4}。

    **僅供診斷，不定義 band 邊界。** band 續用 k_η 是為了不動已發表的數字；
    這個欄位存在的目的是把「兩者差多少」量清楚，讓日後要不要改用 k_d 是資料決定
    而非猜測。k⁴ 權重對高 k 極敏感：譜尾的 FFT corner 噪音會抬高 η，判讀時需與
    `k_cut` 併看。
    """
    k_phys = 2.0 * np.pi * np.asarray(k_arr) / domain_length
    ens_diss = 2.0 * nu * float(np.sum(k_phys ** 4 * np.asarray(E_dns)))
    return float((ens_diss / nu ** 3) ** (1.0 / 6.0) * domain_length / (2.0 * np.pi))


def band_spectral_rel_error(E_pred, E_dns, k_arr, lo, hi, eps: float = 1e-12) -> float:
    """band 內 shell-wise 相對 L2：‖E_p(k)−E_d(k)‖_{2,k∈B} / ‖E_d(k)‖_{2,k∈B}。

    刻意**不用** integrated 形式 |ΣE_p − ΣE_d|/ΣE_d——後者允許 band 內跨 shell 抵消
    （k=6 高估、k=15 低估可互相消掉），與 headline KE MAPE 原本的缺陷同一類。
    band 定義為 lo < k <= hi。
    """
    sel = (k_arr > lo) & (k_arr <= hi)
    if not sel.any():
        return float("nan")
    dp, dd = E_pred[sel] - E_dns[sel], E_dns[sel]
    return float(np.sqrt(np.sum(dp ** 2)) / (np.sqrt(np.sum(dd ** 2)) + eps))


def spectrum_rel_error(E_pred: np.ndarray, E_dns: np.ndarray, eps: float = 1e-12) -> float:
    """energy spectrum E(k) 的 relative-L2 誤差：||E_pred-E_dns||_2 / ||E_dns||_2。

    全 k-shell（含能量主導低頻 + 高頻尾），補 low_band_rel_err（僅低頻積分能量）成完整
    ε_spectrum。E_pred/E_dns 為 compute_energy_spectrum 的逐 shell E(k)。長度不符即 fail-fast。
    """
    ep = np.asarray(E_pred, dtype=float).ravel()
    ed = np.asarray(E_dns, dtype=float).ravel()
    if ep.shape != ed.shape or ep.size == 0:
        raise ValueError(f"E(k) 長度不符或為空: pred={ep.shape} dns={ed.shape}")
    return float(np.sqrt(np.sum((ep - ed) ** 2)) / (np.sqrt(np.sum(ed ** 2)) + eps))


def forcing_mode_coeff_u(
    u: np.ndarray,
    y: np.ndarray,
    k_forcing: float,
    domain_length: float = 1.0,
) -> tuple[float, float]:
    """x-平均後 u(y) 在 Kolmogorov forcing mode 上的複數 Fourier 係數 → (振幅, 相位)。

    與 `compute_metrics` 的 `kf_amp_ratio` 是不同的量：後者是 k-shell 能量比
    E_pred(k_f)/E_dns(k_f)，能譜已丟掉相位。要診斷「模態學到了但相位錯位」這種
    失效，只有複數係數看得見，故兩者並存而非擇一。

    投影 basis 必須與 forcing 同波長。physics 端的 forcing 是
    `A·sin(2π·k_f·y / domain_length)`，所以此處 basis 也吃 domain_length；
    寫死成 1.0 在非單位域會 silent 投影到錯的波數（見對應單元測試）。

    u 的 axis 0 為 x（與 DNS 場 `u[t,x,y]` 慣例一致）。
    """
    u_bar = np.asarray(u).mean(axis=0)
    phase_arg = -2.0 * np.pi * float(k_forcing) * np.asarray(y) / float(domain_length)
    coeff = np.mean(u_bar * np.exp(1j * phase_arg))
    return float(np.abs(coeff)), float(np.angle(coeff))


def per_t_to_arrays(metrics_per_t: list) -> dict:
    """逐時 metrics dict list → npz-ready 欄位陣列（scalar→[T]，list→[T, n]）。

    繪圖端直接吃這份輸出，所以欄位對齊是硬失敗而非補值：某個時間點缺欄位若
    靜默補 NaN，畫出來的曲線會少一段而圖面看不出來；E(k) 長度不一致若讓
    numpy 收成 object array，錯誤會延後到繪圖時才炸。兩者都在此擋掉。
    """
    if not metrics_per_t:
        raise ValueError("metrics_per_t 為空，無可匯出的時間點")
    keys = list(metrics_per_t[0].keys())
    out: dict = {}
    for k in keys:
        col = []
        for i, m in enumerate(metrics_per_t):
            if k not in m:
                raise ValueError(f"欄位 {k!r} 在 t-index {i} 缺席；不補值，請檢查 eval 輸出")
            col.append(m[k])
        if isinstance(col[0], (list, tuple, np.ndarray)):
            lengths = {len(np.asarray(c).ravel()) for c in col}
            if len(lengths) != 1:
                raise ValueError(f"欄位 {k!r} 各時間點長度不一致: {sorted(lengths)}")
        out[k] = np.asarray(col, dtype=float)
    return out


def sparsity_yardsticks(K: int, n_spatial_deriv: int = 2) -> dict:
    """稀疏度標尺：把 sensor budget K 釘在資訊論下界與 Nyquist 解析度上界之間。

    Why（背書 sec:count / tab:kscaling，呼應 Gupta et al. 2026 JFM 對 2D Kolmogorov 的
    determining-node 論述；給 eval/plot 單一真實來源，取代散落的硬編標尺數字）:
      - nyquist_kmax = sqrt(K/π)：K 個點在 2D 的等效 Nyquist 截止波數（解析度上界；k 超過
        此值的小尺度非觀測直接解析，而是模型/physics 外推）。
      - determining_nodes = n_spatial_deriv**2：FT determining-node 概念下對最低階導數結構的
        最寬鬆參照（黏性項 ∇²→2 → 2×2=4）。注意這「非」Foias--Temam (1984) 證的 bound——真實
        determining-node 數依 Grashof number、黏性、forcing、domain，湍流 Re 下遠大於 4；此值
        僅作 margin-above-floor 的 order-of-magnitude 參照，不承載任何 headline metric。
      - determining_ratio = K / determining_nodes：K 距該參照的倍數（margin above floor）。

    刻意不重算 DS critical resolution：它依賴 (Re, n_f) 與 forward solver，本研究 regime
    （Re≤1e6, k_f=2, QR 不規則 K）與文獻（Re≤2000, n_f=4-5, 規則 4×4）不符，於此偽造會誤導。
    """
    if K <= 0 or n_spatial_deriv <= 0:
        raise ValueError(f"K({K}) 與 n_spatial_deriv({n_spatial_deriv}) 須為正整數")
    determining_nodes = int(n_spatial_deriv) ** 2
    return {
        "K": int(K),
        "nyquist_kmax": float(np.sqrt(K / np.pi)),
        "determining_nodes": determining_nodes,
        "determining_ratio": float(K / determining_nodes),
    }


def _compute_metrics_projection(
    u_pred: np.ndarray, v_pred: np.ndarray,
    u_dns: np.ndarray, v_dns: np.ndarray,
    domain_length: float = 1.0,
    p_pred: Optional[np.ndarray] = None,
    p_dns: Optional[np.ndarray] = None,
    periodic: bool = True,
    dns_x=None, dns_y=None,
    Lx: float = 1.0, Ly: float = 1.0,
    mask=None,
    kf: int = 2,
    nu: Optional[float] = None,
) -> dict:
    """完整 metrics: per-channel + KE rel-err + divergence L2 + vorticity rel-err + E(k) slope。
    Shapes: 全部 (Nx, Ny)。

    periodic=True（預設）保持 Kolmogorov 行為（np.roll div/vorticity + E(k) spectrum）。
    periodic=False（cylinder）走 np.gradient + body mask；非方域時跳過 E(k) spectrum。
    新增 ke_pred_over_ref = 遮罩內 integrated KE 比值（CEXP-002 的主判據）。
    """
    # 遮罩（mask=None → 全場；cylinder 傳 body_mask）。u/v/KE/div/vorticity 統一遮罩語意，
    # 避免速度 rel-L2 含 body 內格點而其餘量已遮罩的不一致。
    ke_sel = np.ones_like(u_pred, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    u_err = np.linalg.norm((u_pred - u_dns) * ke_sel) / (np.linalg.norm(u_dns * ke_sel) + 1e-12)
    v_err = np.linalg.norm((v_pred - v_dns) * ke_sel) / (np.linalg.norm(v_dns * ke_sel) + 1e-12)
    # vector-velocity rel-L2：把 (u,v) 當一個向量場評，而非 u/v 兩個各自的分數。
    # 速度場是本任務重建的「基本狀態變數」，故此量是 primary field fidelity；
    # u/v 分開的數字降為診斷（分量誤差不等權時才需要看）。
    uv_err = (np.sqrt(np.sum(((u_pred - u_dns) * ke_sel) ** 2)
                      + np.sum(((v_pred - v_dns) * ke_sel) ** 2))
              / (np.sqrt(np.sum((u_dns * ke_sel) ** 2)
                         + np.sum((v_dns * ke_sel) ** 2)) + 1e-12))

    # KE
    ke_pred = 0.5 * (u_pred ** 2 + v_pred ** 2)
    ke_dns = 0.5 * (u_dns ** 2 + v_dns ** 2)
    ke_err = (np.linalg.norm((ke_pred - ke_dns) * ke_sel)
              / (np.linalg.norm(ke_dns * ke_sel) + 1e-12))
    ke_ratio = float(np.sum(ke_pred * ke_sel) / (np.sum(ke_dns * ke_sel) + 1e-12))

    # pointwise KE 誤差：先逐點取絕對值再空間平均（時間平均由 evaluate_time_series 的
    # agg_keys 完成）。這是 headline KE MAPE 的定義——先塌縮成純量再比會讓「空間分布
    # 錯了但總量對」的失效（渦位置偏移、能量在空間中搬家）被抵消掉，逐點取絕對值則不會。
    #   ke_pw_mape: 純 MAPE 語意。分母是逐點 KE_dns，低能量區權重被放大；Re=1e4 DNS 上
    #               KE_dns 無零點（min 1.1e-7 vs mean 1.3e-1），eps 由 1e-12 掃到 1e-4
    #               讀數僅 12.95%→11.43%，故良態，沿用 repo 慣例 eps=1e-12。
    #   ke_pw_nmae: normalized MAE。同樣禁止抵消，但分母是遮罩內 mean(KE_dns)，
    #               不放大低能量區、無 eps 依賴。
    ke_d_sel = ke_dns[ke_sel]
    ke_abs_diff = np.abs(ke_pred - ke_dns)[ke_sel]
    ke_pw_mape = float(np.mean(ke_abs_diff / (np.abs(ke_d_sel) + 1e-12)))
    ke_pw_nmae = float(np.mean(ke_abs_diff) / (np.mean(np.abs(ke_d_sel)) + 1e-12))

    # divergence (應 ≈ 0 for incompressible)
    div_pred = compute_divergence_l2(u_pred, v_pred, domain_length, periodic, dns_x, dns_y, Lx, Ly, mask)
    div_dns = compute_divergence_l2(u_dns, v_dns, domain_length, periodic, dns_x, dns_y, Lx, Ly, mask)

    # vorticity
    omega_pred = compute_vorticity(u_pred, v_pred, domain_length, periodic, dns_x, dns_y, Lx, Ly)
    omega_dns = compute_vorticity(u_dns, v_dns, domain_length, periodic, dns_x, dns_y, Lx, Ly)
    omega_err = (np.linalg.norm((omega_pred - omega_dns) * ke_sel)
                 / (np.linalg.norm(omega_dns * ke_sel) + 1e-12))

    # enstrophy（純量總量）: Z = ½·mean_space(ω²)。enstrophy_rel_err = |Z_pred − Z_dns|/Z_dns。
    # 與 omega_rel_err（渦量「場」的 L2）互補：此量的是「總 enstrophy 量級」重建得多準，
    # 是純量對純量的相對誤差（turbulence 重建常見的 enstrophy rel-err 定義）。
    # 遮罩語意與 u/v/KE/ω 一致（cylinder 只取 body 外格點；Kolmogorov mask=None → 全場）。
    ens_pred = 0.5 * float(np.mean((omega_pred ** 2)[ke_sel]))
    ens_dns = 0.5 * float(np.mean((omega_dns ** 2)[ke_sel]))
    ens_err = abs(ens_pred - ens_dns) / (ens_dns + 1e-12)

    # Energy spectrum 僅在週期方域有效（np.roll + Nx=Ny FFT）
    spectrum_ok = periodic and (u_pred.shape[0] == u_pred.shape[1])

    result = {
        "uv_rel_err": float(uv_err),
        "u_rel_err": float(u_err),
        "v_rel_err": float(v_err),
        "ke_rel_err": float(ke_err),
        "ke_pw_mape": ke_pw_mape,
        "ke_pw_nmae": ke_pw_nmae,
        "ke_pred_over_ref": ke_ratio,
        "ke_pred_mean": float(ke_pred[ke_sel].mean()),
        "ke_dns_mean": float(ke_dns[ke_sel].mean()),
        "omega_rel_err": float(omega_err),
        "enstrophy_rel_err": float(ens_err),
        "div_pred_l2": float(div_pred),
        "div_dns_l2": float(div_dns),
    }
    if spectrum_ok:
        k_arr, E_pred = compute_energy_spectrum(u_pred, v_pred)
        _, E_dns = compute_energy_spectrum(u_dns, v_dns)
        low_mask = k_arr <= 5  # 低頻 band (主能量區)
        low_band_rel_err = abs(E_pred[low_mask].sum() - E_dns[low_mask].sum()) / (E_dns[low_mask].sum() + 1e-12)
        result["low_band_rel_err"] = float(low_band_rel_err)
        result["spectrum_rel_err"] = spectrum_rel_error(E_pred, E_dns)  # 完整 ε_spectrum（全 k-shell rel-L2）
        # shell-wise band 譜誤差（low/mid/high）。取代 integrated band 形式——後者
        # 允許 band 內跨 shell 抵消（實測：band 內 ±50% 互相抵消時 integrated 讀
        # 0.000e+00 而 shell-wise 讀 0.500）。band 邊界是 **k_η 的分數**（見下方
        # _bands 與 dissipation_wavenumber），**不是**早期的固定整數 k<=5/16。
        # k_cut：DNS 譜的動態範圍下限（診斷用）。E(k) 的 bin 涵蓋到對角 corner
        # （N=256 → k_max=182），但高 k shell 的能量是 FFT corner 的數值噪音而非物理
        # 訊號（實測 57 個 shell < 1e-20、1 個恰為 0）。用 DNS 定義以跨方法可比。
        _kcut = float(k_arr[E_dns > 1e-6 * E_dns.max()].max())
        result["k_cut"] = _kcut
        # band 邊界由耗散尺度 k_η 定義（見 dissipation_wavenumber 的 why）。
        # nu 未提供時不產生這些欄位——不退回固定邊界，避免同名不同義的靜默誤讀。
        if nu is not None:
            _keta = dissipation_wavenumber(k_arr, E_dns, nu, domain_length)
            result["k_eta"] = _keta
            result["kcut_over_keta"] = _kcut / (_keta + 1e-12)
            # Kraichnan 尺度併行輸出（診斷）：band 邊界仍由 _keta 定義，見兩支函式的
            # docstring。存 ratio 是為了讓「換成 k_d 會差多少」一眼可讀。
            _kd = enstrophy_dissipation_wavenumber(k_arr, E_dns, nu, domain_length)
            result["k_d_kraichnan"] = _kd
            result["kd_over_keta"] = _kd / (_keta + 1e-12)
            _bands = (("low", 0.0, 0.1), ("mid", 0.1, 0.4), ("high", 0.4, 1.0))
            for _name, _f0, _f1 in _bands:
                result[f"band_rel_err_{_name}"] = band_spectral_rel_error(
                    E_pred, E_dns, k_arr, _f0 * _keta, _f1 * _keta)
        # γ(k)：相位/結構相干性。與上面的 band 譜誤差**配對**解讀——γ 對純幅值衰減
        # 不敏感，單獨看 γ 會把一個能量嚴重不足的低通預測誤判為「結構正確」。
        _, gamma_k = spectral_coherence(u_pred, v_pred, u_dns, v_dns)
        result["gamma_k"] = gamma_k.tolist()
        # band 摘要用 DNS 能量加權：高 k shell 的 DNS 能量極小、γ 在那裡雜訊大，
        # 等權平均會被這些 shell 主導而失去意義。band 邊界與譜誤差一致（k_η-based）。
        if nu is not None:
            for _name, _f0, _f1 in _bands:
                _sel = ((k_arr > _f0 * _keta) & (k_arr <= _f1 * _keta)
                        & np.isfinite(gamma_k))
                _w = E_dns[_sel]
                result[f"gamma_{_name}"] = (
                    float(np.sum(gamma_k[_sel] * _w) / np.sum(_w))
                    if _sel.any() and np.sum(_w) > 1e-30 else float("nan"))
        result["E_pred_k"] = E_pred.tolist()
        result["E_dns_k"] = E_dns.tolist()
        # 論文 tab:main_metrics 兩個診斷（§5 定義，僅週期方域；用 eval grid + 同 DNS 場保證一致）:
        #   div_ratio    = ‖∇·u_pred‖₂ / ‖∇u_DNS‖_F（div_pred = ‖∇·u_pred‖₂/√N → ×√size 還原分子）
        #   kf_amp_ratio = E_pred(k_f)/E_dns(k_f)，k_f = Kolmogorov forcing 波數（k-shell 能量比）
        grad_F_dns = compute_grad_frobenius(u_dns, v_dns, domain_length)
        result["div_ratio"] = float(div_pred * np.sqrt(u_pred.size) / (grad_F_dns + 1e-12))
        if 0 <= kf < len(E_pred):
            result["kf_amp_ratio"] = float(E_pred[kf] / (E_dns[kf] + 1e-12))
        # forcing-mode 複數係數（振幅 + 相位）。能譜比丟掉相位，看不出「模態學到了
        # 但相位錯位」；此處補上，pred/dns 各存一份供繪圖端相減。
        y_axis = np.linspace(0.0, domain_length, u_pred.shape[1], endpoint=False)
        amp_p, ph_p = forcing_mode_coeff_u(u_pred, y_axis, kf, domain_length)
        amp_d, ph_d = forcing_mode_coeff_u(u_dns, y_axis, kf, domain_length)
        result["kf_mode_amp_pred"] = amp_p
        result["kf_mode_amp_dns"] = amp_d
        result["kf_mode_phase_pred"] = ph_p
        result["kf_mode_phase_dns"] = ph_d
    if p_pred is not None and p_dns is not None:
        # gauge-corrected：壓力 gauge 任意，各自減空間均值再比
        pp = p_pred - p_pred.mean()
        pd = p_dns - p_dns.mean()
        result["p_rel_err"] = float(np.linalg.norm(pp - pd) / (np.linalg.norm(pd) + 1e-12))
    return result


def compute_metrics(
    u_pred: np.ndarray, v_pred: np.ndarray,
    u_dns: np.ndarray, v_dns: np.ndarray,
    domain_length: float = 1.0,
    p_pred: Optional[np.ndarray] = None,
    p_dns: Optional[np.ndarray] = None,
    periodic: bool = True,
    dns_x=None, dns_y=None,
    Lx: float = 1.0, Ly: float = 1.0,
    mask=None,
    kf: int = 2,
    nu: Optional[float] = None,
) -> dict:
    """Compatibility adapter for callers that still consume storage-key dictionaries."""
    return _compute_metrics_projection(
        u_pred, v_pred, u_dns, v_dns, domain_length,
        p_pred, p_dns, periodic, dns_x, dns_y, Lx, Ly, mask, kf, nu,
    )


def energy_timeseries_errors(ke_t_pred, ke_t_dns, eps: float = 1e-12) -> dict:
    """scalar 能量時序 E(t) 的相對誤差指標。

    把每幀的空間平均 KE 當成一條時間序列 E(t) 來評，量的是「全域能量動力學」，
    與 ke_rel_err（每幀整場 Frobenius L2、再對 t 平均的場誤差）是不同面向。

    Args:
        ke_t_pred, ke_t_dns: shape [T]，每幀 E(t) = 0.5 * <u²+v²>_space。
    Returns:
        ke_t_mape_spatialmean:
                        MAPE = <|E_pred-E_dns|/E_dns>_t。**已非 headline 定義**：先把整場
                        塌縮成純量再比，空間分布錯誤會被抵消（合成 5% rms 擾動實測僅讀
                        0.49%）。保留此值僅為與既有已發表數字對照；headline KE MAPE 現由
                        compute_metrics 的 pointwise ke_pw_mape 提供。
        ke_t_rel_l2:    relative L2(0,T) = ||E_pred-E_dns||_2 / ||E_dns||_2（GPT 建議主指標）
        ke_t_rel_linf:  relative L∞ = max|E_pred-E_dns| / max|E_dns|（最壞時刻）
        ke_t_final_rel: final-time = |E_pred(T)-E_dns(T)| / |E_dns(T)|（long-time drift）
    """
    ep = np.asarray(ke_t_pred, dtype=float).ravel()
    et = np.asarray(ke_t_dns, dtype=float).ravel()
    if ep.shape != et.shape or ep.size == 0:
        raise ValueError(f"E(t) 長度不符或為空: pred={ep.shape} dns={et.shape}")
    mape = float(np.mean(np.abs(ep - et) / (np.abs(et) + eps)))
    rel_l2 = float(np.sqrt(np.sum((ep - et) ** 2)) / (np.sqrt(np.sum(et ** 2)) + eps))
    rel_linf = float(np.max(np.abs(ep - et)) / (np.max(np.abs(et)) + eps))
    final_rel = float(abs(ep[-1] - et[-1]) / (abs(et[-1]) + eps))
    return {
        "ke_t_mape_spatialmean": mape,
        "ke_t_rel_l2": rel_l2,
        "ke_t_rel_linf": rel_linf,
        "ke_t_final_rel": final_rel,
    }


def aggregate_metric_rows(
    metrics_per_t: list[dict[str, Any]],
    ke_t_pred: np.ndarray,
    ke_t_dns: np.ndarray,
    *,
    pressure_evaluated: bool = False,
) -> dict[str, Any]:
    """Aggregate frame measurements into the established compatibility projection."""
    if not metrics_per_t:
        raise ValueError("metrics_per_t is empty")
    agg_keys = [
        "uv_rel_err", "u_rel_err", "v_rel_err", "ke_rel_err",
        "ke_pw_mape", "ke_pw_nmae", "omega_rel_err", "enstrophy_rel_err",
        "low_band_rel_err", "spectrum_rel_err", "div_pred_l2", "div_dns_l2",
    ]
    if pressure_evaluated:
        agg_keys.append("p_rel_err")
    if "div_ratio" in metrics_per_t[0]:
        agg_keys += ["div_ratio", "kf_amp_ratio"]
    if "band_rel_err_low" in metrics_per_t[0]:
        agg_keys += ["band_rel_err_low", "band_rel_err_mid", "gamma_low", "gamma_mid"]

    # 聚合只吃有限值：ledger 對每個 summary 宣告的聚合基礎就是 "measured values
    # only"（見 _build_ledger），用 np.mean/np.quantile 會讓單一 NaN 幀汙染整條，
    # 與那句宣告自相矛盾。哪幾幀被排除不需另存——ledger 的 sample_count 已經是
    # _measured_frame_count(per_t, key)。全幀皆非有限時保留 NaN（fail loud），不塞 0。
    def _agg(key: str, fn) -> float:
        vals = np.asarray([row[key] for row in metrics_per_t], dtype=float)
        if not np.isfinite(vals).any():
            return float("nan")
        return float(fn(vals))

    metrics_mean = {key: _agg(key, np.nanmean) for key in agg_keys}
    metrics_p90 = {
        key: _agg(key, lambda v: np.nanquantile(v, 0.90)) for key in agg_keys
    }
    ke_t_errors = energy_timeseries_errors(ke_t_pred, ke_t_dns)
    ke_t_errors["ke_t_mape"] = metrics_mean["ke_pw_mape"]
    ke_t_errors["ke_t_nmae"] = metrics_mean["ke_pw_nmae"]
    ke_t_errors["ke_mape_def"] = "pointwise_v2"

    band_high_diag = None
    if "band_rel_err_high" in metrics_per_t[0]:
        high_error = np.asarray(
            [row["band_rel_err_high"] for row in metrics_per_t], dtype=float,
        )
        high_coherence = np.asarray(
            [row["gamma_high"] for row in metrics_per_t], dtype=float,
        )
        n_valid = int(np.isfinite(high_error).sum())
        band_high_diag = {
            "n_valid_frames": n_valid,
            "n_total_frames": len(metrics_per_t),
            "k_cut_min": float(np.min([row["k_cut"] for row in metrics_per_t])),
            "k_cut_max": float(np.max([row["k_cut"] for row in metrics_per_t])),
            "band_rel_err_high_over_valid": (
                float(np.nanmean(high_error)) if n_valid else None
            ),
            "gamma_high_over_valid": (
                float(np.nanmean(high_coherence))
                if np.isfinite(high_coherence).any() else None
            ),
            "note": ("high band = (0.4 k_eta, 1.0 k_eta]；僅在該區間有 DNS 訊號的幀"
                     "（k_cut > 0.4 k_eta）才有效。band 邊界逐幀隨 k_eta 變動，"
                     "故不作為 headline，僅供診斷"),
        }
    return {
        "ke_t_pred": np.asarray(ke_t_pred).tolist(),
        "ke_t_dns": np.asarray(ke_t_dns).tolist(),
        "metrics_per_t": metrics_per_t,
        "metrics_mean": metrics_mean,
        "metrics_p90": metrics_p90,
        "ke_t_errors": ke_t_errors,
        "band_high_diag": band_high_diag,
    }


def evaluate_field_series(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_reference: np.ndarray,
    v_reference: np.ndarray,
    *,
    context: EvaluationContext,
    provenance: SourceProvenance,
    p_pred: np.ndarray | None = None,
    p_reference: np.ndarray | None = None,
    dns_x: np.ndarray | None = None,
    dns_y: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    kf: int = 2,
    run_measurements: Mapping[str, float] | None = None,
) -> tuple[MetricArtifact, dict[str, Any]]:
    """Deep seam: field series in, semantic artifact plus its projection out.

    Returns `(artifact, projection)`. Both faces come from one aggregation, so the
    established storage-key dictionary is a projection of the canonical record
    rather than a parallel hand-assembled one. Producers that must keep writing the
    established schema pass the projection to `write_compatibility_projection`;
    nothing needs a definition-id-to-storage-key table, which would be a second
    hand-maintained surface free to drift away from `_METRIC_DEFINITIONS`.
    """
    fields = tuple(np.asarray(x) for x in (u_pred, v_pred, u_reference, v_reference))
    if len({field.shape for field in fields}) != 1 or fields[0].ndim != 3:
        raise ValueError("velocity field series must share shape [T, Nx, Ny]")
    if fields[0].shape[0] != len(context.times):
        raise ValueError("field-series length does not match evaluation context times")
    if context.grid_shape is not None and fields[0].shape[1:] != context.grid_shape:
        raise ValueError("field grid does not match evaluation context grid_shape")
    if context.pressure_evaluated:
        if p_pred is None or p_reference is None:
            raise ValueError("pressure_evaluated requires predicted and reference pressure fields")
        if np.asarray(p_pred).shape != fields[0].shape or np.asarray(p_reference).shape != fields[0].shape:
            raise ValueError("pressure field series must match velocity field shape")

    rows: list[dict[str, Any]] = []
    ke_pred = np.empty(fields[0].shape[0], dtype=float)
    ke_reference = np.empty(fields[0].shape[0], dtype=float)
    for index, time in enumerate(context.times):
        row = _compute_metrics_projection(
            fields[0][index], fields[1][index], fields[2][index], fields[3][index],
            context.domain_length,
            (np.asarray(p_pred)[index] if p_pred is not None else None),
            (np.asarray(p_reference)[index] if p_reference is not None else None),
            context.periodic, dns_x, dns_y, context.Lx, context.Ly, mask, kf,
            context.viscosity,
        )
        row["t"] = float(time)
        rows.append(row)
        ke_pred[index] = row["ke_pred_mean"]
        ke_reference[index] = row["ke_dns_mean"]
    evaluation = aggregate_metric_rows(
        rows, ke_pred, ke_reference,
        pressure_evaluated=context.pressure_evaluated,
    )
    artifact = build_metric_artifact(
        evaluation, context=context, provenance=provenance,
        run_measurements=run_measurements,
    )
    return artifact, evaluation


def read_metric_summary(
    payload: Mapping[str, Any],
    definition_id: str,
    *,
    legacy_interpretation: LegacyInterpretation | None = None,
) -> MetricSummary:
    """Read one semantic summary from a current metric artifact projection."""
    if payload.get("schema_version") == METRIC_ARTIFACT_SCHEMA_VERSION:
        artifact = MetricArtifact.from_dict(payload)
        matches = [s for s in artifact.summaries if (
            s.definition_id == definition_id or s.summary_definition_id == definition_id
        )]
        if len(matches) != 1:
            if not matches:
                raise MetricUnavailable(f"{definition_id} is unavailable in canonical artifact")
            raise ValueError(f"canonical artifact has multiple summaries for {definition_id}")
        summary = matches[0]
        return (replace(summary, definition_id=definition_id)
                if summary.summary_definition_id == definition_id else summary)
    if definition_id == VORTICITY_REL_ERR_TIME_MEAN_V1:
        return MetricSummary(
            definition_id=definition_id,
            value=float(payload["metrics_mean"]["omega_rel_err"]),
            provenance="embedded",
        )
    if definition_id == KE_T_MAPE_POINTWISE_V2:
        ke_t_errors = payload["ke_t_errors"]
        marker = ke_t_errors.get("ke_mape_def")
        if marker != "pointwise_v2":
            if marker is None:
                raise AmbiguousMetricSemantics(
                    "legacy metric artifact does not identify ke_t_mape semantics"
                )
            raise MetricUnavailable(
                f"{KE_T_MAPE_POINTWISE_V2} is unavailable; artifact declares "
                f"ke_mape_def={marker!r}"
            )
        return MetricSummary(
            definition_id=definition_id,
            value=float(ke_t_errors["ke_t_mape"]),
            provenance="embedded",
        )
    if definition_id != KE_T_MAPE_SPATIALMEAN_V1:
        raise KeyError(f"unknown metric definition: {definition_id}")

    ke_t_errors = payload.get("ke_t_errors")
    if ke_t_errors is None:
        if (definition_id == KE_T_MAPE_SPATIALMEAN_V1
                and legacy_interpretation is not None
                and "metrics_per_t" in payload):
            if legacy_interpretation.definition_id != definition_id:
                raise ValueError("legacy interpretation definition does not match request")
            rows = payload["metrics_per_t"]
            values = energy_timeseries_errors(
                [row["ke_pred_mean"] for row in rows],
                [row["ke_dns_mean"] for row in rows],
            )
            return MetricSummary(
                definition_id=definition_id,
                value=values["ke_t_mape_spatialmean"],
                provenance="legacy_interpretation",
                legacy_interpretation=legacy_interpretation,
            )
        raise KeyError("ke_t_errors")
    if "ke_t_mape_spatialmean" not in ke_t_errors:
        if "ke_mape_def" in ke_t_errors:
            raise MetricUnavailable(
                f"{KE_T_MAPE_SPATIALMEAN_V1} is unavailable; artifact declares "
                f"ke_mape_def={ke_t_errors['ke_mape_def']!r} but has no "
                "ke_t_mape_spatialmean value"
            )
        if "ke_t_mape" in ke_t_errors and "ke_mape_def" not in ke_t_errors:
            if legacy_interpretation is not None:
                if legacy_interpretation.definition_id != definition_id:
                    raise ValueError(
                        "legacy interpretation definition does not match request"
                    )
                return MetricSummary(
                    definition_id=definition_id,
                    value=float(ke_t_errors["ke_t_mape"]),
                    provenance="legacy_interpretation",
                    legacy_interpretation=legacy_interpretation,
                )
            raise AmbiguousMetricSemantics(
                "legacy metric artifact does not identify ke_t_mape semantics; "
                "provide an explicit legacy interpretation"
            )
        raise KeyError("ke_t_mape_spatialmean")

    value = float(ke_t_errors["ke_t_mape_spatialmean"])
    return MetricSummary(
        definition_id=definition_id,
        value=value,
        provenance="embedded",
    )
