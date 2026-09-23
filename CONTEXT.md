# PI-CON Evaluation

This context names the persisted evidence produced when PI-CON and its baselines are evaluated against reference flow fields.

## Language

**Metric artifact**:
A logical evaluation record containing per-time measurements, aggregate summaries, metric semantics, and provenance. Its JSON and array files are compatible projections of the same record.
_Avoid_: Metrics JSON, result dictionary

**Metric measurement**:
A metric value associated with one evaluation time and the conditions under which it is defined.
_Avoid_: Per-time value, metric row

**Metric definition**:
The stable, versioned scientific meaning of a metric, including its formula and the conditions under which it is defined. It is distinct from the metric's display label and from the artifact's storage format.
_Avoid_: Metric key, column name, display name

**Metric availability**:
The recorded state of a metric measurement: measured, not applicable to the evaluation conditions, or unavailable from the supplied evidence. A numerical failure is an error, not an availability state.
_Avoid_: Missing value, null metric

**Diagnostic series**:
A derived, non-scalar sequence such as an energy spectrum that supports metric interpretation and figures but is not aggregated as a scalar metric measurement.
_Avoid_: Array metric, field artifact

**Cost measurement**:
A run-level observation of the resources an evaluated reconstruction consumed, such as wall-clock seconds per field. It is recorded in the metric artifact with its own definition and unit, and it has no per-time counterpart, so it is summarized as a single run-scalar observation rather than aggregated over evaluation times. It is not source provenance: it is consumed as data, including as a figure axis.
_Avoid_: Timing metadata, benchmark note, wall time

**Metric summary**:
An aggregate derived from metric measurements in the same metric artifact according to a summary definition.
_Avoid_: Headline number, aggregate dictionary

**Summary definition**:
The stable meaning of an aggregation rule, including which metric definition it summarizes, which measurements are eligible, and how many samples contributed.
_Avoid_: Summary key, aggregation helper

**Field artifact**:
A persisted reconstructed flow field used for spatial analysis or figures; it is not part of the metric artifact.
_Avoid_: Field metrics, metric arrays

**Evaluation context**:
The physical and numerical conditions required to interpret metric measurements, such as the domain convention, periodicity, viscosity, coordinates, mask, and evaluation times.
_Avoid_: Metric options, extra kwargs

**Legacy metric artifact**:
A metric artifact produced before its metric semantics were recorded explicitly and therefore requiring an explicit interpretation when read.
_Avoid_: Old metrics, unversioned result

**Legacy interpretation**:
An explicit assignment of metric definitions to a legacy metric artifact. It is recorded as provenance and must not be inferred automatically from filenames, paths, or timestamps.
_Avoid_: Legacy guess, automatic upgrade

**Source provenance**:
The recorded lineage of the producer and inputs from which a metric artifact was created, including checkpoint verification or baseline configuration and any legacy interpretations.
_Avoid_: Extra metadata, run notes

**Evaluation run recorder**:
The single owner of the stamp-and-name-and-write lifecycle that surrounds the metric-artifact write. Constructed once per evaluation run with a producer identity and output location, it assembles the source provenance for each evaluated unit, guarantees the resolved evaluation protocol is embedded in it, reproduces the frozen artifact filename from ordered identity components, and persists the artifact through the sole write path. It does not own reconstruction, protocol resolution, the summary row, or the compatibility projection, which stay with each producer.
_Avoid_: Metric writer, provenance stamper, eval helper

**Evaluation result table**:
The read-side counterpart to the evaluation run recorder: a uniform, semantics-checked table over a set of producer projections, keyed by (method, Reynolds, split). It owns the method taxonomy and the "no method column → this producer's method id" identity rule, and it reads every metric through the write-side semantic layer so a renamed or dropped projection key fails loudly rather than yielding a silent wrong number. It does not own aggregation statistics, plotting, or the compatibility-projection write.
_Avoid_: Metrics dict, results dataframe, comparison rows

# PI-CON Training Configuration

This context names the boundary between configuration intent and values derived while assembling a training run.

## Language

**Effective configuration**:
The complete, validated, case-specific statement of configuration intent after all configuration layers have been resolved. It is structurally immutable and does not contain raw TOML or an argparse namespace.
_Avoid_: Effective dictionary, merged args, config blob

**Configuration layer**:
An ordered source of candidate values for canonical effective-configuration fields. The current order is schema default, TOML, then CLI; a future explicit override layer may be appended only through the same validation and safety policy.
_Avoid_: Merge source, override dictionary

**Case policy**:
The Kolmogorov- or Cylinder-specific adapter that owns CLI shape, compatibility defaults, TOML field mapping, validation rules, and construction of its effective-configuration type. A case policy does not own the shared layer engine.
_Avoid_: Case branch, is-cylinder switch

**Configuration provenance**:
A sidecar record of every value observed for each canonical field, including the final source, overwritten sources, ignored TOML keys, and not-yet-supported TOML keys. Numerical consumers read the effective configuration and do not unwrap provenance.
_Avoid_: Config metadata, debug dictionary

**Run specification**:
Values derived during assembly from effective configuration plus loaded data or environment constraints. For example, the requested sensor-query count belongs to effective configuration, while the population-clamped count belongs to the run specification.
_Avoid_: Mutated config, final effective args

**CLI control**:
An operational acknowledgement or source locator, such as `--allow-cpu` or `--config`, that is evaluated at the CLI boundary and cannot be bypassed by a configuration override layer.
_Avoid_: Highest-priority config
