# ADR 0002: Make the metric artifact the sole write path for evaluation results

- Status: Accepted
- Date: 2026-08-06

## Context

`pi_lnn_jax/metric_artifact.py` was introduced as the canonical record for evaluation
results, and `evaluate_field_series` was written as its deep seam: reconstructed and
reference field series in, semantic artifact out. That seam had **no production caller**.

Fourteen scripts imported the module, but only `evaluate_exp245.py` and
`evaluate_multi_re.py` crossed its write surface. Five producers — `evaluate_baselines.py`,
`classical_baselines_fair.py`, `eval_gappy_cross_re.py`, `train_baseline_shred.py`, and
`cost_accuracy.py` — each ran their own per-frame loop, aggregated with a second
implementation (`baseline_eval.mean_metrics`), and hand-assembled a JSON dictionary.

The consequences were concrete rather than theoretical:

- The metric semantics marker `ke_mape_def` was written as a literal string by hand in
  `classical_baselines_fair.py` and `backfill_ksweep_metrics.py`, while
  `_common.ke_mape.read_ke_mape` — written specifically to make that marker a
  precondition of reading a value — had zero script consumers. This is the same root
  cause recorded as TD-17.
- `compare_baselines.py` merged the hand-rolled schemas with artifact-produced rows into
  the paper's four-way comparison table.
- Two aggregation implementations coexisted with different failure semantics: a metric
  that was `None` in every frame produced a silent `None` from one and a `TypeError`
  from the other.
- `json.dump` with its default `allow_nan=True` wrote literal `NaN` into published
  metric files, which is not valid JSON.

## Decision

`metric_artifact.evaluate_field_series` is the sole path by which an evaluated
reconstruction becomes a persisted result. It returns `(artifact, projection)`; both
faces come from one aggregation.

### Granularity and identity

One artifact per evaluated unit: `(method × Re)` for the baseline evaluators, and
`(method × Re × modes)` for the cost-accuracy sweep. This follows the precedent already
set by `evaluate_multi_re.py`.

- Reynolds number is recorded canonically as `EvaluationContext.viscosity = 1/Re`.
- Method, split membership, and evaluation-protocol settings are recorded in
  `SourceProvenance.details`.
- Artifact filenames extend the projection's stem, following
  `{stem}_re{Re}[_{method}].metric_artifact.json`.

### The established schema becomes a projection

Every producer keeps writing its previous JSON file, with the same name, the same keys,
and the same key order, through `write_compatibility_projection`. The values are read
out of the projection returned by the seam, so the record and the file cannot drift
apart. No downstream consumer changes.

### Run-level cost measurements

Quantities such as reconstruction wall time have no per-frame counterpart and are not
produced by `compute_metrics`. They are carried by a `run_measurements` channel on
`build_metric_artifact`, keyed by a registry (`_RUN_MEASUREMENT_DEFINITIONS`) that
refuses unknown keys, and they are summarized as a `run_scalar` observation. Their
definitions carry a truthful formula and a real unit rather than the
`compute_metrics[...]` projection template. The domain term is **Cost measurement**,
recorded in `CONTEXT.md`.

They are recorded as measurements rather than provenance because they are consumed as
data: the cost-accuracy curve plots one of them on its x axis.

### Summary availability

A non-finite summary of a metric in `_ELIGIBILITY_NAN_KEYS` is an availability state,
not an error, matching what the per-time layer already did:

- no frame had a defined value → `NOT_APPLICABLE`;
- some frames had defined values → `UNAVAILABLE`;
- `sample_count` records how many frames were defined.

`MetricSummary` gains the invariant `MetricMeasurement` already had: a non-measured
summary must not carry a number. Non-finite summaries outside the eligibility set — and
non-finite run measurements — remain hard failures.

## Compatibility constraints

This is an architecture change. It preserves:

- output file names, key sets, and key order of every established projection;
- CLI names, defaults, and messages;
- `METRIC_ARTIFACT_SCHEMA_VERSION`, so artifacts already written on lab-server remain
  readable by `read_metric_summary` and `MetricArtifact.from_dict`;
- the numerical definition of every metric.

Two behavioral changes are intended and are not preserved:

1. A metric column that is undefined for the evaluation conditions is now written as
   `null` rather than as a literal `NaN`. The previous form was invalid JSON and read
   as a number downstream.
2. `evaluate_exp245.py` and `evaluate_multi_re.py` inherit the summary-availability
   change. A run that would previously have aborted on a non-finite band or coherence
   summary now records that summary as unavailable and continues.

## Alternatives rejected

### Project the artifact back through a definition-id to storage-key table

Rejected because it would create a second hand-maintained mapping alongside
`_METRIC_DEFINITIONS`, free to drift from it. Returning the aggregation that produced
the artifact needs no table and cannot disagree with the record.

### One artifact bundle per script run

Rejected because it would require a new dimension inside `MetricArtifact` for method and
Reynolds number, contradicting the existing design in which one record describes one
evaluation, and widening the interface without adding behavior behind it.

### Record cost in `SourceProvenance.details`

Rejected because a number that reaches a figure needs a definition and a unit. Placing
it in provenance would reproduce TD-17 in a new location.

### Bump the schema version to add a run-measurement field

Rejected because `MetricArtifact.from_dict` refuses unrecognized schema versions, so
every artifact already written would stop being readable. Cost measurements ride in the
existing `summaries` list instead.

### Keep `mean_metrics` alongside the seam

Rejected: two aggregators with different failure semantics is the condition this ADR
removes. It was retained until the acceptance runs below had compared old and new
outputs, then deleted along with the tests that existed only to back that substitution.

## Verification obligations

Completed:

- characterize the previous per-method row of `classical_baselines_fair.py` by copying
  the pre-migration aggregation verbatim into a test, and require the migrated path to
  reproduce it key for key, in order, to within 1e-12, with and without a viscosity;
- establish that `compute_metrics` forwards verbatim to `_compute_metrics_projection`,
  so per-frame values are unchanged for every producer;
- establish that `mean_metrics` and `aggregate_metric_rows` agree on the published keys;
- establish that passing whole `[T, Nx, Ny]` series to the seam reproduces the previous
  per-frame calls, with a pollution probe that swaps the velocity channels and requires
  the comparison to notice. Every migrated producer now slices its arrays this way, and
  a swap would not crash;
- pin the summary-availability states, the run-measurement registry, the unchanged
  schema version, and the definition-family boundary;
- provide the acceptance verdict as testable code rather than as a job-script heredoc:
  `_common.ab_compare.metrics_projection_diff` compares two projections key by key with a
  tolerance, reports structural drift, classifies `NaN → null` separately from a moved
  number, announces which keys it skipped, and refuses a comparison whose surface is
  empty. Reachable from a job as `ab_compare.py metrics --base … --head …`;
- guard the seam with an AST sentinel over the seven metric producers
  (`tests/test_no_metric_write_bypass.py`), and demonstrate it fires against the
  pre-migration revision of all five migrated scripts;
- run the full test suite.

- run the four deterministic producers on lab-server through Slurm as a true A/B — the
  same data and machine, `main` in one worktree and this branch in another — and compare
  every key. The historical artifacts on lab-server were **not** usable as a base: those
  for `evaluate_baselines` and `eval_gappy_cross_re` date from 2026-06-14/15 and predate
  both the JAX port and the 2026-08-01 KE-MAPE redefinition, and `cost_accuracy` had none
  at all. Diffing against them would have measured two months of unrelated change.

  Result — jobs 4927/4931, 4928/4932, 4929/4933, 4930/4934 on `acmt20`, base `9adf4b8`
  (= `main` plus the two new templates) against head `3c12915`: **every key identical
  within 1e-12 for all four**. Evidence in `knowledge/superpowers/evidence/metrics_ab_*.json`.
  Skipped keys are recorded in each evidence file rather than left implicit: wall-clock
  measurements, and the absolute config path, which necessarily differs between the two
  worktrees. The head leg wrote 33 canonical artifacts and the base leg none, confirming
  the artifacts are produced by this change and not by something already present;
- delete `mean_metrics` and the tests that existed only to back its substitution;
- confirm the two changes that reach beyond the migrated producers are inert on real data:

  *Read side.* `MetricSummary` gained the invariant that a measured summary carries a
  finite value, and `read_metric_summary` builds legacy summaries with that default.
  Reading 55 pre-existing artifacts on lab-server (25 canonical, 30 legacy `metrics.json`)
  gives **identical results before and after**: 88 read, 22 refused, 0 non-finite measured.
  The 22 refusals are `AmbiguousMetricSemantics` and a missing `ke_t_errors` — the
  module declining to guess at unmarked legacy files, which it did before this change too.

  *Write side.* `evaluate_exp245.py` and `evaluate_multi_re.py` were not migrated but
  inherit the summary-availability change. Rebuilding the artifact from a real
  `evaluate_exp245` product (`abl_b3_s4`, 101 evaluated times) reproduces every shared
  summary under both revisions, and none of the 101 frames carries a non-finite value in
  the eligibility set — so on this path the new branch is not even reached.

  Both probes are recorded in `knowledge/superpowers/evidence/metrics_ab.log`.

Outstanding:

- `train_baseline_shred.py` cannot be accepted by re-running it: a fresh run retrains the
  model, so a key-by-key comparison would conflate the metric path with where that
  training landed. Its metric path is instead backed offline by the field-series
  equivalence above, which covers exactly the transformation applied to it — the previous
  per-frame calls over `pred[i, :, :, c]` against the migrated whole-series slices. **This
  producer's acceptance is weaker than the other four**: nothing verifies its reconstruction
  or its published row end to end, only that the metric path is unchanged;
- the A/B covered Re=10000 for `classical_baselines_fair` and `cost_accuracy`, exercising
  all three methods and all five mode counts, but not the five Reynolds numbers behind the
  existing `tab_fair_crossre_*` files. It establishes that the migration moved nothing; it
  does not reproduce those published files.
