# ADR 0005: Give the read side a single semantics-checked owner

- Status: Accepted
- Date: 2026-08-11

## Context

ADR-0002 made `evaluate_field_series` the sole *write* path; ADR-0004 gave the
stamp/name/write *lifecycle* an owner (`EvaluationRunRecorder`). Both are write-side.
The **read** side — the scripts that load producer projections and synthesize paper
tables and cross-run aggregates — had no counterpart. Each reader hand-rolled the read:

- `compare_baselines.py` hard-picked `d["rows"]` and a fixed set of storage keys, hardcoded
  the method taxonomy (`_METHODS` / `_LABELS`), and carried the "a projection whose rows have
  no `method` column is pi-con" identity rule as a local heuristic. The (method × Re × split)
  view and the per-method/split means were re-derived inline.
- `aggregate_pv_campaign.py` read `d["metrics_mean"].get(k, nan)` per run. A **renamed or
  dropped** projection key silently became `nan` (behind a warning that continued anyway),
  skewing a §6/§7 number rather than failing.
- `_common.ke_mape.read_ke_mape` — the one adapter built to make metric semantics a
  precondition of reading a value (ADR-0002, TD-17) — had **zero production callers**.

So the read side lacked the guarantee the write side has: metric access spoke in storage
keys hardwired into each script, and a projection-key drift produced a wrong number quietly
instead of a loud failure. The taxonomy and the definition↔key mapping had no home.

## Decision

`pi_lnn_jax/result_table.py` owns the read side, mirroring the write side: it turns a set of
producer projections into a uniform, semantics-checked **(method × Re × split) result
table**. This is the read counterpart to ADR-0004's recorder.

### The reader owns loading, taxonomy, and semantics-checked access

`ResultTable.from_projections([...])` normalizes each producer projection into evaluated units
keyed by `(method, Reynolds, split)`. It owns, in one place:

- **The method taxonomy** — method ids, display labels, and their canonical (paper-column)
  order — and the "no `method` column → this producer's declared method id" rule as an explicit
  `default_method`, not a heuristic. A row that carries neither raises, rather than the old
  silent `"?"`.
- **Semantics-checked metric access** via `read_metric_value`: reading a metric speaks in a
  **metric definition id**, not a storage key. The KE-MAPE family routes through
  `read_metric_summary` (the `ke_t_mape` dual-semantics rule); other registered metrics do a
  strict `metrics_mean` read. A missing/renamed storage key raises `MetricUnavailable`; a
  genuine null returns `None`. This is the "no value without semantics" guarantee, on read.

### The definition↔key mapping is derived, not hand-copied

The definition-id ↔ storage-key tables are derived from `metric_artifact._METRIC_DEFINITIONS`,
the write-side registry — so the read side cannot drift from the write schema. This mirrors
ADR-0002's single-source principle on the read side.

### Empty cell vs drift is a first-class distinction

`ResultTable.value(...)` returns `None` when the evaluated unit is absent (an empty table
cell) but **raises** when the unit is present and the metric's storage key has drifted. The
old readers collapsed both into `None`/`nan`. Separating them is the point: an absent cell is
expected; a drifted key is a bug that must not become a silent number.

### The reader does not own statistics, plotting, or the write

Aggregation statistics (mean / sd / Welch) stay in the campaign aggregator; plotting stays in
the plot scripts; the compatibility-projection *write* stays with the recorder / write side
(ADR-0004). The table owns reading and identity, nothing more.

### Location: in the package, not `scripts/_common/`

`result_table.py` lives in `pi_lnn_jax/` for the same reason the write seams do: it is tied to
the metric-artifact schema and must not drift from it (`scripts/CLAUDE.md` §1).

## Compatibility constraints

A behaviour-preserving wide refactor (expand → migrate → contract). It preserves, for valid
inputs:

- every number `compare_baselines` and `aggregate_pv_campaign` produce — the printed tables,
  the `compare_4way.json` (`{metric, methods, rows, summary}`, key-for-key), and the §6/§7 σ
  aggregation are byte-identical (golden diffs below);
- CLI names, defaults, and the figure — the reader is an internal seam.

One behavioral change is intended and is *not* preserved: a **renamed / dropped projection
key** now raises (`MetricUnavailable` / `KeyError`) where the old readers silently yielded
`None` or `nan`. The producers' registered columns are always present in practice (Kolmogorov
computes all of them; the picon rows always carry the key, value possibly null), so valid
inputs are unaffected — only genuine drift trips the guard.

## Alternatives rejected

### A per-metric read helper instead of a table

Rejected for the table-building consumers. A bare `read_metric_value` helper (which the plot
scripts, reading a single metric across files, correctly use) leaves the taxonomy, the split
derivation, and the pi-con identity rule re-carved in every reader. The (method × Re × split)
table is what `compare_baselines` and `aggregate_pv` actually build; owning it removes the
re-carving. Plot scripts are deliberately **not** migrated — they already read single metrics
through the typed seam and do not need the table.

### Route readers through the existing `ke_mape` adapter

Rejected. `read_ke_mape` is a single-metric adapter with zero callers; `read_metric_value`
subsumes its ke_t_mape dual-semantics behavior. The adapter and its test are retired in the
contract slice.

### Hand-copy the definition↔key table into the read module

Rejected. A hand-copied table drifts from `_METRIC_DEFINITIONS`; deriving it cannot. The
coupling to a private registry name is intentional and load-bearing.

### Put it in `scripts/_common/`

Rejected. The single criterion is whether drifting from the schema would be wrong. The metric
definitions and the `read_metric_summary` seam are in the package because they must not drift;
the table inherits that constraint (mirrors ADR-0004's placement decision).

## Verification obligations

Completed:

- unit tests through the interface (`tests/test_result_table.py`, 14): the table serves
  `(method, Re, split) → value`, method/split iteration, and the taxonomy/pi-con identity;
- pollution probes: a renamed metric key raises rather than returning a silent `None`; a
  genuine null returns `None`; a row with no `method` and no default raises; a duplicate
  `(method, Re, split)` and a Re spanning multiple splits raise; the ke_t_mape dual-semantics
  route through `read_metric_summary` and an unmarked legacy artifact raises
  `AmbiguousMetricSemantics`;
- per-consumer golden diffs on synthetic inputs: `compare_baselines` stdout and
  `compare_4way.json` **identical** before/after (null cells, taxonomy column order, byte-equal
  float means); `aggregate_pv_campaign` 15-run aggregation **output-identical** (§6 table, §7
  σ / ratio / Welch);
- the read-side sentinel (`tests/test_no_metric_read_bypass.py`) guards that the migrated
  consumers import `result_table` and do not hand-pick `["metrics_mean"]` for a metric read,
  with a self-discrimination test; it mirrors `tests/test_no_metric_write_bypass.py`;
- the full test suite is green (921 passed) after retiring `_common.ke_mape`.

Not done, by decision:

- plot readers (`plot_ksweep`, `plot_crossre_sensor_budget`, …) are **not** migrated: they read
  a single metric via `read_metric_summary` (already the typed seam), and the table abstraction
  would be over-applied to a single-metric read.
