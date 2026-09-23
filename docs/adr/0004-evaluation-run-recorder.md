# ADR 0004: Give the evaluation stamp/write lifecycle a single owner

- Status: Accepted (expand slice; migration pending)
- Date: 2026-08-09

## Context

ADR-0002 made `metric_artifact.evaluate_field_series` the sole path by which an
evaluated reconstruction becomes a persisted result. It made the *write* single-sourced.
It did not give an owner to the lifecycle that surrounds that write: assembling
`SourceProvenance` (producer, code revision, dirty flag, inputs, and the resolved
evaluation protocol embedded in `details`), naming the artifact by the frozen convention,
and persisting it. That tail is copied by hand across all seven metric producers.

The consequences are concrete, not theoretical:

- Provenance assembly is hand-written seven times. A producer that forgets to stamp the
  evaluation protocol (ADR-0003) or the code revision emits an artifact with a **silent
  provenance hole**, and that stamp travels with the number into a paper table. Nothing
  catches it.
- The filename convention documented by ADR-0002
  (`{stem}_re{Re}[_{method}][_m{modes}].metric_artifact.json`, and the bare
  `metric_artifact.json` for the single-record case) has **no owner**: it lives as a format
  string in each caller.
- Changing the provenance contract means editing seven scripts.
- `cost_accuracy.py` had already hand-carved a local recorder (`_score`) for exactly this
  lifecycle. The abstraction is proven by need; it is merely re-carved per producer.

## Decision

`pi_lnn_jax/evaluation_run.py` owns the per-evaluated-unit tail of the evaluation
lifecycle through a stateful `EvaluationRunRecorder`. This completes ADR-0002's logic:
ADR-0002 owns the write; this recorder owns the stamp-and-name-and-write lifecycle that
reaches it.

### The recorder owns the stamp/write tail; the producer keeps what varies

The recorder owns: `SourceProvenance` assembly; the guarantee that the resolved
evaluation protocol is embedded in provenance; artifact naming; the call to
`write_metric_artifact`; and pass-through of run-level cost measurements to the write seam.

Each producer keeps only what genuinely differs between producers: how it reconstructs the
field (the reconstruction signatures do not unify), how it resolves the protocol and aligns
(the existing evaluation-protocol seam), its own summary row (the columns differ —
`cost_accuracy` carries `modes` / `fit_s` / `recon_s_per_field`), and the writing of its
compatibility projection.

### Stateful recorder, constructed once per run

`EvaluationRunRecorder(producer, out_path)` is built once per evaluation run and captures
the code revision and dirty flag once, from the recorder module's own repository location.
Per evaluated unit the caller invokes `.record(...) -> projection`. This gives the strongest
locality: producer identity, code revision, and the naming convention are stated once rather
than once per evaluated unit. The recorder returns the compatibility projection so the
caller can compose its own summary row; the recorder does not own the summary row or the
projection write.

### The protocol is a required argument, so its stamp cannot be dropped

`protocol` is a required parameter of `.record()`, and the recorder — not the caller —
writes `evaluation_protocol = protocol.to_provenance()` into `SourceProvenance.details`. A
`details_extras` mapping that carries the reserved key `evaluation_protocol` is refused, so
a caller cannot silently override the guaranteed stamp. This closes the silent-provenance
hole at the point where it would open.

### Frozen naming, reproduced from ordered identity components

The recorder owns the naming function, but the caller supplies the ordered suffix
components (`RunArtifactIdentity` with optional `re` / `method` / `modes`), and the recorder
reproduces each producer's current frozen filename verbatim:

- `re` + `method` [+ `modes`] → `{stem}_re{Re}_{method}[_m{modes}]`
- `method` only (single-Re) → `{stem}_{method}` (no `_re`)
- `re` only → `{stem}_re{Re}`
- none (single record) → `metric_artifact.json`

The shapes are not unified into one template. The granularity differences are real (ADR-0002
froze filenames by granularity); forcing a canonical shape would change output filenames.

### Location: in the package, not `scripts/_common/`

The recorder lives in `pi_lnn_jax/` for the same reason the protocol and write seams it
binds do: drifting from training is exactly the failure it exists to prevent
(`scripts/CLAUDE.md` §1).

## Compatibility constraints

This is an architecture change delivered as a wide refactor (expand → migrate → contract).
It is behaviour-preserving. It preserves, once migration lands:

- every artifact filename, reproduced verbatim from the frozen conventions;
- every metric value, projection key and its order, and the artifact schema — the recorder
  delegates the computation and write to `evaluate_field_series` and `write_metric_artifact`
  unchanged;
- the `SourceProvenance` structure of every producer, key for key (`freeze_provenance_details`
  sorts keys, so injecting the protocol stamp last does not reorder anything);
- CLI names, defaults, and messages — the recorder is an internal seam with no CLI surface.

The expand slice adds the module, its unit tests, this ADR, and a glossary term, with **zero
callers**. It changes no output. The AST sentinel
(`tests/test_no_metric_write_bypass.py`) is not touched in this slice: the recorder is not a
`scripts/` producer, so it is out of the sentinel's scope, and blessing the recorder as a
sanctioned write path belongs to the contract slice once no producer hand-writes the write
anymore.

## Alternatives rejected

### A pure per-unit function instead of a stateful recorder

Rejected. Producer identity, code revision, and the naming convention are constants of a
run. A pure function would take them on every call, re-deriving or re-passing the revision
per evaluated unit — the repetition this ADR removes. The stateful recorder states them once.

### A fat recorder that also owns reconstruction

Rejected. The five field-series producers reconstruct through irreconcilable signatures
(baseline interpolators, classical LSQ solves, gappy-POD with a shared SVD sliced per mode
count, a trained SHRED forward). Folding reconstruction into the recorder via a callback
would turn it into a shallow module of optional flags — the exact anti-pattern the deep-module
framing warns against. Reconstruction stays with the producer.

### Unify the filename into one canonical shape

Rejected. ADR-0002 froze filenames by granularity, and the granularity differences are real
(single-Re carries no `_re`; the cost sweep carries `_m{modes}`; `evaluate_exp245` writes a
bare `metric_artifact.json`). A canonical shape would change published filenames — a
userspace change on the same footing as a changed value.

### Put the recorder in `scripts/_common/`

Rejected. The single criterion is whether drifting from training would be wrong. The protocol
and write seams the recorder binds are both in the package precisely because they must not
drift; the recorder inherits that constraint.

## Verification obligations

Completed (this slice):

- unit tests through the recorder interface (`tests/test_evaluation_run.py`): a normal
  `.record()` writes an artifact whose provenance carries the producer, a non-empty code
  revision, and the evaluation protocol; the frozen filename is reproduced for the four
  identity shapes (re+method, method-only, with modes, bare single record); the returned
  projection equals what `evaluate_field_series` returns; a run measurement travels through
  to the artifact as a cost measurement;
- pollution probes: omitting `protocol` or `inputs` raises loudly (a required kwarg, and an
  explicit `None` is refused); `details_extras` carrying the reserved `evaluation_protocol`
  key is refused, so the stamp cannot be overridden; swapping the velocity reference changes
  the projection, so a wrong reference is not invisible to the seam.

Outstanding (migration slice):

- per-producer CPU output A/B during migration — for each of the five field-series producers,
  every projection key and the provenance must be identical before and after adopting the
  recorder, on the same data and machine, through the existing `ab_compare` projection diff
  (ADR-0002's acceptance mode). Deferred to the migrate tasks, one vertical slice per producer;
- `evaluate_exp245` and `evaluate_multi_re` adopting the recorder is deferred to the
  GPU-verifiable window, together with the checkpoint-restore slice (they need a GPU venv and a
  checkpoint to A/B); `train_baseline_shred`'s acceptance stays weaker (its reconstruction is
  not verified end to end, only that its metric path is unchanged), matching the boundary
  ADR-0002 already set;
- the contract slice updates the AST sentinel so the recorder is a sanctioned write path and
  producers can no longer hand-write `write_metric_artifact`.
