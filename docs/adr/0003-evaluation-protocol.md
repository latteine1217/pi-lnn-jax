# ADR 0003: Make the evaluation protocol an explicit, single-sourced decision

- Status: Accepted
- Date: 2026-08-07

## Context

Evaluation samples sensors and reference DNS onto a shared time axis. How that
sampling relates to the cadence a model was *trained* on was never expressed in code.
It lived in three mutually incompatible places:

- `scripts/slurm/submit_crossre.sh` grepped `time_strides` out of the TOML with `sed`
  and refused to submit without it;
- `scripts/slurm/eval_exp301_enstrophy.sbatch` carried a comment reminding the operator
  to pass a matching `--time-stride`;
- `scripts/slurm/eval_dropout_sweep.sbatch.tmpl` stated the opposite — that its eval
  stride is deliberately unrelated to the training `time_strides`.

All three are correct. They describe three different protocols, and nothing recorded
which one a given run was using.

Three separate alignment mechanisms had also grown: value matching against the full DNS
time axis (`baseline_eval.match_sensor_dns_times`, 5 producers), and two variants that
derived an integer sub-stride and asserted `allclose(dns_t, sensor_time)`
(`evaluate_exp245`, `evaluate_multi_re`).

The cost is measured, not hypothetical. Of 27 T20-family `final_eval` artifacts, **12
were evaluated at a stride different from the one their config declared** — mostly
training 8 against evaluation 2, one at 1. The `allclose` guard cannot catch this:
when sensors and DNS are both loaded at the same wrong stride they remain mutually
consistent. Recorded as TD-4 in `knowledge/codebase/technical-debt.md`.

## Decision

`pi_lnn_jax/evaluation_protocol.py` owns the protocol: how sensors are strided, how DNS
is strided, and whether the two axes must align. It lives in the package rather than
`scripts/_common/` because drifting from the training cadence is exactly the failure it
exists to prevent (`scripts/CLAUDE.md` §1).

### Three modes, first-class, no default

- `follow_training` — sampling comes from the training config's `time_strides`. An
  explicit CLI stride that disagrees is refused rather than silently preferred.
- `fixed_grid` — the evaluation grid is fixed regardless of training cadence. Requires
  an explicit stride *and* a stated reason. The snapshot-density family sweeps
  `time_strides` from 1 to 20 as its independent variable; if evaluation followed
  training there, the `st20` arm would be scored on a grid 20× coarser than `st1` and
  the arms would not be comparable.
- `sensor_time_independent` — the sensor axis is deliberately non-uniform and unrelated
  to the DNS grid (the intermittent evaluation). `evaluate_exp245` already described
  this as "a different evaluation protocol, not a relaxation of alignment".

The mode is required with no default. A default is the failure mode being removed: it
does not crash, it silently produces numbers at a different time resolution.

### One source, not one value

`time_strides` takes at least seven distinct values across 82 configs (`[1]`×50,
`[8]`×24, `[2]`×18, `[4]`, `[20]`, `[4,4,4,4,4]`, `[2,4]`), because the correct stride
depends on each sensor set's native cadence. What is unified is the **source**, not the
number. `TRAINING_DEFAULT_TIME_STRIDE` copies the training fallback rather than defining
its own, so the two cannot diverge on configs that omit the key.

### One aligner

`load_for_evaluation` performs the only alignment: per-frame value matching, fail-fast on
an out-of-tolerance gap or a duplicate mapping. It also produces the `EvaluationContext`,
collapsing seven hand-built construction sites.

### The protocol travels with the artifact, not the projection

The resolved protocol and its stated basis are recorded in the metric artifact's
`SourceProvenance`. They are **not** added to the compatibility projection.

## Compatibility constraints

Preserved: every metric value, every compatibility-projection key and its order, and all
existing output file names.

Deliberately changed:

1. **`--protocol` is required on all seven producers**, and every Slurm entry point
   declares it. This is a caller-interface change; it is the point of the ADR.
2. **`--sensor-time-stride` / `--time-stride` no longer carry a numeric default.**
   Passing one is an assertion; omitting it defers to the protocol.
3. **Configurations whose evaluation stride disagreed with their training declaration
   now fail** instead of producing numbers. This is the path that produced the 12
   mismatched artifacts.

## Alternatives rejected

### Unify the stride to one set of values across all evaluators

Rejected: not implementable. The correct stride is per-sensor-set, and forcing one value
would make most evaluations wrong rather than consistent.

### Read the TOML and always enforce agreement

Rejected: the snapshot-density family requires a fixed evaluation grid *because* its
training cadence is the independent variable. Enforcement without a `fixed_grid` mode
would break a working ablation.

### Model the protocol as two orthogonal axes (grid source × alignment policy)

Rejected: the product yields six combinations, at least three of which nobody has
validated or can state a physical meaning for. A three-valued enumeration names only
what exists; a fourth real case earns a fourth name.

### Keep the `sed` in `submit_crossre.sh`

Rejected: the judgment it encoded was right, but a shell script protects only its own
path. Every other caller was unprotected, which is how the 12 artifacts happened.

### Add the protocol to the compatibility projection as well

Rejected on human review (2026-08-07): adding a key to the projection **is** a userspace
change, on the same footing as a changed value. The artifact is a new canonical record
nothing previously consumed; the projection exists in order not to change.
`tests/test_projection_keys_frozen.py` freezes each producer's projection key set.

## Verification obligations

Completed:

- pin protocol resolution, the three modes, and the fail-fast paths (29 tests);
- establish that `load_for_evaluation` reproduces `load_aligned_re` field-for-field on
  real Re=10000 T20 data across three settings (`max|Δ| = 0`, including the
  101×256×256 field arrays);
- establish that where the two `allclose` mechanisms accept a run, value matching selects
  the identical frames (0 violations of 8 measured cases), and that `choose_grid_stride`
  matches `evaluate_multi_re`'s inline grid logic (108 combinations, 0 differences);
- pre-register the A/B expectations **before** running them
  (`knowledge/superpowers/evidence/protocol_ab_preregistration.md`);
- run the four CPU producers as a true A/B on lab-server (jobs 4944–4957): every
  projection key identical. Evidence in `knowledge/superpowers/evidence/protocol_ab_*.json`;
- guard the caller interface (`tests/test_slurm_declares_protocol.py`) and the projection
  key sets (`tests/test_projection_keys_frozen.py`), each with a discrimination check.

Outstanding:

- **`evaluate_exp245` and `evaluate_multi_re` have not been A/B'd.** They are the
  headline paths and this change replaced their alignment mechanism. The pre-registered
  prediction is "every key unchanged"; its support is the mechanism argument and the
  local frame-selection measurement, neither obtained on their own production path.
  Running it needs a GPU venv and a checkpoint.
- **The 12 mismatched historical artifacts are not re-evaluated.** Whether to re-run them,
  and what it would mean for the conclusions drawn from them, is a claims-level decision
  recorded in TD-4. This change prevents recurrence; it does not correct the past.
