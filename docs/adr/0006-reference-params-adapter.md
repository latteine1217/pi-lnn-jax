# ADR 0006: Give the reference-params template an owner

- Status: Accepted (code landed; verified 2026-09-02, jobs 5425/5452/5472 — two of three producers bit-identical on CPU, `evaluate_multi_re` still open)
- Date: 2026-08-29

## Context

`ckpt.restore_eval_params` was carved as the single entry point by which an eval or
diagnostic script obtains checkpoint params. Its two gates — `verify_params_tree`
(against `reference_params`) and the construction fingerprint (against `model`) — are
both required arguments, on the principle that an optional gate is a decorative gate.

Requiring `reference_params` had a consequence the original change did not address: it
pushed the *construction* of that template back onto every caller. Four scripts each
wrote the same nine-line dance —

```
scripts/evaluate_exp245.py:221-232        rng → split → T_total → init_xy → init_t → model.init(7 args)
scripts/evaluate_multi_re.py:141-148      rng          → T_total → init_xy → init_t → model.init(re_norm=0.5)
scripts/diag_trainable_fourier.py:191-197 split[1]     → T_total → init_xy → init_t → model.init
scripts/dump_cp_fields.py:79-83           rng          → T_total inline → init_xy → init_t → model.init
```

— and the four are similar but not identical: one splits the RNG and uses the second
key, one splits and discards, two do not split; `re_norm` is read from config in two,
hardcoded to `0.5` in one, and derived from a hardcoded `log(10000)` in the fourth;
`T_total` carries its own fallback in each.

None of those divergences is a defect. Measured on CPU before this change: the RNG key,
the seed, `T_total`, and `re_norm` all produce **identical parameter trees** (105 leaves,
same paths and shapes). The only input that changes the tree is the trailing channel
dimension of `sensor_vals`.

So the load-bearing dimension of a nine-line ritual appeared in none of the four
callers' intent, while four things that appeared prominently in all of them were inert.
Worse, the one mistake the ritual can actually produce — a wrong leading dimension —
is invisible to the gate it feeds: `verify_params_tree` compares paths and shapes, and
neither varies with the number of query points.

## Decision

`ckpt.reference_params_for(model, sensor_vals, sensor_pos, sensor_time)` owns the
construction of the reference-params template.

The interface states only what is load-bearing. The caller supplies the model and the
sensor arrays it is about to evaluate with — things it already holds and already knows
to be correct. Query points (`xy`, `t_q`), the init seed, and `re_norm` are fixed inside
the implementation, because none of the three operators' `__call__` uses their *values*
to determine any parameter shape.

`restore_eval_params` is unchanged. Its signature is documented as a contract in two
places (`CLAUDE.md` §7.2, `scripts/CLAUDE.md` §5); widening it to absorb the template
would be a userspace change on a different footing from adding an adapter beside it.
Whether to fold the adapter in later is deliberately left open.

### Three callers migrate; the fourth deliberately does not

`evaluate_exp245`, `evaluate_multi_re`, and `dump_cp_fields` consume the template only
as a template: it reaches `verify_params_tree`, and `mgr.restore` is called with
`reference_state=None`, so no value from it survives into the restored params.

`diag_trainable_fourier` does **not** migrate. It reads the *values* of its init params
— `_find_frequency_matrix(init_params)` produces `B_init`, and `_downstream_gate` reads
the gate weights; both feed the report's `init`, `gate_init`, and `shift` fields. What
that script needs is the model's actual initialization under the training seed, not a
shape template. Migrating it would silently change every number in three report blocks.
The distinction is recorded as a comment at the site.

This is the difference the adapter's name has to carry: it produces a *reference
template*, not *an initialization*.

## Compatibility constraints

Behaviour-preserving for the three migrated callers. Preserved:

- the parameter tree each caller produces, path for path and shape for shape (pinned by
  test, see below);
- every metric value, projection key, artifact filename, and CLI surface — the adapter
  is an internal seam with no CLI;
- `restore_eval_params`' signature, its two gates, and its provenance return.

Not preserved, and intentionally so: the *values* of the reference params for the three
migrated callers now come from a fixed seed. This is inert — those values reach only
`verify_params_tree`, which compares structure. `diag_trainable_fourier`, the one caller
for which the values are not inert, is not migrated.

Orphans removed with the ritual: an unused `seed` local in `evaluate_exp245` (the
dropout path uses `args.dropout_seed`, a different variable) and an unused `import jax`
in `evaluate_multi_re` and `dump_cp_fields`.

## Alternatives rejected

### Widen `restore_eval_params` to take the sensors and build the template itself

Deeper, and possibly right. Rejected for this change because that signature is a
documented contract in two files and four callers; changing it is a userspace edit that
should be made on its own evidence, after the adapter has shown the four sites really do
unify. Expand → migrate → contract; this is the expand slice.

### Give the adapter a `seed` parameter so `diag_trainable_fourier` can migrate too

Rejected. A caller that needs specific initialization *values* is not asking for a
reference template, and a `seed` parameter would blur exactly the distinction that makes
the adapter safe to use without thinking. Four callers is not a target; three that
genuinely share a need is the seam.

### Keep the ritual and add a comment warning about the trailing dimension

Rejected. The failure it guards against is invisible to the gate the ritual feeds, so a
comment is the weakest possible instrument. Putting the dimension in the interface makes
it the caller's declared input rather than an incidental property of a dummy array.

## Verification obligations

Completed:

- pin the equivalence through the interface (`tests/test_restore_eval_params.py`, 9 new):
  each of the four historical dances is replayed verbatim — RNG split or not, `re_norm`
  `0.5` vs config-derived, per-site `T_total` fallback — and required to produce the same
  path→shape index as the adapter; the adapter is exercised for all three archs;
- discrimination: the trailing channel dimension (`sensor_value_dim` 2 vs 3) **does**
  change the tree, so the equivalence tests are not two agreeing no-ops;
- pin the claim the fixed query points rest on: query counts 1 / 3 / 64 leave the tree
  unchanged. If an arch ever makes the query dimension reach a parameter shape, this
  test goes red before the adapter can produce a wrong template;
- establish that the migrated callers cannot be affected by the changed values:
  `restore_eval_params` passes `reference_state=None` to `mgr.restore`, and uses
  `reference_params` only through `verify_params_tree`;
- full local test suite: 1072 passed, 8 skipped (the pre-existing arm64/x86_64
  init-digest skips), `ruff` clean of new findings on every touched file.

Run 2026-09-02 (`scripts/slurm/verify_adapter_ab.sbatch` job 5425,
`verify_adapter_control.sbatch` job 5452; evidence `adapter_ab_*_5425.json`,
`adapter_ctrl_5452.json`):

- the plan above could not be executed as written. `artifacts/kolmogorov/exp245_seed42`,
  which produced `artifacts/cp/seed42.npz`, no longer exists, and the nearest surviving
  checkpoint of that generation carries `query_decoder.mha_out_proj` — removed
  2026-06-23 — so `verify_params_tree` refuses it. Substituting a similarly-named
  directory would not establish identity: `ARCH` is an sbatch variable and reaches
  neither the config nor `summary.json`. The comparison was therefore rebuilt as
  before/after: the same checkpoint (`ksweep_k200_s42`) evaluated by the pre-adapter
  tree (`fd44c70`) and by main;
- **both migrated producers clear, up to platform noise, and no further.** The
  pre-registered criterion — every projection key equal to 1e-12 — is not attainable on
  this platform and was not met: `pred_u` and `pred_v` differ by 2.390e-04 and 3.008e-04
  (5.9e-04 and 1.0e-03 of their RMS). The first run omitted the determinism control that
  §7.1 of `CLAUDE.md` requires, and so produced no usable verdict. With the control
  (job 5452), running main against *itself* reproduces those two figures to seven
  significant digits, and the projection comparison reports the same mismatch count (4)
  for identical code as for before/after. The adapter's contribution is not
  distinguishable from GPU non-determinism. Every non-prediction array — `dns_*`,
  `x_grid`, `y_grid`, `sensor_pos`, `re_value` — is bit-identical in both comparisons.

CPU run 2026-09-02 (`scripts/slurm/verify_adapter_cpu.sbatch` job 5472, node
`acmt01`, x86_64; evidence `adapter_cpu_5472.json`):

- **bit-identical.** The determinism control runs first and comes back clean — main
  against itself gives zero projection-key mismatches and zero differing arrays — so the
  noise floor on this node is exactly zero, and the same zero for pre-adapter against
  main is equivalence rather than an unresolved comparison. No unexpected keys on either
  side. Three sides, one job, one node: §7.1 also warns that CPU digests do not compare
  across architectures.
- the checkpoint had to be produced inside the job. Feeding the GPU-trained
  `ksweep_k200_s42` to CPU JAX fails in orbax with `sharding passed to deserialization
  ... Got None` (job 5459); `verify_ckpt_compat.sbatch.tmpl` had already met this and
  solves it by training a few steps on CPU, which this job copies rather than reinventing.
- **the cost of that: the comparison runs at fixture scale** (`_ledger_single_re.toml`,
  `d_model=32`, 10 steps), not on the paper checkpoint. What the adapter has to preserve
  is the template-construction path, which does not vary with model size — but this is
  narrower evidence than the GPU pair, and must not be restated as bit-identity on the
  paper checkpoint. Each side took ~805 s.

Outstanding:
- **`evaluate_multi_re` is unverified.** No multi-Re checkpoint exists on lab-server —
  only single-Re `crossre_*` artifacts. Running it against one of those would produce a
  pass that means nothing, so it was skipped rather than substituted.
