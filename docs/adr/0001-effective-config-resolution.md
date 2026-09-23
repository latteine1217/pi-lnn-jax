# ADR 0001: Replace pipeline input resolvers with typed effective configuration

- Status: Accepted
- Date: 2026-08-04

## Context

The Kolmogorov and Cylinder pipelines previously carried three overlapping input representations: an argparse namespace, the loader result, and a partial effective-value dictionary. Consumers selected among those representations with string keys and one assembly path mutated the dictionary after clamping the sensor-query count.

The 2026-07-29 and 2026-07-30 pipeline design documents deliberately made the old resolvers the behavior authority while the training pipeline was being extracted. That decision protected bit-identical behavior during the extraction, but it also prevented the input boundary from becoming independently understandable. Candidate 3 explicitly reopens and replaces that decision while retaining its compatibility obligations.

## Decision

`pi_lnn_jax/config.py` owns a shared ordered-layer engine, provenance sidecar, policy protocol, and typed resolution errors. It does not import either training case.

Each case owns one local policy and one case-specific effective-configuration type:

- `pi_lnn_jax/pipeline/kolmogorov/config.py`
- `pi_lnn_jax/pipeline/cylinder/config.py`

The ordered configuration layers are:

1. compatibility defaults;
2. TOML;
3. CLI;
4. a future explicit override layer, not exposed by this change.

Existing TOML unknown-key and not-yet-supported-key warnings remain unchanged. A future programmatic override layer must reject unknown fields. Operational controls such as the config source locator and CPU safety acknowledgement are not overrideable configuration fields.

Resolution returns a clean typed configuration and a separate provenance record. Provenance stores the complete shadow chain for each canonical field. Downstream assembly and execution receive only the typed configuration.

Effective configuration records requested intent. Data-dependent and environment-dependent values belong to the assembled run specification. In particular, sensor-query population clamping no longer mutates configuration.

The two old resolver modules are deleted. No production wrapper, alias, or executable test copy remains.

## Compatibility constraints

This is an architecture-only change. It preserves:

- CLI names, defaults, one-way boolean behavior, messages, and exit codes;
- TOML insertion-order flattening and cross-section overwrite behavior;
- schema coercion and validation;
- historical case-specific fallback values;
- list and tuple runtime types;
- Cylinder guard order and backend query behavior;
- training RNG consumption and numerical assembly order.

The effective-configuration dataclasses are structurally immutable. Layer values are defensively copied, while existing list and tuple types are retained for userspace compatibility.

## Alternatives rejected

### Keep adapters around the old resolvers

Rejected because the old modules would remain the actual behavior authority and every new field would require synchronizing two interfaces.

### One shared case-neutral configuration with optional fields

Rejected because it would create a shallow union containing many invalid states and encourage case conditionals in the shared engine.

### Put provenance into every field value

Rejected because wrapping each value would pollute numerical consumers with `.value` access and make the interface shallower.

### Deep-freeze every container

Rejected for this change because converting lists to tuples would alter existing runtime types. Deep immutability requires a separately reviewed userspace change.

## Verification obligations

- characterize representative precedence, warning, type, and error behavior before deletion;
- compare all repository TOML files against the previous resolver outputs on CPU;
- run CLI parity, replay, init-digest, targeted, and full test suites;
- scan production and tests for removed resolver symbols and string-key access;
- run the existing lab-server Kolmogorov and Cylinder verification jobs through Slurm.
