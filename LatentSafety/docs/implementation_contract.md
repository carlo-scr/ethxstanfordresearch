# Implementation and artifact contract

This contract lets the representation and control workstreams proceed independently. Changes to it
require both owners to review the same example artifact.

## Environment adapter

Every task adapter must expose, for each complete trajectory:

- stable `trajectory_id`, timestep, environment seed, and scenario/shift labels;
- privileged physical state or belief target used only for supervision/evaluation;
- deployed observation and optional observation history;
- action, next state/observation, reward, termination, and truncation;
- canonical signed safety margin in declared state units;
- unsafe indicator with an explicit boundary convention;
- safe-action margins over the frozen audit grid, or a callable verified optimizer; and
- renderer/environment version plus a deterministic configuration hash.

Train, validation, radius-calibration, and test splits operate on whole trajectory IDs. The
`AuditRecord` validator rejects a trajectory present in more than one split.

## Frozen representation adapter

Each checkpoint must provide:

```text
encode(observation_or_history) -> latent
predict(latent, action) -> next-latent distribution or point
decode(latent) -> optional reconstructed observation
metadata() -> architecture, dimension, normalization, training objective, checkpoint rule
```

The adapter must state whether the latent is deterministic or stochastic and which object the
distance/uncertainty set applies to. History length and recurrent reset semantics are part of the
checkpoint identity.

Checkpoint selection uses validation utility and a predeclared Pareto rule. Final test safety labels
may not select checkpoints, radii, stopping times, or regularization weights.

## Per-example audit table

The portable core schema is `latent_safety.records.AuditRecord`:

| Field | Meaning |
|---|---|
| `sample_id` | immutable trajectory/timestep-derived ID |
| `trajectory_id` | split and bootstrap unit |
| `split` | train, validation, calibration, or test |
| `safety_margin` | signed physical/canonical margin |
| `latent` | frozen encoder output used by the method |
| `observation_latent` | raw-observation/oracle comparison embedding, when defined |
| `state_latent` | privileged ground-truth-state comparison embedding, when defined |
| `action_safety_margins` | one value per immutable action-grid entry, when defined |

Large images and states live in versioned arrays referenced by ID rather than duplicated in JSONL.
Every paper witness must resolve back to the original state, observation, trajectory, and model
checkpoint.

## Checkpoint manifest

Minimum fields:

- experiment/config schema version and SHA-256;
- Git revision and dirty flag;
- path-free installed-distribution snapshot and SHA-256 (plus the cluster lock/container when
  available);
- accelerator, deterministic backend flags, seed, wall time, and failure status;
- dataset/split manifest hashes;
- model and optimizer state hashes;
- checkpoint-selection metric and selected epoch; and
- parent checkpoint for fine-tuned arms.

## Aggregate result table

One row per frozen checkpoint and evaluation condition, never one row per frame. Required columns:

- task, shift, representation family, dimension, training arm, and seed;
- reconstruction/rollout/probe utility metrics;
- operational radius/uncertainty calibration ID;
- static robust-defect curve summary and witness count;
- observation-level comparison;
- action-conflict rate, common-action slack, and the mean across trajectories of the
  within-trajectory 95th-percentile required violation;
- certificate coverage/false-safe/false-reject metrics where applicable;
- episode violation, severity, return, success, intervention, and latency for E5; and
- run/evidence manifest paths.

Frame- or episode-level tables remain available for block bootstrap and audits but are not treated
as independent trained-model replicates.

## Preconfirmation weight-freeze input

The E2 weight freeze consumes exactly 288 validation-only aggregate rows: 18 paired `none` controls
and 270 learned-arm cells (`3 arms x 5 positive weights x 3 domains x 2 families x 3 pilot seeds`).
Each row records the safety endpoint, reconstruction and rollout errors, eligible-center coverage,
median nonself neighborhood mass, empty-trajectory count, and explicit run/coverage completion.
Predicted-profile rows additionally record the five-teacher gate, normalized p95 error, and assembled
label-manifest SHA-256. All five weights in one domain/family/data-seed cell must reference the same
label manifest.

Every completed indexed job terminates in a self-authenticating
`e2_preconfirmation_task_handoff_v1` manifest. `scripts/aggregate_e2_preconfirmation.py` reopens the
underlying run, dataset, checkpoint, evaluation (where applicable), post-fit validation, audit, and
profile handoff files; rejects any missing, duplicate, failed, split-leaking, or checksum-mismatched
cell; and emits `e2_preconfirmation_validation_observations_v1`. That artifact contains exactly 288
selector observations and 288 self-hashed radius sidecars. The file-backed selector accepts only
this canonical authenticated aggregate, never an unauthenticated row list.

`src/latent_safety/analysis/matched_radius.py` freezes the validation-only matched-radius reducer.
For `N` exactly paired samples it defines each eligible center's nonself neighborhood mass as
`(|B_delta(z_i) intersect I_0| - 1)/(N - 1)`, where the reference population is first restricted
to individually finite-grid-viable samples and eligibility then requires a viable nonself neighbor.
It takes the median across eligible centers, fixes the
no-supervision reference at relative radius 0.05 on the grid `[0, 0.01, 0.02, 0.05, 0.10]`, and
chooses the learned radius only by smallest relative mass mismatch, with smaller-radius tie-break.
Each arm's radius scale is its validation median pairwise latent distance on at most 1024 samples
selected by SHA-256 of sample ID; no safety value enters scale or radius selection. Zero or undefined
support fails closed. The safety endpoint is the equal-trajectory mean of within-trajectory linear
p95 eligible-center required common-action violation. `scripts/reduce_preconfirmation_radius.py`
accepts only paired validation JSONL records, recomputes both curves, verifies physical pairing,
pins file/config/implementation hashes, rejects nonvalidation access, reports empty trajectories and
sign retention, refuses overwrite, and marks its standalone output diagnostic rather than evidence
eligible. The canonical aggregator independently reopens the task manifests and record files.
Its `selector_radius_provenance` object carries the fixed control relative radius 0.05, selected
learned relative radius, both absolute radii, both arm-audit SHA-256 values, physical-pairing SHA-256,
relative mass mismatch, and strict-sign result. The 288-row aggregator must authenticate and retain
that object for every learned row, and the weight selector must serialize the selected candidate's
complete object into the freeze artifact; `safety` and `neighborhood_mass` alone are insufficient to
reconstruct the frozen radius.

`src/latent_safety/analysis/preconfirmation.py` applies the utility, coverage, no-empty-trajectory,
matched-mass, and profile gates. Matched-mass safety must improve strictly over paired `none` in each
pilot seed; selection then minimizes the three-seed mean safety endpoint, the worse of the two
paired-none utility ratios, and finally the positive weight. The zero-denominator rule admits only a
zero learned error and records the tied ratio as one. `scripts/select_preconfirmation_weights.py`
refuses incomplete or duplicate factorials, pins the input checksum, and emits all 18 stratum
freezes or a fail-closed readiness result. Every selected weight retains three complete seed-level
radius/audit-hash freezes. It never consumes calibration or final-test labels.

## Confirmatory analysis input

The frozen E2 analysis consumes exactly one complete row for every
`domain x model_family x arm_or_derived_view x seed` cell: three domains, two families, the proposed
profile arm plus four comparators/views, and seeds 100--107, for 240 rows total. Each row must contain
finite nonnegative safety, reconstruction, and rollout endpoints plus explicit run-complete and
coverage-complete flags. Missing, duplicate, failed, or unexpected cells abort analysis.

`src/latent_safety/analysis/confirmatory.py` converts that factorial into exactly 36 elementary
bounds. `scripts/run_confirmatory_analysis.py` is the file-backed entrypoint and records the input
SHA-256 in its result. Within seed/domain it averages AE and beta-VAE differences equally, then uses
the frozen 100,000-resample paired-seed bootstrap. Notebook views may exercise synthetic regression
fixtures, but only a checksum-pinned complete final-test input may support a promoted result.

The predicted-profile component is implemented across `profile_protocol.py`, `profile_teacher.py`,
`profile_artifacts.py`, `profile_coverage.py`, and `profile_downstream.py`: the exact 330-teacher
plan, five-fold held-out-label assembly, separate observed-rollout validation bundles, nearest-rank
p95/sign gates, distinct-target fitting, and validation-only post-fit physical audit all have
authenticated handoffs. The FCSRL numerical contract in `fcsrl_protocol.py` and
`fcsrl_training.py` implements the 63-atom symlog projection, ten-transition recursion, inclusive
ending mask, deterministic sequence windows, EMA encoder, categorical head/loss, exact
checkpoint/resume state, and per-run regression fixture. Both learned arms enter the same exact
288-task validation-only plan and task-manifest boundary; the matched-radius reducer, authenticated
aggregator, and selector freeze all 18 stratum weights or fail closed. Reduced CPU chains test this
plumbing only. No production teacher grid, complete preconfirmation grid, GPU comparison, or smoke
metric is promoted as learned evidence.

## Paper promotion rule

A plotting script consumes only frozen aggregate manifests. A result may enter a figure/table after:

1. all planned seeds have a success/failure record;
2. the claim registry names the producing command and evidence;
3. a second collaborator reproduces the aggregate from per-example records; and
4. the witness images/states pass a sensor-versus-encoder attribution review.
