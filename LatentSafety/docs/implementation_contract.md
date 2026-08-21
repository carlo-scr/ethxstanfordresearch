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

## Paper promotion rule

A plotting script consumes only frozen aggregate manifests. A result may enter a figure/table after:

1. all planned seeds have a success/failure record;
2. the claim registry names the producing command and evidence;
3. a second collaborator reproduces the aggregate from per-example records; and
4. the witness images/states pass a sensor-versus-encoder attribution review.
