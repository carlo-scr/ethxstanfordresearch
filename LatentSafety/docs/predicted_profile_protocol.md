# Nonprivileged predicted-profile protocol

Status: prospectively frozen design with all single-cell execution components implemented. The
single-task runner trains one fold teacher and atomically hands off its held-out prediction shard;
`scripts/assemble_profile_labels.py` authenticates and assembles exactly five complementary shards;
`scripts/run_profile_coverage_gate.py` executes the registered observed-rollout teacher gate; and
`scripts/run_predicted_profile_arm.py` trains the distinct downstream arm from the authenticated
cross-fitted target key. A real reduced CPU chain has exercised all four stages, including an honest
failed coverage gate from one-epoch random teachers. That engineering result is not evidence. No
production teacher cell or production coverage gate has been executed, and this document does not
authorize final-test access.

`scripts/plan_profile_teachers.py` materializes the exact preflight matrix:
`3 domains x 2 model families x 11 data seeds x 5 folds = 330` teacher tasks. It records each base
config hash, deterministic train/held-out trajectory-set hashes, teacher seed, architecture, and
isolated output path, and pins a per-task SHA-256 in addition to the whole-plan SHA-256. Each task is
independently runnable. The implementation is component-ready, but confirmation remains blocked on
executing and passing every required production cell and gate.

The runner verifies the plan self-digest, each task digest, every source-config digest, and the
canonical 330-task expansion before importing PyTorch. It refuses any existing output path.
Production tasks materialize only `train` and ordinary `validation`; the selected cross-fit fold is
excluded from fitting and emitted only as an unvalidated prediction shard. Calibration and final-test
trajectories are not generated. The sample field allowlist is pixel history, next pixel history,
behavior action, and observed safety margin; hidden state is not exposed to the model, and known-
dynamics counterfactual profiles are not constructed. The data seed and teacher optimization seed
remain distinct.

For an actual model-path check, `--engineering-smoke` derives a one-epoch, 20-trajectory, CPU-only
run while retaining four-frame pixel history and the planned domain/family/fold. The handoff records
`evidence_eligible = false`; reduced smoke trajectory hashes are never accepted as production hashes.

## Target

For each registered constant action `a`, the teacher predicts

`g_H(x, a) = min_{0 <= t <= H} h(x_t^(a))`

from pixel history, behavior actions, and observed safety-margin labels only. It may not use hidden
physical state, known dynamics, or oracle counterfactual profiles for fitting, checkpoint selection,
or arm selection. Controlled-domain oracle profiles remain diagnostic-only.

## Cross-fitted label construction

- Use exactly five trajectory folds inside the training split. Sort trajectory IDs, shuffle with
  Python `random.Random(20260822)`, and assign shuffled index modulo five.
- For held-out fold `k`, train teacher `k` only on the other four folds. Its seed is
  `10000 + 10 * data_seed + k`.
- Match the downstream model family and use the existing `LatentWorldModel` with `stack_h4`, latent
  dimension 8, hidden dimension 128, and transition hidden dimension 192.
- Use the `h_prediction` objective with safety weight 0.5 and otherwise inherit the frozen domain
  base training config. Train 30 epochs; select the earliest checkpoint minimizing ordinary
  validation `world_model_utility` within that family.
- At inference use evaluation mode, the posterior mean, and deterministic latent transitions. Roll
  each registered action constantly for the domain horizon and take the minimum predicted margin over
  `t = 0,...,H`. Teacher `k` labels only fold `k`.
- A single checksum-pinned cross-fitted label manifest is shared by all weights and arms in the same
  domain-family-data-seed stratum. Test and calibration trajectories never enter this manifest.

## Validation-only profile gate

Create a `coverage_validation` split disjoint from training, ordinary validation, radius calibration,
and final test. For each domain and data seed, freeze 200 initial-history bundles from the deployment
generator mix and reuse their identifiers across model families. The frozen seed is
`70000000 + domain_offset + data_seed`, with offsets 0, 100000, and 200000 for cart, pendulum, and
Dubins respectively. A bundle uses the first four behavior-policy frames and branches at timestep
three. Clone each bundle and execute every registered action constantly, with branch process noise
zero, for the domain horizon; record the physical margin at every `t=0,...,H` and retain its minimum.
The known-dynamics `action_safety_profile` label helper is not called. Sorted bundle index modulo
five chooses the corresponding fold teacher; no ensemble is used.

For bundle `b`, define

`e_b = max_a |g_hat_H(b,a) - g_H(b,a)| / margin_scale`,

where the target uses the minimum *observed* `h` on the physical rollout. The registered error is the
nearest-rank empirical p95: after sorting `N=200` bundle errors, use element
`ceil(0.95*N)-1` in zero-based indexing. The arm must have p95 at most 0.10 separately in every
domain-family-data-seed cell: pilot seeds 0–2 and fresh confirmatory seeds 100–107. A failing fresh
teacher is a registered method failure that fails confirmation; it cannot trigger retraining or
reselection. Also report mean componentwise absolute error, per-action and overall sign disagreement,
and false-safe sign error.

## Preconfirmation selection and failure rule

All three paired pilot seeds 0, 1, and 2 must be complete. Within each domain-family stratum, average
the seed-level primary safety endpoint arithmetically. Define reconstruction and maximum-horizon
rollout ratios as the learned arm's mean validation MSE divided by paired `none` mean validation MSE.
If a control denominator is zero, only a zero numerator is eligible and the tied ratio is one;
otherwise the ratio is infinite.

Apply the registered utility, coverage, no-empty-trajectory, and matched-neighborhood-mass gates.
Among eligible positive weights choose, lexicographically: lowest mean primary safety endpoint, lower
maximum of the two utility ratios, then smaller weight. If the profile-error gate or any selection
gate fails in any required cell or stratum, confirmation is cancelled or prospectively reframed
before final-test access. No post hoc teacher, weight, arm, domain, or history substitution is allowed.

## Required manifest fields

The implementation must record the protocol version; trajectory/fold/split checksums; data and
teacher seeds; inherited config checksum; checkpoint epoch and checksum; per-teacher label-shard and
manifest checksums;
bundle IDs; action grid; horizon; `margin_scale`; every per-bundle error; all eligibility decisions;
and the producing command. Downstream fitting uses only `predicted_action_profile_target`; the
privileged `action_safety_margins` training key is rejected. Checkpoint selection uses ordinary
validation world-model utility without predicted-profile targets. Only after checkpoint selection,
the reusable validation audit executes physical constant-action rollouts and writes validation audit
records and rollout summaries; calibration and final test remain unmaterialized. Confirmation
remains blocked until all production teacher cells are executed, all required 200-bundle profile
gates pass, and the separate latent eligible-center coverage and matched-neighborhood-mass gates are
computed successfully. No learned evidence is reported here.
