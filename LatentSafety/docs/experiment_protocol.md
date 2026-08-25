# Experiment protocol

## Research questions, executed controls, and prospective hypotheses

### RQ1 - Measurement validity

Can finite data distinguish representation-induced safety ambiguity from ordinary sampling near a
decision boundary?

H1a–c below are executed internal E0 controls, not prospective or promoted manuscript results. The
claim ledger keeps their finite notebook evidence in `scope = "excluded"`.

- **H1a:** the exact sample witness is zero for an injective control and positive at the known
  margin for an explicitly collapsed encoder.
- **H1b:** a fixed-radius robust-defect curve is invariant to joint rescaling of codes and radius.
- **H1c:** the unthresholded nearest-unsafe-neighbor statistic changes with sample size and class
  balance even when the encoder is injective; it is retained only as a diagnostic.

### RQ2 - Prevalence

At matched validation utility, do ordinary world-model encoders lose safety-relevant information?

- **H2a:** reconstruction and rollout error explain little variation in prospectively frozen static and
  action defects after controlling for task, model family, latent dimension, and seed.
- **H2b:** history reduces the observation/partial-observability component on tasks where velocity
  or occluded hazards matter. The frozen E1 contrasts are `stack_h1` versus `stack_h4` for history
  availability and `stack_h4` versus `gru_h4` for architecture at matched information. The stack
  path uses a fixed-width `(last, last-first)` summary, so h1 and h4 have the same parameter count;
  the recorded model metadata must verify that invariant before interpreting the contrast.
- **H2c:** high safety-probe `R^2` does not imply a small boundary-tail defect.

### RQ3 - Intervention

Can safety-sufficient training improve the frontier without destroying representation utility?

- **H3a:** boundary-aware separation improves static defect more efficiently than a random all-pair
  penalty.
- **H3b:** at matched reconstruction and rollout utility, the selected nonprivileged predicted-profile
  arm improves the action-safety frontier beyond `none`, `h_prediction`, append-true-`h`, and the
  frozen FCSRL feasibility-loss adaptation.
- **H3c:** with the controller or certificate held fixed, the selected intervention reduces downstream
  conservatism or violations at matched task performance, and the defect metrics add predictive value
  beyond reconstruction, rollout, and probe metrics.

## Milestone M0 - Recover or recreate the proposal pilot

The PDF reports numerical results but this workspace contains no source, data, or checkpoints. Do
not treat those results as reproduced. Before rerunning the 96-model grid, recover or explicitly
reconstruct:

- image renderer, state distribution, episode length, and sample counts;
- safety radius and exact handling of boundary states;
- architecture, normalization, optimizer, schedule, early stopping, and checkpoint selection;
- train/validation/test trajectory IDs;
- nearest-neighbor preprocessing and whether self-neighbors were excluded;
- every seed-level metric and the plotting source.

If exact recovery is impossible, rename the study “independent recreation” and remove numerical
comparisons to the PDF.

## E0 - Estimator validation (must pass first)

### Positive controls

- encoder that maps mirrored safe/unsafe states to identical codes;
- tunable near-collision with known robust defect as a function of radius;
- fiber with individually viable states but disjoint finite safe-action sets;
- sensor aliasing before encoding, to test the observation floor.

### Negative/invariance controls

- ground-truth state / identity representation;
- injective orthogonal or invertible transform;
- code translation, rotation, and rescaling with correspondingly transformed uncertainty sets;
- appending irrelevant dimensions;
- raw observation safety oracle;
- independent test sample sizes and safe/unsafe class ratios;
- duplicate removal and trajectory-level split enforcement.

### Estimator outputs

For each pre-specified radius or uncertainty set, retain:

- maximum witnessed safe margin (lower bound on a population supremum only);
- fraction and distribution of safe samples with an unsafe witness;
- witness pair IDs, distances, margins, trajectories, and images;
- 50th/90th/95th/99th percentiles as descriptive estimates;
- sample-size convergence and uncertainty intervals;
- observation-level and encoded-level curves;
- safe-action common-intersection slack; and
- traceable per-center required violations plus the mean across trajectories of each
  trajectory's 95th-percentile value.

The nearest-unsafe 1-NN statistic may be plotted for continuity with the proposal but must be labeled
as a sampling-dependent diagnostic.

## E1 - World-model audit

### Tasks

Start small enough to finish, but ensure distinct failure mechanisms.

| Task | Observation | Safety mechanism | Partial-observability test |
|---|---|---|---|
| Pendulum pixels | rendered frames | angle/velocity envelope | single frame versus 4-frame history |
| Controlled-cart pixels | rendered position frames | rail-position envelope | velocity hidden in one frame |
| Controlled Dubins-navigation pixels | frozen top-down image | arena/collision clearance | moving-obstacle direction hidden in one frame |

Exact environment versions, wrappers, safety margins, action bounds, and termination behavior must
be frozen before main runs. The confirmatory third environment is controlled Dubins navigation;
CarRacing is a stress test only and cannot replace a confirmatory domain after validation begins.

### Representation families

- deterministic convolutional autoencoder with stack or GRU history frontend;
- beta-VAE with the same history controls;
- recurrent state-space world model (planned expansion);
- action-conditioned joint-embedding predictive model (planned expansion);
- ground-truth-state and observation-oracle controls.

Do not add model families until the first two are end-to-end reproducible. The comparison unit is a
frozen encoder checkpoint selected by validation utility without test safety labels.

### Splits

- split by environment seed and complete trajectory;
- dedicate separate train, validation, radius-calibration, and final test trajectories;
- create boundary-stratified test strata without changing the deployment-distribution primary
  metric;
- include at least one dynamics or appearance shift as a secondary robustness condition;
- checksum split manifests.

## E2 - Safety/utility frontier

Compare, with matched architectures and paired seeds:

1. ordinary representation objective;
2. the frozen same-backbone FCSRL feasibility-loss adaptation in
   `docs/fcsrl_baseline_adaptation.md`;
3. auxiliary prediction of `h`;
4. all-pair faithfulness hinge;
5. boundary hard-negative contrastive loss;
6. safe-action-set preservation;
7. oracle `h` appended to the code;
8. specification-agnostic invertible/noncompressive control where feasible.

Sweep regularization weights and report the validation-selected Pareto frontier. The oracle append
tests whether static label preservation alone fixes the downstream problem; it is a critical control,
not the proposed method.

The confirmatory proposed arm uses nonprivileged predicted profiles, not simulator counterfactuals.
Five trajectory folds are formed inside the training split. For each fold, a teacher with the selected
action-conditioned transition and scalar-margin head is trained only on the other folds' pixel histories,
behavior actions, and observed safety-margin labels, then rolls every registered grid action for the
frozen domain horizon to label its held-out fold. `docs/predicted_profile_protocol.md` freezes fold
assignment, teacher seeds/architecture/checkpoints, deterministic rollout, and 200 disjoint coverage
bundles per domain and seed. Each bundle branches under every registered constant action. Eligibility
uses the nearest-rank p95 of the bundle-wise maximum absolute action error divided by the physical
margin scale and requires at most 0.10 in every domain-family-data-seed cell, including pilot seeds
0–2 and fresh confirmatory seeds 100–107; also report mean and sign error. A failing fresh teacher is
a registered method failure and cannot be retrained or reselected. Controlled-domain oracle profiles
are diagnostic-only and never enter fitting or selection.
Known dynamics and privileged state remain oracle-only.
If the 0.10 gate fails in any required cell or domain-by-family validation stratum, the predicted-profile arm
is ineligible and confirmation is cancelled or prospectively reframed before final-test access. Do
not replace it with a post hoc arm.

The three learned core arms use the same pre-confirmation weight grid
{0.001, 0.01, 0.1, 1, 10}. A weight is eligible only when reconstruction and rollout error are each
within 5% of the paired no-supervision arm on validation data. In every validation stratum it must
also retain at least 80% eligible-center coverage, lose no more than five percentage points of
coverage relative to paired no-supervision, and leave no trajectory without an eligible center. On a
validation-only radius grid `{0, 0.01, 0.02, 0.05, 0.10}`, radii are relative to each arm's median
pairwise latent distance over at most 1024 validation samples selected deterministically by the
SHA-256 of sample ID; this scale uses no safety values. The paired no-supervision reference is fixed
prospectively at relative radius 0.05. For `N` paired validation samples, first restrict the
reference population to samples that are individually viable on the frozen finite action grid. A
viable center is eligible when its closed ball, formed only over that viable reference population,
contains at least one viable nonself sample. Its normalized mass is the number of viable nonself
neighbors divided by `N-1`; the arm-level mass is the median over eligible centers. Coverage is the
eligible fraction conditional on the viable population, while viable fraction is reported
separately. Select the learned radius that minimizes relative mass mismatch, breaking ties toward
the smaller learned relative radius, without consulting safety.
The match must be within 5%; undefined or zero support fails closed. The primary safety endpoint is
the arithmetic mean across trajectories of the within-trajectory linearly interpolated p95
eligible-center required common-action violation. It must improve strictly at the matched mass in
each of the three paired pilot seeds. `scripts/reduce_preconfirmation_radius.py` accepts only paired
validation record files, recomputes both arm curves, pins input and implementation hashes, refuses
overwrite, rejects calibration/final-test inputs, and marks the standalone reduction diagnostic and
not evidence eligible. The production aggregator reopens the source plan, task manifests, and paired
record files. Every learned selector row must
carry the control relative radius 0.05, selected learned relative radius, both absolute radii, both
arm-audit hashes, physical-pairing hash, relative mass mismatch, and strict-sign result. The final
selected-weight artifact must retain that selected candidate's complete radius provenance rather than
only its reduced mass and safety values. Freeze the matched radius before
final-test access. Among eligible weights select the lowest primary
validation safety endpoint, then the lower maximum of the reconstruction and rollout ratios to paired
no-supervision, then smaller weight. A zero no-supervision denominator admits only a zero learned-arm
error and defines that tied ratio as one. If no weight passes in a required stratum, that arm fails the
stratum; confirmation proceeds only if every core arm passes every required stratum, otherwise it
is cancelled or prospectively reframed without substitution.
`scripts/select_preconfirmation_weights.py` accepts only the complete 288-row validation factorial
(18 shared controls plus 270 learned-arm/weight rows), requires one checksum-identical assembled
profile-label manifest across all five profile weights in each cell, and emits all 18 weight freezes
or a fail-closed readiness artifact. Its only file-backed input is the self-hashed
`e2_preconfirmation_validation_observations_v1` artifact from
`scripts/aggregate_e2_preconfirmation.py`: exactly 288 observations paired one-for-one with 288
self-hashed radius sidecars derived from checksum-linked `e2_preconfirmation_task_handoff_v1`
manifests. Each selected weight preserves all three seed-level radius and audit-hash freezes.
These tuning and oracle-control jobs are outside the 192 fresh-seed confirmatory core. The primary
confirmatory aggregate weights domain-by-family strata equally, and the joint
safety-superiority/utility-noninferiority criterion must pass in every domain.

The confirmatory family is fixed before final-test access. Within seed and domain, average AE and
beta-VAE differences equally. Compare the predicted-profile arm with `none`, `h_prediction`, the
FCSRL adaptation, and the append-true-`h` view of the paired `none` checkpoint, across three domains
and three endpoints (normalized p95 safety, reconstruction, maximum-horizon rollout): 36 elementary
bounds. Use 100,000 paired-seed bootstrap resamples with seed 20260822 and the nonstudentized
percentile upper quantile at `1 - 0.05/36`. Safety requires both mean profile-minus-comparator difference at most
-0.10 and upper bound below zero; each relative utility degradation requires upper bound at most
0.05. A zero comparator error admits only zero profile-arm error and defines tied relative
degradation as zero. All 36
bounds and all registered runs must pass. Exact sign-flip tests, if exchangeability is
justified, are secondary, cover only the 12 safety contrasts, and use Holm alpha 0.05.

CVRL-BM changes the representation architecture and introduces a safety-bisimulation metric; it is
therefore an architecture-changing sensitivity, not evidence from the core same-backbone comparison.
Full FCSRL, SRPL, and SDQC also couple representation learning to online actor/critic or constrained-RL
updates. Include those end-to-end baselines only in E5 when the downstream RL setting is matched;
do not describe the offline feasibility-loss arm as a reproduction of any full agent.

## E3 - Representation routes

Compare specification-aware and specification-agnostic routes at matched parameter count and
utility. Evaluate transfer to:

- the training specification;
- a held-out threshold/radius of the same physical quantity;
- a distinct safety specification;
- learned/noisy specification values with explicit error bars.

This experiment determines whether the paper can claim more than one fixed known `h`.

## E4 - Dimension and information frontier

Sweep latent dimension only after capacity and optimization are controlled. Report static defect,
action defect, utility, and verification cost. Do not call an observed elbow a “minimal safety
dimension” without a theorem; call it an empirical frontier.

## E5 - Downstream control

This is a gate, not the starting point. Freeze encoders from E2 and apply the same latent certificate
or controller construction to all arms. Primary outcomes:

- sound-certificate coverage/completeness;
- false-certification and false-rejection rates under the declared uncertainty model;
- intervention frequency and feasibility;
- closed-loop constraint violations and minimum physical margin;
- task success/return and control effort.

Static defect should predict coverage; action defect should predict feasibility or violations. Test
incremental predictive value beyond reconstruction, rollout, and probe metrics.

## Statistical protocol

- Use at least eight paired training seeds for primary confirmatory results; three seeds are
  exploratory. Reduce this only with a pre-registered precision/power argument.
- Predefine the safety and utility endpoints and multiplicity family before final-test access.
- For the frozen E2 confirmation, require the complete 240-row analysis input and compute exactly 36
  one-sided Bonferroni upper bounds with 100,000 paired-seed resamples and seed 20260822. Within each
  seed/domain, average the AE and beta-VAE differences equally. All registered runs and all 36 bounds
  must pass; exploratory 95% intervals and standardized effects are descriptive only.
- When paired method labels are exchangeable under a predeclared sharp null, report the exact
  paired sign-flip randomization p-value over seed aggregates. Pairing alone is not an
  exchangeability justification; otherwise omit the p-value and treat the bootstrap as interval
  estimation.
- Holm correction applies only to the optional 12 safety sign-flip tests. It is not the primary
  correction for the 36-bound confirmatory family.
- Fit cross-model association analyses with task/model-family effects; do not pool thousands of
  correlated frames as independent evidence.
- Publish all completed seeds, failures, exclusions, and timeouts. Never select the best seed.
- Freeze analysis code before opening final test results where practical.

## Compute tiers

| Tier | Purpose | Approximate scope | Gate |
|---|---|---|---|
| CPU smoke | metric correctness | deterministic synthetic fixtures | every commit |
| Pilot GPU | decisive method/history gate | 2 tasks x 2 families x 2 histories x 1 dimension x 4 arms x 3 seeds = 96 | E0 pass |
| Main GPU | confirmatory frontier | selected arms/history on 3 domains x 8 fresh paired seeds | pilot effect and budget review |
| Stretch | CarRacing, shifts, downstream | only selected checkpoints | E2 pass |

Record wall time, peak memory, accelerator model, energy estimate if available, and failed runs.

## Kill criteria

Stop or materially reframe if any occurs:

- the identity/injective control retains a nonvanishing “defect” as sample size grows under the
  proposed calibration;
- proposal results cannot be qualitatively recreated after protocol recovery;
- effects vanish under trajectory splits or observation-level controls;
- defect adds no predictive information beyond a calibrated safety probe;
- the proposed intervention is dominated by simply appending/predicting `h`;
- action defect adds no value over static defect on any partially observable task;
- theorem novelty collapses into the latent-safe-set definition of prior work; or
- the main conclusion depends on one task, architecture, or seed.
