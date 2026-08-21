# Experiment protocol

## Research questions and prospectively frozen hypotheses

### RQ1 - Measurement validity

Can finite data distinguish representation-induced safety ambiguity from ordinary sampling near a
decision boundary?

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
- **H3b:** preserving safe-action sets improves action defect beyond appending/predicting `h` alone.
- **H3c:** the intervention reduces downstream certificate conservatism or violations at matched
  task performance.

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
| Car racing/navigation pixels | local or top-down image | collision/track-boundary margin | occlusion and moving obstacle split |

Exact environment versions, wrappers, safety margins, action bounds, and termination behavior must
be frozen before main runs. If CarRacing's confounds dominate, use a controlled Dubins/navigation
task as the third environment and keep CarRacing as stress test.

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
2. FCSRL feasibility-consistent, CVRL-BM safety-bisimulation, SRPL, and SDQC baselines;
3. auxiliary prediction of `h`;
4. all-pair faithfulness hinge;
5. boundary hard-negative contrastive loss;
6. safe-action-set preservation;
7. oracle `h` appended to the code;
8. specification-agnostic invertible/noncompressive control where feasible.

Sweep regularization weights and report the validation-selected Pareto frontier. The oracle append
tests whether static label preservation alone fixes the downstream problem; it is a critical control,
not the proposed method.

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
- Predefine one primary safety and one primary utility endpoint per experiment.
- Report seed-level points, mean/median, 95% paired bootstrap confidence intervals, and standardized
  paired effects. Use environment/trajectory block bootstrap for per-frame quantities.
- When paired method labels are exchangeable under a predeclared sharp null, report the exact
  paired sign-flip randomization p-value over seed aggregates. Pairing alone is not an
  exchangeability justification; otherwise omit the p-value and treat the bootstrap as interval
  estimation.
- Correct confirmatory family-wise comparisons (Holm) and label all others exploratory.
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
