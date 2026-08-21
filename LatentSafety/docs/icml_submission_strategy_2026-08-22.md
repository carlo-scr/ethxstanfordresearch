# ICML submission strategy — 2026-08-22

## Technical summary

**Current evidence status.** The repository supports exact finite examples, internally drafted
theory, estimator controls, and CPU integration tests. It supports **no learned-representation
result yet**: neither the recommended 96-task two-domain decision pilot, the optional 288-task
expansion, any confirmatory run, nor downstream control study has been run on GPU. Proposal-PDF
numbers are unreproduced and must not enter the submission.

**Sharpest target thesis.** The strongest defensible paper is not “safe and unsafe states can
collide in a latent space.” It is:

> A learned history representation can preserve ordinary world-model utility while losing the
> distinctions needed to choose a robustly safe action. We audit common-action obstruction and
> retained viable coverage, quantify robust optimal-margin regret separately, bracket the relevant
> loss with traceable lower witnesses and a non-vacuous verified upper bound, and train
> representations to improve the safety/utility frontier. Across controlled and harder visual-control
> domains, the resulting models improve downstream safety/certificate coverage at matched utility
> beyond scalar-margin supervision and the strongest safety-aware baseline.

This is a **claim to earn**, not a description of current results. If the upper certificate remains
vacuous, replace “bracket” with “audit” and do not use certification language. If downstream and
third-domain results are absent, the package is unlikely to clear the ICML main-track bar.

**Area-chair judgment (inference).** The exact-fiber and dynamic Bellman lemmas are useful
foundations, but not an ICML headline. The credible novelty window is the combination of a
computable representation-level action-sufficiency audit, a learning intervention that beats strong
controls at matched utility, and downstream predictive/control relevance. The official ICML 2026
call asks for original, rigorous, reproducible work of significant ML interest and an eight-page
main paper; use that only as a provisional planning constraint until the 2027 call appears
([ICML 2026 call](https://icml.cc/Conferences/2026/CallForPapers)).

Throughout this document, linked statements about prior work are **source facts**. Labels such as
“AC judgment,” “recommended,” and “go/no-go” are **inferences or prospective design choices**, not
reported findings.

## The novelty boundary is narrow

| Primary-source fact | Collision and required non-claim |
|---|---|
| Lutkus et al. define the sound latent safe set as the complement of encoded unsafe states and explicitly note that safe/unsafe aliases make its pullback conservative ([paper, Sec. V](https://arxiv.org/html/2505.23210#S5)). | Do not claim the first safe/unsafe fiber obstruction, sound latent-set construction, or latent-to-state safety transfer. |
| Feedback refinement relations already formalize controller refinement through static quantizers under nondeterminism ([Reissig et al.](https://arxiv.org/abs/1503.03715)); belief-support shields already restrict POMDP actions to enforce almost-sure reach-avoid safety ([Sheng et al.](https://arxiv.org/abs/2309.10216)). | Do not claim the first common-action condition, robust support action set, or partial-observation shield. |
| Approximate information states already give learnable history compressions, approximate dynamic programs, and policy-loss bounds ([Subramanian et al., JMLR 2022](https://www.jmlr.org/papers/v23/20-1165.html)). Worst-case AIS work treats partially observed uncertain systems and bounded performance loss ([Dave et al.](https://arxiv.org/abs/2301.05089)). | Do not headline a generic history-compression theorem, Hausdorff successor condition, Bellman error propagation, or factor-two policy-loss bound. The current dynamic theorem is a safety-specialized foundation. |
| FCSRL learns feasibility-consistent safety representations on vector and image tasks ([ICML 2024](https://proceedings.mlr.press/v235/cen24b.html)); CVRL-BM learns safety-bisimulation visual representations ([publisher DOI](https://doi.org/10.1109/TIP.2024.3523798)); explicit learned safety features have already improved safer policy learning ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html)). | Do not claim that prior representation objectives ignore safety or that adding an auxiliary safety loss is novel. These are mandatory comparisons or must be excluded with a precise incompatibility argument. |
| SDQC uses cost/reward Q-supervision to preserve safe offline decisions ([ICML 2025](https://proceedings.mlr.press/v267/yang25aq.html)); PIGDreamer uses privileged representation alignment for safe partial-observation world models ([ICML 2025](https://proceedings.mlr.press/v267/huang25ai.html)). | Do not claim first action-value supervision, cost-aware decoupling, or privileged safety alignment. State whether action-profile labels are privileged and test a learned-profile variant. |
| Latent Safety Filters already perform HJ reachability in visual world-model latents ([RSS 2025](https://www.roboticsproceedings.org/rss21/p113.html)); LatentCBF studies smooth margins and mixed data for latent CBF filtering ([L4DC 2026](https://proceedings.mlr.press/v331/nakamura26a.html)). | Do not claim the first latent safety filter, latent reachability method, latent CBF, or downstream safety controller. Hold the controller fixed and isolate representation quality. |
| A very recent partial-observation CBVF paper combines a minimum-over-time margin, conformal estimator error, finite-horizon probabilistic safety, and a QP filter ([Jahanshahi and Chen, 2026](https://arxiv.org/abs/2608.13819)). | Do not conflate support-robust safety with chance guarantees or claim the first partial-observation margin certificate. Explain the semantic difference and compare where feasible. |

Additional non-claims:

- a finite nearest-neighbor witness is not an exact collision or a population upper certificate;
- calibration by median latent distance removes global rescaling but does not make uncertainty
  balls physically equivalent across nonlinear encoders;
- static margin preservation does not imply Markovity, dynamic action sufficiency, or closed-loop
  safety;
- high probe accuracy or mean pairwise loss does not establish a uniform tail property;
- one known safety specification does not establish specification-agnostic compression;
- an empirical dimension elbow is not a minimal sufficient dimension; and
- no causal claim follows from cross-model correlation between an audit and violations.

## Ranked contribution ladder

| Rank | Package | ICML assessment (inference) | Evidence required |
|---:|---|---|---|
| 1 | **Certified dynamic safety-sufficient representations** | Strongest and most differentiated | An exact common-action/retained-viability target plus a practical lower/upper bracket on dynamic policy loss, with every cover/Lipschitz/search premise checked; a non-vacuous certificate; a training method; three visual domains; strong baselines; downstream gains at matched utility. |
| 2 | **Action-sufficiency learning and downstream validation** | Plausible ICML paper | Traceable finite-sample audit, action-profile or viability-targeted objective, two controlled plus one hard domain, two or more representation families, scalar-margin and safety-aware baselines, fresh eight-seed confirmation, and held-fixed downstream controller. No formal-certification language. |
| 3 | **A rigorous audit/benchmark result** | Borderline but possible if the phenomenon is large | Many frozen encoders, calibrated physical/neighbor-mass sensitivity checks, observation and state controls, and strong out-of-domain prediction of downstream failures beyond rollout/probe metrics. A small two-domain study is insufficient. |
| 4 | **Dynamic theorem plus controlled examples** | Better fit for CDC/L4DC unless strengthened | Independent proof review and a result not subsumed by AIS/Q-abstraction theory; continuous/stochastic or genuinely computable finite-data extension; non-toy control demonstration. |
| 5 | **Static fibers, counterexamples, or a new auxiliary loss alone** | Do not submit to ICML main track | These are supporting material or a workshop note, not the central contribution. |

**Recommended commitment rule.** Aim for Rank 1, budget around Rank 2, and stop the ICML push if
the project cannot reach Rank 2 by the confirmatory freeze.

## Minimal decisive two-domain pilot

The pilot should answer only three questions: (1) is there encoder-induced action ambiguity after
separating observation ambiguity, (2) does history remove the planted partial-observation component,
and (3) does action-profile supervision beat scalar safety supervision without degrading the world
model? Do not spend the full 288-task grid to answer these.

### Frozen 96-train core

| Axis | Values | Reason |
|---|---|---|
| Domain | controlled cart; controlled inverted pendulum | Distinct dynamics and renderers with velocity hidden in one frame. |
| Family | AE; beta-VAE | Two existing, end-to-end families. Compare methods only within a family. |
| History | `stack_h1`; `stack_h4` | Same fixed-width architecture; isolates added temporal information. Defer GRU architecture effects. |
| Latent dimension | 8 only | Avoid dimension-driven multiplicity in the decision pilot. |
| Training arm | none; `h_prediction`; `boundary_contrastive`; `safe_action_profile` | Separates ordinary, scalar/static, boundary/static, and action-aware supervision. |
| Seeds | 0, 1, 2 | Exploratory paired seeds, never confirmatory evidence. |

Total: `2 × 2 × 2 × 1 × 4 × 3 = 96` trains. These cells are a subset of the checked-in grids.

### Metrics and definitions to freeze before launch

1. **Primary safety endpoint:** per-trajectory 95th-percentile required violation for the
   predeclared finite constant-action profile, normalized by the domain's physical `margin_scale`,
   then aggregated by seed. This trajectory-balanced statistic is implemented. The current
   `action_conflict_fraction` is a useful co-primary descriptive measure; the maximum
   `worst_required_violation` is a traceable lower witness but is too sample-size-sensitive to be
   the only inferential endpoint.
2. **Primary utility endpoint:** normalized reconstruction plus rollout prediction, reported as
   separate components. The current `world_model_utility` includes a family-dependent KL term, so
   do not match AE and beta-VAE globally on that raw composite. Matching is within
   domain/family/history, with the composite retained for checkpoint selection only. The
   fail-closed aggregator now retains current-frame reconstruction and maximum-predeclared-horizon
   rollout pixel MSE separately; domain-normalized utility and its non-inferiority margin must still
   be frozen before confirmation.
3. **Radius semantics:** retain the calibration-only `0.05 × median pairwise distance` result as
   exploratory, but require the conclusion to survive the full frozen radius curve, a
   matched-neighborhood-mass analysis, and physically generated perturbation/alias pairs. A
   relative neural radius alone cannot support “robust” or “certified.”
4. **Dynamic semantics:** the existing profiles are worst margins under constant discrete actions
   over horizon 4 (cart) or 6 (pendulum). Call this exactly what it is. Do not identify it with the
   full robust Bellman `Q` or dynamic regret. A separate registered three-step exact oracle now
   bridges both controlled domains to `metrics/dynamic.py`, but it is a theorem fixture and does not
   change the learned-model endpoint's semantics.
5. **Denominators:** report conflict count over individually viable nontrivial neighborhoods,
   number of trajectories, number of states, and coverage of the boundary/challenge stratum.

### Mandatory pilot controls

- visible-observation oracle and privileged-state oracle, with their own declared calibration
  geometry; at finite radius these are controls, not automatically additive lower bounds;
- latent plus true scalar margin, and latent plus true action profile, as privileged no-retraining
  reference views;
- exact planted same-frame aliases with disjoint safe-action sets in both domains;
- h1/h4 parameter-count equality and shuffled-history negative control;
- latent rescaling with radius rescaling, duplicate removal, trajectory splits, and sample-size
  curves;
- shuffled margin/profile labels to detect regularization or capacity effects unrelated to safety;
- predicted action profiles versus privileged true profiles, because the implemented training arm
  currently uses known-dynamics counterfactual labels; and
- all completed, failed, and divergent runs retained in the manifest.

### Prospective pilot go/no-go thresholds

These are recommended defaults to ratify **before opening validation results**, not observed facts.

- **Integrity:** every E0 control passes; all 96 task cells are accounted for; no final-test value is
  used; model size is matched; no effect disappears after trajectory-level aggregation.
- **Partial observability:** `stack_h4` reduces normalized action loss versus `stack_h1` for the
  ordinary arm in both domains, with the expected planted-alias controls recovered. Otherwise the
  current domains do not establish the history thesis.
- **Action-specific value:** `safe_action_profile` beats `h_prediction` on the primary action metric
  in both domains and in at least three of four domain-by-family strata, with no stratum showing a
  practically important reversal. It should also outperform `boundary_contrastive` on action loss;
  the latter may win on static defect.
- **Minimum effect:** use both an absolute reduction of at least `0.10` margin-scale units and a
  relative reduction of at least 20% in conflict fraction, unless the team replaces these values
  with domain-expert MCIDs before training.
- **Utility non-inferiority:** no more than 5% relative degradation in both reconstruction and
  rollout error within domain/family/history. Replace 5% only before launch and document why.
- **Robustness:** the direction holds at three adjacent radii, under matched neighborhood mass, and
  on the boundary/challenge stratum; it is not driven by one seed or one witness.

Passing the pilot authorizes a confirmatory design. It does not authorize a paper result.

## Confirmatory matrix

Freeze the matrix after the pilot and before opening any new test split.

| Workstream | Frozen comparison | Replication | Primary decision |
|---|---|---:|---|
| Core E2 | Three domains: cart, pendulum, and one harder occluded/moving-hazard navigation domain; two representation families; one pilot-selected history mode; ordinary, scalar-margin, proposed action-profile/viability, and one strongest task-compatible safety-aware baseline | 8 **fresh paired** seeds | Proposed method is safety-superior and utility-noninferior to the strongest baseline in every domain; equal-weight aggregate is primary. This is 192 trains before extra ablations. |
| History mechanism | h1 versus h4, plus GRU-h4 as an architecture-only contrast, on the two controlled domains for ordinary and proposed methods | 8 paired seeds | Added information, not parameter count or recurrence alone, explains the action-sufficiency change. |
| Geometry sensitivity | frozen radius curve, matched neighbor mass, physical perturbation pairs, and exact aliases | same checkpoints | Headline direction survives all three operationalizations; certificate language is used only for a verified upper bound. |
| Shift | appearance/nuisance shift and dynamics/challenge shift fixed per domain | same checkpoints | Effect is not confined to the training renderer or behavior-policy density. |
| E5 downstream | one fixed certificate/filter/controller applied to frozen encoders; at least 200 episode rollouts per seed and condition | 8 paired seeds | Lower action regret yields fewer episode violations or more certificate coverage at matched return/control effort. |
| Certificate track | enumerated or verified-cover controlled domains, with lower witness and upper bound | independent proof/code reviewer | Bound is non-vacuous, contains the exact oracle on held-out finite cases, and certifies a nontrivial subset of the full-state viable region. |

The third domain must have a failure mechanism not reducible to one visible scalar. Prefer a
controlled Dubins/navigation environment with occlusion and moving obstacles if CarRacing creates
wrapper and reward confounds. Keep the benchmark version, safety specification, action set, and
termination rules frozen.

### Baselines and ablations that reviewers will ask for

| Question | Required comparison |
|---|---|
| Is scalar safety enough? | `h_prediction`, true `h` appended, and a calibrated safety probe. |
| Is the benefit merely boundary separation? | `boundary_contrastive` and an all-pair/random-pair loss. |
| Is action supervision the real ingredient? | true action-profile append oracle, predicted-profile training, shuffled profiles, horizon/action-grid sensitivity, and full robust-Q oracle where enumerable. |
| Is this already a safety-aware representation method? | At least one faithful FCSRL/CVRL-BM-style objective on the same backbone/data; add official end-to-end baselines if downstream RL is claimed. Explain any adaptation. |
| Is privileged information doing all the work? | privileged true profiles versus learned/model-estimated profiles and PIGDreamer-style privileged alignment where applicable. |
| Is this sensor ambiguity? | same-frame observation oracle, h4 observation/history oracle, and full-state oracle. |
| Is it capacity or optimization? | matched parameter counts, training curves, gradient norms, safety-weight sweep, equal compute, and failed-run ledger. |
| Is the representation simply noncompressive? | latent dimension sweep and an injective/noncompressive control where feasible. |
| Does one specification overfit? | held-out threshold of the same quantity and one distinct safety specification; otherwise state that the method is specification-specific. |

## Statistical analysis contract

- **Inferential unit:** paired training seed. Aggregate frames within trajectories and trajectories
  within each seed first. Never use frames as independent replicates.
- **Primary contrast:** proposed method minus the strongest validation-selected baseline, paired on
  data seed, initialization, architecture, domain, and history. Domain/family strata receive equal
  weight; do not let the largest trajectory set dominate.
- **Estimation:** report every seed, paired mean and median difference, 95% paired block-bootstrap
  interval, and Cohen's `d_z`. Keep the current 10,000-resample deterministic implementation.
- **Testing:** if method labels are exchangeable under the sharp null, use the exact paired
  sign-flip/randomization distribution over all `2^8` seed signs. Otherwise treat the bootstrap as
  interval estimation and do not manufacture a p-value. Apply Holm to the predeclared confirmatory
  family.
- **Joint success:** require safety superiority **and** utility non-inferiority using a frozen
  utility margin. This is an intersection-union decision; passing only one component is failure.
- **Tail metrics:** bootstrap trajectories inside seed when estimating a tail quantile, then use
  seed-level estimates for method inference. Report maxima as witnessed lower bounds, not stable
  population estimates.
- **Downstream association:** compare a base model using reconstruction, rollout, and safety-probe
  metrics with a model that adds static/action audit metrics. Use leave-one-domain-out prediction;
  advancement requires at least 10% lower held-out prediction error with a paired interval excluding
  zero. This is predictive evidence, not causality.
- **Failures:** rerun only documented infrastructure failures. Optimization divergence is a method
  outcome. Require eight valid paired seeds for the confirmatory contrast and include a worst-case
  sensitivity analysis for missing outcomes.
- **Blinding:** model/frontier selection uses validation only. Hash the selection rule and included
  run ledger; then unblind final test once. The current runner computes test audits during training,
  so implement a procedural or code-level two-stage blind before confirmation.

## Compute staging and stop rules

The checked-in Slurm request is one GPU for at most four hours per task; all GPU-hour values below
are **request ceilings**, not measured runtimes.

1. **Instrumentation gate:** validate the implemented centered-neighborhood violation score and
   trajectory-balanced tail, verify the now-separated reconstruction and maximum-predeclared-
   horizon rollout utility fields, complete the failure ledger and blinded confirmatory path, and
   run one short GPU integration per domain. Stop if artifacts or controls fail. Do not relabel the
   finite constant-action profile as Bellman regret.
2. **Eight-cell shakedown:** seed 0, ordinary arm, both domains × families × h1/h4. Verify runtime,
   memory, loss scales, data coverage, and audit denominators.
3. **One-seed method screen:** complete all 32 seed-0 pilot cells. Diagnose optimization only; make
   no scientific selection from a single seed.
4. **Decisive pilot:** run the remaining 64 cells for 96 total. Ceiling: 384 requested GPU-hours,
   or roughly 48 hours at concurrency eight if every task uses the full allocation.
5. **Expansion:** do not launch the existing 288-task full grids unless the 96-cell gate passes.
   GRU and dimension are mechanism/sensitivity expansions, not automatic requirements.
6. **Confirmation:** the 192-train core has a nominal ceiling of 768 requested GPU-hours under the
   current allocation, before external baselines. Benchmark actual time from the pilot and approve
   a written budget before launch.
7. **Downstream/verification:** evaluate only frozen selected checkpoints. No additional encoder
   tuning after test unblinding.

Immediate stop/reframe conditions are: a safety effect vanishes under trajectory aggregation;
history does not resolve planted aliases; the observation/state controls explain the entire effect;
scalar-margin or a prior baseline dominates the proposed arm; gains require more than the utility
margin; results depend on one family/domain/seed; the physical-radius sensitivity reverses the
conclusion; or the downstream metric adds no information beyond rollout and a safety probe.

## Fallback and pivot paths

| Failure mode | Honest pivot |
|---|---|
| Finite upper certificate is vacuous, but learning/downstream results are strong | Submit Rank 2 as an empirical method; describe all audits as witnesses/diagnostics. |
| Proposed method is weak, but the audit robustly predicts failures across many models/domains | Build a broader audit/benchmark paper with a public model zoo and third-party baselines; ICML remains possible but risky. |
| `h_prediction` or append-`h` solves both static and action outcomes | Drop the action-sufficiency novelty claim. Test multiple specifications or stop; do not rename a dominated loss as the contribution. |
| History removes nearly all ambiguity and encoder excess is negligible | Pivot from latent compression to sensing/memory sufficiency, or target a partial-observation control venue. |
| Dynamic theorem remains a specialization of AIS/Q-abstraction theory | Keep it as a lemma/metric derivation. Pursue a stochastic-channel or verified finite-data extension, or omit it from the main paper. |
| Certificate is non-vacuous but empirical gains are modest | Recast for CDC/L4DC around verifiable abstraction and controlled case studies. |
| Effect survives only the two synthetic domains | Add the hard third domain before any main-track submission; otherwise target a workshop or control venue. |
| Action-sufficiency metrics do not improve downstream prediction or control | Remove control-relevance language. If the remaining audit is not independently valuable, stop. |
| No robust effect after controls | Publish nothing as a positive ICML result; retain the repository as a negative-result/internal research artifact. |

## Submission narrative and final gates

The eight-page story should be: **decision-relevant failure** → **obstruction/viability audit and regret bracket** →
**targeted learning method** → **matched-utility evidence** → **held-fixed downstream consequence**.
Static fiber lemmas and counterexamples belong in one compact setup result and the appendix.

Before calling the project an ICML candidate, require all of the following:

1. independent proof and novelty review, including a line-by-line comparison with AIS and
   Q/state-action abstraction results;
2. a non-vacuous certified upper bound, or removal of every certification/guarantee claim tied to
   finite neural data;
3. the pilot thresholds above passed without test access;
4. confirmation on two controlled and one hard visual domain with eight fresh paired seeds;
5. action-aware training beating scalar-margin and a strong safety-aware baseline at matched
   reconstruction and rollout utility;
6. a downstream result under the same frozen controller, plus state/observation/oracle ablations;
7. complete manifests, failed-run ledger, immutable analysis code, and one independent rerun; and
8. every headline statement promoted in `paper/claims.toml` only with its producing command and
   artifact path.

### Questions that can still kill the thesis

- Can the team define a physically meaningful uncertainty set shared across encoders, rather than
  only a latent-distance convention?
- Can the finite-cover/Lipschitz bound be made non-vacuous on more than an enumerated toy domain?
- Can action-relevant supervision be obtained without privileged simulator counterfactuals?
- Does the audit predict failures on a genuinely harder domain after controlling for rollout error
  and a calibrated safety probe?
- Is the resulting method meaningfully different from FCSRL, safety bisimulation, Q-supervision,
  and privileged alignment when implemented on the same backbone and data?

Until these are answered, the honest project status is “well-instrumented high-risk research
program,” not “ICML paper in progress.”
