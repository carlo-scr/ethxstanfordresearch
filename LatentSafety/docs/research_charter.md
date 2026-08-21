# Research charter

## Working title

**Does Safety Survive Encoding? Safety-Sufficient Representations for Certification and Control**

## One-sentence thesis

A representation is adequate for safe control only if its fibers preserve both the safety
specification and the feasible safe-action correspondence; standard reconstruction and prediction
objectives need not control either obstruction. Exact finite oracles and carefully labeled sampled
witnesses can diagnose the loss and provide targeted representation-learning objectives, while
population or controller guarantees require additional coverage and dynamics premises.

## Why the proposal needs this refinement

The original proposal's static insight is useful but not sufficient as a headline. Lutkus et al.
already define the sound latent safe set as the complement of encoded unsafe states and explicitly
note that a shared encoding for safe and unsafe states makes its pullback a strict
under-approximation ([arXiv:2505.23210, Section V](https://arxiv.org/abs/2505.23210)). A quantitative
margin theorem and systematic empirical audit may still be new, but “safe/unsafe collisions cause
conservatism” cannot be claimed as the first observation.

There is also a trivial static solution when one scalar specification is known: encode
`E(x) = h(x)`. That preserves the label but may destroy the Markov property, prediction utility, and
which action is safe. A high-bar paper should characterize the information a representation needs
for *safe decisions*, not only safe-set classification.

## Formal object

Keep the information pipeline explicit:

```text
physical state/history x  -- observation O -->  observable y  -- encoder E -->  latent z
```

This supports two non-additive comparisons:

1. **Observation ambiguity floor.** If two physical states are safety-distinct but observationally
   identical, no downstream encoder can recover the missing information.
2. **Encoder-induced ambiguity.** Since `E o O` has fibers at least as coarse as `O`, deterministic
   post-processing cannot improve the exact defect. Compare both quantities rather than attributing
   sensor aliasing to the learned representation.

For partially observable tasks, `x` should be a history or belief state. A memoryless image encoder
and a recurrent/history encoder are separate experimental conditions.

### Axis A: safe-set separation

For region `K`, safety margin `h`, and representation `R`, define

```text
eps_sep(R) = sup { h(x) : h(x) >= 0 and some x' has h(x') < 0 and R(x') = R(x) }.
```

The operational version replaces equality with a pre-specified latent uncertainty set or radius.
Its radius must be fixed in physical/deployment terms or calibrated without test labels; raw latent
distances can be changed by arbitrary rescaling.

Because this number changes if `h` is rescaled, the default theoretical margin should be signed
distance to the unsafe set under a declared state metric. Domain-specific `h` values are allowed,
but every result must then be labeled `h`-relative and must not compare incompatible units.

### Axis B: safe-action consistency

Let `U_safe(x)` be the actions satisfying a chosen one-step, CBF, robust MPC, or viability
condition. For a latent fiber `F_z = {x : R(x)=z}`, define its robust feasible actions as

```text
U_safe^R(z) = intersection_{x in F_z} U_safe(x).
```

If every state in a fiber is individually viable but this intersection is empty, no deterministic
memoryless policy `pi(z)` can be safe for every state in the fiber. A quantitative action defect is
the minimum worst-state constraint violation of a common action. This is the primary upgrade over
static label preservation.

## Candidate theorem package

T1--T5 have internal proof drafts and executable finite checks but remain
`theorem_needs_proof` until independent review.

- **T1 - Static frontier.** The infimum completeness margin attainable by any sound latent safe-set
  certificate equals `eps_sep`. In general, impossibility holds for margins strictly below the
  supremum and achievability strictly above it; equality needs an attainment convention.
- **T2 - Data processing.** If `R_2 = T o R_1`, then exact safe-set defect cannot decrease. This
  yields an observation-level lower bound for every learned encoder.
- **T3 - Action-conflict impossibility.** A fiber with individually nonempty safe-action sets and
  empty intersection makes uniform safety impossible for every deterministic memoryless latent
  policy. Conversely, nonempty intersections plus regularity/measurable-selection assumptions give
  a safe latent selector for the audited condition.
- **T4 - Average-loss separation.** Expected reconstruction, rollout, or pairwise safety losses do
  not uniformly control either supremum defect without tail/coverage assumptions.
- **T5 - Positive control.** A boundary-separation modulus controls robust set defect; preservation
  of the safe-action correspondence controls action defect. A finite verified cover gives a
  dimension-dependent exact bound, while i.i.d. violation scores give a distribution-free
  probabilistic tolerance statement.
- **T7/T8 - Finite dynamic audit.** On a fully enumerated robust history game, arbitrary code-policy
  margin loss admits a path-dependent Bellman bound, while zero first-action obstruction is
  equivalent to global preservation of every full-history-viable history by one time-indexed code
  policy. Robust optimal-margin regret, sign obstruction, and retained viability are separate.
  These are candidate foundations with direct approximate-information-state and symbolic-control
  overlap, not a current novelty claim.

## Empirical thesis

At matched held-out world-model utility, ordinary learned representations exhibit materially
different static and action defects; these defects predict certificate coverage or closed-loop
violations better than reconstruction/rollout metrics; a safety-sufficient objective shifts the
safety/utility Pareto frontier rather than merely improving a probe score.

## What would count as an ICML-level result

The minimum convincing package is:

1. a corrected theorem package covering both static and action ambiguity;
2. an estimator validation suite where identity/injective controls converge to zero and constructed
   collapses are recovered;
3. two controlled pixel domains plus one harder domain with a distinct failure mechanism, and at
   least two representation families;
4. matched-utility comparisons with at least eight paired seeds for primary results;
5. an intervention that improves a prospectively frozen safety/utility frontier;
6. one downstream demonstration tying representation defect to certificate coverage or violations;
7. all pilot and main-study claims reproducible from manifests.

## Novelty boundary

Do **not** claim:

- the first latent CBF, latent safety filter, or transfer theorem;
- the first observation that safe/unsafe states can collide under an encoder;
- that a nearest-neighbor label mismatch is an exact collision;
- that a probe with high `R^2` establishes uniform safety information;
- that static safety-faithfulness makes latent dynamics Markov or a controller safe; or
- that the proposed quotient dimension is minimal without an admissible-encoder lower bound.

Candidate differentiators are a traceable action-sufficiency audit with a non-vacuous verified
upper route, a learned intervention evaluated at matched world-model utility against current
safety-aware baselines, and demonstrated downstream relevance. The exact static/data-processing and
finite Bellman lemmas are supporting foundations rather than headline novelty.

The novelty-safer primary thesis is the history/belief version under partial observability. The
memoryless static defect remains a motivating special case, not the sole headline contribution.

## Gated plan

| Gate | Question | Pass criterion | If it fails |
|---|---|---|---|
| G0 | Can the proposal pilot be reproduced? | Original artifact recovery or independent recreation matches qualitative ordering | Remove every pilot number and treat PDF as motivation only |
| G1 | Does the estimator measure its stated object? | Correct behavior under identity, collision, rescaling, class-balance, and sample-size controls | Redesign metric before training models |
| G2 | Is the phenomenon ordinary? | Robust effect on at least 2 tasks x 2 model families, paired CIs excluding negligible effect | Reframe as theorem/method paper or stop |
| G3 | Is it actionable? | Intervention Pareto-dominates or materially extends baseline frontier | Keep audit paper only if effect is very strong |
| G4 | Is the theorem package nontrivial and correct? | Independent proof review; no reliance on unverified regularity | Target control venue with narrower claims |
| G5 | Does it matter downstream? | Defect predicts coverage/violations beyond standard metrics | Do not claim control relevance |

## Deliberate exclusions for the first submission

- full long-horizon viability-kernel preservation in continuous state spaces;
- real-robot experiments before the simulator protocol is stable;
- claims for unknown or learned safety specifications without an explicit error composition;
- universal specification-agnostic safety under compression; and
- a benchmark zoo larger than the team can reproduce carefully.
