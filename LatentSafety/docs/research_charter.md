# Research charter

## Working title

**Do Safe Choices Survive Encoding? Auditing and Repairing Safe-Action Sufficiency in Pretrained Visual Representations**

## One-sentence thesis

A visual representation is safe-action sufficient for a declared specification, horizon, and
action/skill library only when every viable history represented together retains a common feasible
choice. Reconstruction, semantic invariance, and latent prediction need not preserve that
correspondence. Exact controls and traceable neighborhood audits diagnose the loss; pre-bottleneck
Common-Action Geometry Repair targets it while retaining each pretrained family's native utility.
Population and controller guarantees still require additional coverage and dynamics premises.

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

Under identical causal information and matched representation rate, pretrained reconstructive
(Cosmos), predictive (V-JEPA), and semantic (DINO) representations may retain different amounts of
safe-choice information. The common-action defect should predict held-fixed-controller violations
beyond native utility and scalar probes. A pre-bottleneck set-wise repair should shift the
within-family safety/utility frontier and survive removal of its training head.

## What would count as an ICML-level result

The minimum convincing package is:

1. a corrected theorem package covering both static and common-action ambiguity;
2. exact toy controls used only as estimator and observation-floor unit tests;
3. identical-information and matched-rate audits of Cosmos Tokenizer, V-JEPA 2/2.1, and DINOv3;
4. branchable manipulation and embodied-navigation domains with simulator-grounded profiles;
5. within-family repair comparisons against frozen-head, native-adapter, scalar-feasibility,
   pairwise, post-code, and safety-aware baselines;
6. matched-utility comparisons with fresh paired seeds, head removal, and a held-fixed downstream
   controller;
7. specification and distribution shift, plus complete reproducibility manifests.

## Novelty boundary

Do **not** claim:

- the first latent CBF, latent safety filter, or transfer theorem;
- the first observation that safe/unsafe states can collide under an encoder;
- that a nearest-neighbor label mismatch is an exact collision;
- that a probe with high `R^2` establishes uniform safety information;
- that static safety-faithfulness makes latent dynamics Markov or a controller safe; or
- that the proposed quotient dimension is minimal without an admissible-encoder lower bound.

Candidate differentiators are a traceable safe-action-sufficiency audit, a set-wise pre-bottleneck
repair evaluated within pretrained family against strong safety-aware and geometry-only controls,
and demonstrated downstream relevance after head removal. The exact static/data-processing and
finite Bellman lemmas are supporting foundations rather than headline novelty.

The novelty-safer primary thesis is the history/belief version under partial observability. The
memoryless static defect remains a motivating special case, not the sole headline contribution.

## Gated plan

| Gate | Question | Pass criterion | If it fails |
|---|---|---|---|
| G0 | Is accessible information matched? | Same causal span and modalities; DINO receives the registered temporal adapter; raw-history and privileged-state controls pass | Attribute the effect to sensing/history, not encoding |
| G1 | Is the profile oracle trustworthy? | Exhaustive simulator branches reproduce known controls and teacher error is reported with an abstention band | Do not train or audit on predicted profiles |
| G2 | Is the defect reproducible? | Robust effect in both primary domains and at least two pretrained families under matched rate and neighborhood mass | Reframe as a narrow case study or stop |
| G3 | Does the audit matter? | It predicts held-fixed-controller violations beyond native utility, action prediction, and scalar probes | Do not claim downstream relevance |
| G4 | Is the encoder repaired? | Pre-bottleneck repair beats frozen-head, native-adapter, pairwise, and post-code controls after head removal | Reframe as supervision/readout, not representation repair |
| G5 | Does it transfer? | Gains survive fresh controllers, appearances, hazard types, thresholds, horizons, and denser action libraries | State specification/domain dependence or stop |

## Deliberate exclusions for the first submission

- full long-horizon viability-kernel preservation in continuous state spaces;
- real-robot experiments before the simulator protocol is stable;
- claims for unknown or learned safety specifications without an explicit error composition;
- universal specification-agnostic safety under compression; and
- a benchmark zoo larger than the team can reproduce carefully.
