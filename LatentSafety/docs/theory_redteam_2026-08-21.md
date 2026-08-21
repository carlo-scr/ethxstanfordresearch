# Theory red-team record — 2026-08-21

An independent adversarial pass reviewed every result in `docs/theory_v1.md`. This record preserves
the material findings and prevents the corrected overstatements from reappearing.

## Corrections made

1. **Deterministic decision quantifier.** The original draft incorrectly said a particular
   `pi(z)` is uniformly safe iff the common-action set is nonempty. The corrected equivalence is
   `pi(z)` safe iff `pi(z)` belongs to that set; only the existential statement is equivalent to
   nonemptiness.
2. **Randomization semantics.** The finite-action minimax quantity averages loss over the sampled
   action before taking the worst compatible state. It is now called pointwise expected-loss slack.
   Under strong/pathwise semantics, expectation is taken over the action-wise worst fiber loss and
   the optimum equals deterministic slack, so randomization gives no benefit.
3. **Measurability.** The randomized theorem now declares a measurable action space and Borel
   probability in the compact-metric case.
4. **Continuous-action caveat.** Compactness or coercivity alone is no longer presented as enough
   for positive slack; lower semicontinuity and uniform loss calibration are substantive.
5. **Notation.** The quadratic projection caveat, observation-fiber domain, extended-real slacks,
   and LP variable domains are now explicit.

## Results that survived the audit

- exact sound-set existence and all endpoint cases;
- equality of infimum completeness thresholds and exact defect;
- deterministic data processing and the observation floor;
- deterministic action-conflict impossibility after the quantifier repair;
- randomized fixed-fiber impossibility under strong, countable, or compact-closed conditions;
- closed-convex pointwise projection;
- finite-action pointwise expected-loss sandwich, strong-slack equality, and finite-fiber LP dual.

## Novelty warning

The exact-fiber results are mathematically useful but elementary relative to control abstraction,
state--action abstraction, action-sufficient representations, and belief-support shielding. In
particular, feedback refinement relations already study controller refinement through quantized
state and admissible inputs; POMDP shields already restrict actions over belief supports. These
lemmas should support—not headline—the submission.

The ICML-level target remains one of: a nontrivial approximate/dynamic safety-sufficiency theorem;
a finite-data certificate with a computable search guarantee; or a strong matched-utility learning
result that survives observation-level controls and predicts downstream safety.
