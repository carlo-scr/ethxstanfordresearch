# Theory audit

This document records proof obligations and known flaws in the August 21, 2026 proposal. It is a
blocker list, not reviewer-facing prose. The original PDF remains unchanged for provenance.

## Severity A: invalid or materially overstated as written

### A1. Endpoint in the soundness-completeness theorem

The proposal defines unsafe states with `h(x') < 0`, so the witness set need not be closed and the
supremum need not be attained. From `m <= eps_sep` one cannot always choose a witness with
`h(x) >= m` when `m = eps_sep`.

**Safe statement:** no sound certificate is `m`-complete for `m < eps_sep`; the fiber-infimum
certificate is complete for `m > eps_sep`. Thus the *infimum* attainable completeness margin is
`eps_sep`. Handle equality only with an attainment assumption or a revised strict-completeness
definition.

### A2. Differential kernel does not imply a crossing fiber

`v in ker dE_x` alone does not produce another state with the same code. For example, `E(x)=x^3`
has zero derivative at the origin but is injective. The geometric proposition needs a local
constant-rank/submersion condition or an explicit differentiable curve contained in the encoder
fiber. Bounds must normalize the curve tangent and state all neighborhood assumptions.

### A3. The nearest-neighbor pilot is not an exact-defect lower bound

A safe sample whose nearest held-out code is unsafe generally has a *different* code. Even an
injective identity map produces cross-boundary nearest neighbors in finite samples. Therefore the
rank-based `k=1` statistic is not a witness for `eps_sep` and “latent-indistinguishable” is not
supported.

**Required repair:** report distances and evaluate a radius-indexed robust defect at radii fixed
before test evaluation. Alternatively, solve a verified/adversarial collision problem. Include
identity, known-collision, rescaling, sample-density, class-balance, and observation-oracle controls.

### A4. Minimal latent dimension is not established

The quotient construction proves that `q` dimensions suffice for that chosen quotient; it does not
prove minimality. For one known scalar margin, `E=h` is a one-dimensional faithful encoder under
the current static definition. A meaningful lower bound must restrict admissible encoders to also
preserve dynamics, utility, a specification family, or the safe-action correspondence.

### A5. Static faithfulness is not sufficient for safe control

States can share `h` yet require disjoint safe actions because their dynamics or relative degree
differ. Preserving `h` neither creates Markov latent dynamics nor makes a CBF-QP feasible. Separate
static set certification from dynamic/action sufficiency and add an action-fiber condition if the
paper claims safe control relevance.

### A6. The defect depends on the numerical parameterization of `h`

Replacing `h` by `c h` rescales `eps_sep` without changing the safe set. The quantity is therefore
`h`-relative, not a property of the safe set alone. Prefer signed distance to the unsafe set under a
declared state metric for cross-task theory, or retain physical `h` units and avoid
invariant-sounding claims.

### A7. The set-theoretic optimum may not belong to a certificate class

The fiber infimum can be discontinuous even when `E` and `h` are continuous. The frontier theorem
is exact over arbitrary latent subsets/functions, not necessarily differentiable CBFs, neural
networks, polynomials, or verifiable Lipschitz classes. It gives a universal lower bound for every
class, but not class-wise achievability without additional regularity.

### A8. A latent-distance margin does not imply the claimed state margin

Faithfulness gives `|h(x)-h(x')| <= kappa ||E(x)-E(x')||`. It implies that a positive *state*
margin forces some latent separation from unsafe codes; the reverse lower bound on `h` does not
follow from latent separation alone. If “latent margin” means the factored certificate value, it is
already `h` and no division by `kappa` appears. Define the margin precisely or remove the sentence.

## Severity B: repairable constants or conditions

### B1. Cover-bound factor

Approximating two arbitrary states by two cover centers contributes twice. If `E(x)=E(x')` and
centers `p,q` lie within `r`, then `||E(p)-E(q)|| <= 2 L_E r`. The exact-radius bound is

```text
2 L_h r + 2 kappa L_E r,
```

and the robust-radius version is `2 L_h r + kappa(delta + 2 L_E r)`, before adding any verified
cover-pair slack. The code reference implementation tests this constant.

### B2. Metric entropy claim

`N = O((diam(K)/r)^m)` requires a stated covering/doubling-dimension condition and constants. An
informal “intrinsic dimension of supp rho” is insufficient, especially when exact claims use a
region `K` larger than distributional support.

### B3. Safety-faithfulness is sufficient, not weakest

Preserving every pairwise difference in `h` is stronger than needed for one-sided cross-boundary
separation. Describe it as a convenient sufficient condition. Compare it empirically with a
boundary-focused/hard-negative objective.

### B4. Expected penalty is not verification

Equation (2) is an expected training loss. A small mean can hide rare, large violations—the same
failure mode used in the proposal's impossibility theorem. Keep empirical loss, uniform cover
verification, and probabilistic tolerance claims visibly separate.

### B5. Operational radius is not scale invariant

An encoder can multiply every code by a constant without changing its information. A fixed raw
`delta` then changes the reported robust defect. The protocol must anchor uncertainty in a deployed
noise/set propagation model, use a validation-only calibration, or report the whole radius curve
with explicit rescaling controls. Never choose the radius that looks best on test data.

The pairwise training loss has the same scale escape: multiplying every code by a large constant
can drive its hinge toward zero. Constrain/normalize latent scale or use an uncertainty-set
formulation invariant to the allowed reparameterizations.

### B6. Observation and encoder aliasing are conflated

For image input, the relevant representation is `E(O(x))`. A high average `R^2(h | image)` does not
prove zero worst-case observation defect. Establish an observation-level ceiling using the same
worst-case/robust metric, then quantify what additional ambiguity the encoder introduces.

### B7. One-step prediction separation needs a defined system

The smooth-tube construction sketches small expected prediction loss but does not specify the
dynamics, predictor, or how collided states with distinct successors are handled. State an explicit
system and construct the predictor, or limit the theorem to reconstruction and give prediction as a
separate corollary with its assumptions.

The proposed reflection construction also needs replacement: a reflection is injective, while a
fold such as an absolute value is nonsmooth and naive mollification can remove exact collisions. An
explicit smooth nonmonotone bump map can create equal-code pairs, but its dimensions, decoder, and
latent predictor must be written out.

### B8. The order-statistic wrapper assumes an exact confounding oracle

The distribution-free tolerance calculation is dimension-free only after observing the true score
`V(x)`. Deciding whether any unsafe preimage lies in a latent neighborhood is the global
verification problem. A finite unsafe reference set produces a lower surrogate, so applying a
one-sided upper tolerance statement to it is unsound. Either verify each score, add a certified
search/cover error, or state the theorem as conditional on oracle scores.

### B9. Conjugacy and safety-defect units cannot be compared directly

A safety margin and a latent dynamics residual live in different units. Any composition needs the
relevant Lipschitz or class-K conversion. Do not state `epsilon_sep <= conjugacy epsilon` merely
because both constants use the same Greek letter.

### B10. Related-work claims are too broad

Safety-aware representations, calibrated latent errors, uncertainty-augmented reachability, and
cost/reward decoupling already exist. The paper may argue that these objectives do not certify
fiber-wise separation, but it cannot say current representation objectives universally ignore
safety. The EEG “When Certificates Fail” preprint is not evidence of a latent-control failure.

### B11. The quotient embedding dimension is underspecified

A `q`-dimensional quotient/manifold need not admit a co-Lipschitz embedding into `R^q`. The safe
claim is only that any ambient dimension `d` admitting the stated embedding of `pi(K)` is
sufficient. It is neither an achieved `d=q` statement nor a lower bound without additional
topological and admissibility assumptions.

## Proof checklist

- [ ] Definitions distinguish exact, robust, empirical-witness, and probabilistic defects.
- [ ] All suprema/infima have explicit empty-set and endpoint conventions.
- [ ] Region `K`, support of `rho`, safe boundary, and unsafe strictness are consistent.
- [ ] Data-processing monotonicity under deterministic post-processing is proved.
- [ ] Static frontier theorem is independently proof-checked.
- [ ] Geometric result uses constant-rank/fiber-curve assumptions.
- [ ] Action-conflict theorem states policy class, action space, uncertainty, and selection
      regularity.
- [ ] Expected-loss counterexample specifies architecture dimensions and dynamics.
- [ ] Cover theorem has correct factors and covering-number assumptions.
- [ ] Tolerance theorem separates exact violation scores from approximate score computation.
- [ ] Learned-`h` error is composed into every claimed state-safety margin.
- [ ] No downstream control claim follows only from static `h` preservation.
