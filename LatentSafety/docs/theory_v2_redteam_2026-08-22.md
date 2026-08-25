# Dynamic theory v2 red-team — 2026-08-22

## Verdict

**Core finite theorem:** mathematically sound under the stated finite, fully enumerated,
fixed-action assumptions. I found no counterexample to the Bellman viability equivalence, the
fiber-regret characterization, coarsening monotonicity, the local/global composition bound, the
closed endpoint, the midpoint identity, or either tightness construction.

**Submission status:** not ready to headline an ICML paper. The strongest issue is not a false
inequality; it is that `rho` is an optimal-margin regret, not an exact safety-preservation defect.
Its minimax selector can choose an unsafe action even when a common safe action exists. The generic
Bellman/Hausdorff/factor-two material is also already substantially contained in approximate
information-state and Q-abstraction theory. The finite-cover routine is useful arithmetic but is
not yet a data-derived certificate.

This is an adversarial internal check, not external peer review and not a substitute for review by
a control theorist.

Files checked without modifying them:

- `docs/theory_v2_dynamic.md`
- `paper/sections/theory_v2_dynamic.tex`
- `src/latent_safety/metrics/dynamic.py`
- `tests/test_dynamic_theory.py`
- `docs/literature_map.md`
- `paper/references.bib`

## Severity summary

| ID | Severity | Finding | Consequence |
|---|---|---|---|
| F1 | **High** | `rho` and its minimax selector optimize margin regret, not exact safety preservation | A safe representation-level action may exist while the constructed policy is unsafe |
| F2 | **High** | The generic theory has a direct novelty collision with worst-case AIS and Q-abstraction results | Do not claim the Bellman, Hausdorff, or factor-two results as standalone novelty |
| F3 | **Medium** | The paper proof of the particular q-greedy policy bound invokes a theorem for a different policy | The statement is true, but the written proof does not establish it |
| F4 | **Medium** | The finite-cover API hard-codes Euclidean latent distance while the theorem permits an arbitrary declared metric | The implementation can return an invalid bound if the caller interprets it under another metric |
| F5 | **Medium** | The result audits a representation-measurable history policy, not a recursively updateable latent world model | “Information state” or “world-model sufficiency” language would overstate what is proved |
| F6 | **Medium** | The cover routine verifies only finite pair-search arithmetic, not the cover, Lipschitz, or uniform-Q premises | It is not yet the nontrivial finite-data certificate needed for novelty |
| F7 | **Low** | The coarsening helper requires postprocessing on the terminal layer although no stage-0 regret exists | It can report a false negative for the theorem it purports to check |
| F8 | **Low** | Test booleans do not check every displayed inequality or the most safety-relevant counterexample | Passing tests provide weaker assurance than their names suggest |
| F9 | **Low** | The main audit accepts Boolean/string margins and Boolean tolerance despite “finite real” semantics | Fail-closed input handling is inconsistent with the cover routine |

## Detailed findings and required fixes

### F1 — `rho` is not an exact safety-sufficiency defect

The note correctly says that zero regret is stronger than sign-only safety, but it does not expose
the more consequential fact: minimizing value regret can select an unsafe action even when a
common safe action is available.

Consider one fiber containing two viable histories and the one-step robust Q table

\[
\begin{array}{c|cc}
 & a_0 & a_1\\\hline
\eta_1 & 0 & 2\\
\eta_2 & 0 & -1.
\end{array}
\]

Then `V(eta_1)=2` and `V(eta_2)=0`. Action `a_0` is safe at both histories. Its worst regret is
two. Action `a_1` has worst regret one, so it is the unique minimax-regret selector, but it is
unsafe at `eta_2`. The implementation reproduces exactly this behavior: `rho=1` and
`J(eta_2)=-1`.

This does not contradict Theorem 1, because `V(eta_2)=0 < B(eta_2)=1`; the theorem simply declines
to certify the history. It does show that the central selector is not an exact safety-preserving
controller and that reducing `rho` need not maximize the retained viable set.

**Required fixes:**

1. Rename `rho` consistently as **robust optimal-margin regret**, not a safety defect or exact
   safety-sufficiency quantity.
2. Add the counterexample above to the theory note and tests.
3. For a safety headline, introduce a sign-specific recursive object. At minimum, distinguish the
   first-action obstruction
   \[
   \kappa_s(z)=\min_a\max_{\eta\in F_{s,z}:V_s(\eta)\ge0}[-Q_s(\eta,a)]_+
   \]
   from value regret (on fibers with no viable histories, set this quantity to zero by convention).
   For an exact representation-policy result, define latent viability
   recursively under the future representation restrictions rather than assuming full-history
   control resumes.
4. Empirically report retained viable volume/false-safe rate in addition to `rho`; otherwise a
   lower regret can conceal worse safety coverage.

### F2 — novelty is blocked by prior theory

The current novelty warning is directionally correct and should be made even firmer.

- [Dave, Venkatesh, and Malikopoulos (TAC 2024)](https://arxiv.org/html/2301.05089v2)
  formulate finite-horizon partially observed non-stochastic control with a maximum instantaneous
  cost, give a memory Bellman recursion, define approximate information states through current-cost
  and Hausdorff successor errors, prove Q/value approximation, and obtain a factor-two policy loss.
  With a constant shift and sign change, maximum-over-time cost is the same objective as
  minimum-over-time margin. The candidate's model-local result is therefore a finite
  safety-specialized instance, not merely thematically adjacent.
- [Subramanian et al. (JMLR 2022)](https://www.jmlr.org/papers/v23/20-1165.html) already treat
  learned history compression, approximate latent dynamic programs, and policy-loss propagation.
  Their expected-return semantics differ, but the history-representation program is established.
- [Abel et al. (ICML 2016)](https://proceedings.mlr.press/v48/abel16.html) explicitly aggregate
  states whose optimal Q profiles are uniformly close and derive near-optimal policy bounds. The
  candidate midpoint and greedy `2 delta` argument is the finite max/min analogue of a standard
  Q-abstraction argument.
- [Abel et al. (AISTATS 2020)](https://proceedings.mlr.press/v108/abel20a.html) give necessary and
  sufficient value-preservation conditions for state/action abstractions and options.
- [Reissig, Weber, and Rungger (TAC 2017)](https://arxiv.org/abs/1503.03715) characterize robust
  controller refinement through static quantizers for set-valued dynamics. Common admissible
  action structure is not new.
- [Sheng, Parker, and Feng (ICRA 2024)](https://arxiv.org/html/2309.10216) use winning regions over
  POMDP belief supports to restrict actions for almost-sure reach-avoid safety. Support-wise action
  safety under partial observation is not new.
- [Jahanshahi and Chen (2026)](https://arxiv.org/html/2608.13819v1) use the same
  minimum-over-time safety margin, partial-observation estimator uncertainty, a finite-horizon
  guarantee, and online filtering, albeit with continuous-time probabilistic semantics.
- The map should also include the older primary literature on controller synthesis for safety via
  approximate bisimulation, for example
  [Girard (Automatica 2012)](https://doi.org/10.1016/j.automatica.2012.02.037), before claiming that
  the remaining exact-regret/path-buffer combination is new.

The exact fiber minimax regret is a clean diagnostic definition, and the branch-local buffer is a
useful audit presentation. On the evidence reviewed, neither is strong enough by itself to support
a theory-contribution claim. Both follow from elementary finite minimax and Bellman-error
arguments.

**Required fixes:** keep the candidate section out of `main.tex`; position it as an analytical
instrument. A defensible submission needs at least one of: (i) a genuine, validated population
upper certificate with non-vacuous scaling; (ii) an exact sign-specific compression/viability
theorem not subsumed by symbolic control/AIS; or (iii) a learned method whose matched-utility
frontier is convincingly better and whose theory directly controls the measured quantity.

### F3 — the q-greedy policy corollary has a proof-routing gap

Theorem 1 is stated for the minimax selector
`a_s^R in argmin_a r_s(z,a)`. Theorem 2 additionally claims that the policy greedy with respect to
an arbitrary approximate `q_s` has loss at most `2 sum_j delta_j`. The displayed one-step argument
correctly shows that this q-greedy action has regret at most `2 delta_s`, but the paper then says to
“apply Theorem 1 stagewise.” Theorem 1 does not apply to that different selector.

The claim is still true: repeat the same induction with

\[
\bar r_s^q(z)=\max_{\eta\in F_{s,z}}
  [V_s(\eta)-Q_s(\eta,a_s^q(z))]\le 2\delta_s,
\]

or state Theorem 1 first for an arbitrary code policy and then specialize it to the minimax and
q-greedy choices.

**Required fix:** generalize the composition lemma or include the missing induction. Add a test that
uses a q-greedy action different from the minimax selector and checks the composed q-policy bound;
current tests only validate the minimax policy.

### F4 — generic metric in the theorem, Euclidean metric in code

Proposition 4 is written for an arbitrary latent metric `d_Z`. The implementation accepts only
vectors and silently computes Euclidean distance. This is not a harmless presentation choice. If a
caller establishes the assumptions in, for example, the infinity norm but the routine searches in
the Euclidean norm, sample pairs satisfying the theorem's radius can be excluded, and the returned
maximum need not upper-bound the required oscillation.

For a concrete failure, take the complete two-point domain (so `kappa=0`), latent vectors
`(0,0)` and `(0.8,0.8)`, exact scalar Q values `0` and `10`, and operational radius one under
`d_Z=||.||_infinity`. The population pair is within radius, so its required oscillation is ten.
The implementation measures Euclidean distance `sqrt(1.28)>1`, excludes the pair, and returns zero.
All remaining declared constants can be valid; the mismatch is solely the unrecorded metric.

**Required fix:** either:

- restrict the theorem-facing routine and documentation explicitly to Euclidean `d_Z`, require
  `L_R` in that same norm, and record the metric in `CoverQRegretCertificate`; or
- accept a validated distance callback/precomputed distance matrix and test at least Euclidean,
  infinity-norm, and non-Euclidean cases.

Also remove the undocumented `+1e-12` radius expansion or expose it in the returned operational
radius. It is conservative for the current maximum, but it means the code is not computing exactly
the mathematical quantity it reports.

### F5 — this is not yet a latent world-model theorem

The policy is allowed to evaluate `R_s(eta)` from the full history at every time. Nothing proves
that the code is recursively updateable from `(z,a,o')`, that a latent successor correspondence is
well-defined, or that planning can occur wholly in latent space. The model-local condition compares
successor sets in a history metric and assumes the full-history oracle `V` is Lipschitz. This is a
valid theorem about representation-measurable history policies, but it is weaker than an
information-state or latent-world-model sufficiency result.

**Required fix:** use “history encoder”/“representation-measurable policy” language. If world-model
sufficiency is claimed, add a recursive update/prediction premise and a latent Bellman operator,
then bound its error relative to the history game. Dave et al.'s AIS conditions are the minimum
comparison point for such an extension.

### F6 — the finite-cover routine is conditional arithmetic, not finite-data certification

The inequality in Proposition 4 is correct if all four premises hold. However, the routine receives
only latent vectors, Q tables, and four declared constants. It neither receives histories nor checks
that the sample is a `kappa`-net; it does not verify `R`- or Q-Lipschitzness; and it cannot establish
the uniform Q-estimation error. A user can enter arbitrarily optimistic constants and receive a
small object named `CoverQRegretCertificate`.

This limitation is acknowledged in prose, so it is not a false theorem. It is nevertheless a
submission-level blocker: the current object is a **conditional bound calculator**, not the
finite-data certificate identified as the novelty route.

**Required fixes:** rename the public result accordingly; serialize explicit premise provenance;
add independent routines or proofs for cover radius, metric domain, Lipschitz constants, and
uniform-Q error; fail closed if any premise lacks provenance. Report vacuity rates and scaling with
dimension. Do not infer a population claim from random train/validation/test histories.

### F7 — coarsening checker has a terminal-layer false negative

`check_dynamic_data_processing` tests whether the coarse code is a deterministic postprocessing of
the fine code on every layer, including `s=0`. The theorem concerns `rho_s` only for `s>=1`. If the
postprocessing relation holds on every controlled layer but fails only on the terminal layer, the
returned stage regrets still satisfy the theorem while `is_deterministic_postprocessing=False` and
`monotone=False`.

**Required fix:** return a per-stage postprocessing flag aligned with each reported stage regret, or
exclude layer zero. Add a terminal-only mismatch regression test.

### F8 — test and audit booleans are incomplete

The random-game test is useful fuzzing and all 100 cases pass, but it recomputes all quantities
through the same implementation. It is not an independent oracle. Specific missing checks are:

- the safety/objective counterexample in F1;
- actual execution of a non-minimax q-greedy policy across two or more stages;
- direct numerical tests of the model-local current-margin/Hausdorff bound;
- an arbitrary-metric cover test or an explicit Euclidean restriction;
- exhaustive enumeration of every representation policy in tiny games;
- a negative check for `0 <= V-J`.

The implementation's `theorem_bound_holds` only flags losses above `B` or the global sum. It does
not flag `V-J < -tolerance`, even though nonnegativity is part of the displayed theorem.
`max_actual_loss` starts at zero and would hide such a negative loss.

**Required fix:** check every displayed inequality explicitly and add an independently implemented
brute-force policy enumerator for tiny games. Keep the random tests as additional fuzzing.

### F9 — inconsistent finite-real validation

The cover helper now rejects Booleans and permissive string coercions, but the main finite-game
audit still uses `float(margins[key])` and `math.isfinite(tolerance)`. It accepts `True` and `"1.0"`
as margins and accepts `True` as a tolerance. This conflicts with the declared finite-real API and
the fail-closed posture.

**Required fix:** route margins and tolerance through the same strict real-number validator used by
the cover helper; add Boolean/string regression tests.

## Theorem-by-theorem proof check

### Proposition 1 — finite-horizon viability: **pass**

At `s=0`, `V_0=G_0`, and `G_0>=0` is equivalent to all compatible physical margins being
nonnegative. At `s>=1`, finiteness attains the action maximum; `Q_s>=0` is equivalent to current
safety and nonnegative `V_{s-1}` at every enumerated successor. A global tie-broken Bellman
selector supplies one deterministic policy across the finite history graph. The closed endpoint is
valid even when the physical-state infimum defining `G` is not attained.

The phrase “every compatible branch” is valid only because `N_s` is assumed complete. The code
checks nonempty declared successors, not physical completeness; that remains a model premise.

### Proposition 2 — zero-regret characterization: **pass**

For every history and action, `V_s-Q_s>=0`. A maximum of finitely many nonnegative terms is zero
exactly when every term is zero, which is exactly membership of the common argmax intersection.
The separately stated sign-only first-action intersection is also correct, but it assumes
full-history optimal control resumes after the action and is not yet an exact recursive latent
policy characterization.

### Proposition 3 — coarsening monotonicity: **pass**

Each fine fiber is contained in a coarse fiber. Pointwise action regrets can only increase when the
maximization set grows; taking the action minimum and fiber maximum preserves the inequality. The
observation-floor corollary is correct only on the same history domain, as the note explicitly
states.

### Theorem 1 — local and global representation loss: **pass**

For a selected action `a`, the induction uses

\[
\min_{\eta'}(V_{s-1}(\eta')-B_{s-1}(\eta'))
\ge \min_{\eta'}V_{s-1}(\eta')-\max_{\eta'}B_{s-1}(\eta')
\]

and `min(g,b-M) >= min(g,b)-M` for `M>=0`. Combining this with the local regret definition gives
the claimed `B_s`; replacing each local term by the stage maximum gives the global sum. The safety
implication at equality is valid because safety is closed. The two-stage fixture realizes loss
`2=rho_1+rho_2` and `J=0` at `V=B=2`.

### Midpoint identity and Theorem 2: **statement pass; paper proof needs F3**

For each finite fiber-action pair, the best constant in uniform norm is the midpoint of its minimum
and maximum, with error half the range. A greedy action for any uniform approximation incurs at
most twice its uniform Q error. The matrix
`[[1,-1],[-1,1]]` gives `rho=2` and `delta*=1`, so the factor is attained. The multi-stage q-greedy
claim is true by a fresh induction, but not by direct invocation of Theorem 1 as currently written.

### Model-local Lipschitz/Hausdorff condition: **pass under stated premises**

For an `L`-Lipschitz function, minima over two nonempty finite sets differ by at most `L` times
their symmetric Hausdorff distance. The scalar minimum is one-Lipschitz in the infinity norm, so
the Q oscillation is bounded by `max(alpha,L beta)`. The midpoint result then gives
`rho <= max(alpha,L beta)` without an extra factor two.

### Proposition 4 — finite cover: **proof pass; operational status limited by F4/F6**

Nearest cover points are within `kappa`; representation Lipschitzness places their codes within
`delta+2 L_R kappa`; two history approximation errors and two Q-estimation errors yield the stated
`2 L_Q kappa+2e_Q`. At `delta=0`, this bounds within-fiber action-wise Q oscillation and hence
`rho`. The quantifiers are correct. The result is only as valid as its global cover and uniform
constants.

### Selector and randomization limits: **pass**

The `(0,1)` action example correctly shows why an infimum-zero regret need not yield an attained
selector. Under strong/pathwise evaluation, a mixture is evaluated by its worst supported action,
so a deterministic singleton support is optimal; expected or chance semantics require another
theorem.

## Checks executed

- All 23 focused tests in `tests/test_dynamic_theory.py` passed after the latest remediation.
- Exhaustive enumeration of all 729 two-action Q tables with three histories and entries in
  `{-1,0,1}` found no violation of `rho <=` maximum within-fiber Q oscillation.
- The supplied one-step factor-two and two-step horizon-sum fixtures were recomputed by hand and by
  the finite checker; both attain the stated bounds.
- The F1 table is now permanent regression coverage: it verifies `rho=1`, the unsafe margin-minimax
  action, `kappa=0`, the distinct common-safe action, and the retained-viability difference between
  the two selectors.
- A two-stage regression executes a q-greedy policy that differs from the minimax selector and
  checks the arbitrary-code-policy and cumulative factor-two bounds.
- Exact finite representation-policy viability is compared with an independently written exhaustive
  policy enumeration, including a case where root first-action obstruction is zero but recursive
  representation-policy viability fails.
- The F7 terminal-only mismatch is now a regression for the corrected semantics: terminal
  postprocessing is reported separately, while every controlled-stage flag and the theorem-level
  monotonicity result remain true.
- Further regressions cover explicit Euclidean metric semantics and radius tolerance, premise
  provenance, strict finite-real validation, the direct Lipschitz/Hausdorff calculation, and every
  displayed inequality `0 <= V-J <= B <=` the stagewise global sum.
- A separate complete-finite-domain verifier now derives the exact cover radius, representation/Q
  Lipschitz constants, and sampled-Q error from an enumerated metric domain, then independently
  checks the Q-coordinate oscillation bound on every enumerated pair within the operational latent
  radius. The folded-interval regression attains its bound.
- Both dynamic and cover calculators now reject finite inputs whose derived regret, distance,
  radius, or Q-bound arithmetic overflows to a non-finite value.
- The complete-finite verifier exposes raw diagnostics but advances each positive enumerated
  Q-coordinate difference one float toward positive infinity before correcting the advertised
  bound. Regressions cover both ordinary cover-arithmetic roundoff and subtraction that rounds
  below the exact-real difference of the stored binary inputs; a large-scale regression confirms
  that the mandatory one-ulp enclosure is not mistaken for failure against an absolute tolerance.
- Primary-source statements above were checked against the linked paper pages/full text on
  2026-08-22.

## Integration gate

The in-repository implementation/semantic findings are remediated: F1 now separates robust
optimal-margin regret from sign obstruction and recursive viability; F3 uses an arbitrary-code-policy
composition theorem; F4 makes the calculator explicitly Euclidean and records its tolerance; F5
uses history-encoder/representation-policy scope; F7 is controlled-stage aligned; and F8/F9 have
focused inequality, counterexample, exhaustive-policy, and strict-input regressions. F6 now has a
verified route for a genuinely complete finite domain, while the original sampled-population path
is resolved only at the naming and fail-closed provenance level: it remains explicitly a
conditional bound calculator, not a verified population certificate.

Keep `paper/sections/theory_v2_dynamic.tex` excluded from `main.tex`. The remaining gates are an
independent proof audit; the unresolved F2 novelty comparison with AIS, Q-abstraction, symbolic
safety synthesis, and shielding; and a genuinely validated population certificate or strong
matched-utility learned-method result. The current conditional calculator must not be renamed or
presented as a finite-data certificate without independent verification of its cover, Lipschitz,
and uniform-error premises.
