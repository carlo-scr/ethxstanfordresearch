# Theory v1: exact set and action frontiers

This note is a proof-grade foundation for the first paper draft. It deliberately separates
set-theoretic facts from claims about learnable, continuous, measurable, or verifiable certificate
classes. All suprema and infima below are taken in the extended reals, with
`sup(empty union {0}) = 0` and `inf(empty) = +infinity`.

## Novelty and scope

The maximal sound latent set

\[
\mathcal S_{\max}=\mathcal Z\setminus R(\{x:h(x)<0\})
\]

and the qualitative loss caused by a safe/unsafe encoding collision are **not** new claims of this
project. Lutkus, Wang, Lindemann, and Tu use precisely the complement-of-encoded-unsafe-set
construction and discuss the resulting strict under-approximation in Section V of
[Latent Representations for Control Design with Provable Stability and Safety
Guarantees](https://arxiv.org/abs/2505.23210). We use that construction as a baseline lemma. The
candidate project differentiators to assess against the literature are the endpoint-correct
quantitative threshold, the observation/encoder decomposition, action-fiber impossibility
(including the randomized-policy qualifications), and a finite-action quantitative slack.

These results concern a declared state, history, or belief domain `K` and a declared one-step or
barrier-style safe-action correspondence. They do not by themselves establish long-horizon
viability, latent Markovianity, or neural-network verifiability.

## 1. Setup and strict conventions

Let

\[
h:\mathcal K\to\mathbb R,\qquad
\mathcal C:=\{x\in\mathcal K:h(x)\ge 0\},\qquad
\mathcal D:=\{x\in\mathcal K:h(x)<0\},
\]

and let `R : K -> Z` be a deterministic representation. Thus the zero level set belongs to the
safe set, while unsafety is strict. For `z` in `R(K)`, its fiber is

\[
F_z^R:=\{x\in\mathcal K:R(x)=z\}.
\]

A latent certificate is an arbitrary subset `S` of `Z`. It is **sound** when

\[
R^{-1}(\mathcal S)\subseteq\mathcal C.
\]

For a finite margin `m >= 0`, distinguish two completeness conventions:

- **closed `m`-completeness:** `{x : h(x) >= m} subset R^{-1}(S)`;
- **open `m`-completeness:** `{x : h(x) > m} subset R^{-1}(S)`.

The distinction matters at the safe boundary and whenever a supremum is attained. Define the set
of conflicting safe margins

\[
\Gamma_R:=\left\{h(x):x\in\mathcal C,\ \exists x'\in\mathcal D
\text{ with }R(x')=R(x)\right\}
\]

and its exact, `h`-relative fiber defect

\[
\varepsilon_{\rm sep}(R):=\sup(\Gamma_R\cup\{0\})\in[0,+\infty].
\]

The union with `{0}` gives defect zero when there is no conflict and keeps boundary-only conflicts
at defect zero. It does **not** mean that a boundary-only collision is harmless under closed
zero-completeness.

Any unrestricted latent function `b : Z -> R` induces the set `S_b={z:b(z)>=0}`. Conversely, any
latent subset is induced by a two-valued function. The results below therefore apply equally to
unrestricted set-valued certificates and unrestricted certificate functions, but not automatically
to continuous, smooth, polynomial, Lipschitz, or neural certificate classes.

## 2. Exact existence equivalence and endpoints

### Theorem 1 (largest sound set and exact completeness test)

Define

\[
\mathcal S_{\max}:=\mathcal Z\setminus R(\mathcal D).
\]

Then:

1. `S` is sound if and only if `S subset S_max`. Hence `S_max` is the unique largest sound latent
   set.
2. A sound, closed `m`-complete latent set exists if and only if

   \[
   \Gamma_R\cap[m,+\infty)=\varnothing.
   \]

3. A sound, open `m`-complete latent set exists if and only if

   \[
   \Gamma_R\cap(m,+\infty)=\varnothing.
   \]

Whenever either condition holds, `S_max` itself attains the requested completeness.

#### Proof

If a sound set `S` contained some `z in R(D)`, there would be an unsafe `x'` with `R(x')=z`, and
then `x' in R^{-1}(S)`, contradicting soundness. Thus every sound `S` is a subset of `S_max`.
Conversely, if `R(x) in S_max`, then `R(x) notin R(D)`, so `x` cannot be in `D`; hence `x in C`.
Therefore `S_max` and every one of its subsets are sound. This proves part 1.

Because every sound set is contained in `S_max`, a sound set can be closed `m`-complete exactly
when `S_max` is. The latter fails exactly when there is an `x` with `h(x)>=m` whose code lies in
`R(D)`. Such an `x` is precisely a witness with `h(x) in Gamma_R intersect [m,infinity)`. This proves
part 2. Replacing `h(x)>=m` by `h(x)>m` gives part 3. QED.

### Corollary 1 (strictly below, strictly above, and at the endpoint)

Let `epsilon = epsilon_sep(R)`.

- For every finite `m > epsilon`, a sound closed `m`-complete set exists.
- For every `0 <= m < epsilon`, no sound open or closed `m`-complete set exists.
- If `epsilon < infinity`, a sound open `epsilon`-complete set always exists.
- If `epsilon < infinity`, a sound closed `epsilon`-complete set exists if and only if
  `epsilon notin Gamma_R`.
- If `epsilon = infinity`, no finite margin admits open or closed completeness.

#### Proof

No element of `Gamma_R` can exceed its finite supremum, so the relevant intersection in Theorem 1
is empty for `m>epsilon`, and also for the open endpoint `(epsilon,infinity)`. If `m<epsilon`, the
definition of supremum supplies a conflicting margin strictly larger than `m`. At the closed
endpoint, every conflicting margin is at most `epsilon`, so the intersection with
`[epsilon,infinity)` is empty exactly when `epsilon` itself is not a conflicting margin. If the
supremum is infinite, `Gamma_R` is unbounded above, so every finite threshold is crossed. QED.

### Corollary 2 (infimum completeness threshold)

Define the closed and open attainable thresholds

\[
\begin{aligned}
\tau_{\ge}(R)
&:=\inf\{m\ge0:\exists\text{ a sound, closed }m\text{-complete latent set}\},\\
\tau_{>}(R)
&:=\inf\{m\ge0:\exists\text{ a sound, open }m\text{-complete latent set}\}.
\end{aligned}
\]

With `inf(empty)=+infinity`, both thresholds satisfy

\[
\tau_{\ge}(R)=\tau_{>}(R)=\varepsilon_{\rm sep}(R).
\]

Equality here is an **infimum** statement. In the closed convention, the infimum need not be
attained.

#### Proof

If the defect is finite, every margin strictly above it is attainable and no nonnegative margin
strictly below it is attainable, by Corollary 1. Both infima therefore equal the defect, regardless
of endpoint attainment. If the defect is infinite, both attainable sets are empty. QED.

### Endpoint examples

These examples are useful unit tests for both prose and code.

1. **Boundary-only conflict.** Let `K={s,u}`, `h(s)=0`, `h(u)=-1`, and `R(s)=R(u)`. Then
   `Gamma_R={0}` and `epsilon_sep=0`. Open zero-completeness is attainable (it asks to retain no
   positive-margin state), but closed zero-completeness is impossible because it asks to retain
   `s`.
2. **Finite supremum not attained.** Let safe states `s_n`, unsafe states `u_n` for `n>=2`, and a
   safe state `t` satisfy `h(s_n)=1-1/n`, `h(u_n)=-1`, `h(t)=1`, with
   `R(s_n)=R(u_n)=n` and `R(t)=star`. Then `epsilon_sep=1` but `1 notin Gamma_R`; the sound set
   `{star}` is nonvacuously closed `1`-complete.
3. **Finite supremum attained.** Add `s_infinity,u_infinity` with margins `1,-1` and a common code.
   The defect remains one, but closed `1`-completeness is now impossible.

## 3. Deterministic data processing and the observation floor

### Theorem 2 (exact-defect data processing)

Let `R_1 : K -> Z_1` be deterministic and let `T : Z_1 -> Z_2` be any deterministic map. For
`R_2=T compose R_1`,

\[
\Gamma_{R_1}\subseteq\Gamma_{R_2},\qquad
\varepsilon_{\rm sep}(R_1)\le\varepsilon_{\rm sep}(R_2),\qquad
\tau_{\ge}(R_1)\le\tau_{\ge}(R_2).
\]

#### Proof

Every equality `R_1(x)=R_1(x')` implies
`R_2(x)=T(R_1(x))=T(R_1(x'))=R_2(x')`. Hence every safe/unsafe witness for `R_1` is also a witness
for `R_2`, proving inclusion of the conflicting-margin sets. Taking suprema gives the defect
inequality, and Corollary 2 gives the threshold inequality. QED.

### Corollary 3 (observation ambiguity is a floor)

Let `O : K -> Y` be a deterministic observation map and `E : Y -> Z` a deterministic encoder. For
`R=E compose O`,

\[
\varepsilon_{\rm sep}(O)\le\varepsilon_{\rm sep}(E\circ O).
\]

Thus the encoder cannot undo an exact safe/unsafe alias already created by the observation. A
history encoder can improve on a memoryless observation only by changing the input object—from a
single observation to an informative history or belief—not by post-processing the same
memoryless observation.

For action sets, every observation fiber is contained in the corresponding representation fiber:

\[
F_y^O\subseteq F_{E(y)}^{E\circ O}.
\]

Consequently, for the common-action sets defined next,

\[
\mathcal A_{E\circ O}(E(y))\subseteq\mathcal A_O(y),\qquad y\in O(\mathcal K).
\]

This local inclusion is the action analogue of the observation floor.

## 4. Action-fiber impossibility

Let the action space be `U`, and let `U_safe(x) subset U` be the actions satisfying one explicitly
declared audited condition at state/history/belief `x`. For a nonempty fiber, define

\[
\mathcal A_R(z):=\bigcap_{x\in F_z^R}\mathcal U_{\rm safe}(x).
\]

The fiber is **action-conflicted** if every `U_safe(x)` is nonempty but `A_R(z)` is empty. The
individual nonemptiness separates aliasing-induced infeasibility from a state that was infeasible
even with full information.

### Theorem 3 (deterministic memoryless policies)

For a fixed `z in R(K)`, a deterministic memoryless decision `pi(z)` satisfies

\[
\pi(z)\in\mathcal U_{\rm safe}(x)\quad\text{for every }x\in F_z^R
\]

if and only if `pi(z) in A_R(z)`. Consequently, there exists such a uniformly safe decision at
`z` if and only if `A_R(z)` is nonempty. In particular, an action-conflicted fiber makes uniform
safety impossible for every deterministic memoryless latent policy.

#### Proof

The displayed condition says exactly that the one action `pi(z)` belongs to the intersection of all
safe-action sets in the fiber. QED.

Pointwise nonempty intersections for every `z` imply a set-theoretic selector (under the usual
choice convention), but they do not by themselves imply that the selector is Borel measurable,
continuous, representable by a policy class, or numerically computable. Those require separate
regularity assumptions on the correspondence `z -> A_R(z)`.

### Randomized policies: two different safety semantics

Let `(U, Sigma_U)` be a measurable action space. A randomized memoryless decision assigns a
probability measure `mu_z` on it at a fixed code. Assume every `U_safe(x)` and the common set below
belong to `Sigma_U`. There are two non-equivalent robust requirements:

1. **Strong/pathwise uniform safety:** `mu_z(A_R(z))=1`. A sampled action must be safe for all states
   compatible with the code.
2. **Pointwise almost-sure safety:** `mu_z(U_safe(x))=1` for every fixed `x in F_z^R`.

Strong safety implies pointwise safety. For an uncountable fiber, the converse can fail because an
uncountable intersection of probability-one events need not have probability one.

### Theorem 4 (randomization does not repair a robust conflict under stated conditions)

Fix a representation fiber.

1. A strongly safe randomized decision exists if and only if `A_R(z)` is nonempty.
2. If the fiber is finite or countable, a pointwise almost-sure safe randomized decision exists if
   and only if `A_R(z)` is nonempty.
3. More generally, if `U` is a compact metric space, `mu_z` is a Borel probability measure, and
   every `U_safe(x)` is closed, a pointwise almost-sure safe randomized decision exists if and only
   if `A_R(z)` is nonempty, even for an uncountable fiber.

Hence an empty common-action set rules out both deterministic and randomized memoryless robust
safety under either the strong semantics, a countable fiber, or the compact-closed regularity in
part 3.

#### Proof

For part 1, a probability measure cannot assign mass one to the empty set. Conversely, any common
safe action induces a safe Dirac measure.

For part 2, if `mu_z(U_safe(x))=1` for each member of a countable fiber, countable subadditivity of
the complements gives

\[
\mu_z\!\left(\bigcap_{x\in F_z^R}\mathcal U_{\rm safe}(x)\right)=1.
\]

The intersection is therefore nonempty. The converse again follows from a Dirac measure.

For part 3, pointwise probability-one safety implies that every finite intersection of safe-action
sets has probability one and is therefore nonempty. The closed sets `U_safe(x)` thus have the
finite-intersection property. Compactness of `U` implies that their full intersection is nonempty.
The converse is the same Dirac construction. QED.

### Why the randomized qualification is necessary

Take `U=[0,1]`, index the fiber by `x in [0,1]`, and define
`U_safe(x)=[0,1] setminus {x}`. Every set is Borel and individually nonempty, but their intersection
is empty. The uniform distribution nevertheless assigns probability one to `U_safe(x)` for each
fixed `x`. It is not strongly safe. The example violates the closed-set assumption in Theorem 4
and shows that “empty intersection rules out every pointwise-a.s. randomized policy” is false
without qualification.

## 5. Convex common-action projection and least intervention

### Theorem 5 (pointwise least-intervention robust action)

Assume `U subset R^p` and, for a fixed code `z`, every `U_safe(x)` is closed and convex. If
`A_R(z)` is nonempty, then it is closed and convex. For any nominal action `u_nom(z) in R^p`, the
problem

\[
\pi_{\rm LI}(z)
=\mathop{\rm argmin}_{u\in\mathcal A_R(z)}
\frac12\|u-u_{\rm nom}(z)\|_2^2
\]

has a unique solution. It is uniformly safe for the audited condition on the entire fiber and has
minimum Euclidean intervention among all such uniformly safe actions. It is characterized by

\[
\langle u_{\rm nom}(z)-\pi_{\rm LI}(z),
u-\pi_{\rm LI}(z)\rangle\le0
\quad\text{for every }u\in\mathcal A_R(z).
\]

The same result holds for a positive-definite quadratic norm. If its matrix is `Q`, the displayed
variational inequality replaces `u_nom-pi_LI` by `Q(u_nom-pi_LI)`.

#### Proof

An arbitrary intersection of closed convex sets is closed and convex. A minimizing sequence has
bounded objective values and is therefore bounded. In finite dimensions it has a convergent
subsequence; closedness makes the limit feasible, and continuity makes it a minimizer. If two
distinct points `u` and `v` were minimizers, convexity would make their midpoint feasible, while

\[
\left\|\frac{u+v}{2}-u_{\rm nom}\right\|^2
=\frac12\|u-u_{\rm nom}\|^2+\frac12\|v-u_{\rm nom}\|^2
-\frac14\|u-v\|^2
\]

would make the midpoint strictly better, a contradiction. The minimizer is feasible by
construction, and the objective is exactly the intervention magnitude. Finally, apply optimality
to the feasible segment `pi_LI+t(u-pi_LI)` and take the right derivative at `t=0` to obtain the
variational inequality. A positive-definite quadratic is reduced to the Euclidean case by an
invertible linear change of coordinates. QED.

For control-affine, relative-degree-one barrier filters, each exact state often supplies an affine
action constraint, so the common-action problem is a robust convex quadratic program. This
observation is only pointwise. Feasibility, finite reduction of infinitely many fiber constraints,
measurability/continuity as `z` changes, estimation error, and the hypotheses that make the
underlying barrier condition a forward-safety guarantee remain separate obligations.

## 6. Finite-action quantitative slack under distinct randomization semantics

Let `U={1,...,k}` with `1<=k<infinity`. Let
`ell : K x U -> [0,infinity)` be a finite-valued nonnegative violation loss.  For the fixed nonempty
fiber considered below, assume every action-wise robust loss `sup_{x in F} ell(x,a)` is finite; this
avoids undefined `0 * infinity` terms in randomized action-wise objectives.  The intended calibrated
case satisfies

\[
\ell(x,a)=0\quad\Longleftrightarrow\quad a\in\mathcal U_{\rm safe}(x).
\]

For a fixed nonempty fiber `F=F_z^R`, define deterministic and **pointwise expected-loss** minimax
slacks,

\[
\begin{aligned}
\beta_{\rm det}(z)
&:=\min_{a\in\mathcal U}\ \sup_{x\in F}\ell(x,a),\\
\beta_{\rm exp}(z)
&:=\inf_{p\in\Delta_k}\ \sup_{x\in F}
\sum_{a=1}^k p_a\ell(x,a).
\end{aligned}
\]

Here the compatible state is worst-case after the distribution `p` is fixed, but the loss is
averaged over the realized action. This is weaker than strong or worst-supported-action fiber
safety. The corresponding action-wise robust expected loss is

\[
\beta_{\rm ar}(z)
:=\inf_{p\in\Delta_k}\sum_{a=1}^k p_a\sup_{x\in F}\ell(x,a)
=\beta_{\rm det}(z),
\]

because a linear objective over the simplex is minimized at an action attaining the smallest
action-wise supremum. Thus randomization gives no benefit when every realized action is evaluated
against all compatible states. A genuinely worst-supported-action loss also obeys

\[
\beta_{\rm ws}(z)
:=\inf_{p\in\Delta_k}\max_{a:p_a>0}\sup_{x\in F}\ell(x,a)
=\beta_{\rm det}(z),
\]

because a Dirac mass on a deterministic minimizer attains the lower bound and every nonempty
support contains an action whose robust loss is at least that deterministic minimum.

### Theorem 6 (finite-action pointwise-expectation sandwich)

For every such fiber and loss,

\[
\frac{1}{k}\,\beta_{\rm det}(z)
\le \beta_{\rm exp}(z)
\le \beta_{\rm det}(z).
\]

If the fiber is action-conflicted and `ell(x,a)=0` exactly on safe actions, then
`beta_det(z)>0`, hence `beta_exp(z)>0`. For the hard loss
`ell(x,a)=1{a notin U_safe(x)}`, an action-conflicted fiber obeys

\[
\beta_{\rm det}(z)=1,
\qquad
\frac1k\le\beta_{\rm exp}(z)\le1,
\qquad \beta_{\rm ar}(z)=\beta_{\rm ws}(z)=1.
\]

Thus randomization can spread pointwise expected failure across actions and states, but with `k`
actions it cannot drive the worst-compatible-state failure probability below `1/k` on a
conflicted fiber. It does not improve the action-wise robust or worst-supported-action slack.

#### Proof

The upper bound follows by choosing the Dirac distribution on an action attaining
`beta_det`; the minimum is attained because `U` is finite. For any `p in Delta_k`, choose an action
`a_p` with `p_{a_p}>=1/k`. Nonnegativity gives

\[
\begin{aligned}
\sup_{x\in F}\sum_a p_a\ell(x,a)
&\ge \sup_{x\in F}p_{a_p}\ell(x,a_p)\\
&=p_{a_p}\sup_{x\in F}\ell(x,a_p)\\
&\ge\frac1k\min_a\sup_{x\in F}\ell(x,a)
=\frac1k\beta_{\rm det}(z).
\end{aligned}
\]

Take the infimum over `p`. If the common safe set is empty, every action is unsafe for at least one
state in the fiber. Exact zero calibration makes each of the finitely many quantities
`sup_x ell(x,a)` strictly positive, so their minimum is positive. Under hard loss, every one of
those suprema equals one. QED.

### Finite-fiber adversarial witness

If `F={x_1,...,x_n}` is also finite and `L_{ia}=ell(x_i,a)`, finite linear-program duality gives

\[
\beta_{\rm exp}(z)
=\max_{q\in\Delta_n}\ \min_{a\in\mathcal U}
\sum_{i=1}^n q_iL_{ia}.
\]

The primal minimizes `t` subject to `Lp<=t 1` and `p in Delta_k`; its dual maximizes `r` subject to
`L^T q>=r 1` and `q in Delta_n`, with `p,q>=0`, simplex equalities, and unrestricted scalar
variables `t,r`. This form returns a distribution over compatible states that witnesses the
pointwise expected-loss slack and is directly testable in a finite audit.

### Coarsening monotonicity for action slack

If `R_2=T compose R_1`, `z_2=T(z_1)`, and both slacks use the same loss and action space, then

\[
F_{z_1}^{R_1}\subseteq F_{z_2}^{R_2}
\quad\Longrightarrow\quad
\beta_{\rm det}^{R_1}(z_1)\le\beta_{\rm det}^{R_2}(z_2),\qquad
\beta_{\rm exp}^{R_1}(z_1)\le\beta_{\rm exp}^{R_2}(z_2).
\]

This follows because the supremum is taken over a larger set after coarsening, for every fixed
deterministic action or action distribution, and minimization preserves the inequality.

## 7. Precise limits and counterexamples

### 7.1 The static number is `h`-relative

For any constant `c>0`, replacing `h` by `h_tilde=c h` leaves the safe and unsafe sets unchanged but
gives

\[
\varepsilon_{\rm sep}^{\widetilde h}(R)
=c\,\varepsilon_{\rm sep}^{h}(R).
\]

Cross-task comparisons therefore require a canonical margin, such as signed distance under a
declared state metric, or explicit physical units. The defect is not a set invariant.

### 7.2 Static separation alone has a trivial scalar solution

The representation `R(x)=h(x)` has no exact safe/unsafe collision and therefore has static defect
zero. Yet it need not preserve dynamics or safe actions. For example, take safe states `x_1,x_2`
with `h(x_1)=h(x_2)=1`, an unsafe state `u` with `h(u)=-1`, and let `R=h`. If
`U_safe(x_1)={a}` and `U_safe(x_2)={b}` with `a!=b`, the positive code fiber is action-conflicted
despite perfect static separation. This rules out a nontrivial latent-dimension lower bound from
one known scalar safety specification alone and rules out any inference from static defect zero to
safe-control sufficiency.

### 7.3 Exact data processing does not transfer to an uncalibrated radius

For a metric latent space, define a radius defect by allowing
`d(R(x),R(x'))<=delta`. Take one safe state of margin one and one unsafe state, with codes zero and
one. At `delta=0.75` there is no robust witness. The injective post-processing `T(z)=z/2` creates a
radius witness, while `T(z)=2z` removes one in the reverse construction. Exact fibers are unchanged
by either scaling. Thus a fixed raw radius has no representation-independent data-processing law;
it must be tied to a deployment uncertainty set, a scale convention, or appropriate Lipschitz
transport.

### 7.4 Randomized expected safety is weaker than robust safety

The punctured-interval example after Theorem 4 shows that per-state probability-one safety can
coexist with an empty common-action set when an uncountable family of nonclosed safe sets is
allowed. Allowing a small expected loss is weaker still and must not be described as a pathwise or
uniform certificate.

### 7.5 Positive slack needs finiteness or uniform analytic control

An empty intersection need not imply a positive continuous-action minimax margin. In
`U=R^2`, let

\[
A_1=\{(t,0):t\ge0\},\qquad
A_2=\{(t,1/t):t>0\}.
\]

Both sets are nonempty, closed, and disjoint, but their distance is zero. With
`ell(i,u)=dist(u,A_i)`, the points `(n,0)` have worst-set loss at most `1/n`, so the infimum robust
slack is zero and is not attained. The finite-action lower bound does not extend to this
noncompact action space. Positive continuous-action slack needs substantive conditions such as a
compact/coercive domain together with lower semicontinuity and a uniform calibration away from the
safe sets; compactness or coercivity alone is not a sufficient slogan for an arbitrary loss.

### 7.6 Projection assumptions are substantive

If the common-action set is empty, the least-intervention program is infeasible. If it is not
closed, the infimum need not be attained. If it is nonconvex, a projection may not be unique: for
`A={-1,1}` and nominal action zero, both actions are minimizers. Even closed-convex pointwise
projections need not form a measurable or continuous policy when the feasible correspondence
varies irregularly with `z`.

### 7.7 The maximal set need not belong to a certificate class

Theorem 1 optimizes over arbitrary subsets of latent space. It proves a universal lower bound for
every restricted certificate class, but `S_max` need not be closed, smooth-boundary, semialgebraic,
representable by a fixed neural architecture, or amenable to verification. Class-wise
achievability is a separate approximation/regularity theorem and is currently open in this
project.

### 7.8 One-step action sufficiency is not long-horizon safety

All action statements are relative to the declared correspondence `U_safe(x)`. If that
correspondence encodes only an approximate one-step test, finding a common action proves only
common feasibility for that test. A downstream theorem still needs the usual dynamics,
disturbance, model-error, recursive-feasibility, or barrier hypotheses. Likewise, a common current
action does not make the latent process Markov.

### 7.9 Determinism is part of the observation-floor theorem

The exact defect is defined through equality of deterministic codes. Stochastic observations or
encoders require a different object—such as overlap of conditional output laws, a coupling, or a
hypothesis-testing deficiency. Theorem 2 must not be quoted as a proved stochastic-channel result.
The intuitive statement that identical observation laws cannot be disambiguated downstream is
plausible data processing, but its quantitative formulation is outside this v1 package.

## 8. Smooth rare-event separation for average world-model losses

The following explicit construction closes the previously informal average-loss counterexample.
It is a separation lemma, not a claim that localized smooth folds are the dominant empirical
failure mode of trained encoders.

### Proposition 1 (vanishing expected loss with fixed static and action defects)

Fix `a in (0,1)`, let `K=[-1,1]` carry the uniform distribution, set `h(x)=x`, and let the
dynamics be the identity `f(x)=x`. Define the smooth unit bump

\[
\psi(t)=
\begin{cases}
\exp\!\left(1-\frac{1}{1-t^2}\right),& |t|<1,\\
0,& |t|\ge 1,
\end{cases}
\]

and, for `0 < delta < min(a,1-a)`, define

\[
R_\delta(x)
=x-a\psi\!\left(\frac{x-a}{\delta}\right)
+a\psi\!\left(\frac{x+a}{\delta}\right).
\]

With identity decoder `D(z)=z` and identity latent predictor `P(z)=z`:

1. `R_delta` is smooth, `R_delta(a)=R_delta(-a)=0`, and
   `epsilon_sep(R_delta) >= a` for every allowed `delta`;
2. the expected squared reconstruction loss obeys

   \[
   \mathbb E|X-D(R_\delta(X))|^2\le 2a^2\delta\longrightarrow0;
   \]

3. the latent one-step prediction loss is identically zero,
   `P(R_delta(x))=R_delta(f(x))`, while decoded next-state MSE obeys the same
   `2 a^2 delta` bound; and
4. for the two actions `U={-1,+1}` and the safe-action condition `u x >= 0`, the common fiber
   containing `-a` and `a` is action-conflicted, with deterministic margin-loss slack at least
   `a` when the violation loss is `ell(x,u)=max(0,-u x)`.

#### Proof

The standard compactly supported bump `psi` is infinitely differentiable and takes value one at
zero. The width restriction puts the two bump supports inside `[-1,1]` and makes them disjoint.
At `x=a`, the right bump equals one and the left bump vanishes, so `R_delta(a)=0`; the symmetric
calculation gives `R_delta(-a)=0`. Since `a` is safe with margin `a` and `-a` is unsafe, this is a
static collision of margin `a`.

Outside the two support intervals, `R_delta(x)=x`. On their union, whose Lebesgue measure is at
most `4 delta`, disjointness and `0 <= psi <= 1` give `|x-R_delta(x)| <= a`. The uniform density on
`[-1,1]` is `1/2`, hence the reconstruction MSE is at most
`(1/2)(4 delta)a^2=2a^2 delta`. Identity dynamics and prediction give exact equality in latent
space and reduce decoded next-state loss to reconstruction loss.

Finally, at `a` only action `+1` is safe and at `-a` only action `-1` is safe, so their safe-action
sets are individually nonempty and disjoint. For either common deterministic action, one of the
two states incurs violation `a`. QED.

This proposition rules out any theorem that tries to upper-bound a worst-fiber static or action
defect using only an expected reconstruction or prediction loss, without coverage, tail,
regularity, or verified uniform-control assumptions. The executable construction and rate checks
live in `latent_safety.synthetic` and `tests/test_average_loss_counterexample.py`.

## 9. Claim status after this note

Proof-drafted and internally red-teamed within the stated set-theoretic scope:

- endpoint-correct static existence equivalence;
- equality of the infimum completeness threshold and exact `h`-relative defect;
- deterministic exact-defect data processing and the observation floor;
- deterministic action-conflict impossibility;
- randomized impossibility under strong semantics, countable fibers, or compact-closed action
  sets;
- pointwise closed-convex least-intervention projection; and
- finite-action deterministic/pointwise-expected slack bounds, strong-slack equality, and the
  finite-fiber LP dual; and
- a smooth localized-collision construction separating expected reconstruction/prediction loss
  from worst-fiber static and action defects.

Still requiring new assumptions or separate work:

- class-wise achievement by continuous/neural/CBF certificates;
- stochastic observation/encoder data processing;
- robust-radius transport under calibrated uncertainty sets;
- measurable or continuous global selectors across latent codes;
- certified reduction of infinite fiber constraints to finite programs; and
- any implication for long-horizon closed-loop safety.
