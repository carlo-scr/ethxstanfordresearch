# Theory v2 candidate: dynamic margin regret and representation-policy viability

**Status:** candidate theorem package, internally checked on finite models, not independently
reviewed and not yet a novelty claim. This note is deliberately narrower than a general POMDP,
chance-constraint, or infinite-horizon result. It gives an endpoint-correct finite-horizon theorem
for support-robust safety on a finite history domain and records exactly what would be needed to
extend it.

The note separates three objects that must not be conflated. Robust optimal-margin regret measures
value loss from forcing one fiber action. A sign-specific obstruction measures whether one first
action remains safe when full-history optimal control resumes. Exact representation-policy
viability evaluates the entire future policy under the representation restriction. Only the third
is an exact recursive safety-preservation statement. Optimal-margin regret composes into a useful
margin-loss buffer, but its minimax selector can be unsafe even when a common safe action exists.

## 1. Scope and semantics

Fix a finite action set `A` and a horizon `T>=0`. For each `s in {0,...,T}`, let `H_s` be a finite
nonempty set of histories with `s` controls remaining. A member `eta in H_s`
may be the full observation/action history, a belief, or a belief support. The theorem does not
require a Markov physical observation. We call `R_s` a history encoder: no recursive latent update
or standalone latent world model is assumed.

For each history, let `I(eta)` be its nonempty set of compatible physical states and define the
support-robust current margin

\[
G_s(\eta):=\inf_{x\in I(\eta)}h(x).
\]

The convention is closed safety: `h(x) >= 0` is safe. In a finite POMDP one can take `I(eta)` to
be the support of the posterior belief. This is a support-wise/pathwise criterion, not an expected
cost or chance constraint. A very small positive-probability branch remains in scope.

For `s >= 1`, let

\[
N_s(\eta,a)\subseteq H_{s-1}
\]

be the finite, nonempty set of every admissible next history after action `a`. It contains all
declared plant disturbances and all possible next observations. A deterministic history-feedback
policy is a time-indexed family of maps from `H_s` to `A`.

The assumptions used in the main theorem are therefore explicit:

1. finite nonempty history layers and action set;
2. finite real margins;
3. a nonempty, fully enumerated successor correspondence;
4. the same physical action set at every history; and
5. deterministic policies evaluated against every successor branch.

State-dependent action availability can be handled by replacing `A` on a fiber with the
intersection of available actions. An empty intersection has infinite obstruction. We keep a fixed
action set so every quantity below is finite and every optimizer is attained.

## 2. Robust safety Bellman values

Define, for `s >= 1`,

\[
\begin{aligned}
V_0(\eta)&=G_0(\eta),\\
Q_s(\eta,a)
&=\min\!\left\{G_s(\eta),
       \min_{\eta'\in N_s(\eta,a)}V_{s-1}(\eta')\right\},\\
V_s(\eta)&=\max_{a\in A}Q_s(\eta,a).
\end{aligned}
\]

Thus `V_s` is the largest worst-case minimum physical margin that a full-history controller can
maintain over the current state and the next `s` transitions.

### Proposition 1 (closed finite-horizon viability)

For any `eta in H_s`, `V_s(eta) >= 0` if and only if there exists a deterministic
history-feedback policy that keeps `G >= 0` on every compatible branch for the remaining horizon.

#### Proof

At `s=0` this is exactly the definition of `G_0`. For the induction step, finiteness makes the
maximum attainable. The value is nonnegative precisely when some action has nonnegative current
margin and sends every successor to a history with nonnegative `V_{s-1}`. Apply the induction
hypothesis after the next history is observed. The converse follows by reading the same argument
backward. QED.

This equivalence uses the zero endpoint. There is no strictness gap because the action maxima and
finite-successor minima in the dynamic program are attained. The physical-state infimum already
absorbed into `G_s` need not itself be attained.

## 3. Two different first-action quantities

Let `R_s : H_s -> Z_s` be a deterministic history encoder and

\[
F_{s,z}:=\{\eta\in H_s:R_s(\eta)=z\}.
\]

### 3.1 Robust optimal-margin regret

For `s >= 1`, define the local action regret and local/global robust optimal-margin regrets

\[
\begin{aligned}
r_s(z,a)
&:=\max_{\eta\in F_{s,z}}
       \left[V_s(\eta)-Q_s(\eta,a)\right],\\
\rho_s(z;R_s)&:=\min_{a\in A}r_s(z,a),\\
\rho_s(R_s)&:=\max_{z\in R_s(H_s)}\rho_s(z;R_s).
\end{aligned}
\]

Every term is nonnegative. The local number is exactly the smallest worst-history
*optimal-margin* loss when only the next action depends on the code and full-history optimal
control resumes. It is not an exact safety defect. Choose a deterministic tie-broken margin
selector

\[
a_s^\rho(z)\in\arg\min_{a\in A}r_s(z,a).
\]

### Proposition 2 (exact margin-optimality characterization)

For a fixed fiber,

\[
\rho_s(z;R_s)=0
\quad\Longleftrightarrow\quad
\bigcap_{\eta\in F_{s,z}}\arg\max_{a\in A}Q_s(\eta,a)\ne\varnothing.
\]

Thus zero `rho_s` means that one code action is robust-margin optimal for every history in the
fiber. This value-preserving property is stronger than sign preservation.

### 3.2 Sign-specific first-action obstruction

Let `H_s^+(z)={eta in F_{s,z}:V_s(eta)>=0}` contain the full-history viable members. Define

\[
\kappa_s(z;R_s)
:=\min_{a\in A}\max_{\eta\in H_s^+(z)}[-Q_s(\eta,a)]_+,
\]

where the maximum over an empty viable set is zero. Then

\[
\kappa_s(z;R_s)=0
\quad\Longleftrightarrow\quad
\bigcap_{\eta\in H_s^+(z)}\{a:Q_s(\eta,a)\ge 0\}\ne\varnothing.
\]

Thus `kappa_s` is the exact *first-action* safety obstruction when full-history optimal control may
resume afterward. It is different from `rho_s`. For example,

\[
\begin{array}{c|cc}
 &a_0&a_1\\\hline
\eta_1&0&2\\
\eta_2&0&-1
\end{array}
\]

has two viable histories. Action `a_0` is common safe, so `kappa_s=0`. Its worst optimal-margin
regret is two, while action `a_1` has regret one. Therefore the unique `rho_s` minimizer is `a_1`,
which is unsafe at `eta_2`. The margin-loss theorem below remains correct—it declines to certify
`eta_2`, whose optimal value is zero and whose buffer is one—but its minimax selector is not a
maximal-viability controller.

#### Proof

The `rho_s` claim follows because every regret term is nonnegative and vanishes exactly at a
Q-maximizing action. The `kappa_s` claim follows identically from the nonnegative violation terms,
with the stated empty-set convention. QED.

### Proposition 3 (optimal-margin-regret data processing)

If `R_s^c=T_s compose R_s^f` is a deterministic coarsening on the same controlled history layer,
then

\[
\rho_s(R_s^f)\le\rho_s(R_s^c).
\]

#### Proof

Every fine fiber lies inside one coarse fiber. For every action, maximizing a nonnegative regret
over the smaller set cannot exceed maximizing over the larger set. Minimizing over actions
preserves the inequality, and so does the outer maximum over fibers. QED.

The statement is per controlled layer `s>=1`; no postprocessing premise is needed at terminal layer
zero, where no regret is defined. If `R_s=E_s compose O_s` for a deterministic history observation
`O_s`, then `rho_s(O_s) <= rho_s(R_s)` on that same history domain. It does not compare a
memoryless observation to a longer history, because those are different information objects.

## 4. Arbitrary representation-policy composition and exact viability

### 4.1 Composition for an arbitrary code policy

Let `mu_s : Z_s -> A` be any deterministic time-indexed code policy and write

\[
\pi_s^\mu(\eta)=\mu_s(R_s(\eta)).
\]

Its robust value is

\[
\begin{aligned}
J_0^\mu(\eta)&=G_0(\eta),\\
J_s^\mu(\eta)&=\min\!\left\{G_s(\eta),
  \min_{\eta'\in N_s(\eta,\pi_s^\mu(\eta))}J_{s-1}^\mu(\eta')\right\}.
\end{aligned}
\]

Define its selected-action optimal-margin regret and path-dependent buffer by

\[
\begin{aligned}
\epsilon_s^\mu(z)&=r_s(z,\mu_s(z)),
&\epsilon_s^\mu&=\max_z\epsilon_s^\mu(z),\\
B_0^\mu(\eta)&=0,\\
B_s^\mu(\eta)&=\epsilon_s^\mu(R_s(\eta))
 +\max_{\eta'\in N_s(\eta,\pi_s^\mu(\eta))}B_{s-1}^\mu(\eta').
\end{aligned}
\]

### Theorem 1 (arbitrary-code-policy margin loss)

For every `s` and every `eta in H_s`,

\[
0\le V_s(\eta)-J_s^\mu(\eta)
\le B_s^\mu(\eta)
\le\sum_{j=1}^{s}\epsilon_j^\mu.
\]

Therefore the closed endpoint implication

\[
V_s(\eta)\ge B_s^\mu(\eta)
\quad\Longrightarrow\quad
J_s^\mu(\eta)\ge0
\]

holds. The looser condition

\[
V_s(\eta)\ge\sum_{j=1}^{s}\epsilon_j^\mu
\]

also guarantees support-robust safety.

#### Proof

Optimality gives `J_s^mu <= V_s`, so only the upper loss bound is substantive. The claim is exact at
`s=0`. Assume it holds with one fewer control remaining and abbreviate
`a=pi_s^mu(eta)`. By the induction hypothesis,

\[
J_{s-1}^\mu(\eta')\ge V_{s-1}(\eta')-B_{s-1}^\mu(\eta')
\]

for every successor. If `M` is the maximum successor buffer, then

\[
\min_{\eta'}J_{s-1}^\mu(\eta')
\ge\min_{\eta'}V_{s-1}(\eta')-M.
\]

For every real `g,b` and `M >= 0`, `min(g,b-M) >= min(g,b)-M`. Hence

\[
J_s^\mu(\eta)\ge Q_s(\eta,a)-M.
\]

The definition of the selected-action regret gives

\[
Q_s(\eta,a)
\ge V_s(\eta)-\epsilon_s^\mu(R_s(\eta)).
\]

Combining the last two displays gives

\[
V_s(\eta)-J_s^\mu(\eta)\le B_s^\mu(\eta).
\]

The global sum follows because every local term is at most `epsilon_s^mu`. If
`V_s >= B_s^mu`, rearrangement gives `J_s^mu >= 0`, including equality. QED.

The local buffer can be much smaller than the global sum because it follows only branches reached
by the chosen policy. No distributional averaging is used. For the minimax margin selector
`mu_s=a_s^rho`, `epsilon_s^mu=rho_s(R_s)`, recovering the original `rho` buffer. This is a
margin-loss guarantee; it does not say that the selector maximizes retained viable volume.

### 4.2 Exact recursively evaluated representation-policy viability

Define the global first-action obstruction

\[
\kappa_{1:T}^{\max}(R)=\max_{1\le j\le T}\max_{z\in R_j(H_j)}
\kappa_j(z;R_j).
\]

### Proposition 4 (exact global sign preservation)

`kappa_{1:T}^max(R)=0` if and only if there exists one time-indexed representation policy `mu`
such that, simultaneously on every layer,

\[
V_j(\eta)\ge0\quad\Longrightarrow\quad J_j^\mu(\eta)\ge0
\qquad\text{for every }j,\eta.
\]

#### Proof

If every obstruction is zero, select a common-safe action in every fiber. At layer zero the claim
is exact. Inductively, a viable history has nonnegative Q under the selected action, so its current
margin is safe and every successor has nonnegative full-history value. The selected lower-layer
policy preserves all such successors by induction. Conversely, suppose one code policy preserves
all full-history viable histories. At any viable history, replacing that policy's continuation by
the full-history optimum can only increase its first-action value, so
`Q_j(eta,mu_j(R_j(eta)))>=J_j^mu(eta)>=0`. The same code action is therefore safe for every viable
member of the fiber, and every `kappa_j` vanishes. QED.

This proposition is global over all controlled layers and fibers. If only a particular initial
fiber or its reachable subtree matters, a positive obstruction elsewhere is irrelevant; the
fiber-specific object below handles that question.

Let

\[
\Pi_{1:s}^R
:=\prod_{j=1}^s A^{R_j(H_j)}
\]

be the finite class of all time-indexed policies whose stage-`j` action depends only on the
current code. Every `mu in Pi_{1:s}^R` is evaluated recursively by `J_j^mu` above; future actions
remain representation-restricted.

For any nonempty set `S subseteq H_s`, define

\[
g_s(S):=\min_{\eta\in S}G_s(\eta)
\]

and, for an action assignment `u:R_s(S)->A`,

\[
\operatorname{Post}_s^R(S,u)
:=\bigcup_{\eta\in S}N_s\!\left(\eta,u(R_s(\eta))\right).
\]

Because `A` is nonempty, every such assignment extends arbitrarily from `R_s(S)` to all codes in
`R_s(H_s)`; values on codes absent from `S` do not affect the current reachable-set value.

The union is essential: one lower-layer code policy must work jointly for every successor of every
initial history. Define the set-valued representation Bellman recursion

\[
\begin{aligned}
W_0^R(S)&=g_0(S),\\
W_s^R(S)&=
\max_{u\in A^{R_s(S)}}
\min\!\left\{g_s(S),
W_{s-1}^R\!\left(\operatorname{Post}_s^R(S,u)\right)\right\}.
\end{aligned}
\]

### Theorem 2 (exact set-valued representation Bellman recursion)

For every nonempty `S subseteq H_s`,

\[
W_s^R(S)
=\max_{\mu\in\Pi_{1:s}^R}\min_{\eta\in S}J_s^\mu(\eta).
\]

Consequently, `W_s^R(S)>=0` if and only if one deterministic time-indexed representation policy
keeps the margin nonnegative on every branch from every history in `S`. Every maximum is attained.

#### Proof

The base case is the definition of `G_0`. At a controlled layer, restrict a policy's current
assignment to `R_s(S)` and call it `u`; conversely, the arbitrary extension above makes every such
`u` admissible. Write the remaining lower-layer policy as `nu`. For fixed `u,nu`, associativity of
finite minima gives

\[
\min_{\eta\in S}J_s^{(u,\nu)}(\eta)
=\min\!\left\{g_s(S),
\min_{\eta'\in\operatorname{Post}_s^R(S,u)}J_{s-1}^\nu(\eta')\right\}.
\]

For fixed `u`, finiteness and monotonicity of `x mapsto min{g_s(S),x}` let the maximum over `nu`
pass through this scalar minimum. The induction hypothesis then supplies
`W_{s-1}^R(Post_s^R(S,u))`; maximizing over `u` proves the recursion. For a fixed policy, its
recursive value is nonnegative exactly when every current and successor margin is nonnegative.
The minimum over `S` makes the statement simultaneous, and finite maximization gives the endpoint
and attainment. QED.

### Proposition 5 (exact dynamic viability data processing)

Suppose that, on the same history game, `R_j^c=T_j compose R_j^f` for every controlled layer
`1<=j<=s`. Then, for every nonempty `S subseteq H_s`,

\[
W_s^{R^f}(S)\ge W_s^{R^c}(S).
\]

No postprocessing condition is needed at terminal layer zero. The premise is required at every
future controlled layer, not only at the initial layer.

#### Proof

Every coarse policy lifts to the fine representation by
`mu_j^f(z)=mu_j^c(T_j(z))`. The lifted and coarse policies choose the same physical action at every
history, hence have identical recursively evaluated `J` values. The fine policy class contains all
such lifted behavior, so maximizing over it cannot lower the common-set value. QED.

The comparison must use the same initial history set. Monotonicity in the set argument additionally
gives, for a fine code `z_f` and `z_c=T_s(z_f)`,

\[
W_s^{R^f}(F_{s,z_f}^{f})
\ge W_s^{R^c}(F_{s,z_c}^{c}),
\]

because the fine fiber is contained in the corresponding coarse fiber.

Define the exact universal fiber value as the special case

\[
\Lambda_s^R(z)
:=W_s^R(F_{s,z})
=\max_{\mu\in\Pi_{1:s}^R}
  \min_{\eta\in F_{s,z}}J_s^\mu(\eta).
\]

### Corollary 1 (exact finite representation-policy viability)

For a finite game and `s>=1`, `Lambda_s^R(z)>=0` if and only if there exists one deterministic time-indexed
representation policy that keeps the margin nonnegative on every branch from every history in
`F_{s,z}`. All maximizers are attained. The safe initial actions are exactly those appearing at
`(s,z)` in at least one such policy.

#### Proof

For a fixed policy, induction on the displayed recursion for `J_j^mu` shows that
`J_s^mu(eta)>=0` exactly when it is safe on every branch from `eta`. Taking the minimum over the
fiber imposes the guarantee simultaneously on all histories sharing the code. Theorem 2 gives the
existence equivalence and attainment. For a fiber, the exact safe initial actions are

\[
\left\{a\in A:
\min\!\left[g_s(F_{s,z}),
W_{s-1}^R\!\left(\bigcup_{\eta\in F_{s,z}}N_s(\eta,a)\right)\right]\ge0
\right\}.
\]

QED.

This object is not reducible to the one-step `kappa_s`. For example, let two one-step histories
share a code and have Q rows `(1,-1)` and `(-1,1)`. Let a two-step singleton root have either action
lead nondeterministically to both histories. At the root, both full-history continuation values are
one, hence `kappa_2=0`. Yet every code policy chooses one action at the next shared code and incurs
margin minus one on one branch, so `Lambda_2^R=-1`.

Let `U_s={eta:V_s(eta)>=0}` be the full-history viable set and define the recursive representation
shortfall

\[
\chi_s(R)=
\begin{cases}
0,&U_s=\varnothing,\\
[-W_s^R(U_s)]_+,&U_s\ne\varnothing.
\end{cases}
\]

Theorem 2 and Proposition 5 give `chi_s(R^f)<=chi_s(R^c)`. Moreover,

\[
\max_{1\le s\le T}\chi_s(R)=0
\quad\Longleftrightarrow\quad
\kappa_{1:T}^{\max}(R)=0
\quad\Longleftrightarrow\quad
\exists\mu\ \forall s,\eta:\
V_s(\eta)\ge0\Rightarrow J_s^\mu(\eta)\ge0.
\]

When `T=0`, both maxima over controlled layers are defined as zero, the policy family is the empty
product, and the equivalence holds vacuously because `J_0=V_0=G_0`.

For the only non-immediate direction, if `chi_s=0`, some policy safe on all of `U_s` supplies at
its first layer a common action with `Q_s>=J_s^mu>=0` on every viable member of every fiber. Thus
every `kappa_s` vanishes. Proposition 4 then constructs one policy that works simultaneously on
all layers. Policies witnessing separate values of `chi_s` need not themselves be the same.

The safely retainable family

\[
\mathfrak V_s(R)
:=\{S\subseteq U_s:S=\varnothing\ \text{or}\ W_s^R(S)\ge0\}
\]

is downward closed, and deterministic coarsening gives
`mathfrak V_s(R^c) subseteq mathfrak V_s(R^f)`. Its maximum cardinality equals the largest number
of viable histories retained by any one representation policy. This yields an exact data-processing
statement for the retained-viability statistic, not only for optimal-margin regret.

Two coupling qualifications are essential. First, coarsening only at layer `s` does not order
`W_s`: a singleton two-step root can lead to two next histories that a fine lower-layer code
separates and a coarse lower-layer code merges, producing values one and minus one despite
identical root codes. Second, checking `Lambda_s^R(z)>=0` separately for every initial code need
not produce one joint policy. Two different-code roots can each be safe under incompatible choices
at one shared successor code, while `W_s^R(U_s)<0`. The common-set recursion, rather than separate
fiber maxima, detects this conflict.

The finite implementation memoizes this reachable-history-set recursion and, under an explicit
guard, enumerates `mathfrak V_s(R)`. It is checked against independent full-policy enumeration on
every nonempty subset of randomized small games. The older policy enumerator remains as an
independent oracle. Because these routines make exact sign and ordering decisions, they accept
only finite numerical inputs unchanged by binary64 conversion and reject inexact conversions
rather than rounding across a theorem endpoint.

The viable-family checker also reports

\[
\operatorname{Ret}_s(R)
=\max_{\mu\in\Pi_{1:s}^R}
\frac{|\{\eta:V_s(\eta)\ge0,\ J_s^\mu(\eta)\ge0\}|}
{|\{\eta:V_s(\eta)\ge0\}|},
\]

with value one by convention when the denominator is zero. This retained-viability statistic and
false-safe diagnostics should accompany `rho`; optimal-margin regret alone can conceal lost safety
coverage.

The theorem is about representation-measurable history policies. It permits evaluating
`R_s(eta)` from the full history at each time. It does **not** prove that a code is recursively
updateable from `(z,a,o')`, that a latent successor model is Markov, or that planning can be done
entirely in latent space. Those claims require a latent update/transition premise and comparison
to approximate-information-state theory.

## 5. Uniform latent-Q approximation

Define the best time-`s` uniform latent approximation error

\[
\delta_s^*(R_s)
:=\min_{q:Z_s\times A\to\mathbb R}
 \max_{\eta\in H_s,\,a\in A}
 |Q_s(\eta,a)-q(R_s(\eta),a)|.
\]

Because all sets are finite, the minimum is attained. A separate constant may be chosen for every
fiber-action pair, so midpoint approximation gives the exact identity

\[
\delta_s^*(R_s)
=\frac12\max_{z,a}
\left(
\max_{\eta\in F_{s,z}}Q_s(\eta,a)
-\min_{\eta\in F_{s,z}}Q_s(\eta,a)
\right).
\]

### Theorem 3 (Q approximation bounds optimal-margin regret)

For every stage,

\[
\rho_s(R_s)\le 2\delta_s^*(R_s).
\]

More generally, if a particular `q_s(z,a)` satisfies uniform error at most `delta_s`, then the
greedy code-level action `argmax_a q_s(z,a)` has selected-action optimal-margin regret at most
`2 delta_s`. Executing these greedy actions at every stage gives

\[
V_s(\eta)-J_s^q(\eta)\le 2\sum_{j=1}^s\delta_j,
\]

with the analogous local path-sum refinement.

#### Proof

Fix a fiber and let `a_q` maximize `q(z,a)`. For every compatible history,

\[
\begin{aligned}
V_s(\eta)
&=\max_a Q_s(\eta,a)
\le\max_a q(z,a)+\delta_s\\
&=q(z,a_q)+\delta_s
\le Q_s(\eta,a_q)+2\delta_s.
\end{aligned}
\]

Thus the selected action has worst-fiber optimal-margin regret at most `2 delta_s`. Minimize over
actions, maximize over fibers, and use `delta_s=delta_s^*` for the first claim. For the particular
q-greedy policy, Theorem 1 applies directly because it was stated for an arbitrary code policy:
its `epsilon_s^mu` is at most `2 delta_s` at every stage. QED.

Uniform Q preservation is sufficient but not necessary. If the same action is optimal throughout a
fiber, `rho_s` is zero even when suboptimal action values or absolute value offsets vary widely.
This argues for auditing optimal-margin regret in addition to latent-Q prediction error. Neither
quantity replaces the sign obstruction or exact representation-policy viability.

## 6. A model-local sufficient condition

The Q-based quantity is exact but uses the full-history oracle. The following standard-style
condition reduces it to current-margin and successor-set discrepancies.

Give `H_{s-1}` a metric `d_{s-1}`. Assume `V_{s-1}` is `L_{s-1}`-Lipschitz and define

\[
\begin{aligned}
\alpha_s
&:=\max_{R_s(\eta)=R_s(\widetilde\eta)}
 |G_s(\eta)-G_s(\widetilde\eta)|,\\
\beta_s
&:=\max_{\substack{R_s(\eta)=R_s(\widetilde\eta)\\a\in A}}
 d_H^{(s-1)}\!\left(N_s(\eta,a),N_s(\widetilde\eta,a)\right),
\end{aligned}
\]

where `d_H` is symmetric Hausdorff distance. The elementary inequality for infima of a Lipschitz
function over two nonempty sets gives

\[
|Q_s(\eta,a)-Q_s(\widetilde\eta,a)|
\le \max\{\alpha_s,L_{s-1}\beta_s\}.
\]

Therefore

\[
\rho_s(R_s)
\le\max\{\alpha_s,L_{s-1}\beta_s\},
\]

and these local discrepancies may replace `rho_s` in the global safety buffer.

This is an approximate safety-bisimulation-style sufficient condition, not a data-derived result.
One must prove the Lipschitz constant and the Hausdorff successor bound under a physically calibrated
metric. A raw neural latent metric can be arbitrarily rescaled, and finite sampled successors do
not upper-bound a population Hausdorff distance without coverage assumptions.

### Proposition 6 (conditional Euclidean cover-bound calculation)

There is a deterministic route from a verified cover to a population upper bound. Let a possibly
infinite history layer have metric `d_H`; the latent metric in this proposition and its code is
explicitly Euclidean `d_Z(z,z')=||z-z'||_2`. Assume:

1. a finite audited set `S={eta_i}` is an `r_cov`-net of the declared history layer;
2. `R_s` is `L_R`-Lipschitz from `d_H` to `d_Z`;
3. every `Q_s(.,a)` is `L_Q`-Lipschitz under `d_H`; and
4. tabulated values satisfy `|Qhat_s(eta_i,a)-Q_s(eta_i,a)| <= e_Q` uniformly.

For `r >= 0`, compute the finite expanded-radius oscillation

\[
\widehat\Omega_s(r)
:=\max_{\substack{i,j,a:\\
d_Z(R_s(\eta_i),R_s(\eta_j))\le r}}
|\widehat Q_s(\eta_i,a)-\widehat Q_s(\eta_j,a)|.
\]

Pairs with `i=j` make the maximum nonempty. Then every population pair satisfying
`d_Z(R_s(eta),R_s(eta')) <= delta` obeys

\[
|Q_s(\eta,a)-Q_s(\eta',a)|
\le
\widehat\Omega_s(\delta+2L_Rr_{\mathrm{cov}})
+2L_Qr_{\mathrm{cov}}+2e_Q.
\]

In particular, at `delta=0`,

\[
\rho_s(R_s)
\le
\widehat\Omega_s(2L_Rr_{\mathrm{cov}})
+2L_Qr_{\mathrm{cov}}+2e_Q.
\]

Thus verified stagewise premises give a conditional upper bound on optimal-margin regret that can
be substituted into the minimax specialization of Theorem 1.

#### Proof

Choose cover points `eta_i,eta_j` within `r_cov` of the population pair. Representation
Lipschitzness and the triangle inequality put their latent codes within
`delta+2 L_R r_cov`, so their estimated Q difference is included in the finite maximum. Insert
the two true sampled Q values between the population values. The two history approximation terms
sum to at most `2 L_Q r_cov`, and the two Q-estimation terms sum to at most `2 e_Q`. This proves
the first display. At `delta=0` it bounds every within-fiber Q oscillation; Theorem 3 bounds robust
optimal-margin regret by that maximum action-wise oscillation. QED.

The public routine is therefore named a **conditional Euclidean bound calculator**, and its result
status says `conditional_arithmetic_only_not_verified_certificate`. It requires nonempty provenance
for the cover, both Lipschitz constants, Q error, and metric. Its optional radius-comparison
tolerance `tau_rad` is explicit: the finite search uses and records the effective radius
`delta+2 L_R r_cov+tau_rad`; the default is exactly zero.

This is not an i.i.d. generalization theorem or a verified finite-data certificate. A random
dataset is not a certified cover merely because it is large. Establishing the cover with high
probability needs explicit support-density and metric-entropy assumptions; certified neural
Lipschitz constants may also be too loose. Provenance records where a premise was claimed; it does
not validate that claim. The implementation deliberately does not accept an infinity-norm or
arbitrary-metric interpretation of its Euclidean pair search.

### 6.2 Verified complete-finite-domain route

There is now a separate, stronger routine for a domain that is genuinely finite and completely
enumerated. Given its full history-distance matrix, latent vectors, true Q table, and a subset of
sample indices, `calculate_verified_finite_domain_q_bound`:

1. validates the finite metric matrix, with any numerical validation tolerance recorded;
2. computes the exact nearest-sample cover radius on the declared domain;
3. derives the smallest pairwise Euclidean representation and coordinatewise Q Lipschitz
   constants on that domain;
4. computes the exact sampled-Q error against the supplied full Q table;
5. runs the cover arithmetic; and
6. independently enumerates every finite-domain pair and checks the claimed Q-coordinate
   oscillation bound on those within the operational latent radius.

It also reports the domain, sample, latent-dimension, and action counts; raw floating-point local
and global Q oscillations; outward-enclosing versions of both oscillations; and the
bound-to-global-oscillation ratio computed from the outward global value. These fields expose
whether a valid finite-domain bound is uninformative at the declared scale. Metric validation is
cubic in the domain size and is intended for bounded oracle problems.

The raw conditional cover calculation remains visible. Each positive enumerated coordinate
difference is advanced one representable value toward positive infinity, enclosing the exact-real
difference of the stored binary inputs even when subtraction rounds down. A raw cover calculation
that falls below the raw exhaustive oscillation by more than the declared arithmetic tolerance is
rejected. Independently of that diagnostic, the returned verified bound is the maximum of the raw
cover value and the outward local oscillation, and its correction is recorded. The outward step is
not compared with a fixed absolute tolerance because one ulp is scale dependent.

This closes the premises only relative to the supplied complete finite domain. It does not turn a
finite dataset into the population of a continuous system, certify omitted histories or
successors, or establish statistical coverage. The original conditional calculator remains the
correct interface whenever any premise is externally declared rather than exhaustively derived.

## 7. Tightness and adversarial examples

### 7.1 Static safety can be perfect while dynamic safety fails

Take two current histories with identical current margin `G_1=1` and the same code. With actions
`L,R`, let their robust Q table be

\[
\begin{array}{c|cc}
 &L&R\\\hline
\eta_+&1&-1\\
\eta_-&-1&1
\end{array}
\]

Both full-history values equal one. Every code-level deterministic action makes one history's value
minus one, so `rho_1=2`. The best uniform Q error is one, and
`rho_1=2 delta_1^*`; both the factor two and the one-step policy-loss bound are attained. Current
safe-label separation sees no problem.

### 7.2 The horizon sum can be attained at the closed endpoint

A deterministic two-step tree can realize the following Q profiles on the only non-singleton
fibers:

\[
\begin{array}{c|cc}
\text{one step left}&0&1\\\hline
\xi_A&0&1\\
\xi_B&1&-1
\end{array}
\qquad
\begin{array}{c|cc}
\text{two steps left}&0&1\\\hline
\eta_A&1&2\\
\eta_B&2&0.
\end{array}
\]

At each stage the minimax margin selector chooses action zero and has regret one. Starting from
`eta_A`, the full-history value is two while that code-policy value is zero. Thus the loss is
exactly `rho_1+rho_2=2`, and safety holds exactly at the closed buffer endpoint. The executable
fixture is in `tests/test_dynamic_theory.py`.

### 7.3 Q-uniformity is not necessary

For the table

\[
\begin{array}{c|cc}
 &L&R\\\hline
\eta_1&1&0\\
\eta_2&0.8&-1,
\end{array}
\]

action `L` is optimal everywhere, hence `rho_1=0`, but `delta_1^*=0.5` because the right-action
values vary by one. A loss that forces all action values to be uniform can preserve more information
than the safety decision requires.

### 7.4 Why selector assumptions matter

Let a singleton fiber have continuous action space `(0,1)` and `Q(a)=a`. Then
`V=sup_a Q(a)=1` and the infimum first-action regret is zero, but no action attains it. The finite
theorem's conclusion that zero regret produces an exact policy is false without attainment.
On general Borel spaces, even pointwise nonempty argmin sets need not yield a measurable selector
without regularity. With compact actions, appropriate semicontinuity, and a measurable-selection
theorem, an attained measurable version may be possible. Otherwise an `epsilon_s`-minimizer adds
`sum_s epsilon_s` to the safety buffer, and an exact endpoint claim is unavailable.

### 7.5 Randomization semantics

The Bellman value is strong/pathwise: the action realization and every successor branch must be
safe. Randomization cannot improve `max_a Q_s`, because an adversarial evaluation over the support
of a mixed action is no larger than its best deterministic support element. Expected violation or
chance safety is a different objective and is not covered by this theorem.

## 8. Relation to prior work and novelty boundary

This package should not be advertised as the first abstraction or partial-observation performance
bound.

- [Feedback refinement relations](https://arxiv.org/abs/1503.03715) already characterize when
  controllers synthesized on set-valued abstractions can be refined through static quantizers in
  the presence of nondeterminism and state-dependent admissible inputs. Zero optimal-margin regret
  is a task- and horizon-specific optimal-action property, not a replacement for that
  controller-refinement framework.
- [Girard's approximate-bisimulation safety synthesis](https://doi.org/10.1016/j.automatica.2012.02.037)
  predates this work and connects symbolic abstractions to safety-controller synthesis. The exact
  representation-policy viability object must be compared with that literature before any
  sign-specific compression claim.
- [Safe POMDP online planning via shielding](https://arxiv.org/abs/2309.10216) already restricts
  actions using belief-support winning regions to enforce almost-sure reach-avoid specifications.
  The recursion here can audit compression of a history or belief-support shield, but does not make
  shield synthesis new or solve infinite-horizon reach-avoid.
- Approximate state and state-action abstractions already provide value-loss results, including
  [approximate state abstraction](https://proceedings.mlr.press/v48/abel16.html) and
  [value-preserving state-action abstractions](https://proceedings.mlr.press/v108/abel20a.html).
  Q-profile uniformity and horizon-wise error propagation are established ideas.
- [Action-sufficient representations](https://proceedings.mlr.press/v162/huang22f.html) study
  decision-relevant representations under partial observability and structural assumptions.
- [Approximate information states for worst-case control](https://arxiv.org/abs/2301.05089)
  already treats finite-horizon, partially observed, non-stochastic control with a maximum
  instantaneous-cost objective, exact and approximate information states, cost and Hausdorff
  successor errors, Lipschitz value propagation, learnable surrogates, and a factor-two policy-loss
  bound. After the sign change from cost to margin, its maximum-in-time objective is directly
  adjacent to minimum-margin safety. The model-local corollary and factor two here are therefore
  specializations, not novelty.
- [Approximate information states for partially observed planning and RL](https://www.jmlr.org/papers/v23/20-1165.html)
  give learnable history compressions, approximate dynamic programs, and policy-loss bounds in the
  stochastic/expected-return setting. [Worst-case infinite-horizon control using partial
  observations](https://arxiv.org/abs/2303.16321) develops the non-stochastic discounted analogue.
- As of 13 August 2026, [control barrier--value functions under partial observability](https://arxiv.org/abs/2608.13819)
  use the same minimum-over-time margin payoff in continuous time and combine estimator
  uncertainty, conformal prediction, finite-horizon probabilistic safety, and a safety filter.
  Their probabilistic estimator-space semantics differ from the support-robust finite-history
  semantics here, but the work materially raises the empirical and theoretical bar.

The package is presently an analytical instrument, not a theory-contribution claim. Robust
optimal-margin regret, its coarsening law, and the path-dependent buffer are elementary finite
minimax/Bellman consequences. The exact sign-specific representation-policy value is useful, but
its relation to symbolic safety synthesis and belief-support games still requires a line-by-line
novelty audit. The Bellman induction, Hausdorff condition, midpoint argument, and factor two cannot
be claimed as standalone contributions. A headline claim requires a genuinely verified population
upper bound, a sign-specific compression theorem not subsumed by prior control abstraction, or a
learned method with a convincing matched safety--utility frontier.

## 9. Executable finite oracle

`src/latent_safety/metrics/dynamic.py` implements the finite theorem without external dependencies:

- exact robust `V_s` and `Q_s` recursion;
- exact local and global robust optimal-margin regrets;
- sign-specific first-action obstructions and common-safe actions;
- midpoint-optimal uniform latent-Q error;
- the minimax margin selector and arbitrary-code-policy composition audit;
- memoized set-valued representation Bellman values, all-layer recursive shortfalls, and guarded
  viable-history-family enumeration;
- the older guarded exhaustive representation-policy audit as an independent oracle;
- local path-sum and global sum buffers;
- a per-controlled-stage deterministic-coarsening check;
- a conditional Euclidean cover-bound calculator with metric, tolerance, status, and premise
  provenance recorded; and
- a complete-finite-domain verifier that derives the cover, Lipschitz, and sampled-Q premises and
  checks the resulting Q-coordinate oscillation bound on every enumerated pair within the
  operational latent radius.

The cover calculator is Euclidean only. A bound established under another latent norm must not be
passed to it as though the metrics were interchangeable.

`tests/test_dynamic_theory.py` additionally covers the common-safe/minimax-unsafe counterexample,
a non-minimax q-greedy policy across two stages, exact exhaustive representation-policy viability,
set-valued dynamic programming against independent policy enumeration on every nonempty subset of
randomized small games, strict and randomized coarsening monotonicity, future-layer necessity,
initial-fiber union coupling, viable-family heredity, and the recursive-shortfall/global-obstruction
equivalence. It also checks every displayed loss inequality, a direct Lipschitz/Hausdorff
calculation, terminal-only coarsening mismatch, Euclidean metric semantics, explicit radius
tolerance, provenance, and strict finite-real validation, including rejection of non-finite
derived arithmetic. The folded-interval regression checks the complete-finite-domain verifier
against a tight bound.

These are exact checks only because every history, action, margin, and successor is enumerated.
Running the same code on a finite dataset does not certify an unseen continuous system.

## 10. Integration recommendation

1. Keep `paper/sections/theory_v2_dynamic.tex` out of `main.tex` until an independent proof audit
   and a line-by-line comparison with approximate information-state and Q-abstraction theory are
   complete.
2. Add a finite-horizon oracle to the controlled benchmarks. Report static defect, `rho_s`,
   `kappa_s`, exact retained viable fraction where enumeration is feasible, `delta_s^*`, realized
   code-policy loss, and the local buffer as history length changes.
3. Extend the learning pipeline with a horizon-conditioned robust-Q/action-profile head. Compare
   minimax-regret training against full Q-profile prediction; the latter is sufficient but can be
   unnecessarily restrictive.
4. Treat any sampled estimate of `rho_s` as a lower witness unless a declared coverage or verified
   successor-set argument supplies an upper bound. A test-set mean is not the theorem premise.
5. Use a margin with declared physical units, preferably signed distance under a fixed metric.
   Every buffer rescales with `h`.
6. Test horizon scaling and the constructed accumulation example. A claimed improvement should
   reduce optimal-margin regret and safety obstruction at matched predictive/task utility, while
   preserving the result under
   observation-level and full-history controls.
7. For a stronger theorem, target one of two additions: a verified finite-data upper bound for
   `rho_s` under explicit covering/Lipschitz assumptions, or a stochastic-channel version based on
   overlap/deficiency rather than deterministic fibers.

## 11. Claim status

Internally proof-drafted and exhaustively checked on the included finite counterexamples:

- finite support-robust Bellman viability equivalence;
- robust optimal-margin regret and its zero-regret characterization;
- sign-specific first-action obstruction, including the selector-separation counterexample;
- per-controlled-stage monotonicity under deterministic coarsening;
- arbitrary-code-policy local/global loss bounds with the closed endpoint;
- exact exhaustive finite representation-policy viability semantics;
- exact set-valued representation Bellman recursion, recursive shortfall, viable-family heredity,
  and dynamic data processing under refinement at every controlled layer;
- best uniform latent-Q midpoint identity and factor-two bound; and
- the Lipschitz/Hausdorff sufficient condition.

Still open before submission:

- independent theorem review;
- novelty determination against approximate state/information-state abstraction;
- measurable or continuous selector extension;
- finite-data population upper certification;
- stochastic observation-channel formulation;
- infinite-horizon or time-uniform safety; and
- empirical evidence that a learned history encoder controls both optimal-margin regret and safety
  coverage at matched utility.
