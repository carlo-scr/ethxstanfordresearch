# Static theory extensions: stochastic separation and convex-action compression

This note records proof-grade extensions to the static theory. The ingredients—mutual
singularity, total-variation data processing, Pinsker's inequality, and Helly's theorem—are
classical. The contribution claimed here is only their careful specialization to the paper's
soundness/completeness and common-action audit semantics. These results should support the audit;
they are not standalone mathematical novelty claims.

All suprema and infima use extended-real conventions. The safe set is

\[
\mathcal C=\{x:h(x)\ge 0\},\qquad
\mathcal D=\{x:h(x)<0\},
\]

and all completeness margins satisfy `m >= 0`.

## 1. Countable-state stochastic encoders

Let `K` be finite or countable and let `(Z, Sigma)` be an arbitrary measurable output space. For
each `x in K`, let `P_x` be a probability measure on `(Z, Sigma)`. Equivalently, after equipping
`K` with the discrete sigma algebra, `x -> P_x` is a Markov kernel. No topology or density on `Z`
is assumed.

A measurable accepted-code set `S in Sigma` is **almost-surely sound** when

\[
P_x(S)=0\quad\text{for every }x\in\mathcal D.
\]

It is **closed `m`-complete** when `P_x(S)=1` for every `x` with `h(x) >= m`, and **open
`m`-complete** when the same condition holds for every `x` with `h(x) > m`.

Write `P perpendicular Q` when `P` and `Q` are mutually singular. Thus there is a measurable set
`A` with `P(A)=1` and `Q(A)=0`. Define

\[
\Gamma_P^{\rm st}
=\{h(x):h(x)\ge0,\ \exists x'\in\mathcal D
\text{ such that }P_x\not\perp P_{x'}\},
\]

and

\[
\varepsilon_{\rm st}(P)=\sup(\Gamma_P^{\rm st}\cup\{0\}).
\]

This is an almost-sure separation object. It is stronger than small Bayes error, small expected
loss, or high-probability separation.

### Theorem 1: exact stochastic separator and threshold

For either completeness convention, an almost-surely sound and complete measurable set exists if
and only if every required safe law is mutually singular to every unsafe law. Equivalently:

- closed `m`-completeness exists exactly when

  \[
  \Gamma_P^{\rm st}\cap[m,+\infty)=\varnothing;
  \]

- open `m`-completeness exists exactly when

  \[
  \Gamma_P^{\rm st}\cap(m,+\infty)=\varnothing.
  \]

The infimum closed and open thresholds both equal `epsilon_st(P)`. If this number is finite, open
completeness is attainable at the endpoint; closed completeness is attainable there exactly when
`epsilon_st(P)` is not an element of `Gamma_P^st`. If it is infinite, no finite margin is
attainable.

#### Proof

Fix a convention and let `H_m` be the corresponding required safe-state set.

Necessity is immediate. If one measurable `S` has `P_x(S)=1` for a required safe state and
`P_y(S)=0` for an unsafe state, then `P_x` and `P_y` are mutually singular.

For sufficiency, suppose every required-safe/unsafe pair is singular. For every pair `(x,y)` in
`H_m x D`, choose a measurable `S_{x,y}` satisfying

\[
P_x(S_{x,y})=1,\qquad P_y(S_{x,y})=0.
\]

Define

\[
S_x=\bigcap_{y\in\mathcal D}S_{x,y},\qquad
S=\bigcup_{x\in H_m}S_x.
\]

Use `Z` for an empty intersection and the empty set for an empty union. Countability is exactly
what keeps both operations measurable and preserves the probability-one/zero conclusions. For
each required `x`, a countable intersection of `P_x`-probability-one sets has probability one, so
`P_x(S)=1`. For a fixed unsafe `y`, every `S_x` is contained in `S_{x,y}` and therefore has
`P_y`-measure zero; the countable union `S` also has `P_y(S)=0`.

The ray-avoidance statements follow by substituting the closed or open definition of `H_m`.
The threshold and endpoint proof is then identical to the deterministic supremum argument: every
finite margin strictly above the supremum is closed-attainable, every margin strictly below it is
not even open-attainable, no conflict exceeds a finite supremum, and closed endpoint attainment
fails exactly when the supremum itself is a conflicting safe margin.

### Why countability is substantive

Pairwise singularity need not combine into one separator for an uncountable physical-state
family. Let `Z=[0,1]` with its Borel sigma algebra. For every `t in [0,1]`, introduce a required
safe state with law `delta_t`, and introduce one unsafe state with the uniform law `lambda`.
Every `delta_t` is singular to `lambda`. But a measurable set accepted with probability one by
every `delta_t` must contain every point of `[0,1]`, and hence has unsafe probability one under
`lambda`.

The theorem therefore cannot be quoted for a continuous state domain merely because every
individual safe/unsafe pair was shown singular. A replacement would need a common measurable
separator assumption or stronger dominated-family structure.

## 2. Deterministic reduction and stochastic data processing

If a deterministic representation `R` is embedded as `P_x=delta_{R(x)}` and `Sigma` separates
the realized codes (in particular, if `Z` is standard Borel), then

\[
\delta_{R(x)}\perp\delta_{R(y)}
\quad\Longleftrightarrow\quad R(x)\ne R(y).
\]

Consequently `Gamma_P^st=Gamma_R`, and all stochastic existence, threshold, and endpoint
statements reduce exactly to the deterministic theory.

The separation qualification matters on a genuinely arbitrary measurable space. Distinct points
that belong to exactly the same measurable sets induce identical Dirac probability laws. In that
case the law-level theorem correctly treats them as indistinguishable even though raw point
equality would distinguish them. The deterministic paper uses unrestricted latent subsets, which
corresponds to the full power-set sigma algebra and automatically separates points.

Now let `Kappa` be a Markov kernel from `(Z,Sigma)` to another measurable space `(W,T)`, and set

\[
Q_x=P_xKappa.
\]

With total variation normalized as

\[
\|P-Q\|_{\rm TV}=\sup_{A}|P(A)-Q(A)|\in[0,1],
\]

Markov kernels contract total variation:

\[
\|P_xKappa-P_yKappa\|_{\rm TV}
\le \|P_x-P_y\|_{\rm TV}.
\]

For probability laws, mutual singularity is equivalent to total variation one. Therefore a
non-singular pair cannot become singular after stochastic post-processing, and

\[
\Gamma_P^{\rm st}\subseteq\Gamma_Q^{\rm st},\qquad
\varepsilon_{\rm st}(P)\le\varepsilon_{\rm st}(Q).
\]

This is the stochastic analogue of exact deterministic data processing. It does not compare raw
finite radii, which still depend on a chosen geometry and scale.

## 3. Exact total-variation tradeoffs

### Proposition 2: safe/unsafe separator error

For any safe-state law `P` and unsafe-state law `Q`,

\[
\inf_{S\in\Sigma}\big(P(S^c)+Q(S)\big)
=1-\|P-Q\|_{\rm TV}.
\]

The two terms are respectively false rejection of the safe state and false acceptance of the
unsafe state under the decision rule `accept iff z in S`.

#### Proof

For every measurable `S`,

\[
P(S^c)+Q(S)=1-\{P(S)-Q(S)\}.
\]

Because `P-Q` has total mass zero, the supremum of the one-sided difference `P(S)-Q(S)` equals
the stated total variation. Taking the infimum proves the identity. This is an infimum identity;
no unspoken attainment assumption is needed.

In particular, zero total separator error is possible exactly for mutually singular laws.

### Proposition 3: randomized latent-action bridge

Let two physical states `x,y` have disjoint measurable safe-action sets `A_x,A_y`. Let `pi` be
any randomized latent decision kernel from the stochastic code to the action space. Write

\[
M_x=P_x\pi,\qquad M_y=P_y\pi
\]

for the induced action laws and

\[
v_x=M_x(A_x^c),\qquad v_y=M_y(A_y^c)
\]

for their violation probabilities. Then

\[
v_x+v_y\ge 1-\|P_x-P_y\|_{\rm TV}.
\]

Thus no stochastic latent policy—deterministic or randomized—can be simultaneously almost-surely
safe at both states unless their code laws are mutually singular.

#### Proof

Disjointness gives `A_x subset A_y^c`, so

\[
M_x(A_x)-M_y(A_x)
\ge (1-v_x)-v_y
=1-v_x-v_y.
\]

The left side is at most `TV(M_x,M_y)`, and the policy is itself a Markov kernel, so data
processing gives

\[
1-v_x-v_y
\le\|M_x-M_y\|_{\rm TV}
\le\|P_x-P_y\|_{\rm TV}.
\]

Rearrangement proves the result.

The disjoint-set premise is stronger and clearer than merely saying the actions preferred at the
two states differ. If safe sets overlap, the same event argument does not produce this bound
because one common action can be safe at both states.

## 4. Common-prior KL corollary

Suppose `P_x` and `P_y` have finite forward KL divergence to one declared prior `Pi`. Pinsker's
inequality and the total-variation triangle inequality give

\[
\|P_x-P_y\|_{\rm TV}
\le
\sqrt{D_{\rm KL}(P_x\|\Pi)/2}
+\sqrt{D_{\rm KL}(P_y\|\Pi)/2}.
\]

Therefore both the safe/unsafe separator error and the summed violation probability for disjoint
safe-action sets are at least

\[
\left[
1-\sqrt{D_{\rm KL}(P_x\|\Pi)/2}
 -\sqrt{D_{\rm KL}(P_y\|\Pi)/2}
\right]_+.
\]

This is pointwise in the two physical states. An average KL objective does not ensure the premise
at a rare worst-case pair. The lower bound can also be zero and hence vacuous; finite KL by itself
does not prohibit singular laws. For example, the uniform laws on the left and right halves of a
uniform common prior are mutually singular while each has KL `log 2` to that prior.

### Nondegenerate Gaussian consequence

Every Gaussian law on `R^d` with positive-definite covariance has a strictly positive Lebesgue
density. Any two such laws are mutually absolutely continuous and therefore not mutually singular.
Consequently, if there is at least one unsafe state and at least one required safe state, no exact
almost-surely sound and complete separator exists for nondegenerate Gaussian code laws.

The qualifier is essential. Degenerate Gaussians supported on disjoint affine subspaces can be
singular. More importantly for this project, the implemented AE and beta-VAE experiments audit a
deterministic code—specifically the posterior mean. Their exact-fiber and finite-radius results do
not claim to audit sampled posterior distributions. The Gaussian impossibility delineates a
different deployment semantics; it does not invalidate or promote the current empirical results.

## 5. Helly-compressed convex common actions

Let `p >= 1`, and let the action space `U` be a nonempty compact convex subset of `R^p`. Fix one deterministic
representation fiber `F`. For every `x in F`, assume the declared safe-action set

\[
A_x=\mathcal U_{\rm safe}(x)
\]

is nonempty, closed, convex, and contained in `U`. Individual nonemptiness keeps intrinsic state
infeasibility separate from representation-induced common-action conflict.

For nonempty `G subset F`, define the Euclidean geometric slack

\[
\beta_{\rm geo}(G)
=\min_{u\in\mathcal U}\sup_{x\in G}d_2(u,A_x).
\]

The minimum exists: each distance is continuous, their supremum is lower semicontinuous, and `U`
is compact. This number is the smallest uniform action-space inflation needed to create a common
point. It is norm and action-coordinate relative.

### Theorem 4: exact Helly compression

For a nonempty finite or infinite fiber,

\[
\boxed{
\beta_{\rm geo}(F)
=\sup_{\substack{G\subseteq F\\1\le |G|\le p+1}}
\beta_{\rm geo}(G).}
\]

Furthermore:

1. `intersection_{x in F} A_x` is nonempty if and only if every subfamily of at most `p+1`
   safe-action sets intersects.
2. The fiber is action-conflicted if and only if `beta_geo(F)>0`.
3. If `F` is finite, the displayed supremum is a maximum. Some at-most-`p+1` state subfiber
   therefore preserves the exact full-fiber slack.
4. The cardinality `p+1` is sharp.

If `q=dim aff(U)<p`, pass to affine coordinates and replace every `p+1` witness bound by the
sharper `q+1`. The sharpness statement is relative to a full-dimensional actuator set.

#### Proof

For `r >= 0`, define

\[
B_x(r)=\{u\in\mathcal U:d_2(u,A_x)\le r\}.
\]

Each `B_x(r)` is compact and convex. Also,

\[
\beta_{\rm geo}(G)
=\min\{r\ge0:\bigcap_{x\in G}B_x(r)\ne\varnothing\}.
\]

Let `b` be the supremum of `beta_geo(G)` over subfibers with at most `p+1` states. Since enlarging
the state set can only increase the supremum inside the minimization,

\[
b\le\beta_{\rm geo}(F).
\]

Now fix `r>b`. Every at-most-`p+1` subfamily of `{B_x(r):x in F}` intersects. By finite Helly,
every finite subfamily intersects. Because the sets are compact, the finite-intersection property
implies that the whole family intersects. Thus `beta_geo(F) <= r`. Letting `r` decrease to `b`
proves equality.

At `r=0`, `B_x(0)=A_x`. The same Helly plus compact finite-intersection argument proves the exact
common-action equivalence. Attainment of the minimum and closedness show that beta zero is
equivalent to a nonempty common set. When `F` is finite there are only finitely many subfibers of
size at most `p+1`, so the supremum is a maximum.

### Sharpness of `p+1`

Let

\[
B=\{u\in\mathbb R^p:\|u\|_2\le\sqrt p\}.
\]

For `i=1,...,p`, set

\[
A_i=B\cap\{u:u_i\ge1\},
\]

and set

\[
A_{p+1}=B\cap\left\{u:\sum_{i=1}^p u_i\le p-1\right\}.
\]

All sets are nonempty compact convex subsets of `B`. Their full intersection is empty because the
first `p` inequalities force the coordinate sum to be at least `p`. If `A_{p+1}` is omitted, the
all-ones vector is feasible. If `A_j` is omitted, the vector with coordinate `j` equal to zero and
all other coordinates equal to one is feasible. Thus every `p`-subfamily intersects, and no
universal witness bound smaller than `p+1` is possible.

## 6. Counterexamples to stronger action variants

### Convexity cannot be removed

In one action dimension, the closed nonconvex sets

\[
\{0,1\},\qquad\{1,2\},\qquad\{0,2\}
\]

intersect pairwise but have empty triple intersection. The dimension-one Helly witness size two
therefore fails without convexity.

### Empty intersection need not imply positive distance without compactness

In `R^2`, let

\[
A_1=\{(t,0):t\ge0\},\qquad
A_2=\{(t,s):t>0,\ s\ge1/t\}.
\]

The second set is the closed convex epigraph of the extended-real convex function `1/t` on the
positive half-line. The sets are disjoint, but points `(n,0)` and `(n,1/n)` show their distance is
zero. Hence their geometric slack is zero and is not attained. Compactness or a separate
attainment/uniform-separation assumption is needed for “conflict iff positive slack.”

### Infinite families need a compact anchor

The closed convex sets `A_n=[n,+infinity)` in `R` have nonempty every finite intersection, but
their full intersection is empty. Thus finite Helly alone cannot pass from all finite subfamilies
to an arbitrary infinite fiber. Compact `U` supplies the missing finite-intersection argument.

### The slack is not a physical violation without calibration

Rescaling action coordinates rescales `beta_geo`. A paper may call it a geometric relaxation or
action-space distance, but not a barrier-margin loss, failure probability, or actuator effort
without a declared norm and conversion. Likewise, exact convex safe sets and exact optimization
are premises. A sampled state set or floating solver residual is only finite-reference evidence
unless a verification argument closes the population and numerical gaps.

### No global policy regularity follows

The theorem is pointwise in one representation fiber. It neither proves that a common-action
selector across codes is measurable or continuous nor supplies recursive feasibility, forward
invariance, learned-model error composition, or long-horizon safety.

## 7. Audit and manuscript implications

The stochastic results add an exact law-level boundary:

- deterministic fibers are the Dirac special case;
- stochastic post-processing cannot create almost-sure separability;
- total variation gives the exact minimum total two-state separator error;
- disjoint safe-action sets turn the same total variation into a randomized-policy violation lower
  bound; and
- nondegenerate Gaussian laws cannot exactly separate any required-safe/unsafe pair.

The finite executable oracle now mirrors these premises rather than treating approximate row sums
as exact laws. It evaluates the exact rational numbers represented by binary64 inputs, applies a
deterministic one-pivot correction only when the row is within the declared normalization tolerance
and positive support can be preserved, and otherwise fails closed. Markov composition and TV
contraction are checked on the unprojected exact rational product; the normalization tolerance is
never reused as an arithmetic tolerance. The public binary64 postprocessed kernel is only a
support-preserving exact-sum projection. The executed theory notebook exhausts 486 closed and 486
open separator cases, 36 TV/error pairs, 81 randomized-policy action-tradeoff cases, 81 one-hot
reductions, and 243 Markov postprocessings, plus a common-prior KL/Pinsker fixture. A Bayes half-error
that alone underflows is reported as unavailable with an explicit positivity/underflow flag rather
than being exposed as a false zero.

The convex-action result gives a continuous-action audit route that does not rely on a coarse
action grid when exact convex safe sets are available. In a finite verified fiber, exhaustive or
separation-oracle-backed optimization can return at most `p+1` state identifiers as a complete
qualitative witness and as an exact geometric-slack witness. For an infinite fiber, existence of a
small witness does not by itself provide an algorithm for finding it. For affine barrier
constraints and compact convex actuator limits, a verified separation oracle could supply the
missing search premise. No such implementation or population certificate is claimed by the
current experiments.

Registered manuscript labels:

- `thm:stochastic-separation`;
- `prop:stochastic-tv`;
- `thm:helly-action`.

The corresponding claim records deliberately remain `theorem_needs_proof` pending independent
proof and novelty review. The paper should describe the Helly statement as a classical theorem
specialized to fiber audits and the stochastic statements as measure-theoretic extensions, not as
the first use of singularity, total variation, Pinsker, or Helly theory.
