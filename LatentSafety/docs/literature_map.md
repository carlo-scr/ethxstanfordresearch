# Literature map and novelty boundary

Last primary-source sweep: **2026-08-22**. “Latent safety” is not one settled field; papers differ in
representation, safety semantics, and assurance strength. Expected constrained return, per-step
calibration, and persistent forward invariance are not interchangeable.

## Taxonomy

| Dimension | Variants that must not be conflated |
|---|---|
| Representation | state/pixel latent; memoryless/history/belief; reward/cost split; uncertainty-augmented; constraint-conditioned |
| Safety goal | expected CMDP cost; state-wise constraint; invariance/viability; HJ reachability; OOD/chance risk |
| Assurance | empirical; model-conditional; marginal statistical; time-uniform; formal state-space transfer |
| Failure source | sensor aliasing; encoder fibers; non-Markov latent; dynamics error; constraint error; OOD; solver approximation |

## Closest work

| Work | Verified contribution | Consequence for this project |
|---|---|---|
| [Lutkus et al., latent representations with provable stability and safety](https://arxiv.org/html/2505.23210) | Defines `S_z=(E(S_x^c))^c`, explicitly notes safe/unsafe shared codes cause strict under-approximation, and transfers latent barrier guarantees under conjugacy error | Current proposal C1 is a quantitative corollary, not a safe headline novelty |
| [FCSRL, ICML 2024](https://proceedings.mlr.press/v235/cen24b.html) | Learns feasibility-consistent safety-aware representations on vector and image tasks | Mandatory feasibility-aware baseline; a new auxiliary safety loss is not sufficient novelty |
| [SDQC, ICML 2025](https://proceedings.mlr.press/v267/yang25aq.html) | Learns reward/cost-decoupled Q-supervised representations for safe offline RL and policy preservation | Compare directly; distinguish persistent invariance and fiber-wise safe-action preservation |
| [CVRL-BM, TIP 2025](https://doi.org/10.1109/TIP.2024.3523798) | Safety-bisimulation objective for compact visual representations | “Existing objectives ignore safety” is false; position operational fiber auditing instead |
| [Safety Representations for Safer Policy Learning, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html) | Learns explicit safety features to reduce violations | Append/predict-safety baselines are mandatory |
| [PIGDreamer, ICML 2025](https://proceedings.mlr.press/v267/huang25ai.html) | Privileged representation alignment for safe partially observable RL | Closest training strategy for a history/belief pivot |
| [Latent Safety Filters, RSS 2025](https://www.roboticsproceedings.org/rss21/p113.html) | HJ reachability in visual world-model latents for difficult hazards | Downstream reachability baseline; do not claim first latent safety filtering |
| [UNISafe, CoRL 2025](https://proceedings.mlr.press/v305/seo25a.html) | Conformal OOD threshold and uncertainty-augmented latent reachability | Separate encoder ambiguity from epistemic/OOD uncertainty |
| [How to Train Your Latent CBF, L4DC 2026](https://proceedings.mlr.press/v331/nakamura26a.html) | Shows classifier margins and safety-policy-only data are poor for smooth CBF filtering; proposes smooth margins and mixed data | Strong baseline for boundary-aware objectives and downstream CBF evaluation |
| [Feedback refinement relations, TAC 2017](https://arxiv.org/abs/1503.03715) | Characterizes controller refinement through a static quantizer for systems with state-dependent admissible inputs | Common-action preservation is established control-abstraction structure, not a standalone novelty claim |
| [Controller synthesis via approximate bisimulation, Automatica 2012](https://doi.org/10.1016/j.automatica.2012.02.037) | Synthesizes safety and reachability controllers through symbolic abstractions related by approximate bisimulation | Exact sign-specific representation viability must be compared with older symbolic-control synthesis, not presented as automatically new |
| [Safe POMDP planning via shielding, ICRA 2024](https://arxiv.org/abs/2309.10216) | Restricts actions using winning regions over POMDP belief supports to enforce almost-sure reach-avoid specifications | Robust action choice under partial-observation support predates this project; compare semantics and scalability |
| [Approximate information states for worst-case control, TAC 2024](https://arxiv.org/abs/2301.05089) | Treats finite-horizon partial observation with maximum instantaneous or terminal cost; approximate information states use cost and Hausdorff successor errors and yield a factor-two policy-loss bound | Direct collision with a generic dynamic/Hausdorff/factor-two theorem; the sign-flipped maximum-cost objective is adjacent to minimum-margin safety |
| [Approximate information states, JMLR 2022](https://www.jmlr.org/papers/v23/20-1165.html) | Gives learnable history compressions, approximate dynamic programs, and bounded policy loss in partially observed systems | History compression plus value-loss propagation is established; differentiate robust safety semantics and computable fiber certificates |
| [Worst-case control using partial observations](https://arxiv.org/abs/2303.16321) | Defines exact and approximate information states for worst-case partially observed control and derives performance-loss bounds | A generic approximate-information-state or horizon-loss theorem is not sufficient novelty; compare assumptions and error recursion line by line |
| [Partial-observation control barrier--value functions](https://arxiv.org/abs/2608.13819) | Uses the same minimum-over-time margin payoff, conformal estimator-error bounds, estimator-space CBVF analysis, a finite-horizon probabilistic safety guarantee, and a QP filter | Very recent adjacent work; distinguish support-robust history-fiber semantics from probabilistic estimator-space guarantees and compare empirically where possible |

## Representation and world-model foundations

- [PlaNet, ICML 2019](https://proceedings.mlr.press/v97/hafner19a.html): stochastic latent dynamics
  and planning from pixels.
- [DeepMDP, ICML 2019](https://proceedings.mlr.press/v97/gelada19a.html): reward/transition
  prediction, bisimulation, and downstream value accuracy.
- [DBC, ICLR 2021](https://openreview.net/forum?id=kmaPnFKCuq): reconstruction-free
  bisimulation representations robust to distractors.
- [LOMPO, L4DC 2021](https://proceedings.mlr.press/v144/rafailov21a.html): latent uncertainty
  for offline image-based RL.
- [Action-Sufficient State Representations, ICML 2022](https://proceedings.mlr.press/v162/huang22f.html):
  minimal decision-relevant structure under causal assumptions.
- [Value-Preserving State-Action Abstractions, AISTATS 2020](https://proceedings.mlr.press/v108/abel20a.html):
  necessary and sufficient conditions for abstractions/options to retain near-optimal behavior.
- [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html): predicts pretrained
  visual features for zero-shot planning.

## Safe world models and pixel RL

- [LAMBDA, ICLR 2022](https://openreview.net/forum?id=PRZoSmCinhf): Bayesian world models with
  optimistic reward and pessimistic constraints.
- [Safe SLAC, ICLR 2023](https://openreview.net/forum?id=b39dQt_uffW): constrained POMDP and
  stochastic latent safety critics.
- [SafeDreamer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html):
  Dreamer-style safe imagination and planning.
- [State-wise safe RL with pixels, L4DC 2024](https://proceedings.mlr.press/v242/zhan24a.html):
  visual latent dynamics, barrier-like function, and policy.
- [ActSafe, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/495b0bb32959448b8b5b21ae9c0e9e34-Abstract-Conference.html):
  pessimistic safe policies with optimistic active exploration.

## Reachability, calibration, and current latent certificates

- [DeepReach, ICRA 2021](https://arxiv.org/abs/2011.02082): neural high-dimensional HJ
  reachability.
- [Reachability Constrained RL, ICML 2022](https://proceedings.mlr.press/v162/yu22d.html):
  persistent safety via reachability rather than discounted cost.
- [LS3, CoRL 2021/2022](https://proceedings.mlr.press/v164/wilcox22a.html): learned visual latent
  safe sets from successful trajectories.
- [Calibrated safety chances, L4DC 2024](https://proceedings.mlr.press/v242/mao24c.html): conformal
  calibration for image-conditioned safety prediction.
- [Safety certification with CBFs and world models](https://arxiv.org/abs/2507.13871): latent
  control barrier certificates with limited labels.
- [From Points to Sets](https://arxiv.org/abs/2604.05799): zonotope propagation for set-valued
  latent safety evaluation.
- [Pixels to Proofs](https://arxiv.org/abs/2606.15594): conformalized latent constraints and robust
  MPC.
- [Conformal Koopman reachability, L4DC 2026](https://proceedings.mlr.press/v331/nath26a.html):
  conformally inflated learned reach sets with neural verification.
- [Designing Latent Safety Filters Using Pre-Trained Vision Models](https://arxiv.org/abs/2509.14758):
  frozen/fine-tuned vision backbones for constraint, HJ value, and latent dynamics prediction.
- [AnySafe](https://arxiv.org/abs/2509.19555): runtime adaptation to image-parameterized safety
  constraints.

## Open gaps supported by the survey

1. A learnable and finitely auditable characterization of compressed history representations for
   support-robust minimum-margin safety, beyond existing exact/approximate information states and
   belief-support shields.
2. An operational audit showing whether learned pixel/history representations preserve common
   viable actions and retained viability at matched task utility, with robust optimal-margin
   regret, observation-level, and full-history controls reported separately.
3. Finite-data fiber certification with an actual witness/search guarantee rather than only a
   statistical wrapper around an unavailable oracle.
4. End-to-end composition of representation, dynamics, constraint, solver, and feedback errors.
5. Time-uniform calibration under the closed-loop distribution.
6. One representation supporting a class of safety specifications with a nontrivial compression
   theorem.
7. Common evaluation reporting false-safe rate, retained viable volume, conservatism/intervention,
   and task utility under controlled aliasing and OOD shift.

## Recommended primary positioning

**Auditable safety-sufficient world models under partial observability:** learn a compressed
history representation and measure first-action obstruction, retained viability, robust
optimal-margin regret, conservatism, and matched world-model utility without conflating them. Use
current margins, safe-action correspondences, and controlled transitions as training signals, but
compare directly with approximate information-state methods.
The original static defect and memoryless-aliasing example are special cases and empirical
instruments, not the sole novelty claim.

The exact-fiber lemmas in `docs/theory_v1.md` and the dynamic finite-game lemmas in
`docs/theory_v2_dynamic.md` are foundations and debugging tools. Approximate dynamic programming,
Hausdorff successor conditions, and factor-two policy loss already appear in the information-state
literature. Submission-level novelty must therefore come from a finite-data certificate with a
computable search guarantee, a stronger stochastic/compression theorem, or a substantive learned
representation result; endpoint bookkeeping and common-action intersection are not enough.
