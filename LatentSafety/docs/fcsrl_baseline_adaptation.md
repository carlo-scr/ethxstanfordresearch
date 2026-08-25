# Frozen FCSRL feasibility-loss adaptation

Status: executable prospective protocol; no learned-result evidence. The dependency-free 63-atom symlog
projection, categorical mean, ten-transition recursion, and ending-transition mask live in
`src/latent_safety/learning/fcsrl_protocol.py`. `src/latent_safety/learning/fcsrl_training.py` now
implements the frozen posterior-mean EMA encoder, deterministic trajectory-local windows and batch
plans, vectorized targets, categorical loss, and one combined optimizer-step/EMA-update primitive.
The `fcsrl_feasibility_loss_adaptation` arm is now wired into the main trainer with explicit head
width, exact checkpoint/resume state, and a deterministic per-run regression fixture. The
validation-only post-fit physical audit, exact 288-job planner/task handoff, matched-radius reducer,
authenticated aggregator, and five-weight selector now cover its preconfirmation path. The complete
production grid has not run. Consequently, this is an executable engineering arm, not learned-model
or baseline-result evidence.

This document freezes the same-backbone offline comparison described in the manuscript. It is an
adaptation of the feasibility loss in FCSRL, not a reproduction of the complete online FCSRL agent.
The source crosswalk uses the official repository at commit
[`d882b91054d52c7a63a18a21f7db67bbbed82b63`](https://github.com/czp16/FCSRL/tree/d882b91054d52c7a63a18a21f7db67bbbed82b63),
retrieved on 2026-08-22, and the ICML 2024 paper
([Cen et al., 2024](https://proceedings.mlr.press/v235/cen24b.html)).

## Comparison scope

The arm must keep fixed relative to `none`:

- trajectory data, four-way split, preprocessing, and behavior-policy actions;
- encoder/history frontend, latent width, transition and decoder architecture;
- reconstruction, transition, decoder, and beta-VAE KL losses and their weights;
- optimizer, batch/sequence exposure, training steps, checkpoint rule, and evaluation budget.

The only added trainable module is the categorical feasibility head. A stop-gradient EMA copy of
the encoder is added only to form targets. The project retains its existing latent transition MSE;
it does **not** relabel that loss as FCSRL dynamics consistency. The official implementation instead
uses a projected, post-projected predicted latent and an EMA target latent with a masked cosine
objective (`dyn_coef = 0.5`). No FCSRL actor, reward critic, cost critic, policy target, Lagrangian,
or online data collection is transplanted.

## Categorical head

Let

```text
symlog(y) = sign(y) * log(1 + abs(y))
symexp(v) = sign(v) * (exp(abs(v)) - 1).
```

The head emits logits over 63 equally spaced atoms `b_j` in symlog coordinates on `[-2, 4]`.
For probabilities `p_theta,j(z)`, its scalar mean is

```text
F_theta(z) = symexp(sum_j p_theta,j(z) * b_j).
```

For a scalar target `Y`, clip `symlog(Y)` to the support and distribute unit mass linearly between
the two adjacent atoms (one atom at an endpoint). The feasibility loss is the negative log
likelihood/cross-entropy of that projected target. This matches the official `DiscDist` semantics;
it is not scalar regression against the categorical mean.

## Frozen target recursion

Sample a contiguous 10-transition window from the behavior data. Let

- `c_t = 1{h(x_t) < 0}` be the binary physical violation label;
- `d_t` be a true environment termination flag;
- `u_t` be a time-limit truncation flag;
- `bar_g` be the EMA target encoder; and
- `bar_F_theta(H) = F_theta(bar_g(H))`, evaluated with gradients disabled. The head parameters are
  online parameters viewed under stop-gradient; only the encoder is EMA.

For each window, compute

```text
Y_9 = bar_F_theta(H_10)
B_(t+1) = (1 - u_t) * Y_(t+1) + u_t * bar_F_theta(H_(t+1))
Y_t = max(c_t, 0.9 * (1 - d_t) * B_(t+1)),  t = 8, ..., 0.
```

Train the head on the first `K = 4` rollout latents against the corresponding projected targets. In
the executable component, these are `z_0 = g(H_0)` and `z_1, z_2, z_3` produced recursively by the
existing same-backbone transition and the first three behavior actions. Both AE and beta-VAE arms
use the posterior mean for this auxiliary path; the unchanged base objective retains its registered
stochastic/KL semantics.
The ending transition itself has mask one; every position strictly after the first termination or
truncation has mask zero. Targets, masks, categorical projections, and EMA outputs are serialized
for at least one deterministic regression fixture before this arm may enter a sweep.

The target encoder is initialized from the online encoder and updated after each optimizer step by

```text
bar_theta_g <- 0.99 * bar_theta_g + 0.01 * theta_g.
```

The `0.01` source mixing rate is the official TD3 representation update
`soft_update(fixed_encoder, encoder, soft_update_tau * 0.2)` with `soft_update_tau = 0.05`.

For the current controlled generators, safety violation does not itself end an episode. Therefore
the window adapter sets true-termination flags to false and marks only the final generated horizon
transition as a time-limit truncation. It emits one window per original transition, matching the
ordinary arm's base-transition and optimizer-step exposure. Windows near the horizon repeat the
terminal history and a registered padding action only at positions after truncation, which the exact
mask excludes. Histories never cross trajectory or split boundaries. A SHA-256-keyed ordering over
`(seed, epoch, window index)` makes batch plans independent of ambient random state.

The categorical-head hidden width is frozen explicitly as `objective.fcsrl_head_hidden_dim = 64` in
each domain's base learning config and remains visible in resolved configs and checkpoints. The
training path supplies the unchanged world-model base loss and requires the optimizer to contain
every trainable backbone and head parameter; it applies the EMA update only after a successful
optimizer step.

Every FCSRL run writes `fcsrl_regression_fixture.json` before optimization. It records one declared
window identity, violations, termination/truncation flags, all ten targets and masks, all ten
63-atom projections, EMA latents, encoder-state hashes, and the checkpoint epoch from which the
fixture was formed. Checkpoints include the head, EMA encoder, optimizer, scaler, Python/Torch/CUDA
RNG state, and train-loader generator state. Resume accepts only the identical resolved config and
data manifest, except that the total epoch budget may increase; tests compare resumed and
uninterrupted final states exactly.

## Weight selection and confirmation freeze

The added loss weight is selected without final-test labels from
`{0.001, 0.01, 0.1, 1, 10}`. A candidate is eligible only when both validation reconstruction
error and validation maximum-predeclared-horizon rollout error are at most 5% worse than the paired
`none` arm within domain/model-family/history and the common coverage, no-empty-trajectory, and
matched-neighborhood-mass gates pass. Among eligible candidates, select the lowest primary validation
safety endpoint, then the lower maximum paired-none-normalized reconstruction/rollout ratio, then
smaller weight. If no candidate is eligible, the adaptation fails that stratum rather than changing
the rule. Freeze the selected weight before the eight-seed confirmatory run and open final-test labels
only once.

## Interpretation and non-core baselines

The target is a discounted behavior-policy future-violation score. It is neither the manuscript's
robust counterfactual action profile nor a Bellman-regret certificate. Any comparison must use the
label “FCSRL feasibility-loss adaptation.”

CVRL-BM introduces a safety-bisimulation representation metric and changes the representation
objective/architecture, so it is a secondary architecture-changing sensitivity rather than a core
same-backbone arm. Full FCSRL, SRPL, and SDQC comparisons belong in downstream E5 only if the same
online constrained-RL setting, interaction budget, and evaluation protocol can be matched.

## Official-source crosswalk

| Frozen choice | Official source at the pinned commit | Adaptation decision |
|---|---|---|
| 63 atoms; support `[-2, 4]` | [`hyper_params/TD3Repr_Lag.yaml`](https://github.com/czp16/FCSRL/blob/d882b91054d52c7a63a18a21f7db67bbbed82b63/hyper_params/TD3Repr_Lag.yaml) | Retained exactly |
| feasibility discount `0.9`; return length 10; unroll length 4 | same YAML and [`td3_lag_repr.py`](https://github.com/czp16/FCSRL/blob/d882b91054d52c7a63a18a21f7db67bbbed82b63/fcsrl/agent/td3_lag_repr.py) | Retained exactly |
| symlog/symexp categorical mean and interpolated log probability | [`solver.py`](https://github.com/czp16/FCSRL/blob/d882b91054d52c7a63a18a21f7db67bbbed82b63/fcsrl/utils/solver.py) | Retained exactly |
| target-encoder source mixing `0.05 * 0.2 = 0.01` | [`td3_lag_repr.py`](https://github.com/czp16/FCSRL/blob/d882b91054d52c7a63a18a21f7db67bbbed82b63/fcsrl/agent/td3_lag_repr.py) and [`misc.py`](https://github.com/czp16/FCSRL/blob/d882b91054d52c7a63a18a21f7db67bbbed82b63/fcsrl/utils/misc.py) | Retained exactly |
| encoder objective `0.5 * dynamics + 1.0 * feasibility` | `td3_lag_repr.py` | Dynamics term not transplanted; feasibility weight is validation-selected under the frozen same-backbone rule |
| TD3 actor/critics and Lagrangian | `td3_lag_repr.py` | Excluded; this paper's comparison is offline representation learning |
