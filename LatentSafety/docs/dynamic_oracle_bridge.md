# Dynamic oracle bridge to the controlled domains

**Status:** exhaustive registered finite nominal theorem check; not learned-model evidence and not
a population or deployment certificate.

## Question

Can the finite-history Bellman oracle separate three ideas on the repository's controlled cart and
inverted pendulum: privileged state, a memoryless rendered observation, and a short causal frame
history?

The executable answer is yes on a deliberately bounded problem. For each domain,
`build_controlled_dynamic_tree` starts from two hand-selected histories with the same current
rendered frame and different hidden velocity. It enumerates all `3^3` action sequences from each
root under the registered three-action grid and noise-free nominal dynamics. The layers contain
`(54, 18, 6, 2)` histories from zero to three remaining controls, or 80 path-specific histories.

The roots are algebraically reachable in one nominal step from a safe predecessor under action
zero. That is not a data-coverage claim. They were not sampled from the behavior-policy dataset;
in particular, the cart root speeds exceed the noise-free generator's bounded velocity envelope.
The Gaussian training noise has unbounded support but supplies neither exact coverage nor a robust
finite successor set.

## Registered problem and code equality

The builder fails closed unless the horizon is three and the dynamics, renderer, and action
parameters match the shipped configurations. The complete `DataConfig` fingerprints are:

- cart: `46997bc3fef0f3ca7f7789ff560ae90fac9732a6e9514919b8b8bf558d17d991`;
- pendulum: `575d545e0a5d5e0087f777d3a6bc44f6c6e1eb60ba697c71fd9bc9ef68caf3d9`.

The three representation controls are:

1. `privileged_kinematic_state_exact`: exact Python-float equality on position/velocity or
   angle/angular velocity. Renderer nuisance is omitted because it affects neither kinematic
   dynamics nor safety, although its phase does evolve.
2. `memoryless_rendered_frame_exact`: exact equality of the complete rendered frame tuple, without
   hashing. Scene geometry has already been quantized by the configured 32-by-32 raster.
3. `two_frame_rendered_history_exact`: exact equality of the previous and current complete frame
   tuples. It contains no privileged state and no previous action, matching the observation-only
   information available to the E1 history encoders.

The two current frame tuples are equal in each domain; the previous frames differ. “Exact” here
describes code equality and exhaustive branch enumeration, not symbolic arithmetic. Dynamics use
binary floating point and `sin`; inequalities are checked at the recorded tolerance `1e-12`.

## Quantities

For a representation `R_s`:

- `rho_s` is robust **optimal-margin regret** from forcing a common first action on a fiber before
  resuming full-history optimal control. It is not a safety defect.
- `kappa_s` is the sign-specific first-action obstruction over viable histories: zero exactly when
  each viable fiber has a common action with `Q_s >= 0`; positive means at least one viable fiber
  has no common safe first action.
- `delta_s*` is the best uniform representation-level Q approximation error on the finite layer.
- `loss_s` is the largest realized loss of the selected minimax-`rho` code policy.
- `B_s` is that policy's largest path-dependent loss bound; `global_s` is the looser cumulative
  sum of stagewise `rho` values.

The selected minimax-margin policy and the least-violating `kappa` policy are not policies that
maximize the number of retained histories. Separately, the finite audit checks the global sign
characterization by backward induction: an exact viability-preserving time-indexed code policy
exists for every full-history-viable history iff every stage has `kappa_s = 0`.

## Registered three-step output

Values are ordered by remaining controls `(1, 2, 3)`. “Retained” is the fraction of
full-history-viable histories kept viable by the selected minimax-margin policy, not a maximum over
policies.

| Domain / representation | `rho` | `kappa` | `delta*` | `loss` | `B / global` | selected retained | viability-preserving code policy exists |
|---|---:|---:|---:|---:|---:|---:|---:|
| cart / privileged state | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(1,1,1)` | yes |
| cart / memoryless frame | `(0,0,.035853)` | `(0,0,.017987)` | `(.048294,.035853,.041263)` | `(0,0,.035853)` | `(0,0,.035853)` | `(1,1,0)` | no |
| cart / two-frame history | `(0,0,0)` | `(0,0,0)` | `(.042809,.035853,0)` | `(0,0,0)` | `(0,0,0)` | `(1,1,1)` | yes |
| pendulum / privileged state | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(1,1,1)` | yes |
| pendulum / memoryless frame | `(0,0,.030506)` | `(0,0,.015351)` | `(.005133,0,.042645)` | `(0,0,.030506)` | `(0,0,.030506)` | `(1,1,.5)` | no |
| pendulum / two-frame history | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(0,0,0)` | `(1,1,1)` | yes |

All reported composition inequalities and `rho_s <= 2 delta_s*` checks pass at tolerance
`1e-12`. The nonzero `delta_s*` with zero `rho_s` in some lower cart layers is intentional: a Q
profile can vary inside a fiber while retaining a common optimal action.

At the cart roots, the full-history values are `0.028686` and `0.017865`; their safe first-action
sets are `{+1}` and `{-1}`. The memoryless minimax-margin action is `0`, giving policy values
`-0.007166` and `-0.017987`. Thus every common action is unsafe for at least one viable root, and
`kappa_3 = 0.017987`.

At the pendulum roots, the values are `0.015151` and `0.039440`; their safe first-action sets are
`{+1}` and `{-1, 0}`. The common minimax-margin action is `0`, giving `-0.015351` and `0.008933`.
Again the common-safe set is empty, with `kappa_3 = 0.015351`.

Privileged state and the two-frame observation history have `rho_s = kappa_s = 0` on every layer;
their constructed code policies match the optimal values at both registered roots. This is an
existence/result on the finite representation fibers, not evidence that a trained H4 encoder
learns the same distinction.

## Scope boundary

This is an exhaustive floating-point check for exactly the hand-selected histories, registered
finite action grid, three controls, declared code equality, and singleton successors of the
noise-free nominal transition. Every action branch in that object is present, so it is stronger
than checking sampled rollouts from those roots.

It does not cover the continuous state population, arbitrary histories, continuous actions, the
training configuration's nonzero Gaussian process noise, physical model mismatch, learned
encoders, or deployment. Extending the result requires a declared disturbance set or chance
criterion, a coverage argument over initial histories, and a separately validated learned-code
audit.

## Reproduce

From the `LatentSafety` root:

```bash
PYTHONPATH=src python3 scripts/run_dynamic_oracle.py --horizon 3
```

The generated and executed notebook `05_dynamic_regret_oracle.ipynb` presents the same calculation
with bounded tables and plots. The dependency-free module and regression tests are the source of
truth.
