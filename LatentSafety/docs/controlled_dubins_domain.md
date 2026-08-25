# Controlled Dubins-navigation pixel domain

Status: implemented synthetic environment and CPU integration smoke; no pilot or confirmatory result.

This is the frozen third controlled domain for preconfirmation and, subject to every registered
gate, the 192-run fresh-seed confirmation. It is a local deterministic fixture rather than a Gym
wrapper or a claim of realistic autonomous-driving fidelity. The base configuration is
`configs/e1_world_models/torch_dubins_pilot.toml`.

## State, action, and dynamics

The physical state is `(x, y, heading, obstacle_phase, obstacle_direction, nuisance_phase)`. The
vehicle has constant forward speed `0.55`, step size `0.10`, and steering-rate gain `1.40`. Its
finite controls are `{-1, 0, 1}`. One circular obstacle moves on a fixed orbit in either direction;
the direction is sampled per trajectory and is not marked in a single frame. The nominal
known-dynamics profile sets process noise to zero and holds one action constant for six steps.

The versioned renderer and dynamics constants are:

| Quantity | Frozen value |
|---|---:|
| Safe arena half-width | `1.20` |
| Render half-width | `1.60` |
| Obstacle orbit radius | `0.42 x safe half-width` |
| Obstacle radius | `0.16 x safe half-width` |
| Vehicle collision radius | `0.055 x safe half-width` |
| Obstacle phase increment | `0.21 rad/step` with direction in `{-1, 1}` |
| Profile horizon | `6` |

The signed margin is the minimum of square-arena clearance and vehicle-to-moving-disc collision
clearance. Thus `h >= 0` means both constraints hold. Mechanical clipping occurs only at 95% of the
larger render extent, outside the declared safe arena.

## Observation and partial observability

The top-down RGB frame displays arena walls, vehicle pose and heading, and the obstacle's current
position. It deliberately omits the obstacle direction/velocity. Two states with identical vehicle
pose, obstacle phase, and nuisance phase but opposite obstacle direction therefore render the same
current frame and can have different future profiles. A causal history exposes the obstacle motion.
The controlled observation-oracle feature map retains only the visible pose and obstacle positions;
the privileged state control additionally includes obstacle velocity.

The generator assigns whole trajectories to train, ordinary validation, calibration, and final
test splits before samples are constructed. It mixes nominal starts, wall challenges, and
moving-obstacle challenges. All state, profile, scenario, split, renderer, and dynamics fields enter
the canonical dataset-manifest hash.

## Executable checks

The dependency-free domain tests cover deterministic generation, geometry, manifest identity,
render bounds, the opposite-motion single-frame alias, and history separation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_dubins_learning -v
```

The real model-path smoke uses the research environment and exercises rendered dataset creation,
one forward/backward epoch, utility-only checkpoint selection, every split, audit records, and
latent/pixel rollouts:

```bash
PYTHONPATH=src python scripts/run_e1_torch.py \
  --config configs/e1_world_models/torch_dubins_pilot.toml \
  --smoke --device cpu --output /tmp/latent-safety-dubins-smoke
```

The 2026-08-22 workspace run completed successfully. This engineering check does not establish the
four-frame information gate, predicted-profile accuracy, matched-utility eligibility, robustness,
or any learned-representation safety effect. Those remain prospective and must use fresh output
directories and immutable manifests.
