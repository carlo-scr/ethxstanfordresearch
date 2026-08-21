# Synthetic controlled-pendulum video domain

Status: executable benchmark fixture; not paper evidence and not a Gym/Gymnasium wrapper.

The `controlled_pendulum_video` task is a second known-dynamics test bed for the E1 world-model
pipeline. Its state is `x_t = (theta_t, omega_t, phi_t)`, where zero angle is upright and `phi_t`
controls a trajectory-specific visual nuisance. The image shows the rod angle but not angular
velocity. This creates an intentional partial-observability test: one frame can alias states with
opposite velocities, while a short history can carry directional information.

The discrete-time fixture uses

```text
omega_(t+1) = damping * omega_t
                + dt * (sin(theta_t) + control_gain * action_t + noise_t)
theta_(t+1) = wrap_to_[-pi,pi)(theta_t + dt * omega_(t+1)).
```

The signed state margin is exactly

```text
h(x_t) = angle_limit - abs(wrap_to_[-pi,pi)(theta_t)).
```

For each configured discrete action, the action-profile label is the minimum of this margin over
a finite deterministic rollout that holds that action constant and sets process noise to zero.
The oracle uses the true angular velocity. It is therefore a reproducible known-model supervision
target, but it is not an infinite-horizon or arbitrary-policy safety certificate.

The dataset additionally emits `observation_features`, a controlled observation-oracle feature
map. It contains the sine/cosine direction of the pendulum for every frame in the deployed padded
history. It excludes velocity and visual nuisance and is not a replacement for evaluating the
learned pixel representation. Its purpose is to measure how much of an observed safety conflict
already exists before learned compression.

For a separate privileged ground-truth control, each sample emits `state_features`: pendulum
direction as sine/cosine plus angular velocity. (The cart counterpart is position plus velocity.)
This vector omits nuisance but includes hidden velocity, so it must not be presented as deployable.

## Commands

Validate the paper-scale plan without importing PyTorch:

```bash
python scripts/run_e1_torch.py \
  --config configs/e1_world_models/torch_pendulum_pilot.toml \
  --dry-run
```

Run the one-epoch CPU engineering check (use a fresh output path):

```bash
python scripts/run_e1_torch.py \
  --config configs/e1_world_models/torch_pendulum_pilot.toml \
  --smoke --output runs/e1_world_models/pendulum_smoke_<run_id>
```

Materialize the 48-task pendulum half of the two-domain decision pilot and submit its Slurm array:

```bash
python scripts/plan_e1_sweep.py \
  --grid configs/e1_world_models/decisive_pendulum_grid.toml \
  --output runs/plans/e1_decisive_pendulum.json
sbatch experiments/e1_world_models/slurm_decisive_pendulum_array.sh
```

The 144-task `pilot_pendulum_grid.toml` is a recurrence/dimension expansion to run only if the
96-task cross-domain decision gate passes.

Only whole trajectories cross the train/validation/calibration/test splitter. The dataset manifest
records the task, dynamics, rendering, state fields, label meaning, noise convention, split IDs,
and complete generated trajectory data under a canonical SHA-256 digest.

## Intended use and limitations

- Use it as a controlled cross-domain check for hidden-velocity aliasing and safety supervision.
- Do not describe it as a standard pendulum benchmark or infer real-system robustness from it.
- The renderer and dynamics are intentionally simple; conclusions need stronger visual domains
  and downstream closed-loop experiments before supporting an ICML-level empirical claim.
- Test-set audits remain descriptive until model/arm selection is frozen on validation data.
