# E1 - World-model representation audit

## Current implementation

Two end-to-end synthetic tasks are implemented: a controlled cart rendered as RGB video and a
controlled inverted pendulum with distinct dynamics and rendering. Their single frames hide
linear or angular velocity. Each factorial design has three explicit history modes: `stack_h1` is a true
single-frame/memoryless control, while `stack_h4` and `gru_h4` receive the same four-frame input and
isolate the effect of recurrence from the effect of history availability. The pipeline provides
whole-trajectory train/validation/calibration/test splits, AE and beta-VAE families, latent dynamics
and decoder heads, four safety-objective arms, validation-selected checkpoints, hashed manifests,
rollout evaluation, and traceable static/action audit records.

The recommended decision pilot is:

`2 domains × 2 model families × 2 capacity-matched history modes × 1 latent dimension × 4 safety arms × 3 seeds = 96 tasks`.

The two 48-task plans are `decisive_cart_grid.toml` and `decisive_pendulum_grid.toml`. They answer
the history/action-supervision question before spending compute on recurrence and dimension
sensitivity. The larger expansion available after that gate is, per domain:

`2 model families × 3 history modes × 2 latent dimensions × 4 safety arms × 3 seeds = 144 tasks`.

There are therefore 96 tasks in the recommended gate and 288 in the optional full expansion. This
is a controlled two-task pilot, not a confirmatory result or a reproduction of the proposal's
historical 96-model grid. History modes are jointly encoded in every task's metadata and output
path; `stack_h1` and `stack_h4` must never be pooled merely because both use the stack architecture.

The `stack_h1`/`stack_h4` comparison uses the same fixed-width `(last, last-first)` feature summary
and has identical model parameter counts. The h1 temporal-difference block is exactly zero, while
h4 can expose motion. Model sizes are still recorded and must be checked in every aggregate.

## What has actually run

The notebook contains a real PyTorch forward/backward check. One-epoch CPU integrations have
completed for both tasks, including checkpoint, evaluation, learned-latent, visible-observation,
and privileged-state audit paths. They validate plumbing only: no GPU task in either factorial
pilot has been run, and the CPU smoke values must not enter a paper table.

## GPU commands

Install the research dependencies, commit the repository, create the deterministic plan from that
clean revision, and inspect one task:

```bash
python3 -m pip install -e '.[research]'
mkdir -p runs/plans runs/slurm
python scripts/plan_e1_sweep.py \
  --grid configs/e1_world_models/decisive_cart_grid.toml \
  --output runs/plans/e1_decisive_cart.json
python scripts/run_e1_sweep_task.py \
  --plan runs/plans/e1_decisive_cart.json \
  --index 0 \
  --device cuda \
  --dry-run

python scripts/plan_e1_sweep.py \
  --grid configs/e1_world_models/decisive_pendulum_grid.toml \
  --output runs/plans/e1_decisive_pendulum.json
```

After reviewing `runs/plans/e1_decisive_cart.json`, run one indexed task with the same command
without `--dry-run`. The task runner rechecks the plan and source hashes, regenerates the canonical
factorial plan to reject a manually edited-and-rehashed file, constrains outputs below
`runs/e1_world_models`, and requires the same clean committed Git revision recorded at planning
time. Regenerate plans after every commit. `--allow-unversioned` exists only for disposable
engineering checks.

A task is marked successful only after the required audit suite finishes. Audit failure or
`--skip-audit` leaves a fail-closed status, and `sweep_task_manifest.json` records the plan, task,
and completed views. For Slurm, first adapt account, partition, and environment setup in the two
decisive-pilot scripts; set `LATENT_SAFETY_PYTHON` if `python` is not the intended environment, then:

```bash
sbatch experiments/e1_world_models/slurm_decisive_cart_array.sh
sbatch experiments/e1_world_models/slurm_decisive_pendulum_array.sh
```

Only if the predeclared 96-task gate passes, generate the optional `pilot_grid.toml` and
`pilot_pendulum_grid.toml` plans and use `slurm_array.sh` / `slurm_pendulum_array.sh` for the
144-task-per-domain recurrence and dimension expansion.

Each array permits eight concurrent jobs, so submitting both permits up to sixteen. The trainer
refuses to overwrite a nonempty output directory. Keep failures and scheduler logs; archive a
failed directory and generate a clean output root before a documented retry—never silently replace
or cherry-pick a seed.

After the complete array finishes, build the validation-only checkpoint table, seed-aggregated
frontier, and paired safety-arm comparisons:

```bash
python scripts/aggregate_e1_sweep.py \
  --plan runs/plans/e1_decisive_cart.json \
  --output runs/analysis/e1_pilot_validation.json
python scripts/aggregate_e1_sweep.py \
  --plan runs/plans/e1_decisive_pendulum.json \
  --output runs/analysis/e1_pendulum_pilot_validation.json
```

This command exits nonzero while recording every missing or failed task if the plan is incomplete.
`--allow-partial` is an explicitly labelled exploratory diagnostic, never a substitute for the
complete aggregate. Test keys are not accessed by default; use `--unblind-test` only after freezing
the validation-selected rule and included-run ledger. Even then, test values are reporting fields
and cannot alter the validation frontier or selection.

Aggregate eligibility also requires the runner-written orchestration record to identify the exact
source-plan hash and task ID, report no failed audit, and record the canonical completed audit set.
Every arm requires `latent`; the baseline `none` arm additionally requires `observation_oracle`,
`state_oracle`, `latent_plus_margin`, and `latent_plus_action_profile`. A trainer manifest with a
manually retained `success` status but missing audit completion metadata is rejected.

Utility reporting is deliberately separated. Each checkpoint row, seed aggregate, and paired-arm
comparison includes current-frame `reconstruction_mse` and
`rollout_pixel_mse_at_max_horizon` in addition to the family-dependent `utility_loss` composite.
The rollout summary is fixed to the largest predeclared horizon in the run's resolved config; every
row also retains the pixel MSE and case count at every configured horizon. Missing, non-finite, or
negative utility values, zero-case rollouts, and any mismatch between configured and evaluated
horizons fail closed. The composite remains the validation checkpoint/frontier selection metric, so
this reporting addition does not retroactively change selection.

## Frozen pilot analysis

- Radius scale: calibration-only median pairwise latent distance, without safety labels.
- Relative radii: `[0, 0.01, 0.02, 0.05, 0.10]`; primary radius: `0.05`.
- Validation static-defect budget: `0.10`.
- Inference: paired seed/block bootstrap, 10,000 resamples, seed `20260821`.
- Exact paired sign-flip p-values are allowed only with a predeclared sharp-null exchangeability
  justification; with three pilot seeds their resolution is inherently too coarse for confirmation.
- Model and frontier selection: validation only; frames are never independent replicates.
- Advance gate: at least one safety arm shifts the matched-utility validation frontier while E0
  controls remain green.

`notebooks/03_pilot_analysis_template.ipynb` encodes this selection discipline. Its current values
are a synthetic regression fixture, not pilot results. The task runner does compute test audits, so
the confirmatory phase needs a procedural or code-level blinding step before test unblinding.

## Next empirical gate

Run both controlled tasks and retain the implemented ground-truth-state and visible-observation
controls; then add the strongest relevant representation baselines. Use the pilot to narrow the grid, then extend selected
paired comparisons to eight seeds (or justify a different count through a preregistered precision
analysis). Only after freezing the analysis should final test results be opened.

Primary deliverable: one row per frozen checkpoint containing separate reconstruction and rollout
utility metrics, the selection-only composite, safety-probe metrics, static robust-defect curves,
action-conflict metrics, the trajectory-balanced 95th-percentile required-violation summary,
observation-level controls, and provenance.

**Exit gate:** a non-negligible effect appears on at least two tasks and two representation families,
survives trajectory splitting and observation controls, and is stable over eight paired seeds (or a
different count justified by a preregistered precision analysis).
