# Contributing

## Before implementing a new idea

1. Add or update a falsifiable hypothesis in `docs/experiment_protocol.md`.
2. Add the intended paper statement to `paper/claims.toml` with status `hypothesis`.
3. Specify negative controls, primary metric, stopping rule, and compute budget.
4. Obtain review from one theory owner and one experiment owner for changes that alter a claim.

## Run contract

Every non-smoke run must retain:

- the resolved configuration;
- Git revision and dirty-state flag;
- seed and deterministic/nondeterministic backend settings;
- dependency and hardware metadata;
- dataset version, split IDs, and checksums;
- scalar metrics and per-example safety witnesses; and
- stdout/stderr or scheduler logs.

Generated outputs belong under `runs/` or `results/` and are ignored by default. Promote only small,
reviewed aggregate manifests needed to reproduce paper tables and figures.

## Review standard

- Tests must include an identity/injective negative control and a constructed aliasing positive
  control for any new defect estimator.
- Numerical thresholds must be justified in physical units or by a pre-registered calibration rule;
  do not tune a latent radius on the test set.
- A claim cannot move to `verified` until an independent collaborator reruns the producing command.
- Changes to theorem statements must update `docs/theory_audit.md` and the manuscript together.

