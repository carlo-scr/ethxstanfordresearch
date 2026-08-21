# E0 - Estimator validation

**Question:** does the measurement distinguish real representation aliasing from finite-sample
boundary geometry?

## Implemented and executed

`notebooks/00_e0_estimator_validation.ipynb` is an executed, machine-checked semantic validation.
Together with the deterministic unit tests and controlled double-integrator benchmark, it covers:

- exact injective and folded representations;
- a tunable near-collision threshold with a known radius curve;
- joint latent/radius rescaling invariance;
- sampling-density and class-balance sensitivity of nearest-unsafe-neighbor diagnostics;
- exact finite action-fiber conflicts;
- a saturated observation with planted static defect `0.5`; and
- hidden velocity with planted one-step action violation `0.25`, repaired by two-frame history.

Run the current gate with:

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m latent_safety.cli pilot-smoke \
  --config configs/pilot/smoke.toml \
  --output /tmp/latent-safety-smoke.json
python3 scripts/validate_notebooks.py --require-executed
```

## Status and remaining gate

The reference semantics pass, which is enough to begin an exploratory engineering pilot. E0 is not
yet a population-certification result. Still required are:

- continuous injective maps at increasing held-out sample sizes;
- boundary-stratified held-out trajectories and class-ratio sensitivity intervals;
- irrelevant-coordinate, translation, and rotation controls in the learned audit path;
- raw-observation versus encoded-observation floor comparisons; and
- a declared search or covering procedure that turns finite witnesses into a justified bound.

A nearest unsafe sample is only a finite diagnostic. Its absence does not certify unsampled states,
and its distance is not an exact-collision lower bound without a fixed uncertainty radius.

**Exit gate:** identity/injective defect converges to its known value, constructed defects are
recovered within predeclared tolerance, and every reported bound has the correct finite-sample
semantics. Large or confirmatory GPU studies remain blocked until this stronger gate passes; a small
exploratory pipeline pilot may proceed while completing it.
