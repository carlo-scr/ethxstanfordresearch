# Research notebooks

The notebooks are reviewable experiment logs, generated from
`scripts/build_notebooks.py` and executed in fresh kernels by `scripts/execute_notebooks.py`.
Persisted outputs are intentionally small. Synthetic-control results are measurement/theorem checks,
not evidence about trained models.

From the repository root, in the optional research environment:

```bash
make notebooks-refresh RESEARCH_PYTHON=python
```

The unified target uses one explicit interpreter for generation, fresh-kernel execution, and
validation. The three individual scripts remain available for selective rebuilds and debugging.
Byte identity has been verified across repeated runs in the current resolved environment; because
the research extras presently specify compatible lower bounds rather than a lockfile, cross-version
byte identity is not claimed.

Notebook order:

1. `00_e0_estimator_validation.ipynb`: exact, robust-radius, rescaling, sampling, and action controls.
2. `01_e1_torch_world_model_pilot.ipynb`: CPU smoke and GPU handoff for learned encoders.
3. `02_theory_counterexamples.ipynb`: finite exhaustive theorem checks and counterexamples.
4. `03_pilot_analysis_template.ipynb`: paired inference, the exact 288-row validation-only weight freeze, generic validation frontier, and unblinding gate.
5. `04_two_domain_oracle_controls.ipynb`: exact cart/pendulum observation aliases, privileged-state controls, and history-resolved action conflicts.
6. `05_dynamic_regret_oracle.ipynb`: registered finite nominal action trees, optimal-margin regret, sign obstruction, retained viability, and compositional bounds in both domains.
7. `06_confirmatory_inference_preflight.ipynb`: complete synthetic 240-row preflight of the frozen 36-bound confirmatory API and its fail-closed coverage rule.
8. `07_method_protocol_smoke.ipynb`: a real configured profile-teacher split, the 200-bundle gate boundary, a validation-only matched-radius reducer contract smoke, dependency-free/tensor FCSRL target parity, deterministic FCSRL batching, and a controlled-Dubins causal-history alias smoke.

Regenerate before execution so the source cells match the checked-in generator. GPU runs should
write manifests and metrics under ignored `runs/`; only promote immutable summaries through the
claim registry after the experiment gate passes.
