# Research notebooks

The notebooks are reviewable experiment logs, generated from
`scripts/build_notebooks.py` and executed in fresh kernels by `scripts/execute_notebooks.py`.
Persisted outputs are intentionally small. Synthetic-control results are measurement/theorem checks,
not evidence about trained models.

From the repository root, in the optional research environment:

```bash
python scripts/build_notebooks.py
python scripts/execute_notebooks.py
python3 scripts/validate_notebooks.py --require-executed
```

Notebook order:

1. `00_e0_estimator_validation.ipynb`: exact, robust-radius, rescaling, sampling, and action controls.
2. `01_e1_torch_world_model_pilot.ipynb`: CPU smoke and GPU handoff for learned encoders.
3. `02_theory_counterexamples.ipynb`: finite exhaustive theorem checks and counterexamples.
4. `03_pilot_analysis_template.ipynb`: paired inference, validation frontier, and unblinding gate.
5. `04_two_domain_oracle_controls.ipynb`: exact cart/pendulum observation aliases, privileged-state controls, and history-resolved action conflicts.
6. `05_dynamic_regret_oracle.ipynb`: registered finite nominal action trees, optimal-margin regret, sign obstruction, retained viability, and compositional bounds in both domains.

Regenerate before execution so the source cells match the checked-in generator. GPU runs should
write manifests and metrics under ignored `runs/`; only promote immutable summaries through the
claim registry after the experiment gate passes.
