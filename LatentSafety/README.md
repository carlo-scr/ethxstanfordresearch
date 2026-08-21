# Does Safety Survive Encoding?

Research code and paper workspace for studying the certifiability limits of learned
representations. The working question is:

> How much safety margin is irretrievably lost when a state or observation is compressed into a
> learned latent representation, and can that loss be measured and prevented without giving up
> useful compression?

## Status

This repository is an **auditable, GPU-ready research skeleton**, not a finished method or a
reproduction of the proposal's pilot. Deterministic measurement controls, finite theorem checks,
six executed notebooks, and one-epoch PyTorch integrations for two controlled pixel domains are
complete. The recommended 96-task two-domain decision pilot, optional 288-task expansion, and all
confirmatory experiments remain unrun, so there is no
learned-representation safety result yet.

The only pre-existing artifact was `ETH_Proposal-2.pdf`; its 48-model and 96-model numerical claims
have no accompanying code, data, or checkpoints in this workspace and are tracked as
`pilot_unreproduced` in `paper/claims.toml`. The new 96-task decision pilot (and optional 288-task
factorial expansion) is an independent two-domain study, not a reproduction of those historical
runs.

The current scientific position is deliberately narrower and more defensible than “latent safety is
solved”:

1. quantify static safe-set ambiguity induced by a representation;
2. distinguish ambiguity already present in observations from additional ambiguity introduced by
   the encoder;
3. validate any finite-sample estimator against injective and known-collision controls;
4. test whether specification-aware training improves the safety/utility frontier; and
5. only then study downstream latent control.

Start with the dated [progress and GPU handoff](docs/progress_2026-08-21.md). Then see the
[research charter](docs/research_charter.md),
[literature map](docs/literature_map.md),
[theory audit](docs/theory_audit.md),
[dynamic-theory red team](docs/theory_v2_redteam_2026-08-22.md),
[finite dynamic-oracle bridge](docs/dynamic_oracle_bridge.md),
[ICML submission strategy](docs/icml_submission_strategy_2026-08-22.md),
[experiment protocol](docs/experiment_protocol.md), and
[five-month roadmap](docs/roadmap.md). The
[implementation contract](docs/implementation_contract.md) defines the handoff between the ML and
control workstreams.

## Quick start

The core smoke test uses only the Python 3.11+ standard library.

```bash
cd LatentSafety
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m latent_safety.cli pilot-smoke \
  --config configs/pilot/smoke.toml \
  --output /tmp/latent-safety-smoke.json
python3 scripts/validate_claims.py
python3 scripts/validate_configs.py
python3 scripts/validate_notebooks.py --require-executed
```

The six persisted notebooks cover synthetic estimator controls, a real PyTorch forward/backward
integration check, finite theorem checks, a fail-closed analysis regression fixture, a smooth
average-loss counterexample, and a finite-tree dynamic oracle. They do not report GPU pilot
evidence.

For editable installation and development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
make check
```

The scientific stack is intentionally optional so that the protocol and metric tests remain usable
without a GPU environment:

```bash
python3 -m pip install -e '.[research]'
```

The exploratory GPU grid can be planned without importing PyTorch:

```bash
mkdir -p runs/plans
python scripts/plan_e1_sweep.py \
  --grid configs/e1_world_models/decisive_cart_grid.toml \
  --output runs/plans/e1_decisive_cart.json
python scripts/plan_e1_sweep.py \
  --grid configs/e1_world_models/decisive_pendulum_grid.toml \
  --output runs/plans/e1_decisive_pendulum.json
python scripts/run_e1_sweep_task.py \
  --plan runs/plans/e1_decisive_cart.json --index 0 --device cuda --dry-run
```

Before any write-producing sweep, commit the repository and regenerate both plans from that clean
revision; evidence runs reject dirty, revision-mismatched, or non-canonical plans.

Do not launch the full grid before reading the
[E1 handoff](experiments/e1_world_models/README.md) and checking the frozen gates.

## Repository map

```text
configs/       Versioned experiment specifications.
docs/          Thesis, novelty boundary, theory audit, and evaluation protocol.
experiments/   One gated workstream per experiment family (E0-E5).
paper/         Claim registry, bibliography, and compileable manuscript skeleton.
scripts/       Repository-level validation commands.
src/           Reusable metric and protocol implementation.
tests/         Deterministic negative controls and theorem-adjacent unit tests.
notebooks/     Executed synthetic controls, theory checks, and GPU-ready experiment handoffs.
```

## Non-negotiable research rules

- A nearest unsafe neighbor is a diagnostic, **not** an exact latent collision and not a lower
  bound on the exact defect without a fixed radius.
- Expected pairwise faithfulness loss is a training objective, **not** a uniform certificate.
- Static safe-set representability does not imply action sufficiency, dynamics conjugacy, or
  closed-loop safety.
- Results enter the paper only through `paper/claims.toml`, with a producing command, immutable
  manifest, and evidence path.
- Report every seed and failed run. Split by trajectories or environments, never by adjacent image
  frames.

## Venue target

The working target is ICML 2027. The manuscript uses a neutral LaTeX class until official 2027
style files and policies exist. The experiment plan assumes an eight-page main narrative, matching
the 2026 call, but this must be rechecked when the 2027 call is published.

## License

No open-source license is granted yet. The collaborators should resolve institutional ownership and
release terms before public distribution.
