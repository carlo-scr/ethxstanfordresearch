# Does Safety Survive Encoding?

Research code and paper workspace for studying safe-action sufficiency in pretrained visual
representations. The working question is:

> When a visual representation preserves pixels, semantics, or predictable features, does it also
> preserve which actions remain safe---and can we repair the encoder before that information is
> irretrievably lost?

## Status

This repository is an **auditable, GPU-ready research skeleton**, not a finished method or a
reproduction of the proposal's pilot. Deterministic measurement controls, finite theorem checks,
eight executed notebooks, and one-epoch PyTorch integrations for three controlled pixel domains are
complete. Both learned E2 arm paths execute through validation-only post-fit audits on CPU, and the
preconfirmation planner, task handoff, matched-radius reduction, aggregation, and weight freeze are
implemented fail-closed. Those AE/$\beta$-VAE cart/pendulum/Dubins components are now precursor
plumbing. The primary prospective study audits and repairs Cosmos Tokenizer, V-JEPA 2/2.1, and
DINOv3 on manipulation and embodied navigation. Its dependency-light technical core is now
implemented: viable-only buffered audits, the set-wise repair objective, profile and retention
losses, an identity-initialized internal bottleneck, action-conditioned training head, parameter and
gradient guards, whitelist encoder export, common adapter contracts, and a checksum-pinned
diagnostic audit entrypoint. Real checkpoint extractors, simulator adapters, GPU training, and
evaluation are not yet implemented or run, so there is no pretrained-representation safety result.

The only pre-existing artifact was `ETH_Proposal-2.pdf`; its 48-model and 96-model numerical claims
have no accompanying code, data, or checkpoints in this workspace and are tracked as
`pilot_unreproduced` in `paper/claims.toml`. The subsequently designed 96-task decision pilot and
optional 288-task expansion are independent precursor studies, not reproductions of those
historical runs and not substitutes for the new pretrained-representation evaluation.

The current scientific position is deliberately narrower and more defensible than “latent safety is
solved”:

1. define safe-action sufficiency relative to a safety specification, horizon, and action library;
2. distinguish ambiguity already present in observations from additional encoder-induced loss;
3. audit reconstructive, predictive, and semantic pretrained features under matched information and
   representation rate;
4. update modules before the audited bottleneck with a direct set-wise common-action objective; and
5. remove the training head and test a fresh readout under a held-fixed downstream controller.

Start with the dated [progress and GPU handoff](docs/progress_2026-08-21.md). Then see the
[research charter](docs/research_charter.md),
[literature map](docs/literature_map.md),
[theory audit](docs/theory_audit.md),
[dynamic-theory red team](docs/theory_v2_redteam_2026-08-22.md),
[finite dynamic-oracle bridge](docs/dynamic_oracle_bridge.md),
[controlled Dubins domain contract](docs/controlled_dubins_domain.md),
[ICML submission strategy](docs/icml_submission_strategy_2026-08-22.md),
[experiment protocol](docs/experiment_protocol.md),
[nonprivileged predicted-profile specification](docs/predicted_profile_protocol.md),
[FCSRL baseline-adaptation specification](docs/fcsrl_baseline_adaptation.md),
[pretrained-representation implementation](docs/foundation_implementation.md), and
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

The eight persisted notebooks cover synthetic estimator controls, a real PyTorch forward/backward
integration check, finite theorem checks, fail-closed pilot and confirmatory inference fixtures,
two-domain static and dynamic oracles, and executable predicted-profile, matched-radius, FCSRL, and
controlled-Dubins protocol checks. They do not report GPU pilot evidence.

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

The precursor exploratory GPU grid can be planned without importing PyTorch:

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

The distinct E2 preconfirmation handoff can likewise be inspected without launching a job:

```bash
python scripts/plan_profile_teachers.py \
  --output runs/e2_frontier/profile_teacher_plan.json
python scripts/plan_e2_preconfirmation.py \
  --profile-teacher-plan runs/e2_frontier/profile_teacher_plan.json \
  --output runs/plans/e2_preconfirmation.json
python scripts/run_e2_preconfirmation_task.py \
  --plan runs/plans/e2_preconfirmation.json --index 0 --device cuda --dry-run
```

The planner reports missing production profile artifacts rather than pretending readiness. A real
indexed execution additionally requires the frozen inputs and an exactly matching clean revision.

Before any write-producing sweep, commit the repository and regenerate both plans from that clean
revision; evidence runs reject dirty, revision-mismatched, or non-canonical plans.

Do not launch the full grid before reading the
[E1 handoff](experiments/e1_world_models/README.md) and checking the frozen gates.
The legacy 192-task confirmation matrix is encoded in
[`configs/e2_frontier/confirmatory_core.toml`](configs/e2_frontier/confirmatory_core.toml), but its
production teacher cells, 200-bundle coverage gates, complete 288-job validation-only weight
freeze, and fresh confirmation teachers remain explicit launch blockers. The two learned arm paths
and third domain are implemented and CPU-smoked, but none has passed its production-scale gates.
This matrix does not implement the new pretrained-family study and must not be launched as if it
were the ICML confirmatory design.

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
