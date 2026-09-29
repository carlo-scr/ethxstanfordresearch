# CIRA stage 1a: MuJoCo puck strike

Simulation code for the first experiment of *Adapting Before Contact* (see `../proposal`, `../paper_corl`).
A puck is struck once toward a target near a table edge. The context is a surface (felt, rubber,
acrylic) and a load (0.2, 0.6, 1.0 kg). A frozen dynamics model is trained on rubber at 0.6 kg only.
Stage 1a works on the true puck state `[x, y, vx, vy]`, so it tests the inference machinery
without representation learning. Stage 1b swaps the state for DINOv2 or Cosmos latents of the
rendered frames.

## Quickstart (Mac)

```bash
cd AdaptationContextualInference/code
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                                   # physics, memories, filter (2 s)
.venv/bin/python scripts/train_frozen.py                        # frozen model -> runs/frozen.pt (~5 min)
.venv/bin/python scripts/run_experiment.py --name quick         # 8 methods x 2 streams x 150 episodes (~2 min)
.venv/bin/python scripts/plot_results.py --name quick
.venv/bin/python scripts/make_media.py all                      # stills, observation grid, video -> runs/media
```

Full sweep for the paper figure: `run_experiment.py --name main --episodes 500 --streams 10 --rho 0.111 0.5 0.8 0.95`.
`runs/` is git-ignored. Rerunning an experiment command skips jobs that already finished.

## Colab notebook

`notebooks/colab_stage1a.ipynb` walks through the whole experiment. It covers an interactive strike
widget, a physics check, the camera view, deployment streams, frozen-model training and diagnostics,
and a step-by-step look inside CIRA (belief heatmap, side-by-side video). It then runs the main
experiment, ablations and a config playground, and exports the results.

1. Push your commits (Colab clones the repo), then open the notebook in Colab:
   File → Open notebook → GitHub → `carlo-scr/ethxstanfordresearch`.
2. Add a GitHub token with read access as the Colab secret `GITHUB_TOKEN` (key icon in the left bar),
   or paste it when asked. The token is not stored in the clone's `.git/config`.
3. Pick a profile in the settings cell (`smoke` ≈ 5 min, `quick` ≈ 30 min, `full` = the paper sweep,
   hours) and use Runtime → Run all.

With `USE_DRIVE` on, `runs/` lives in `MyDrive/cira_runs`. Every finished job is written at once,
so rerunning after a disconnect only runs what is missing. A GPU runtime speeds up frozen-model
training and rendering; the experiment jobs are CPU-bound (free Colab has 2 cores, so the M4 is
faster for big sweeps). The notebook also runs locally (`uv pip install -e ".[dev,notebook]"`, then
open it in Jupyter or VS Code from `code/notebooks`).

## Layout

| file | contents |
|---|---|
| `cira/env.py` | MuJoCo scene (plain and hero styles), contexts, strike episode |
| `cira/stream.py` | context streams with geometric dwell times and cue reliability ρ |
| `cira/frozen.py` | frozen dynamics + heteroscedastic noise model, feature map φ, structured prior |
| `cira/memories.py` | Bayesian linear residual memories (predictive-weight updates, merging), cue models |
| `cira/agents.py` | CIRA and baselines: Frozen, Reset-SGD, Continual-SGD, Cue-RLS, Mixture, CIRA-NoCue, Oracle |
| `cira/planner.py` | CEM on the belief-weighted cost Σ_c π(c) J_c(a) |
| `cira/runner.py` | one method on one stream → per-episode metrics |
| `cira/render.py` | hero/plain rendering from logged poses, ghost trails, overlays, mp4 |
| `cira/media.py` | teaser strip, observation grid, side-by-side comparison video |
| `cira/experiment.py` | parallel, resumable grids of variants (method + config overrides) × ρ × streams |
| `cira/plots.py` | all figures (results, physics check, streams, residual and belief heatmaps) |
| `scripts/` | thin command-line wrappers: train, run, plot, media |
| `notebooks/colab_stage1a.ipynb` | the full walkthrough (Colab or local) |

Metrics per episode: `fc_err` (belief-weighted rollout error of the committed strike [m]),
`strike_err` (error of the mixture residual prediction at contact, the paper's b_e, normalized
units), `success` (stops within 5 cm, on the table), `off_table`, `p_true` (pre-contact belief on
the memories whose majority label is the true context), `n_memories`. `first_occurrence` marks the
first dwell block of each context, so "returns" means recall is possible.

## Design decisions made while getting stage 1a to work

Each fixed a concrete failure seen in traces. Several are candidates for the method section.

1. **Explicit Coulomb friction.** MuJoCo's soft frictional contact made the puck hop by 2–5 mm
   under strikes, and per-step deceleration varied by up to 50% on felt. The mat contact is now
   frictionless and friction is applied as −μmg·v/|v| (`env.py`). Deceleration now matches μg to
   machine precision.
2. **Features split the frozen prediction** into an action-driven part f(z,a) − f(z,0) and a passive
   part f(z,0) − z (`frozen.features`). A mass change rescales the first (weight m0/m − 1), a friction
   change the second (μ/μ0 − 1), so all weights are O(1). With φ = [a, f − z, 1] the mass weight was
   about 20, far outside the prior, and memories could not represent light pucks. The same split applies to latent world models.
3. **Structured prior:** each residual coordinate mainly rescales its own action-driven and passive
   change (variance 1), with variance 0.01 on all other weights (`structured_prior_var`).
   `CiraConfig.prior="isotropic"` restores the generic prior.
4. **Episode-conditional scoring.** Within an episode each hypothesis is scored by a copy of its
   memory that has seen all earlier steps of the episode at full weight. This is exact while the
   context is fixed within an episode. Without it, the new-context hypothesis stays at its broad prior
   predictive and never wins, so felt at 1.0 kg was merged into rubber at 1.0 kg. **The library
   is still updated with predictive weights q_t**, so Theorem 1 is unaffected.
5. **Per-episode random effect** on the scorers, on the friction (passive) weights only. Friction
   jitters about 3% between episodes; otherwise the new-context hypothesis wins by fitting that jitter.
   A random effect on the action weights lets the training memory absorb a 40% mass change from
   a single strike step, so those weights get only the impulse execution noise.
6. **New memories start from the new-context scorer**, i.e. the posterior given the creation
   episode under the hypothesis that won. Starting from the q-weighted candidate left new memories near
   the prior, and short episodes (puck leaves the table after 2–3 steps) produced duplicates.
7. **Sticky prior as a stay probability**: π_e = s·π_{e−1} + (1 − s)·CRP(counts), s = 0.8. With the
   count form (n_c + κ·prev), counts outgrew κ after a few episodes and switches stayed 50/50.
8. **Stick–slip σ inflation:** transitions below 0.3 m/s get 10× σ, because stopping inside a step
   breaks linearity. This is the paper's "σ is large during stick–slip", set by hand because a noise model
   fitted on the training condition cannot see it.
9. **Merging** memories that never differ by more than 3 sd on recent feature vectors (from the proposal).

## Current status (one quick run: 2 streams × 150 episodes, ρ = 0.8)

| method | fc_err / Frozen | success | success on returns | off table | memories |
|---|---|---|---|---|---|
| Frozen = Reset-SGD | 1.00 | 9% | 0% | 48% | – |
| Continual-SGD | 0.56 | 9% | 3% | 22% | – |
| Cue-RLS | 0.73 | 2% | 0% | 30% | – |
| Mixture (MOLe-style) | 0.63 | 16% | 26% | 6% | 6 |
| CIRA-NoCue | 0.31 | 30% | 19% | 6% | 15 |
| **CIRA** | 0.33 | **45%** | **42%** | 8% | 12 |
| Oracle | 0.13 | 72% | 82% | 3% | – |

Too little data for claims, but Observation 1 holds exactly (Reset-SGD = Frozen) and the ordering
matches H1. Open issues:

- **Over-segmentation:** 12–15 memories for 7 contexts over 150 episodes. On a hand-made 18-episode
  stream (`make_media.py video`) it finds exactly one memory per context. Duplicates mostly come from the
  light 0.2 kg pucks, whose episodes end after a few steps, and from memories created after one short
  block that are overconfident at other strike strengths. This is the false-context rate from the proposal's
  Risks section; a merge rule that tolerates under-trained memories is the next thing to try.
- **Cue helps success but not fc_err** at this sample size. Needs the full ρ sweep.
- **Hedging:** the belief-weighted cost makes plans cautious whenever any hypothesis predicts the puck
  leaving the table. Rational, but it keeps CIRA far from Oracle. A CVaR or chance-constrained cost is an easy ablation.
- Not implemented yet: factored prior (composition test), safety filter with per-memory ACI,
  value-of-information probing, the recall/learning decomposition, stage 1b encoders.
