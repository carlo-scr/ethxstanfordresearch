"""Parallel, resumable experiment grids (used by scripts/run_experiment.py and the Colab notebook).

A *variant* is a label, a method and config overrides, e.g.
    Variant("CIRA (filtered weights)", "CIRA", {"cira.weights": "filtered"})
Each (variant, rho, stream) job is written to runs/<name>/jobs/ as soon as it finishes, so an
interrupted grid (Colab disconnect) resumes where it stopped.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cira.agents import METHODS
from cira.config import ExperimentConfig


@dataclass(frozen=True)
class Variant:
    label: str
    method: str
    overrides: dict = field(default_factory=dict)


def apply_overrides(cfg: ExperimentConfig, overrides: dict) -> ExperimentConfig:
    """Return a copy of cfg with {"section.field": value} overrides, e.g. {"cira.prior": "isotropic"}."""
    cfg = replace(cfg)
    for key, value in overrides.items():
        section, name = key.split(".")
        setattr(cfg, section, replace(getattr(cfg, section), **{name: value}))
    return cfg


def default_variants(methods=METHODS) -> list[Variant]:
    return [Variant(m, m) for m in methods]


_FROZEN = {}


def _job(job):
    variant, rho, seed, n_episodes, frozen_path = job
    import torch

    torch.set_num_threads(1)
    from cira.frozen import FrozenModel
    from cira.runner import run_stream
    from cira.stream import make_stream

    if frozen_path not in _FROZEN:
        _FROZEN[frozen_path] = FrozenModel.load(frozen_path)
    cfg = apply_overrides(ExperimentConfig(), variant.overrides)
    stream = make_stream(cfg.env, replace(cfg.stream, n_episodes=n_episodes, rho=rho), seed=seed)
    t0 = time.time()
    rows, _ = run_stream(variant.method, stream, _FROZEN[frozen_path], cfg)
    for r in rows:
        r.update(method=variant.label, base_method=variant.method, rho=rho, stream=seed)
    return pd.DataFrame(rows), time.time() - t0


def _slug(variant: Variant, rho: float, seed: int, n: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{variant.label}__rho{rho:.3f}__s{seed}__n{n}")


def run_grid(
    name: str,
    variants: list[Variant],
    rhos=(0.8,),
    streams: int = 2,
    episodes: int = 150,
    frozen_path: str = "runs/frozen.pt",
    workers: int | None = None,
    runs_dir: str = "runs",
    log=print,
):
    """Run every (variant, rho, stream); returns (episodes DataFrame, summary DataFrame)."""
    out = Path(runs_dir) / name
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - (1 if (os.cpu_count() or 2) > 4 else 0))
    meta = dict(variants=[asdict(v) for v in variants], rhos=list(rhos), streams=streams, episodes=episodes,
                config=asdict(ExperimentConfig()))
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    jobs = [(v, float(r), s, episodes, frozen_path) for r in rhos for s in range(streams) for v in variants]
    todo = [j for j in jobs if not (out / "jobs" / f"{_slug(j[0], j[1], j[2], episodes)}.csv").exists()]
    todo.sort(key=lambda j: j[0].method.startswith(("CIRA", "Mixture")), reverse=True)  # slow ones first
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} on {workers} workers")
    t0 = time.time()
    if todo:
        # spawn: children import cira fresh (clean GL/torch state), safe inside notebooks
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_job, j): j for j in todo}
            for i, f in enumerate(as_completed(futures), 1):
                v, rho, seed, n, _ = futures[f]
                df, dt = f.result()
                df.to_csv(out / "jobs" / f"{_slug(v, rho, seed, n)}.csv", index=False)
                log(f"[{i}/{len(todo)}] {v.label:28s} rho={rho:.2f} stream={seed}  {dt:5.0f}s  (elapsed {time.time() - t0:.0f}s)")
    df = pd.concat(
        [pd.read_csv(out / "jobs" / f"{_slug(j[0], j[1], j[2], episodes)}.csv") for j in jobs], ignore_index=True
    )
    df.to_csv(out / "episodes.csv", index=False)
    summary = summarize(df, [v.label for v in variants])
    summary.to_csv(out / "summary.csv", index=False)
    return df, summary


def summarize(df: pd.DataFrame, order=None) -> pd.DataFrame:
    """Per (method, rho): episode means per stream, then mean and 95% interval over streams."""

    def per_stream(g):
        ret = ~g.first_occurrence
        return pd.Series(
            dict(
                fc_err=g.fc_err.mean(),
                fc_err_returns=g.fc_err[ret].mean(),
                strike_err=g.strike_err.mean(),
                success=g.success.mean(),
                success_returns=g.success[ret].mean(),
                off_table=g.off_table.mean(),
                p_true=g.p_true.mean() if "p_true" in g and g.p_true.notna().any() else np.nan,
                n_memories=g.n_memories.iloc[-1] if "n_memories" in g and g.n_memories.notna().any() else np.nan,
                plan_ms=g.plan_ms.mean(),
            )
        )

    ps = df.groupby(["method", "rho", "stream"]).apply(per_stream, include_groups=False).reset_index()
    frozen = df[df.method == "Frozen"].groupby(["rho", "stream"]).fc_err.mean().rename("fc_frozen")
    if len(frozen):
        ps = ps.join(frozen, on=["rho", "stream"])
        ps["fc_rel"] = ps.fc_err / ps.fc_frozen
        ps = ps.drop(columns="fc_frozen")
    metrics = [c for c in ps.columns if c not in ("method", "rho", "stream")]
    g = ps.groupby(["method", "rho"])[metrics]
    out = g.mean()
    sem = g.sem()
    for m in ("fc_err", "fc_rel", "success", "success_returns"):
        if m in out:
            out[f"{m}_ci"] = 1.96 * sem[m].fillna(0.0)
    out["n_streams"] = g.size()
    out = out.reset_index()
    order = list(order or []) + [m for m in METHODS if m not in (order or [])]
    rank = {m: i for i, m in enumerate(order)}
    out["_r"] = out.method.map(lambda m: rank.get(m, len(rank)))
    return out.sort_values(["rho", "_r"]).drop(columns="_r").reset_index(drop=True)
