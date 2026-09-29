"""Deployment streams: which context is active in each episode, and what the cue shows.

Contexts persist for geometrically distributed dwell times. The cue shows the true context's
appearance with probability rho and each other context's appearance with probability
(1 - rho) / (K - 1), so rho = 1/K is uninformative. In stage 1a an appearance is a fixed random
vector per surface and per load; in stage 1b it is replaced by encoder features of the rendered mat
and puck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cira.config import EnvConfig, StreamConfig

APPEARANCE_SEED = 12345


@dataclass
class Stream:
    ctx: np.ndarray  # (N,) context id = surface * n_loads + load
    surface: np.ndarray
    load: np.ndarray
    shown: np.ndarray  # context id whose appearance the cue shows
    cue: np.ndarray  # (N, cue_dim)
    start: np.ndarray  # (N, 2)
    seed: np.ndarray  # (N,) per-episode nuisance seed, shared by all methods
    rho: float

    def __len__(self) -> int:
        return len(self.ctx)


def appearances(env_cfg: EnvConfig, dim_per_factor: int):
    rng = np.random.default_rng(APPEARANCE_SEED)
    return rng.normal(size=(len(env_cfg.surfaces), dim_per_factor)), rng.normal(size=(len(env_cfg.loads), dim_per_factor))


def context_ids(env_cfg: EnvConfig, include_train: bool = True) -> list[int]:
    n_l = len(env_cfg.loads)
    train = env_cfg.train_context[0] * n_l + env_cfg.train_context[1]
    ids = list(range(len(env_cfg.surfaces) * n_l))
    return ids if include_train else [i for i in ids if i != train]


def make_stream(env_cfg: EnvConfig, cfg: StreamConfig, seed: int, schedule: np.ndarray | None = None) -> Stream:
    """Sample a stream. Pass `schedule` (context id per episode) to fix the context sequence."""
    rng = np.random.default_rng(seed)
    n_l = len(env_cfg.loads)
    ids = context_ids(env_cfg, cfg.include_train_context)
    if schedule is None:
        ctx = np.empty(cfg.n_episodes, int)
        cur = rng.choice(ids)
        for e in range(cfg.n_episodes):
            if e > 0 and rng.random() < 1.0 / cfg.mean_dwell:
                cur = rng.choice([i for i in ids if i != cur])
            ctx[e] = cur
    else:
        ctx = np.asarray(schedule, int)
    n = len(ctx)
    shown = ctx.copy()
    for e in range(n):
        if rng.random() > cfg.rho:
            shown[e] = rng.choice([i for i in ids if i != ctx[e]])
    s_app, l_app = appearances(env_cfg, cfg.cue_dim_per_factor)
    cue = np.concatenate([s_app[shown // n_l], l_app[shown % n_l]], 1)
    cue += rng.normal(0.0, cfg.cue_noise, cue.shape)
    start = np.stack([rng.uniform(*env_cfg.start_x, n), rng.uniform(*env_cfg.start_y, n)], 1)
    return Stream(
        ctx=ctx,
        surface=ctx // n_l,
        load=ctx % n_l,
        shown=shown,
        cue=cue,
        start=start,
        seed=rng.integers(0, 2**31, n),
        rho=cfg.rho,
    )
