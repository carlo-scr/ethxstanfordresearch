"""Run one method on one stream and record per-episode metrics."""

from __future__ import annotations

import time

import numpy as np

from cira.agents import CIRA, make_agent
from cira.config import ExperimentConfig
from cira.env import PuckEnv
from cira.frozen import FrozenModel
from cira.planner import CEMPlanner
from cira.stream import Stream


def first_occurrence_flags(ctx: np.ndarray) -> np.ndarray:
    """True for every episode in the first dwell block of its context."""
    seen, flags, cur_first = set(), np.zeros(len(ctx), bool), False
    for e, c in enumerate(ctx):
        if e == 0 or c != ctx[e - 1]:
            cur_first = c not in seen
            seen.add(c)
        flags[e] = cur_first
    return flags


def run_stream(
    method: str,
    stream: Stream,
    frozen: FrozenModel,
    cfg: ExperimentConfig,
    record: bool = False,
    fall_steps: int = 12,
):
    env = PuckEnv(cfg.env)
    planner = CEMPlanner(frozen, cfg.env, cfg.planner)
    agent = make_agent(method, cfg.cira, cfg.sgd, stream.cue.shape[1])
    n_l = len(cfg.env.loads)
    first = first_occurrence_flags(stream.ctx)
    rows, pre_contact, episodes = [], [], []
    for e in range(len(stream)):
        ctx = int(stream.ctx[e])
        rng = np.random.default_rng(stream.seed[e])
        env.set_context(ctx // n_l, ctx % n_l, rng)
        env.reset(stream.start[e])
        z0 = env.observe(rng)

        agent.begin_episode(stream.cue[e], ctx)
        hyps = agent.hypotheses()
        if isinstance(agent, CIRA):
            pre_contact.append(dict(agent.pre_contact))
        t0 = time.perf_counter()
        action, pred, jhat, probs = planner.plan(z0, hyps, np.random.default_rng([int(stream.seed[e]), 1]))
        plan_ms = 1e3 * (time.perf_counter() - t0)

        r = env.strike(action, rng, fall_steps if record else 0, z0=z0)
        strike_err = np.nan
        if len(r.actions):
            PHI, V, SIG = frozen.residuals(r.states[:-1], r.actions, r.states[1:])
            m = sum(p * W @ PHI[0] for p, W in hyps)
            strike_err = float(np.linalg.norm(m - V[0]))
            agent.observe_episode(PHI, V, SIG)
        agent.end_episode(ctx)

        # Expected trajectory error under the pre-contact belief, over the steps the puck stayed on the table.
        T = min(len(r.states) - 1, cfg.planner.horizon)
        per_h = np.linalg.norm(pred[:, 1 : T + 1] - r.states[None, 1 : T + 1, :2], axis=2).mean(1) if T else np.nan
        fc_err = float(probs @ per_h) if T else np.nan
        stop_err = float(np.linalg.norm(r.final_xy - np.array(cfg.env.target)))
        rows.append(
            dict(
                method=method,
                episode=e,
                ctx=ctx,
                surface=ctx // n_l,
                load=ctx % n_l,
                cue_correct=bool(stream.shown[e] == ctx),
                first_occurrence=bool(first[e]),
                fc_err=fc_err,
                strike_err=strike_err,
                stop_err=np.nan if r.off_table else stop_err,
                off_table=bool(r.off_table),
                success=bool(not r.off_table and stop_err < cfg.env.success_tol),
                impulse_x=float(action[0]),
                impulse_y=float(action[1]),
                predicted_cost=jhat,
                plan_ms=plan_ms,
                **agent.diagnostics(),
            )
        )
        if record:
            episodes.append(dict(qpos=r.qpos, ctx=ctx, action=action, off_table=r.off_table, final_xy=r.final_xy))

    if isinstance(agent, CIRA):
        labels = agent.memory_labels()
        # Memory 0 is the fixed training-condition memory by construction (evaluation-only labels).
        labels[0] = cfg.env.train_context[0] * n_l + cfg.env.train_context[1]
        for row, pre in zip(rows, pre_contact):
            row["p_true"] = float(sum(p for mid, p in pre.items() if mid >= 0 and labels[mid] == row["ctx"]))
            row["p_novel"] = float(pre[CIRA.NOVEL])
    elif method == "Oracle":
        for row in rows:
            row["p_true"] = 1.0
    extras = dict(pre_contact=pre_contact, episodes=episodes)
    if isinstance(agent, CIRA):
        extras["memory_labels"] = labels
    return rows, extras
