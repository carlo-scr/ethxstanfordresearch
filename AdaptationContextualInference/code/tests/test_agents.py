"""Synthetic residual streams: no simulator or frozen model needed."""

import numpy as np

from cira.agents import CIRA
from cira.config import CiraConfig
from cira.frozen import K_RESIDUAL as K
from cira.frozen import P_FEATURES as P


def synthetic(n_episodes=40, steps=15, seed=0):
    rng = np.random.default_rng(seed)
    Ws = [np.zeros((K, P)), 0.8 * rng.normal(size=(K, P)), 0.8 * rng.normal(size=(K, P))]
    apps = rng.normal(size=(3, 4)) * 2
    ctx = np.repeat(rng.integers(0, 3, n_episodes // 5), 5)
    for c in ctx:
        phis = rng.normal(size=(steps, P))
        vs = phis @ Ws[c].T + 0.05 * rng.normal(size=(steps, K))
        yield c, apps[c] + 0.1 * rng.normal(size=4), phis, vs


def test_cira_recalls_contexts_and_learns_cues():
    agent = CIRA(CiraConfig(prior="isotropic", merge_z2=0.0), cue_dim=4)
    hits, total = 0, 0
    for e, (c, cue, phis, vs) in enumerate(synthetic(n_episodes=60)):
        agent.begin_episode(cue, c)
        if e >= 30:
            labels = agent.memory_labels()
            top = max(agent.pre_contact, key=agent.pre_contact.get)
            hits += top >= 0 and labels.get(top) == c
            total += 1
        agent.observe_episode(phis, vs, np.full_like(vs, 0.05))
        agent.end_episode(c)
    assert len(agent.mems) <= 5, "too many memories for 3 contexts"
    assert hits / total > 0.8, f"cue-driven recall too low: {hits}/{total}"
