import numpy as np

from cira.config import EnvConfig
from cira.env import PuckEnv


def test_sliding_distance_matches_coulomb_friction():
    env = PuckEnv()
    for s, (_, mu) in enumerate(env.cfg.surfaces):
        env.set_context(s, 1)
        env.reset((-0.45, 0.0))
        v0 = 0.8
        r = env.strike((v0 * env.cfg.loads[1], 0.0))
        dist = r.final_xy[0] + 0.45
        assert not r.off_table
        assert abs(dist - v0**2 / (2 * mu * 9.81)) < 0.08 * dist + 0.005


def test_mass_scales_launch_speed():
    env = PuckEnv()
    speeds = []
    for load in range(3):
        env.set_context(2, load)
        env.reset((-0.45, 0.0))
        env.step(np.array([0.4, 0.0]))
        # add back the friction that acted during the rest of the first step
        c = env.cfg
        speeds.append(env.data.qvel[0] + env.mu * 9.81 * (c.dt - c.impulse_steps * c.timestep / 2))
    masses = np.array(EnvConfig().loads)
    assert np.allclose(np.array(speeds) * masses, 0.4, rtol=0.02)


def test_puck_stays_flat_and_leaves_table():
    env = PuckEnv()
    env.set_context(2, 0)
    env.reset((-0.45, 0.0))
    r = env.strike((1.5, 0.0))
    assert r.off_table
    assert np.all(np.abs(r.states[:, 1]) < 0.05)
