"""MuJoCo puck strike on interchangeable mats.

A single planar impulse is applied to a puck; it then slides until it stops or leaves the table.
Context = (surface, load): the surface sets the sliding friction and mat texture, the load sets the
puck mass and colour.

Sliding friction is applied as an explicit Coulomb force -mu m g v/|v| while the puck rests on the
mat; MuJoCo's contact is frictionless and only supports the puck. With MuJoCo's soft frictional
contact, strikes made the puck hop by several millimetres and the per-step deceleration varied by
up to 50% on felt, which buried the friction signal the experiment is about. MuJoCo still handles
support, the fall over the edge and rendering. The same scene is built in two styles: "plain" (what a world model would see in
stage 1b; fixed top camera, no shadows) and "hero" (shadows, reflections, perspective cameras) for
figures. Hero frames are re-rendered from logged qpos, so they never influence the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cira  # noqa: F401  (GL backend selection)
import mujoco
import numpy as np

from cira.config import EnvConfig

ASSETS = Path(__file__).parent / "assets"
FRICTION_EPS = 0.02  # m/s
LOAD_RGBA = ("0.93 0.90 0.80 1", "0.95 0.55 0.18 1", "0.16 0.22 0.38 1")


def _look_at(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    fmt = lambda v: " ".join(f"{x:.4f}" for x in v)  # noqa: E731
    return f'pos="{fmt(pos)}" xyaxes="{fmt(right)} {fmt(up)}"'


def scene_xml(cfg: EnvConfig, style: str = "plain") -> str:
    hero = style == "hero"
    cx = (cfg.edge_x - cfg.half_l) / 2
    hx = (cfg.edge_x + cfg.half_l) / 2
    hw = cfg.half_w
    r, hh = cfg.puck_radius, cfg.puck_half_height

    if hero:
        visual = """
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient=".28 .28 .28" diffuse=".3 .3 .3" specular="0 0 0"/>
    <map znear="0.01" haze="0.2"/>"""
        lights = """
    <light name="key" pos="-0.6 -0.9 2.4" dir="0.25 0.35 -1" castshadow="true" diffuse=".75 .75 .72" specular=".2 .2 .2"/>
    <light name="fill" pos="1.5 1.2 1.8" dir="-0.6 -0.5 -1" castshadow="false" diffuse=".25 .25 .28" specular="0 0 0"/>"""
        refl = {"felt": 0.0, "rubber": 0.03, "acrylic": 0.25}
        floor_mat = '<material name="floor" texture="floor" texrepeat="6 6" texuniform="true" reflectance="0.08"/>'
        sky = '<texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".86 .89 .93" width="512" height="512"/>'
    else:
        visual = """
    <global offwidth="512" offheight="512"/>
    <quality shadowsize="0" offsamples="4"/>
    <headlight ambient=".65 .65 .65" diffuse=".35 .35 .35" specular="0 0 0"/>"""
        lights = ""
        refl = {"felt": 0.0, "rubber": 0.0, "acrylic": 0.0}
        floor_mat = '<material name="floor" rgba=".85 .85 .85 1"/>'
        sky = ""

    mats = "\n    ".join(
        f'<texture name="tex_{n}" type="2d" file="{ASSETS / (n + ".png")}"/>'
        f'<material name="mat_{n}" texture="tex_{n}" texrepeat="3 3" texuniform="true" reflectance="{refl[n]}"/>'
        for n, _ in cfg.surfaces
    )
    pucks = "\n    ".join(
        f'<material name="puck_{i}" rgba="{rgba}" specular="0.3" shininess="0.5"/>'
        for i, rgba in enumerate(LOAD_RGBA[: len(cfg.loads)])
    )
    legs = "\n      ".join(
        f'<geom type="box" size="0.025 0.025 0.355" pos="{x:.3f} {y:.3f} -0.395" material="wood" contype="0" conaffinity="0"/>'
        for x in (cx - hx + 0.06, cx + hx - 0.06)
        for y in (-hw + 0.06, hw - 0.06)
    )
    tx, ty = cfg.target
    return f"""
<mujoco model="puck_strike">
  <option timestep="{cfg.timestep}" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual>{visual}
  </visual>
  <asset>
    {sky}
    <texture name="floor" type="2d" builtin="checker" rgb1=".93 .93 .93" rgb2=".87 .87 .88" width="512" height="512"/>
    {floor_mat}
    <material name="wood" rgba=".55 .40 .28 1" specular="0.1"/>
    {mats}
    {pucks}
  </asset>
  <worldbody>{lights}
    <geom name="floor" type="plane" size="30 30 0.1" pos="0 0 -0.75" material="floor"/>
    <body name="table">
      <geom name="mat" type="box" size="{hx} {hw} 0.003" pos="{cx} 0 -0.003" material="mat_{cfg.surfaces[0][0]}"
            priority="1" condim="1"/>
      <geom name="tabletop" type="box" size="{hx} {hw} 0.017" pos="{cx} 0 -0.023" material="wood"/>
      {legs}
      <geom name="edge" type="box" size="0.006 {hw} 0.0006" pos="{cfg.edge_x - 0.006} 0 0.0006" rgba=".85 .12 .12 1" contype="0" conaffinity="0"/>
      <geom name="target" type="cylinder" size="{cfg.success_tol} 0.0005" pos="{tx} {ty} 0.0005" rgba=".85 .12 .12 .35" contype="0" conaffinity="0"/>
    </body>
    <body name="puck" pos="{np.mean(cfg.start_x)} 0 {hh + 1e-4}">
      <freejoint name="puck"/>
      <geom name="puck" type="cylinder" size="{r} {hh}" mass="{cfg.loads[0]}" material="puck_0" condim="1"/>
    </body>
    <camera name="top" pos="0 0 1.75" fovy="45"/>
    <camera name="hero" {_look_at((-0.62, -1.3, 1.25), (0.08, 0.0, -0.2))} fovy="40"/>
    <camera name="edge" {_look_at((1.35, -0.95, 0.45), (0.42, 0.0, -0.12))} fovy="40"/>
  </worldbody>
</mujoco>
"""


@dataclass
class StrikeResult:
    states: np.ndarray  # (T+1, 4) observed [x, y, vx, vy]
    actions: np.ndarray  # (T, 2) impulse at t=0, zeros after
    qpos: np.ndarray  # (F, 7) free-joint pose at every observation step, incl. the fall
    final_xy: np.ndarray
    off_table: bool
    executed_impulse: np.ndarray


class PuckEnv:
    def __init__(self, cfg: EnvConfig | None = None, style: str = "plain"):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(scene_xml(self.cfg, style))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.mat_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mat")
        self.puck_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "puck")
        self.puck_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.mat_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"mat_{n}") for n, _ in self.cfg.surfaces]
        self.puck_mat_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"puck_{i}") for i in range(len(self.cfg.loads))
        ]
        self._base_mass = float(m.body_mass[self.puck_bid])
        self._base_inertia = m.body_inertia[self.puck_bid].copy()
        self.context = (0, 0)
        self.mu = self.cfg.surfaces[0][1]

    # ------------------------------------------------------------------ context & state
    def set_context(self, surface: int, load: int, rng: np.random.Generator | None = None) -> None:
        cfg, m = self.cfg, self.model
        jitter = 1.0 + (rng.normal(0.0, cfg.friction_jitter) if rng is not None else 0.0)
        self.mu = cfg.surfaces[surface][1] * jitter
        m.geom_matid[self.mat_gid] = self.mat_ids[surface]
        mass = cfg.loads[load]
        m.body_mass[self.puck_bid] = mass
        m.body_inertia[self.puck_bid] = self._base_inertia * mass / self._base_mass
        m.geom_matid[self.puck_gid] = self.puck_mat_ids[load]
        mujoco.mj_setConst(m, self.data)
        self.context = (surface, load)

    def reset(self, start_xy, vel_xy=(0.0, 0.0)) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = [start_xy[0], start_xy[1], self.cfg.puck_half_height + 1e-4]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:2] = vel_xy
        mujoco.mj_forward(self.model, d)

    def true_state(self) -> np.ndarray:
        return np.array([*self.data.qpos[:2], *self.data.qvel[:2]])

    def observe(self, rng: np.random.Generator | None) -> np.ndarray:
        s = self.true_state()
        if rng is not None:
            s[:2] += rng.normal(0.0, self.cfg.obs_pos_noise, 2)
            s[2:] += rng.normal(0.0, self.cfg.obs_vel_noise, 2)
        return s

    def on_table(self) -> bool:
        x, y = self.data.qpos[:2]
        c = self.cfg
        return (-c.half_l <= x <= c.edge_x) and abs(y) <= c.half_w and self.data.qpos[2] > -0.01

    # ------------------------------------------------------------------ dynamics
    def step(self, impulse=None) -> None:
        """Advance one observation step; an impulse (N s) is spread over the first impulse_steps."""
        c, d = self.cfg, self.data
        force = None if impulse is None else np.asarray(impulse) / (c.impulse_steps * c.timestep)
        weight = self.mu * self.model.body_mass[self.puck_bid] * 9.81
        for k in range(c.substeps):
            d.xfrc_applied[self.puck_bid, :] = 0.0
            if force is not None and k < c.impulse_steps:
                d.xfrc_applied[self.puck_bid, :2] = force
            if self.on_table() and d.qpos[2] > c.puck_half_height - 0.003:
                v = d.qvel[:2]
                # Coulomb friction, regularized below FRICTION_EPS so it stops without chattering.
                d.xfrc_applied[self.puck_bid, :2] -= weight * v / max(np.linalg.norm(v), FRICTION_EPS)
            mujoco.mj_step(self.model, d)
        d.xfrc_applied[self.puck_bid, :] = 0.0

    def execute_noise(self, impulse, rng: np.random.Generator | None) -> np.ndarray:
        j = np.asarray(impulse, float).copy()
        if rng is None:
            return j
        scale = 1.0 + rng.normal(0.0, self.cfg.impulse_noise)
        ang = np.deg2rad(rng.normal(0.0, self.cfg.impulse_angle_noise_deg))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        return scale * rot @ j

    def strike(
        self, impulse, rng: np.random.Generator | None = None, fall_steps: int = 0, z0=None, rest_steps: int = 2
    ) -> StrikeResult:
        """One single-contact episode. Transitions are recorded only while the puck is on the table.

        Pass the observation the plan was made from as z0, so that it is also the first state.
        """
        c = self.cfg
        z0 = self.observe(rng) if z0 is None else np.asarray(z0, float)
        executed = self.execute_noise(impulse, rng)
        states, actions, qpos = [z0], [], [self.data.qpos[:7].copy()]
        off, slow = False, 0
        for t in range(c.max_steps):
            self.step(executed if t == 0 else None)
            qpos.append(self.data.qpos[:7].copy())
            if not self.on_table():
                off = True
                break
            states.append(self.observe(rng))
            actions.append(np.asarray(impulse, float) if t == 0 else np.zeros(2))
            slow = slow + 1 if np.linalg.norm(self.data.qvel[:2]) < c.stop_speed else 0
            if slow >= rest_steps:
                break
        if off:
            for _ in range(fall_steps):
                self.step(None)
                qpos.append(self.data.qpos[:7].copy())
        return StrikeResult(
            states=np.array(states),
            actions=np.array(actions).reshape(-1, 2),
            qpos=np.array(qpos),
            final_xy=self.data.qpos[:2].copy(),
            off_table=off,
            executed_impulse=executed,
        )

    def set_qpos(self, q) -> None:
        self.data.qpos[:7] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
