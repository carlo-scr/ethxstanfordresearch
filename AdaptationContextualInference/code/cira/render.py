"""Figure and video rendering from logged puck poses (never used inside the experiment loop)."""

from __future__ import annotations

from pathlib import Path

import cira  # noqa: F401  (GL backend selection)
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cira.config import EnvConfig
from cira.env import LOAD_RGBA, PuckEnv


def context_name(env_cfg: EnvConfig, ctx: int) -> str:
    n_l = len(env_cfg.loads)
    return f"{env_cfg.surfaces[ctx // n_l][0]} · {env_cfg.loads[ctx % n_l]:.1f} kg"


class SceneRenderer:
    """Renders logged free-joint poses of the puck in the 'hero' or 'plain' scene style."""

    def __init__(self, env_cfg: EnvConfig | None = None, style: str = "hero", width: int = 1280, height: int = 720):
        self.env = PuckEnv(env_cfg, style)
        self.renderer = mujoco.Renderer(self.env.model, height=height, width=width)

    def frame(self, qpos, ctx: int, camera: str = "hero", ghosts=None, arrow=None) -> np.ndarray:
        """ghosts: (n, 3) earlier puck positions drawn as fading copies; arrow: (from_xyz, to_xyz)."""
        n_l = len(self.env.cfg.loads)
        self.env.set_context(ctx // n_l, ctx % n_l)
        self.env.set_qpos(qpos)
        self.renderer.update_scene(self.env.data, camera=camera)
        scn = self.renderer.scene
        rgb = np.array([float(x) for x in LOAD_RGBA[ctx % n_l].split()[:3]])
        c = self.env.cfg
        if ghosts is not None and len(ghosts):
            for k, p in enumerate(ghosts):
                if scn.ngeom >= scn.maxgeom:
                    break
                alpha = 0.08 + 0.35 * (k + 1) / len(ghosts)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([c.puck_radius, c.puck_half_height * 0.98, 0.0]),
                    np.asarray(p, float),
                    np.eye(3).ravel(),
                    np.array([*rgb, alpha], np.float32),
                )
                scn.ngeom += 1
        if arrow is not None and scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.85, 0.12, 0.12, 0.9], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, np.asarray(arrow[0], float), np.asarray(arrow[1], float))
            scn.ngeom += 1
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def overlay(frame: np.ndarray, title: str, lines=(), bars=None, title_color=(30, 30, 30)) -> np.ndarray:
    """Draw a title, text lines and an optional belief bar chart [(label, p, highlight)] on a frame."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    s = H / 448
    pad = int(12 * s)
    big, small = _font(int(22 * s)), _font(int(15 * s))
    d.rounded_rectangle([pad, pad, pad + int(330 * s), pad + int((34 + 20 * len(lines)) * s)], int(8 * s), fill=(255, 255, 255, 215))
    d.text((pad + int(10 * s), pad + int(6 * s)), title, font=big, fill=title_color)
    for i, line in enumerate(lines):
        d.text((pad + int(10 * s), pad + int((34 + 20 * i) * s)), line, font=small, fill=(60, 60, 60))
    if bars:
        bw, bh = int(170 * s), int(16 * s)
        x0 = W - pad - bw - int(118 * s)
        y0 = pad
        box_h = int(30 * s + (bh + 5 * s) * len(bars))
        d.rounded_rectangle([x0 - int(10 * s), y0, W - pad, y0 + box_h], int(8 * s), fill=(255, 255, 255, 215))
        d.text((x0, y0 + int(5 * s)), "belief before contact", font=small, fill=(60, 60, 60))
        for i, (label, p, hi) in enumerate(bars):
            y = y0 + int(26 * s) + i * (bh + int(5 * s))
            d.rectangle([x0, y, x0 + bw, y + bh], fill=(225, 225, 228, 255))
            d.rectangle([x0, y, x0 + int(bw * p), y + bh], fill=(34, 139, 84, 255) if hi else (120, 124, 135, 255))
            d.text((x0 + bw + int(8 * s), y), label, font=small, fill=(40, 40, 40))
    return np.asarray(img)


def write_video(path: Path, frames, fps: int = 25) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=16)
