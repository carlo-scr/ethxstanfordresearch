"""Stills and videos built from logged puck poses (used by scripts/make_media.py and the notebook)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cira.config import ExperimentConfig
from cira.env import PuckEnv
from cira.render import SceneRenderer, context_name, overlay, write_video


def strike_image(renderer: SceneRenderer, env: PuckEnv, ctx: int, impulse, start=(-0.45, 0.0), label=True):
    """Simulate one strike without noise and render its end state with a ghost trail."""
    cfg = env.cfg
    n_l = len(cfg.loads)
    env.set_context(ctx // n_l, ctx % n_l)
    env.reset(start)
    r = env.strike(np.asarray(impulse, float), None, fall_steps=15)
    h = cfg.puck_half_height
    d = np.asarray(impulse, float) / (np.linalg.norm(impulse) + 1e-9)
    arrow = ((start[0] - 0.16 * d[0], start[1] - 0.16 * d[1], h), (start[0] - 0.06 * d[0], start[1] - 0.06 * d[1], h))
    img = renderer.frame(r.qpos[-1], ctx, "hero", ghosts=r.qpos[:-1:2, :3], arrow=arrow)
    if r.off_table:
        outcome = "slides off the table"
    else:
        err = np.linalg.norm(r.final_xy - cfg.target)
        outcome = f"stops {100 * err:.0f} cm from target" + (" (hit)" if err < cfg.success_tol else "")
    if label:
        img = overlay(img, context_name(cfg, ctx), [f"impulse ({impulse[0]:.2f}, {impulse[1]:.2f}) N s", outcome])
    return img, r, outcome


def teaser_strip(cfg: ExperimentConfig, out: Path, impulse=(1.25, 0.0)) -> np.ndarray:
    env, hero = PuckEnv(cfg.env), SceneRenderer(cfg.env, "hero", 1600, 900)
    n_l, panels = len(cfg.env.loads), []
    for s, (name, mu) in enumerate(cfg.env.surfaces):
        ctx = s * n_l + cfg.env.train_context[1]
        img, _, outcome = strike_image(hero, env, ctx, impulse, label=False)
        Image.fromarray(img).save(out / f"strike_{name}.png")
        panels.append(overlay(img[::2, ::2], f"{name} (mu = {mu})", [f"same strike, {outcome}"]))
    hero.close()
    strip = np.concatenate(panels, 1)
    Image.fromarray(strip).save(out / "teaser_strip.png")
    return strip


def obs_grid(cfg: ExperimentConfig, out: Path, size: int = 224) -> np.ndarray:
    """What a world model sees in stage 1b: the plain top view for every context."""
    plain = SceneRenderer(cfg.env, "plain", size, size)
    n_s, n_l = len(cfg.env.surfaces), len(cfg.env.loads)
    q = np.array([-0.4, 0.0, cfg.env.puck_half_height, 1, 0, 0, 0])
    rows = [
        np.concatenate([overlay(plain.frame(q, s * n_l + l, "top"), context_name(cfg.env, s * n_l + l)) for l in range(n_l)], 1)
        for s in range(n_s)
    ]
    plain.close()
    grid = np.concatenate(rows, 0)
    Image.fromarray(grid).save(out / "obs_grid.png")
    return grid


def comparison_video(cfg: ExperimentConfig, stream, runs: dict, path: Path, width: int = 800, height: int = 448,
                     log=print) -> Path:
    """Side-by-side video of recorded runs {method: run_stream(..., record=True)} on the same stream."""
    hero = SceneRenderer(cfg.env, "hero", width, height)
    colors = {"CIRA": (20, 110, 60), "CIRA-NoCue": (190, 80, 30), "Reset-SGD": (150, 40, 40)}
    frames = []
    for e in range(len(stream)):
        ctx = int(stream.ctx[e])
        cue_note = "" if stream.shown[e] == ctx else "  (misleading cue)"
        panels = []
        for m, (rows, extras) in runs.items():
            row, ep = rows[e], extras["episodes"][e]
            if row["off_table"]:
                outcome = "missed: slid off the table"
            else:
                outcome = f"{'hit' if row['success'] else 'missed'}: {row['stop_err'] * 100:.0f} cm from target"
            bars = None
            if extras.get("pre_contact"):
                labels = extras["memory_labels"]
                pre = sorted(extras["pre_contact"][e].items(), key=lambda kv: -kv[1])[:3]
                bars = [("new context" if mid < 0 else context_name(cfg.env, labels[mid]), p, mid >= 0 and labels[mid] == ctx)
                        for mid, p in pre]
            lines = [f"episode {e + 1}: {context_name(cfg.env, ctx)}{cue_note}", outcome]
            imgs = [overlay(hero.frame(q, ctx, "hero"), m, lines, bars, colors.get(m, (30, 30, 30))) for q in ep["qpos"]]
            panels.append(imgs + [imgs[-1]] * 12)
        n = max(len(p) for p in panels)
        frames.extend(np.concatenate([p[min(i, len(p) - 1)] for p in panels], 1) for i in range(n))
        if log and (e + 1) % 5 == 0:
            log(f"rendered {e + 1}/{len(stream)} episodes")
    hero.close()
    write_video(path, frames)
    return path
