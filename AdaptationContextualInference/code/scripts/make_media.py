"""Figures and videos for the puck strike.

    python scripts/make_media.py stills   # same strike on felt / rubber / acrylic, with ghost trails
    python scripts/make_media.py obs      # what a world model sees in stage 1b: plain top view, all contexts
    python scripts/make_media.py video    # Reset-SGD vs CIRA side by side on a short recurring stream
    python scripts/make_media.py all

Outputs go to runs/media/. Hero renders are re-rendered from logged poses, so they never touch the experiment.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from cira.config import ExperimentConfig
from cira.media import comparison_video, obs_grid, teaser_strip

OUT = Path("runs/media")
DEMO_SCHEDULE = [5] * 3 + [6] * 3 + [5] * 3 + [2] * 3 + [6] * 3 + [2] * 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stills", "obs", "video", "all"])
    ap.add_argument("--frozen", default="runs/frozen.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    if args.what in ("stills", "all"):
        teaser_strip(cfg, OUT)
        print(f"saved {OUT / 'teaser_strip.png'}")
    if args.what in ("obs", "all"):
        obs_grid(cfg, OUT)
        print(f"saved {OUT / 'obs_grid.png'}")
    if args.what in ("video", "all"):
        from cira.frozen import FrozenModel
        from cira.runner import run_stream
        from cira.stream import make_stream

        frozen = FrozenModel.load(args.frozen)
        stream = make_stream(cfg.env, replace(cfg.stream, rho=0.95), seed=7, schedule=np.array(DEMO_SCHEDULE))
        runs = {m: run_stream(m, stream, frozen, cfg, record=True) for m in ("Reset-SGD", "CIRA")}
        print(f"saved {comparison_video(cfg, stream, runs, OUT / 'reset_vs_cira.mp4')}")


if __name__ == "__main__":
    main()
