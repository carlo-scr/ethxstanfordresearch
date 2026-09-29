"""Stage-1a experiment: every method on every (rho, stream seed), in parallel and resumable.

    python scripts/run_experiment.py --name quick --episodes 150 --streams 2 --rho 0.8
    python scripts/run_experiment.py --name main  --episodes 500 --streams 10 --rho 0.111 0.5 0.8 0.95

Writes runs/<name>/jobs/*.csv (one per method x rho x stream), episodes.csv, summary.csv, config.json.
Rerunning the same command skips finished jobs.
"""

import argparse

import pandas as pd

from cira.agents import METHODS
from cira.experiment import default_variants, run_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="quick")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--rho", nargs="+", type=float, default=[0.8])
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--frozen", default="runs/frozen.pt")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    _, summary = run_grid(args.name, default_variants(args.methods), args.rho, args.streams, args.episodes,
                          args.frozen, args.workers)
    cols = ["method", "rho", "fc_err", "fc_rel", "strike_err", "success", "success_returns", "off_table", "p_true", "n_memories"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
