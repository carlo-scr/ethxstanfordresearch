"""Diagnostic figures for a finished run (PDF + PNG in runs/<name>/figures).

    python scripts/plot_results.py --name main

For the paper, point pgfplots at runs/<name>/summary.csv instead so fonts match the LaTeX.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from cira import plots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="quick")
    ap.add_argument("--rho", type=float, default=0.8, help="cue reliability for the per-rho panels")
    args = ap.parse_args()
    run = Path("runs") / args.name
    df, summary = pd.read_csv(run / "episodes.csv"), pd.read_csv(run / "summary.csv")
    out = run / "figures"
    out.mkdir(exist_ok=True)
    plots.setup()
    panels = [("success", lambda ax: plots.success_bars(summary, ax, args.rho)),
              ("rolling", lambda ax: plots.rolling_error(df, ax, args.rho))]
    if summary.rho.nunique() > 1:
        panels.insert(0, ("fc_vs_rho", lambda ax: plots.fc_vs_rho(summary, ax)))
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
        draw(ax)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}")
        plt.close(fig)
        print(f"saved {out / name}.pdf/.png")


if __name__ == "__main__":
    main()
