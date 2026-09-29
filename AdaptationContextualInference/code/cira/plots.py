"""Figures for scripts/plot_results.py and the notebook.

Colour carries meaning only where it helps: CIRA blue, CIRA-NoCue orange, baselines grey with
direct labels, Oracle dashed. Surfaces (a categorical factor) use the first three palette slots.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cira.agents import METHODS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED, GRID = "#9a9892", "#0b0b0b", "#52514e", "#e6e5e0"
SURFACE_COLORS = (BLUE, ORANGE, AQUA)
STYLE = {
    "CIRA": dict(color=BLUE, lw=2.0, zorder=5),
    "CIRA-NoCue": dict(color=ORANGE, lw=2.0, zorder=4),
    "Oracle": dict(color=GREY, lw=1.5, ls="--", zorder=3),
    "Reset-SGD": dict(color=GREY, lw=1.4, ls=":", zorder=2),
}
HIGHLIGHT = ("CIRA", "CIRA-NoCue")


def style(m):
    return STYLE.get(m, dict(color=GREY, lw=1.2, alpha=0.8, zorder=2))


def setup():
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 10,
    })


# ---------------------------------------------------------------------------- results
def fc_vs_rho(summary: pd.DataFrame, ax) -> None:
    for m in [m for m in METHODS if m in set(summary.method)]:
        s = summary[summary.method == m].sort_values("rho")
        st = style(m)
        ax.plot(s.rho, s.fc_rel, marker="o", ms=4, **st)
        if "fc_rel_ci" in s and len(s) > 1:
            ax.fill_between(s.rho, s.fc_rel - s.fc_rel_ci, s.fc_rel + s.fc_rel_ci, color=st["color"], alpha=0.12, lw=0)
        last = s.iloc[-1]
        ax.annotate(m, (last.rho, last.fc_rel), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK if m in HIGHLIGHT else MUTED)
    ax.set_xlabel("cue reliability ρ")
    ax.set_ylabel("first-contact error / Frozen")
    ax.set_title("First-contact prediction error")
    ax.set_ylim(bottom=0)


def success_bars(summary: pd.DataFrame, ax, rho: float, metric: str = "success_returns", title=None,
                 highlight=HIGHLIGHT) -> None:
    s = summary[np.isclose(summary.rho, rho)].sort_values(metric)
    colors = [style(m)["color"] if m in highlight and m in STYLE else (BLUE if m in highlight else GREY) for m in s.method]
    y = np.arange(len(s))
    ax.barh(y, s[metric], color=colors, height=0.6)
    ci = s.get(f"{metric}_ci")
    if ci is not None and (ci > 0).any():
        ax.errorbar(s[metric], y, xerr=ci, fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=2)
    for yi, v, c in zip(y, s[metric], ci if ci is not None else np.zeros(len(s))):
        ax.text(v + c + 0.01, yi, f"{100 * v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, s.method)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    what = "on returning contexts" if metric.endswith("returns") else "overall"
    ax.set_title(title or f"Strike success {what} (ρ = {rho:g})")
    ax.grid(axis="y", visible=False)


def rolling_error(df: pd.DataFrame, ax, rho: float, window: int = 15,
                  shown=("Reset-SGD", "Continual-SGD", "CIRA-NoCue", "CIRA")) -> None:
    d = df[np.isclose(df.rho, rho) & (df.stream == df.stream.min())]
    for m in shown:
        s = d[d.method == m].sort_values("episode")
        if not s.empty:
            ax.plot(s.episode, s.fc_err.rolling(window, min_periods=1).mean(), **style(m), label=m)
    first = d[d.method == shown[0]].sort_values("episode")
    for e in first.episode[first.ctx.diff().fillna(0) != 0]:
        ax.axvline(e, color="#d8d7d0", lw=0.6, zorder=0)
    ax.set_xlabel("episode (thin lines: context switches)")
    ax.set_ylabel(f"first-contact error [m], {window}-ep. mean")
    ax.set_title("Error over one stream")
    ax.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------- environment
def physics_check(env_cfg, ax, speeds=np.linspace(0.3, 2.0, 8)) -> None:
    """Simulated vs Coulomb stopping distance v0^2 / (2 mu g) for each surface."""
    from cira.env import PuckEnv

    env = PuckEnv(env_cfg)
    load = env_cfg.train_context[1]
    for s, (name, mu) in enumerate(env_cfg.surfaces):
        dist = []
        for v0 in speeds:
            env.set_context(s, load)
            env.reset((-0.55, 0.0), (v0, 0.0))
            r = env.strike((0.0, 0.0))
            dist.append(r.final_xy[0] + 0.55 if not r.off_table else np.nan)
        c = SURFACE_COLORS[s]
        theory = speeds**2 / (2 * mu * 9.81)
        ax.plot(speeds, theory, color=c, lw=1.2, alpha=0.6)
        ax.plot(speeds, dist, "o", color=c, ms=4)
        k = np.flatnonzero(theory <= 1.15)[-1]  # label the last point inside the axis
        ax.annotate(f"{name} (μ={mu})", (speeds[k], theory[k]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("initial speed [m/s]")
    ax.set_ylabel("sliding distance [m]")
    ax.set_ylim(0, 1.2)
    ax.set_title("Physics check: dots = MuJoCo, lines = v²/(2μg)")


def stream_timeline(stream, env_cfg, ax) -> None:
    from cira.render import context_name

    ids = sorted(set(stream.ctx))
    ypos = {c: i for i, c in enumerate(ids)}
    e = np.arange(len(stream))
    ok = stream.shown == stream.ctx
    ax.scatter(e[ok], [ypos[c] for c in stream.ctx[ok]], s=10, color=INK, label="cue shows true context", zorder=3)
    ax.scatter(e[~ok], [ypos[c] for c in stream.ctx[~ok]], s=28, marker="x", color=ORANGE, label="misleading cue", zorder=4)
    ax.set_yticks(range(len(ids)), [context_name(env_cfg, c) for c in ids])
    ax.set_xlabel("episode")
    ax.set_title(f"A deployment stream (ρ = {stream.rho:g})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", visible=False)


def residual_heatmap(matrix: np.ndarray, env_cfg, ax, title="Mean |residual| / σ of the frozen model") -> None:
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 * matrix.max() else INK)
    ax.set_xticks(range(len(env_cfg.loads)), [f"{m:.1f} kg" for m in env_cfg.loads])
    ax.set_yticks(range(len(env_cfg.surfaces)), [n for n, _ in env_cfg.surfaces])
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def belief_heatmap(pre_contact, labels, stream, env_cfg, ax) -> None:
    """Pre-contact belief over memories (rows, named by majority context) for every episode."""
    from cira.render import context_name

    ids = sorted({m for pre in pre_contact for m in pre if m >= 0}, key=lambda m: (labels.get(m, 99), m))
    rows = ids + [-1]
    B = np.array([[pre.get(m, 0.0) for pre in pre_contact] for m in rows])
    ax.imshow(B, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    names = [f"#{m} {context_name(env_cfg, labels[m])}" if labels.get(m, -1) >= 0 else f"#{m}" for m in ids]
    ax.set_yticks(range(len(rows)), names + ["new context"])
    # mark the true context: a dot on every row whose memory is labelled with it
    for e, c in enumerate(stream.ctx[: len(pre_contact)]):
        for r, m in enumerate(ids):
            if labels.get(m) == c:
                ax.plot(e, r, ".", color=ORANGE, ms=3)
    ax.set_xlabel("episode (orange dots: memories of the true context)")
    ax.set_title("CIRA's belief before contact")
    ax.grid(False)
