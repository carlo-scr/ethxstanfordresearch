"""Deterministic synthetic controls with known representation defects."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AliasingFixture:
    margins: tuple[float, ...]
    faithful_latents: tuple[tuple[float, float], ...]
    collapsed_latents: tuple[tuple[float, float], ...]


def smooth_unit_bump(value: float) -> float:
    """A compactly supported C-infinity bump with value one at the origin."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if abs(value) >= 1.0:
        return 0.0
    return math.exp(1.0 - 1.0 / (1.0 - value * value))


def localized_collision_encoder(
    state: float,
    *,
    collision_margin: float,
    width: float,
) -> float:
    """Smoothly map ``+/-collision_margin`` to zero while staying identity elsewhere.

    On ``[-1, 1]``, require ``0 < width < min(a, 1-a)`` for ``a=collision_margin``.  The two bump
    supports are then disjoint and contained in the state interval.  With the identity decoder and
    uniform state distribution, squared reconstruction loss is at most ``2*a**2*width`` even
    though the states ``-a`` and ``+a`` have exactly the same code.
    """

    state = float(state)
    collision_margin = float(collision_margin)
    width = float(width)
    if not all(math.isfinite(value) for value in (state, collision_margin, width)):
        raise ValueError("state, collision_margin, and width must be finite")
    if not 0.0 < collision_margin < 1.0:
        raise ValueError("collision_margin must lie in (0, 1)")
    if not 0.0 < width < min(collision_margin, 1.0 - collision_margin):
        raise ValueError(
            "width must lie in (0, min(collision_margin, 1-collision_margin))"
        )
    right = smooth_unit_bump((state - collision_margin) / width)
    left = smooth_unit_bump((state + collision_margin) / width)
    return state - collision_margin * right + collision_margin * left


def localized_collision_mse_bound(*, collision_margin: float, width: float) -> float:
    """Upper bound the uniform-state identity-decoder MSE of the localized collision."""

    # Reuse the encoder's complete parameter validation at a state outside both supports.
    localized_collision_encoder(
        0.0,
        collision_margin=collision_margin,
        width=width,
    )
    return 2.0 * float(collision_margin) ** 2 * float(width)


def signed_action_margins(state: float) -> tuple[float, float]:
    """Margins for actions ``(-1,+1)``; opposite nonzero states need opposite actions."""

    state = float(state)
    if not math.isfinite(state):
        raise ValueError("state must be finite")
    return (-state, state)


def make_aliasing_fixture(
    *, n_pairs: int, max_margin: float, seed: int
) -> AliasingFixture:
    """Create mirrored safe/unsafe pairs that collide only under ``collapsed_latents``."""

    if n_pairs < 2:
        raise ValueError("n_pairs must be at least two")
    if max_margin <= 0.0:
        raise ValueError("max_margin must be positive")

    generator = random.Random(seed)
    margins: list[float] = []
    faithful: list[tuple[float, float]] = []
    collapsed: list[tuple[float, float]] = []
    for index in range(n_pairs):
        margin = max_margin * (index + 1) / n_pairs
        nuisance = generator.uniform(-1.0, 1.0)
        margins.extend((margin, -margin))
        faithful.extend(((margin, nuisance), (-margin, nuisance)))
        collapsed.extend(((margin, nuisance), (margin, nuisance)))
    return AliasingFixture(
        margins=tuple(margins),
        faithful_latents=tuple(faithful),
        collapsed_latents=tuple(collapsed),
    )
