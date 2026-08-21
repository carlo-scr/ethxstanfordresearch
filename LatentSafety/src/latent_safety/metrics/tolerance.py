"""Distribution-free one-sided tolerance calculations for precomputed violation scores."""

from __future__ import annotations

import math


def tolerance_confidence(*, sample_count: int, tail_mass: float, order: int = 1) -> float:
    """Confidence that the ``order``-th largest score leaves at most ``tail_mass`` above it.

    This is ``P[Binomial(N, alpha) >= order]``. It assumes the supplied violation scores are i.i.d.
    draws of the *population score*. If each score itself uses an approximate neighbor search or a
    finite unsafe reference set, that inner approximation needs separate analysis.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must be in (0, 1)")
    if not 1 <= order <= sample_count:
        raise ValueError("order must be in [1, sample_count]")
    lower_tail = sum(
        math.comb(sample_count, index)
        * tail_mass**index
        * (1.0 - tail_mass) ** (sample_count - index)
        for index in range(order)
    )
    return max(0.0, min(1.0, 1.0 - lower_tail))


def required_maximum_samples(*, tail_mass: float, failure_probability: float) -> int:
    """Exact sample count for a maximum-based one-sided tolerance threshold."""

    if not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must be in (0, 1)")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must be in (0, 1)")
    return math.ceil(math.log(failure_probability) / math.log(1.0 - tail_mass))

