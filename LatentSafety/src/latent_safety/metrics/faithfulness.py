"""Training diagnostics and cover bounds for safety-faithful representations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Vector = Sequence[float]


@dataclass(frozen=True)
class FaithfulnessAudit:
    pair_count: int
    mean_squared_hinge: float
    maximum_violation: float
    violating_fraction: float


def _distance(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("latent vectors must have the same dimension")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def pairwise_faithfulness_audit(
    latents: Sequence[Vector], margins: Sequence[float], *, kappa: float
) -> FaithfulnessAudit:
    """Evaluate ``(|h_i-h_j| - kappa ||z_i-z_j||)_+`` on observed pairs.

    A small empirical mean is only a training diagnostic. It is not a uniform certificate on the
    state space; use a verified cover, formal verifier, or explicitly probabilistic claim.
    """

    if kappa < 0.0 or not math.isfinite(kappa):
        raise ValueError("kappa must be finite and non-negative")
    if len(latents) != len(margins):
        raise ValueError("latents and margins must have the same length")

    violations: list[float] = []
    for left in range(len(latents)):
        for right in range(left + 1, len(latents)):
            violation = max(
                0.0,
                abs(float(margins[left]) - float(margins[right]))
                - kappa * _distance(latents[left], latents[right]),
            )
            violations.append(violation)

    count = len(violations)
    return FaithfulnessAudit(
        pair_count=count,
        mean_squared_hinge=(sum(value**2 for value in violations) / count if count else 0.0),
        maximum_violation=max(violations, default=0.0),
        violating_fraction=(sum(value > 0.0 for value in violations) / count if count else 0.0),
    )


def cover_robust_defect_bound(
    *,
    cover_radius: float,
    margin_lipschitz: float,
    encoder_lipschitz: float,
    kappa: float,
    latent_radius: float = 0.0,
) -> float:
    """Bound robust defect from pairwise faithfulness verified on an r-cover.

    The triangle inequality gives

    ``2 L_h r + kappa (delta + 2 L_E r)``.

    The two factors of two are intentional: each arbitrary state is approximated by its own cover
    point. This function does not check that the cover or all cover-point pairs were actually
    verified.
    """

    values = (cover_radius, margin_lipschitz, encoder_lipschitz, kappa, latent_radius)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("bound parameters must be finite and non-negative")
    return 2.0 * margin_lipschitz * cover_radius + kappa * (
        latent_radius + 2.0 * encoder_lipschitz * cover_radius
    )
