"""Finite-sample witnesses for static safety ambiguity.

The population robust defect at radius ``delta`` is a supremum over pairs in a region of interest.
An observed pair inside that fixed radius is therefore a valid *witness* and its safe margin lower
bounds the population supremum, assuming the supplied margins and latent distances are exact.

This module intentionally keeps the O(n^2) reference implementation dependency-free. Large-scale
experiments should provide a tested nearest-neighbor backend without changing these semantics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median

Vector = Sequence[float]


@dataclass(frozen=True)
class DefectEstimate:
    """Observed robust-defect witness at one pre-specified latent radius."""

    delta: float
    witness_margin: float
    confounded_safe_count: int
    safe_count: int
    confounded_fraction: float
    margin_quantile: float
    quantile_level: float


@dataclass(frozen=True)
class NearestUnsafeDiagnostic:
    """Scale-dependent nearest-neighbor diagnostic, never an exact-collision certificate."""

    safe_count: int
    cross_boundary_nearest_count: int
    cross_boundary_fraction: float
    max_margin: float
    distances: tuple[float, ...]


@dataclass(frozen=True)
class DefectWitness:
    """One safe sample and its closest audited unsafe neighbor inside the radius."""

    safe_index: int
    unsafe_index: int
    distance: float
    safe_margin: float
    unsafe_margin: float


@dataclass(frozen=True)
class DetailedDefectEstimate:
    """Robust-defect summary plus a bounded, traceable witness list."""

    estimate: DefectEstimate
    witnesses: tuple[DefectWitness, ...]


def _as_vectors(latents: Sequence[Vector], margins: Sequence[float]) -> list[tuple[float, ...]]:
    if len(latents) != len(margins):
        raise ValueError("latents and margins must have the same length")
    if not latents:
        raise ValueError("at least one latent is required")

    vectors = [tuple(float(value) for value in vector) for vector in latents]
    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("latent vectors must be non-empty")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("all latent vectors must have the same dimension")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("latent vectors must contain only finite values")
    if any(not math.isfinite(float(margin)) for margin in margins):
        raise ValueError("margins must contain only finite values")
    return vectors


def _distance(left: Vector, right: Vector) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _upper_quantile(values: Sequence[float], level: float) -> float:
    if not 0.0 <= level <= 1.0:
        raise ValueError("quantile level must be in [0, 1]")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(level * len(ordered)) - 1)
    return ordered[index]


def empirical_robust_defect(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    delta: float,
    quantile: float = 0.99,
) -> DefectEstimate:
    """Return safe-margin witnesses with an unsafe sample within a fixed radius.

    Safe samples use ``h >= 0`` and unsafe samples use ``h < 0``. ``witness_margin`` is a
    finite-sample lower bound on the population *supremum* defect at this exact, externally chosen
    radius. ``margin_quantile`` is descriptive only unless paired with a valid sampling/tolerance
    argument. It is not automatically a population bound.
    """

    return empirical_robust_defect_details(
        latents,
        margins,
        delta=delta,
        quantile=quantile,
        max_witnesses=0,
    ).estimate


def empirical_robust_defect_details(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    delta: float,
    quantile: float = 0.99,
    max_witnesses: int = 100,
) -> DetailedDefectEstimate:
    """Return the summary and representative sample-level witnesses.

    At most one unsafe partner is retained per confounded safe sample: the closest one, with ties
    broken by input index. The returned list is ordered by decreasing safe margin and increasing
    distance before truncation. Counts always use all audited samples, so changing
    ``max_witnesses`` cannot change the metric.
    """

    if delta < 0.0 or not math.isfinite(delta):
        raise ValueError("delta must be finite and non-negative")
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be non-negative")
    vectors = _as_vectors(latents, margins)
    safe_indices = [index for index, margin in enumerate(margins) if margin >= 0.0]
    unsafe_indices = [index for index, margin in enumerate(margins) if margin < 0.0]

    confounded_margins: list[float] = []
    witnesses: list[DefectWitness] = []
    tolerance = 0.0 if delta == 0.0 else max(1.0, delta) * 1e-12
    for safe_index in safe_indices:
        inside = [
            (distance, unsafe_index)
            for unsafe_index in unsafe_indices
            if (distance := _distance(vectors[safe_index], vectors[unsafe_index]))
            <= delta + tolerance
        ]
        if not inside:
            continue
        distance, unsafe_index = min(inside)
        safe_margin = float(margins[safe_index])
        confounded_margins.append(safe_margin)
        witnesses.append(
            DefectWitness(
                safe_index=safe_index,
                unsafe_index=unsafe_index,
                distance=distance,
                safe_margin=safe_margin,
                unsafe_margin=float(margins[unsafe_index]),
            )
        )

    safe_count = len(safe_indices)
    confounded_count = len(confounded_margins)
    estimate = DefectEstimate(
        delta=delta,
        witness_margin=max(confounded_margins, default=0.0),
        confounded_safe_count=confounded_count,
        safe_count=safe_count,
        confounded_fraction=(confounded_count / safe_count if safe_count else 0.0),
        margin_quantile=_upper_quantile(confounded_margins, quantile),
        quantile_level=quantile,
    )
    ordered = sorted(
        witnesses,
        key=lambda witness: (
            -witness.safe_margin,
            witness.distance,
            witness.safe_index,
            witness.unsafe_index,
        ),
    )
    return DetailedDefectEstimate(estimate=estimate, witnesses=tuple(ordered[:max_witnesses]))


def empirical_defect_curve(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    deltas: Iterable[float],
    quantile: float = 0.99,
) -> tuple[DefectEstimate, ...]:
    """Evaluate the finite-sample witness over a pre-registered radius grid."""

    radius_grid = tuple(float(delta) for delta in deltas)
    if tuple(sorted(set(radius_grid))) != radius_grid:
        raise ValueError("deltas must be unique and sorted in increasing order")
    estimates = tuple(
        empirical_robust_defect(
            latents,
            margins,
            delta=delta,
            quantile=quantile,
        )
        for delta in radius_grid
    )
    if any(
        later.witness_margin + 1e-12 < earlier.witness_margin
        for earlier, later in zip(estimates, estimates[1:])
    ):
        raise AssertionError("robust defect must be monotone in delta")
    return estimates


def median_pairwise_distance(latents: Sequence[Vector]) -> float:
    """Return a deterministic scale diagnostic; never use test data to tune a certificate radius."""

    vectors = _as_vectors(latents, [0.0] * len(latents))
    distances = [
        _distance(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return median(distances) if distances else 0.0


def nearest_unsafe_diagnostic(
    latents: Sequence[Vector], margins: Sequence[float]
) -> NearestUnsafeDiagnostic:
    """Compute nearest-unsafe distances for plotting and estimator sanity checks.

    Every finite continuous sample has nearest neighbors. A cross-boundary neighbor does not imply
    equal latent codes, so this output must not be reported as an exact-defect lower bound.
    """

    vectors = _as_vectors(latents, margins)
    safe_indices = [index for index, margin in enumerate(margins) if margin >= 0.0]
    unsafe_indices = [index for index, margin in enumerate(margins) if margin < 0.0]
    nearest_distances: list[float] = []
    margins_with_unsafe_neighbor: list[float] = []

    for safe_index in safe_indices:
        unsafe_distance = min(
            (_distance(vectors[safe_index], vectors[index]) for index in unsafe_indices),
            default=math.inf,
        )
        other_safe_distance = min(
            (
                _distance(vectors[safe_index], vectors[index])
                for index in safe_indices
                if index != safe_index
            ),
            default=math.inf,
        )
        nearest_distances.append(unsafe_distance)
        if unsafe_distance < other_safe_distance:
            margins_with_unsafe_neighbor.append(float(margins[safe_index]))

    count = len(margins_with_unsafe_neighbor)
    safe_count = len(safe_indices)
    return NearestUnsafeDiagnostic(
        safe_count=safe_count,
        cross_boundary_nearest_count=count,
        cross_boundary_fraction=(count / safe_count if safe_count else 0.0),
        max_margin=max(margins_with_unsafe_neighbor, default=0.0),
        distances=tuple(nearest_distances),
    )
