"""Finite-action audit for safety-feasible action information lost inside encoder fibers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Vector = Sequence[float]


@dataclass(frozen=True)
class ActionFiberAudit:
    """Exact-code action-conflict summary for a finite state/action audit set."""

    fiber_count: int
    nontrivial_fiber_count: int
    individually_viable_fiber_count: int
    conflicting_fiber_count: int
    conflict_fraction: float
    worst_required_violation: float


@dataclass(frozen=True)
class ActionNeighborhoodWitness:
    """One radius-neighborhood whose states admit no common audited action."""

    center_index: int
    member_indices: tuple[int, ...]
    best_common_margin: float
    required_violation: float


@dataclass(frozen=True)
class RadiusActionNeighborhoodAudit:
    """Common-action audit over latent balls centered at every supplied state."""

    delta: float
    state_count: int
    nontrivial_neighborhood_count: int
    individually_viable_neighborhood_count: int
    conflicting_neighborhood_count: int
    conflict_fraction: float
    worst_required_violation: float
    mean_required_violation: float
    tail_quantile: float
    tail_required_violation: float
    center_required_violations: tuple[float | None, ...]
    witnesses: tuple[ActionNeighborhoodWitness, ...]


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _code_key(vector: Vector, *, digits: int | None) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("latent codes must be finite")
    if digits is None:
        return values
    return tuple(round(value, digits) for value in values)


def audit_exact_action_fibers(
    latents: Sequence[Vector],
    action_safety_margins: Sequence[Sequence[float]],
    *,
    rounding_digits: int | None = None,
) -> ActionFiberAudit:
    """Audit whether states sharing a code have a common safe action.

    ``action_safety_margins[i][a] >= 0`` means action ``a`` is safe for state ``i`` under the
    chosen one-step or barrier condition. A fiber is a conflict when every state is individually
    viable but no single action is safe for all states. In that case no deterministic memoryless
    policy using only the shared code can satisfy all audited states.

    The result is exact only for the supplied states, exact/rounded code grouping, action grid, and
    safety condition. Continuous-action experiments need optimization or verification, not a coarse
    grid presented as proof.
    """

    if len(latents) != len(action_safety_margins):
        raise ValueError("latents and action_safety_margins must have the same length")
    if not latents:
        raise ValueError("at least one state is required")
    latent_dimension = len(latents[0])
    if latent_dimension == 0 or any(len(latent) != latent_dimension for latent in latents):
        raise ValueError("latent vectors must be non-empty and share one dimension")
    action_count = len(action_safety_margins[0])
    if action_count == 0:
        raise ValueError("at least one action is required")
    if any(len(row) != action_count for row in action_safety_margins):
        raise ValueError("every state must use the same action grid")
    if any(
        not math.isfinite(float(value)) for row in action_safety_margins for value in row
    ):
        raise ValueError("action safety margins must be finite")

    fibers: dict[tuple[float, ...], list[int]] = {}
    for index, latent in enumerate(latents):
        fibers.setdefault(_code_key(latent, digits=rounding_digits), []).append(index)

    nontrivial = 0
    individually_viable = 0
    conflicting = 0
    worst_required_violation = 0.0
    for indices in fibers.values():
        if len(indices) < 2:
            continue
        nontrivial += 1
        viable_indices = tuple(
            index
            for index in indices
            if max(float(value) for value in action_safety_margins[index]) >= 0.0
        )
        if len(viable_indices) < 2:
            continue
        individually_viable += 1
        robust_action_margins = [
            min(
                float(action_safety_margins[index][action])
                for index in viable_indices
            )
            for action in range(action_count)
        ]
        best_common_margin = max(robust_action_margins)
        if best_common_margin < 0.0:
            conflicting += 1
            worst_required_violation = max(worst_required_violation, -best_common_margin)

    return ActionFiberAudit(
        fiber_count=len(fibers),
        nontrivial_fiber_count=nontrivial,
        individually_viable_fiber_count=individually_viable,
        conflicting_fiber_count=conflicting,
        conflict_fraction=(conflicting / individually_viable if individually_viable else 0.0),
        worst_required_violation=worst_required_violation,
    )


def audit_radius_action_neighborhoods(
    latents: Sequence[Vector],
    action_safety_margins: Sequence[Sequence[float]],
    *,
    delta: float,
    max_witnesses: int = 100,
) -> RadiusActionNeighborhoodAudit:
    """Audit common finite-grid actions in every fixed-radius latent neighborhood.

    The reference population is first restricted to individually viable states. A neighborhood is
    then centered at each viable observed state and contains every viable supplied state within
    Euclidean distance ``delta``. It is conflicting when its shared best action has negative
    worst-state margin. This ordering keeps physical infeasibility separate from
    representation-induced conflict and detects higher-order conflicts that pair-only checks can
    miss.

    The result is a finite-reference witness, not a certificate for unsampled states. A continuous
    action space requires verified optimization rather than presenting a coarse grid as exact.
    """

    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be non-negative")
    if len(latents) != len(action_safety_margins):
        raise ValueError("latents and action_safety_margins must have the same length")
    if not latents:
        raise ValueError("at least one state is required")
    dimension = len(latents[0])
    if dimension == 0 or any(len(latent) != dimension for latent in latents):
        raise ValueError("latent vectors must be non-empty and share one dimension")
    vectors = [tuple(float(value) for value in latent) for latent in latents]
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("latent codes must be finite")
    action_count = len(action_safety_margins[0])
    if action_count == 0 or any(len(row) != action_count for row in action_safety_margins):
        raise ValueError("every state must use the same non-empty action grid")
    rows = [tuple(float(value) for value in row) for row in action_safety_margins]
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("action safety margins must be finite")

    tolerance = 0.0 if delta == 0.0 else max(1.0, delta) * 1e-12

    def distance(left: Vector, right: Vector) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))

    nontrivial = 0
    viable = 0
    conflicting = 0
    worst_violation = 0.0
    witnesses: list[ActionNeighborhoodWitness] = []
    center_required_violations: list[float | None] = [None] * len(vectors)
    viable_indices = tuple(index for index, row in enumerate(rows) if max(row) >= 0.0)
    for center_index in viable_indices:
        center = vectors[center_index]
        members = tuple(
            index
            for index in viable_indices
            if distance(center, vectors[index]) <= delta + tolerance
        )
        if len(members) < 2:
            continue
        nontrivial += 1
        viable += 1
        common_action_margins = [
            min(rows[index][action] for index in members) for action in range(action_count)
        ]
        best_common_margin = max(common_action_margins)
        center_required_violations[center_index] = max(0.0, -best_common_margin)
        if best_common_margin < 0.0:
            conflicting += 1
            required_violation = -best_common_margin
            worst_violation = max(worst_violation, required_violation)
            witnesses.append(
                ActionNeighborhoodWitness(
                    center_index=center_index,
                    member_indices=members,
                    best_common_margin=best_common_margin,
                    required_violation=required_violation,
                )
            )

    ordered = sorted(
        witnesses,
        key=lambda witness: (
            -witness.required_violation,
            witness.center_index,
            witness.member_indices,
        ),
    )
    viable_violations = [
        value for value in center_required_violations if value is not None
    ]
    tail_quantile = 0.95
    return RadiusActionNeighborhoodAudit(
        delta=delta,
        state_count=len(vectors),
        nontrivial_neighborhood_count=nontrivial,
        individually_viable_neighborhood_count=viable,
        conflicting_neighborhood_count=conflicting,
        conflict_fraction=(conflicting / viable if viable else 0.0),
        worst_required_violation=worst_violation,
        mean_required_violation=(
            sum(viable_violations) / len(viable_violations)
            if viable_violations
            else 0.0
        ),
        tail_quantile=tail_quantile,
        tail_required_violation=_linear_quantile(viable_violations, tail_quantile),
        center_required_violations=tuple(center_required_violations),
        witnesses=tuple(ordered[:max_witnesses]),
    )
