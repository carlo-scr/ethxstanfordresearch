"""Traceable finite-radius safe-action-sufficiency audits for pretrained features."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from latent_safety.records import AuditRecord, validate_records

Vector = Sequence[float]


@dataclass(frozen=True)
class CommonActionWitness:
    """One viable neighborhood with no common action at the requested buffer."""

    center_index: int
    center_sample_id: str
    member_indices: tuple[int, ...]
    member_sample_ids: tuple[str, ...]
    best_action_index: int
    best_common_margin: float
    required_violation: float


@dataclass(frozen=True)
class CommonActionNeighborhoodAudit:
    """Finite-reference buffered common-action audit with explicit coverage status."""

    delta: float
    gamma: float
    sample_count: int
    action_count: int
    viable_sample_count: int
    viable_fraction: float
    eligible_center_count: int
    eligible_given_viable: float | None
    conflicting_center_count: int
    conflict_fraction: float | None
    mean_required_violation: float | None
    tail_quantile: float
    tail_required_violation: float | None
    worst_required_violation: float | None
    median_viable_nonself_neighborhood_mass: float | None
    trajectory_count: int
    evaluable_trajectory_count: int
    empty_trajectory_ids: tuple[str, ...]
    trajectory_balanced_tail_required_violation: float | None
    status: str
    center_required_violations: tuple[float | None, ...]
    center_best_actions: tuple[int | None, ...]
    center_member_counts: tuple[int, ...]
    witnesses: tuple[CommonActionWitness, ...]


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    if nonnegative and normalized < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return normalized


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("a quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distance(left: Vector, right: Vector) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def audit_common_action_neighborhoods(
    latents: Sequence[Vector],
    action_safety_margins: Sequence[Sequence[float]],
    *,
    delta: float,
    gamma: float = 0.0,
    sample_ids: Sequence[str] | None = None,
    trajectory_ids: Sequence[str] | None = None,
    comparison_group_ids: Sequence[str] | None = None,
    tail_quantile: float = 0.95,
    max_witnesses: int = 100,
) -> CommonActionNeighborhoodAudit:
    """Audit viable-only neighborhoods using held-out physical action profiles.

    The function first constructs ``I_gamma = {i: max_a q_i(a) >= gamma}``, then forms each
    neighborhood only over members of ``I_gamma``.  This order prevents a physically infeasible
    point from hiding a representation-induced conflict between viable points.  The closed audit
    ball uses ``distance <= delta``; the differentiable training loss intentionally uses a strict
    radius through its compact-support hinge.

    If any supplied trajectory has no eligible center, the trajectory-balanced endpoint is
    ``None`` and ``status`` records incomplete coverage rather than returning a misleading zero.
    ``comparison_group_ids`` can prohibit edges between independently declared physical groups.
    """

    radius = _finite(delta, label="delta", nonnegative=True)
    buffer = _finite(gamma, label="gamma", nonnegative=True)
    quantile = _finite(tail_quantile, label="tail_quantile", nonnegative=True)
    if quantile > 1.0:
        raise ValueError("tail_quantile must lie in [0, 1]")
    if isinstance(max_witnesses, bool) or not isinstance(max_witnesses, int):
        raise ValueError("max_witnesses must be an integer")
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be nonnegative")
    if len(latents) != len(action_safety_margins) or not latents:
        raise ValueError("latents and action_safety_margins must have the same positive length")

    vectors = tuple(tuple(float(value) for value in vector) for vector in latents)
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("latent vectors must be non-empty and share one dimension")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("latent vectors must be finite")

    profiles = tuple(
        tuple(float(value) for value in profile) for profile in action_safety_margins
    )
    action_count = len(profiles[0])
    if action_count < 1 or any(len(profile) != action_count for profile in profiles):
        raise ValueError("action profiles must be non-empty and share one action dimension")
    if any(not math.isfinite(value) for profile in profiles for value in profile):
        raise ValueError("action profiles must be finite")

    count = len(vectors)
    resolved_sample_ids = (
        tuple(str(index) for index in range(count))
        if sample_ids is None
        else tuple(str(value) for value in sample_ids)
    )
    resolved_trajectory_ids = (
        tuple("reference" for _ in range(count))
        if trajectory_ids is None
        else tuple(str(value) for value in trajectory_ids)
    )
    resolved_group_ids = (
        None
        if comparison_group_ids is None
        else tuple(str(value) for value in comparison_group_ids)
    )
    for label, values in (
        ("sample_ids", resolved_sample_ids),
        ("trajectory_ids", resolved_trajectory_ids),
    ):
        if len(values) != count or any(not value for value in values):
            raise ValueError(f"{label} must contain one non-empty value per sample")
    if len(set(resolved_sample_ids)) != count:
        raise ValueError("sample_ids must be unique")
    if resolved_group_ids is not None and (
        len(resolved_group_ids) != count or any(not value for value in resolved_group_ids)
    ):
        raise ValueError("comparison_group_ids must contain one non-empty value per sample")

    viable_indices = tuple(
        index for index, profile in enumerate(profiles) if max(profile) >= buffer
    )
    closed_radius = radius if radius == 0.0 else math.nextafter(radius, math.inf)
    center_violations: list[float | None] = [None] * count
    center_best_actions: list[int | None] = [None] * count
    center_member_counts = [0] * count
    witnesses: list[CommonActionWitness] = []
    eligible_values: list[float] = []
    viable_masses: list[float] = []
    trajectory_values: dict[str, list[float]] = {}

    for center_index in viable_indices:
        members = tuple(
            index
            for index in viable_indices
            if (
                resolved_group_ids is None
                or resolved_group_ids[index] == resolved_group_ids[center_index]
            )
            and _distance(vectors[center_index], vectors[index]) <= closed_radius
        )
        center_member_counts[center_index] = len(members)
        if len(members) < 2:
            continue
        common_margins = tuple(
            min(profiles[index][action] - buffer for index in members)
            for action in range(action_count)
        )
        best_action = max(range(action_count), key=lambda action: (common_margins[action], -action))
        best_margin = common_margins[best_action]
        violation = max(0.0, -best_margin)
        center_violations[center_index] = violation
        center_best_actions[center_index] = best_action
        eligible_values.append(violation)
        trajectory_values.setdefault(resolved_trajectory_ids[center_index], []).append(violation)
        comparable_viable_count = sum(
            resolved_group_ids is None
            or resolved_group_ids[index] == resolved_group_ids[center_index]
            for index in viable_indices
        )
        if comparable_viable_count > 1:
            viable_masses.append((len(members) - 1) / (comparable_viable_count - 1))
        if violation > 0.0:
            witnesses.append(
                CommonActionWitness(
                    center_index=center_index,
                    center_sample_id=resolved_sample_ids[center_index],
                    member_indices=members,
                    member_sample_ids=tuple(resolved_sample_ids[index] for index in members),
                    best_action_index=best_action,
                    best_common_margin=best_margin,
                    required_violation=violation,
                )
            )

    trajectories = tuple(sorted(set(resolved_trajectory_ids)))
    empty_trajectories = tuple(
        trajectory for trajectory in trajectories if trajectory not in trajectory_values
    )
    trajectory_tails = tuple(
        _linear_quantile(trajectory_values[trajectory], quantile)
        for trajectory in trajectories
        if trajectory in trajectory_values
    )
    if not eligible_values:
        status = "no_eligible_centers"
    elif empty_trajectories:
        status = "trajectory_coverage_incomplete"
    else:
        status = "ok"

    ordered_witnesses = sorted(
        witnesses,
        key=lambda witness: (
            -witness.required_violation,
            witness.center_sample_id,
            witness.member_sample_ids,
        ),
    )
    conflict_count = len(witnesses)
    eligible_count = len(eligible_values)
    return CommonActionNeighborhoodAudit(
        delta=radius,
        gamma=buffer,
        sample_count=count,
        action_count=action_count,
        viable_sample_count=len(viable_indices),
        viable_fraction=len(viable_indices) / count,
        eligible_center_count=eligible_count,
        eligible_given_viable=(
            eligible_count / len(viable_indices) if viable_indices else None
        ),
        conflicting_center_count=conflict_count,
        conflict_fraction=(conflict_count / eligible_count if eligible_count else None),
        mean_required_violation=(statistics.fmean(eligible_values) if eligible_values else None),
        tail_quantile=quantile,
        tail_required_violation=(
            _linear_quantile(eligible_values, quantile) if eligible_values else None
        ),
        worst_required_violation=(max(eligible_values) if eligible_values else None),
        median_viable_nonself_neighborhood_mass=(
            float(statistics.median(viable_masses)) if viable_masses else None
        ),
        trajectory_count=len(trajectories),
        evaluable_trajectory_count=len(trajectory_tails),
        empty_trajectory_ids=empty_trajectories,
        trajectory_balanced_tail_required_violation=(
            statistics.fmean(trajectory_tails)
            if trajectory_tails and not empty_trajectories
            else None
        ),
        status=status,
        center_required_violations=tuple(center_violations),
        center_best_actions=tuple(center_best_actions),
        center_member_counts=tuple(center_member_counts),
        witnesses=tuple(ordered_witnesses[:max_witnesses]),
    )


def audit_common_action_records(
    records: Iterable[AuditRecord],
    *,
    delta: float,
    gamma: float = 0.0,
    tail_quantile: float = 0.95,
    max_witnesses: int = 100,
) -> CommonActionNeighborhoodAudit:
    """Apply the foundation audit to portable records with traceable identifiers."""

    materialized = tuple(sorted(validate_records(records), key=lambda record: record.sample_id))
    if any(record.action_safety_margins is None for record in materialized):
        raise ValueError("every foundation audit record requires action_safety_margins")
    return audit_common_action_neighborhoods(
        [record.latent for record in materialized],
        [record.action_safety_margins or () for record in materialized],
        delta=delta,
        gamma=gamma,
        sample_ids=[record.sample_id for record in materialized],
        trajectory_ids=[record.trajectory_id for record in materialized],
        tail_quantile=tail_quantile,
        max_witnesses=max_witnesses,
    )
