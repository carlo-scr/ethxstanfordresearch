"""Deterministic transition records for synthetic benchmark experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .double_integrator import ControlledDoubleIntegrator, DoubleIntegratorState
from .observations import ObservationSpec


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    """A named open-loop trajectory with an explicit dataset split."""

    trajectory_id: str
    initial_state: DoubleIntegratorState
    actions: tuple[float, ...]
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        actions = tuple(float(action) for action in self.actions)
        if not actions:
            raise ValueError("actions must contain at least one transition")
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True, slots=True)
class BenchmarkTransitionRecord:
    """One transition with privileged labels and a causal learner observation.

    ``action_safety_margins`` contains exact one-step next-position margins for every grid action
    at ``state``.  ``next_safety_margin`` is the entry corresponding to the chosen action.  These
    labels do not assert recursive feasibility beyond the recorded step.
    """

    sample_id: str
    trajectory_id: str
    split: str
    step_index: int
    state: DoubleIntegratorState
    observation: tuple[float, ...]
    action: float
    next_state: DoubleIntegratorState
    safety_margin: float
    next_safety_margin: float
    action_grid: tuple[float, ...]
    action_safety_margins: tuple[float, ...]


def rollout(
    system: ControlledDoubleIntegrator,
    spec: RolloutSpec,
    *,
    observation_spec: ObservationSpec | None = None,
) -> tuple[BenchmarkTransitionRecord, ...]:
    """Generate a deterministic open-loop rollout in the order actions are supplied."""

    observer = observation_spec or ObservationSpec.position_only()
    states = [spec.initial_state]
    applied_actions: list[float] = []
    records: list[BenchmarkTransitionRecord] = []

    for step_index, requested_action in enumerate(spec.actions):
        state = states[-1]
        action = system.validate_action(requested_action)
        observation = observer.observe(states, applied_actions)
        action_margins = system.one_step_action_margins(state)
        next_state = system.step(state, action)
        action_index = system.action_grid.index(action)
        next_margin = action_margins[action_index]
        direct_next_margin = system.position_safety_margin(next_state)
        if abs(next_margin - direct_next_margin) > 1e-12:
            raise AssertionError("analytic and transition-derived one-step margins disagree")
        records.append(
            BenchmarkTransitionRecord(
                sample_id=f"{spec.trajectory_id}:{step_index:06d}",
                trajectory_id=spec.trajectory_id,
                split=spec.split,
                step_index=step_index,
                state=state,
                observation=observation,
                action=action,
                next_state=next_state,
                safety_margin=system.position_safety_margin(state),
                next_safety_margin=next_margin,
                action_grid=system.action_grid,
                action_safety_margins=action_margins,
            )
        )
        states.append(next_state)
        applied_actions.append(action)
    return tuple(records)


def build_dataset(
    system: ControlledDoubleIntegrator,
    rollout_specs: Iterable[RolloutSpec],
    *,
    observation_spec: ObservationSpec | None = None,
) -> tuple[BenchmarkTransitionRecord, ...]:
    """Concatenate named rollouts without shuffling or implicit randomness.

    The caller controls trajectory order and train/test allocation explicitly.  Duplicate
    trajectory identifiers are rejected so records cannot silently collide or leak across splits.
    """

    specs = tuple(rollout_specs)
    if not specs:
        raise ValueError("rollout_specs must contain at least one trajectory")
    trajectory_ids = [spec.trajectory_id for spec in specs]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory_id values must be unique")
    return tuple(
        record
        for spec in specs
        for record in rollout(system, spec, observation_spec=observation_spec)
    )

