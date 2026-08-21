"""Paired counterexamples with analytically known representation defects."""

from __future__ import annotations

from dataclasses import dataclass

from .double_integrator import ControlledDoubleIntegrator, DoubleIntegratorState
from .observations import ObservationSpec


@dataclass(frozen=True, slots=True)
class ObservationAliasingPair:
    """Two distinct states mapped to exactly the same learner observation.

    All ground-truth summaries are computed from the system rather than copied as fixture labels.
    Safe means a non-negative closed-set margin.  An action conflict occurs only when both states
    have at least one safe grid action but no grid action is one-step safe for both.
    """

    name: str
    system: ControlledDoubleIntegrator
    observer: ObservationSpec
    left_state: DoubleIntegratorState
    right_state: DoubleIntegratorState

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.left_observation != self.right_observation:
            raise ValueError("states in an ObservationAliasingPair must share an observation")
        if self.left_state == self.right_state:
            raise ValueError("an aliasing pair must contain distinct physical states")

    @property
    def left_observation(self) -> tuple[float, ...]:
        return self.observer.observe((self.left_state,))

    @property
    def right_observation(self) -> tuple[float, ...]:
        return self.observer.observe((self.right_state,))

    @property
    def observation(self) -> tuple[float, ...]:
        """Return the exact code shared by both states."""

        return self.left_observation

    @property
    def safety_margins(self) -> tuple[float, float]:
        return (
            self.system.position_safety_margin(self.left_state),
            self.system.position_safety_margin(self.right_state),
        )

    @property
    def action_safety_margins(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            self.system.one_step_action_margins(self.left_state),
            self.system.one_step_action_margins(self.right_state),
        )

    @property
    def has_static_safe_unsafe_collision(self) -> bool:
        """Whether this observation fiber contains both a safe and unsafe state."""

        margins = self.safety_margins
        return any(margin >= 0.0 for margin in margins) and any(
            margin < 0.0 for margin in margins
        )

    @property
    def static_witness_margin(self) -> float:
        """Largest safe margin witnessed in this safe/unsafe two-state collision."""

        if not self.has_static_safe_unsafe_collision:
            return 0.0
        return max(margin for margin in self.safety_margins if margin >= 0.0)

    @property
    def common_one_step_safe_actions(self) -> tuple[float, ...]:
        rows = self.action_safety_margins
        return tuple(
            action
            for action_index, action in enumerate(self.system.action_grid)
            if all(row[action_index] >= 0.0 for row in rows)
        )

    @property
    def individually_one_step_viable(self) -> bool:
        """Whether each state has at least one safe action on the finite grid."""

        return all(max(row) >= 0.0 for row in self.action_safety_margins)

    @property
    def has_one_step_action_conflict(self) -> bool:
        """Whether no deterministic shared-code grid action is safe for both states."""

        return self.individually_one_step_viable and not self.common_one_step_safe_actions

    @property
    def best_common_one_step_margin(self) -> float:
        """Best worst-state next-position margin attainable by one shared grid action."""

        rows = self.action_safety_margins
        return max(
            min(row[action_index] for row in rows)
            for action_index in range(len(self.system.action_grid))
        )

    @property
    def required_one_step_violation(self) -> float:
        """Positive violation forced by the best shared action, or zero if none is forced."""

        return max(0.0, -self.best_common_one_step_margin)


def _fixture_system() -> ControlledDoubleIntegrator:
    return ControlledDoubleIntegrator(
        time_step=1.0,
        position_limit=1.0,
        action_grid=(-2.0, 0.0, 2.0),
    )


def make_static_aliasing_pair() -> ObservationAliasingPair:
    """Return a safe/unsafe collision caused by a saturated position sensor.

    The states at positions ``0.5`` and ``1.5`` both produce observation ``(0.5,)``.  Their current
    position margins are exactly ``0.5`` and ``-0.5``, so the known exact-collision witness is
    ``0.5``.  The pair intentionally has a common one-step safe grid action; it isolates the static
    defect from action conflict.
    """

    return ObservationAliasingPair(
        name="saturated_position_static_collision",
        system=_fixture_system(),
        observer=ObservationSpec.saturated_position(sensor_limit=0.5),
        left_state=DoubleIntegratorState(position=0.5, velocity=0.0),
        right_state=DoubleIntegratorState(position=1.5, velocity=0.0),
    )


def make_action_conflict_pair() -> ObservationAliasingPair:
    """Return two safe position-only aliases with disjoint safe grid actions.

    Both states are at position zero, but their velocities are ``+1.25`` and ``-1.25``.  Under the
    action grid ``(-2, 0, 2)``, the positive-velocity state can only use ``-2`` safely for one step,
    while the negative-velocity state can only use ``+2``.  The best shared action is zero and
    violates the next-position boundary by exactly ``0.25``.  This is a finite-grid, one-step fact,
    not a continuous-action or long-horizon certificate.
    """

    return ObservationAliasingPair(
        name="hidden_velocity_action_conflict",
        system=_fixture_system(),
        observer=ObservationSpec.position_only(),
        left_state=DoubleIntegratorState(position=0.0, velocity=1.25),
        right_state=DoubleIntegratorState(position=0.0, velocity=-1.25),
    )

