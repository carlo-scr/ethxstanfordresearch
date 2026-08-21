"""A deterministic double-integrator benchmark with exact one-step labels.

The benchmark is deliberately small enough that every transition and safety label can be
checked analytically.  Its safety set is the closed position interval
``[-position_limit, position_limit]``; velocity is unconstrained.  Consequently, the margins in
this module certify only the current position or the position after *one* supplied action.  They
are not long-horizon viability or invariance guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class DoubleIntegratorState:
    """Physical state ``(position, velocity)`` of the one-dimensional system."""

    position: float
    velocity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _finite(self.position, name="position"))
        object.__setattr__(self, "velocity", _finite(self.velocity, name="velocity"))

    def as_tuple(self) -> tuple[float, float]:
        """Return the privileged full-state vector in a stable coordinate order."""

        return (self.position, self.velocity)


@dataclass(frozen=True, slots=True)
class ControlledDoubleIntegrator:
    """Exact discrete-time dynamics with a pre-registered finite action grid.

    For step size ``dt`` and constant acceleration ``a`` over a step,

    ``p_next = p + dt * v + 0.5 * dt**2 * a`` and
    ``v_next = v + dt * a``.

    The action grid is required to be strictly increasing so that all emitted action-margin
    vectors have an unambiguous, deterministic coordinate order.
    """

    time_step: float = 0.1
    position_limit: float = 1.0
    action_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        time_step = _finite(self.time_step, name="time_step")
        position_limit = _finite(self.position_limit, name="position_limit")
        action_grid = tuple(_finite(action, name="action") for action in self.action_grid)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if position_limit <= 0.0:
            raise ValueError("position_limit must be positive")
        if not action_grid:
            raise ValueError("action_grid must contain at least one action")
        if any(right <= left for left, right in zip(action_grid, action_grid[1:], strict=False)):
            raise ValueError("action_grid must be strictly increasing")
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "position_limit", position_limit)
        object.__setattr__(self, "action_grid", action_grid)

    def validate_action(self, action: float) -> float:
        """Return a finite grid action, rejecting off-grid controls without projection."""

        converted = _finite(action, name="action")
        if converted not in self.action_grid:
            raise ValueError(
                f"action {converted!r} is not in the finite action grid {self.action_grid!r}"
            )
        return converted

    def step(self, state: DoubleIntegratorState, action: float) -> DoubleIntegratorState:
        """Apply one zero-order-hold action using the exact discrete dynamics."""

        acceleration = self.validate_action(action)
        dt = self.time_step
        return DoubleIntegratorState(
            position=state.position
            + dt * state.velocity
            + 0.5 * dt * dt * acceleration,
            velocity=state.velocity + dt * acceleration,
        )

    def predecessor(
        self, state: DoubleIntegratorState, previous_action: float
    ) -> DoubleIntegratorState:
        """Invert one transition for a known previous grid action.

        This is useful for constructing reachable history observations.  It is an algebraic
        inverse of :meth:`step`, not a claim that the predecessor itself is safe.
        """

        acceleration = self.validate_action(previous_action)
        dt = self.time_step
        previous_velocity = state.velocity - dt * acceleration
        previous_position = (
            state.position
            - dt * previous_velocity
            - 0.5 * dt * dt * acceleration
        )
        return DoubleIntegratorState(previous_position, previous_velocity)

    def position_safety_margin(self, state: DoubleIntegratorState) -> float:
        """Return ``position_limit - abs(position)`` for the current state."""

        return self.position_limit - abs(state.position)

    def one_step_position_margin(
        self, state: DoubleIntegratorState, action: float
    ) -> float:
        """Return the exact position margin after one action.

        A non-negative result means that the *next position* lies in the closed safety interval.
        No assertion is made about later steps.
        """

        return self.position_safety_margin(self.step(state, action))

    def one_step_action_margins(
        self, state: DoubleIntegratorState
    ) -> tuple[float, ...]:
        """Return one-step position margins aligned with :attr:`action_grid`."""

        dt = self.time_step
        uncontrolled_position = state.position + dt * state.velocity
        acceleration_scale = 0.5 * dt * dt
        return tuple(
            self.position_limit
            - abs(uncontrolled_position + acceleration_scale * acceleration)
            for acceleration in self.action_grid
        )

    def one_step_safe_actions(
        self, state: DoubleIntegratorState
    ) -> tuple[float, ...]:
        """Return grid actions whose exact one-step position margin is non-negative."""

        margins = self.one_step_action_margins(state)
        return tuple(
            action
            for action, margin in zip(self.action_grid, margins, strict=True)
            if margin >= 0.0
        )

