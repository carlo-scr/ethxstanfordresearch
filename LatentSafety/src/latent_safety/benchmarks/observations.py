"""Causal observation maps for the controlled double-integrator benchmark."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .double_integrator import DoubleIntegratorState


class ObservationKind(str, Enum):
    """Supported privileged and velocity-hiding observation families."""

    FULL_STATE = "full_state"
    POSITION_ONLY = "position_only"
    SATURATED_POSITION = "saturated_position"
    POSITION_ACTION_HISTORY = "position_action_history"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """A fixed-dimensional, deterministic observation specification.

    ``POSITION_ONLY`` and ``SATURATED_POSITION`` omit velocity.  The latter additionally clips
    position at ``sensor_limit`` and can therefore create safe/unsafe static aliases.

    ``POSITION_ACTION_HISTORY`` also omits velocity directly.  It emits, in order, a chronological
    window of positions, the intervening previous actions, and the number of real (not padded)
    states in the window.  At the beginning of a trajectory, missing positions repeat the earliest
    available position and missing actions are zero.  The final count makes that padding explicit.
    With a two-state exact history, known action, and noise-free dynamics, velocity is recoverable
    by :func:`reconstruct_current_velocity`.
    """

    kind: ObservationKind
    sensor_limit: float | None = None
    history_length: int | None = None

    def __post_init__(self) -> None:
        kind = ObservationKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is ObservationKind.SATURATED_POSITION:
            if self.sensor_limit is None:
                raise ValueError("saturated position observations require sensor_limit")
            sensor_limit = float(self.sensor_limit)
            if not math.isfinite(sensor_limit) or sensor_limit <= 0.0:
                raise ValueError("sensor_limit must be finite and positive")
            object.__setattr__(self, "sensor_limit", sensor_limit)
        elif self.sensor_limit is not None:
            raise ValueError("sensor_limit is only valid for saturated position observations")

        if kind is ObservationKind.POSITION_ACTION_HISTORY:
            if self.history_length is None or self.history_length < 2:
                raise ValueError("position-action history requires history_length >= 2")
        elif self.history_length is not None:
            raise ValueError("history_length is only valid for history observations")

    @classmethod
    def full_state(cls) -> ObservationSpec:
        """Observe position and velocity."""

        return cls(ObservationKind.FULL_STATE)

    @classmethod
    def position_only(cls) -> ObservationSpec:
        """Observe position while hiding velocity."""

        return cls(ObservationKind.POSITION_ONLY)

    @classmethod
    def saturated_position(cls, sensor_limit: float) -> ObservationSpec:
        """Observe clipped position while hiding velocity."""

        return cls(ObservationKind.SATURATED_POSITION, sensor_limit=sensor_limit)

    @classmethod
    def position_action_history(cls, history_length: int = 2) -> ObservationSpec:
        """Observe a fixed causal window of positions and previous actions."""

        return cls(
            ObservationKind.POSITION_ACTION_HISTORY,
            history_length=history_length,
        )

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension emitted by this specification."""

        if self.kind is ObservationKind.FULL_STATE:
            return 2
        if self.kind in {
            ObservationKind.POSITION_ONLY,
            ObservationKind.SATURATED_POSITION,
        }:
            return 1
        assert self.history_length is not None
        return self.history_length + (self.history_length - 1) + 1

    def observe(
        self,
        state_history: Sequence[DoubleIntegratorState],
        action_history: Sequence[float] = (),
    ) -> tuple[float, ...]:
        """Observe the last state using only the supplied causal history.

        ``action_history[k]`` must be the action taking ``state_history[k]`` to
        ``state_history[k + 1]``.  Dynamics consistency is intentionally not inferred here; rollout
        generation supplies consistent histories, while hand-built counterexamples remain useful.
        """

        states = tuple(state_history)
        actions = tuple(float(action) for action in action_history)
        if not states:
            raise ValueError("state_history must contain at least one state")
        if len(actions) != len(states) - 1:
            raise ValueError("action_history must contain one action between adjacent states")
        if any(not math.isfinite(action) for action in actions):
            raise ValueError("action_history must be finite")

        current = states[-1]
        if self.kind is ObservationKind.FULL_STATE:
            return current.as_tuple()
        if self.kind is ObservationKind.POSITION_ONLY:
            return (current.position,)
        if self.kind is ObservationKind.SATURATED_POSITION:
            assert self.sensor_limit is not None
            clipped = max(-self.sensor_limit, min(self.sensor_limit, current.position))
            return (clipped,)

        assert self.history_length is not None
        window = self.history_length
        retained_states = states[-window:]
        retained_actions = actions[-(window - 1) :]
        valid_state_count = len(retained_states)
        missing_states = window - valid_state_count
        missing_actions = (window - 1) - len(retained_actions)
        positions = (
            (retained_states[0].position,) * missing_states
            + tuple(state.position for state in retained_states)
        )
        padded_actions = (0.0,) * missing_actions + retained_actions
        result = positions + padded_actions + (float(valid_state_count),)
        if len(result) != self.dimension:
            raise AssertionError("history observation dimension invariant failed")
        return result


def reconstruct_current_velocity(
    *,
    previous_position: float,
    current_position: float,
    previous_action: float,
    time_step: float,
) -> float:
    """Recover current velocity from a noise-free two-position, one-action history.

    This follows directly from the exact constant-acceleration discretization.  It should be used
    as a synthetic ground-truth check, not as an estimator guarantee under observation noise or
    model mismatch.
    """

    values = (previous_position, current_position, previous_action, time_step)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("positions, action, and time_step must be finite")
    dt = float(time_step)
    if dt <= 0.0:
        raise ValueError("time_step must be positive")
    displacement_velocity = (float(current_position) - float(previous_position)) / dt
    return displacement_velocity + 0.5 * dt * float(previous_action)

