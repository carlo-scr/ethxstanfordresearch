"""Deterministic controlled-state video data for representation audits.

Two deliberately small, dependency-free domains are supported: a translating cart and an
inverted pendulum.  Both expose position in pixels while hiding velocity, use discrete controls,
and retain the true state solely to construct known-dynamics safety-oracle labels.  They are
synthetic research fixtures, not wrappers around Gym or claims of realistic control fidelity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any

from latent_safety.learning.config import DataConfig, trajectory_split_counts


@dataclass(frozen=True)
class CartState:
    position: float
    velocity: float
    nuisance_phase: float


@dataclass(frozen=True)
class PendulumState:
    """State of the synthetic inverted pendulum, with angle zero pointing upright."""

    angle: float
    angular_velocity: float
    nuisance_phase: float


PhysicalState = CartState | PendulumState


@dataclass(frozen=True)
class ControlledTrajectory:
    trajectory_id: str
    split: str
    scenario: str
    states: tuple[PhysicalState, ...]
    actions: tuple[float, ...]
    safety_margins: tuple[float, ...]
    action_safety_margins: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SampleRef:
    trajectory_index: int
    timestep: int


@dataclass(frozen=True)
class DatasetBundle:
    trajectories: tuple[ControlledTrajectory, ...]
    datasets: dict[str, "ControlledVideoDataset"]
    manifest_payload: dict[str, Any]
    manifest_sha256: str
    split_trajectory_ids: dict[str, tuple[str, ...]]


_PENDULUM_GRAVITY_GAIN = 1.0


def _wrap_angle(angle: float) -> float:
    """Map an angle to ``[-pi, pi)`` with deterministic endpoint handling."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _next_cart_state(
    state: CartState,
    action: float,
    config: DataConfig,
    *,
    noise: float,
) -> CartState:
    velocity = config.damping * state.velocity + config.dt * (
        config.acceleration * action + noise
    )
    position = state.position + config.dt * velocity
    mechanical_limit = 0.95 * config.render_extent
    if abs(position) > mechanical_limit:
        position = math.copysign(mechanical_limit, position)
        velocity = 0.0
    return CartState(
        position=position,
        velocity=velocity,
        nuisance_phase=state.nuisance_phase + 0.17,
    )


def _next_pendulum_state(
    state: PendulumState,
    action: float,
    config: DataConfig,
    *,
    noise: float,
) -> PendulumState:
    # Angle zero is the unstable upright equilibrium, hence the positive sin(theta) term.
    angular_velocity = config.damping * state.angular_velocity + config.dt * (
        _PENDULUM_GRAVITY_GAIN * math.sin(state.angle)
        + config.acceleration * action
        + noise
    )
    angle = _wrap_angle(state.angle + config.dt * angular_velocity)
    return PendulumState(
        angle=angle,
        angular_velocity=angular_velocity,
        nuisance_phase=state.nuisance_phase + 0.17,
    )


def _next_state(
    state: PhysicalState,
    action: float,
    config: DataConfig,
    *,
    noise: float,
) -> PhysicalState:
    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        return _next_cart_state(state, action, config, noise=noise)
    if config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        return _next_pendulum_state(state, action, config, noise=noise)
    raise ValueError(f"unsupported controlled-video task: {config.task}")


def deterministic_step(
    state: PhysicalState, action: float, config: DataConfig
) -> PhysicalState:
    """Apply one noise-free transition on the configured finite action grid.

    This is the public known-dynamics transition used by finite theorem oracles.  It deliberately
    excludes the training generator's process noise; callers that need a robust stochastic model
    must enumerate a successor correspondence instead of treating this nominal step as one.
    """

    if isinstance(action, bool) or not isinstance(action, Real):
        raise ValueError("action must be a real number")
    converted_action = float(action)
    if not math.isfinite(converted_action):
        raise ValueError("action must be finite")
    if converted_action not in config.actions:
        raise ValueError(
            f"action {converted_action!r} is not in the configured grid {config.actions!r}"
        )
    return _next_state(state, converted_action, config, noise=0.0)


def safety_margin(state: PhysicalState, config: DataConfig) -> float:
    """Return the exact signed state margin for the configured synthetic domain."""

    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        coordinate = state.position
    elif config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        coordinate = _wrap_angle(state.angle)
    else:
        raise ValueError(f"unsupported controlled-video task: {config.task}")
    return config.position_limit - abs(coordinate)


def action_safety_profile(
    state: PhysicalState, config: DataConfig
) -> tuple[float, ...]:
    """Known-dynamics worst margins for each constant-action counterfactual.

    The oracle is deterministic: process noise is set to zero and the true hidden velocity is
    used.  It is a finite-horizon model label, not a certificate for arbitrary future policies.
    """

    margins: list[float] = []
    for action in config.actions:
        counterfactual = state
        worst_margin = safety_margin(counterfactual, config)
        for _ in range(config.action_profile_horizon):
            counterfactual = _next_state(counterfactual, action, config, noise=0.0)
            worst_margin = min(worst_margin, safety_margin(counterfactual, config))
        margins.append(worst_margin)
    return tuple(margins)


def _policy(
    state: PhysicalState,
    config: DataConfig,
    generator: random.Random,
    *,
    corrective_probability: float,
) -> float:
    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        predicted = state.position + config.dt * config.damping * state.velocity
    elif config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        predicted = _wrap_angle(
            state.angle + config.dt * config.damping * state.angular_velocity
        )
    else:
        raise ValueError(f"unsupported controlled-video task: {config.task}")
    if predicted > 0.45 * config.position_limit:
        corrective = min(config.actions)
    elif predicted < -0.45 * config.position_limit:
        corrective = max(config.actions)
    else:
        corrective = min(config.actions, key=abs)
    if generator.random() < corrective_probability:
        return float(corrective)
    return float(generator.choice(config.actions))


def generate_trajectories(config: DataConfig, *, seed: int) -> tuple[ControlledTrajectory, ...]:
    """Generate independently seeded trajectories before assigning whole-trajectory splits."""

    prefixes = {
        "controlled_cart_video": "cart",
        "controlled_pendulum_video": "pendulum",
    }
    try:
        prefix = prefixes[config.task]
    except KeyError as error:
        raise ValueError(f"unsupported controlled-video task: {config.task}") from error
    ids = [f"{prefix}-{index:06d}" for index in range(config.trajectories)]
    shuffled_ids = list(ids)
    split_generator = random.Random(seed ^ 0x5AFE71)
    split_generator.shuffle(shuffled_ids)
    counts = trajectory_split_counts(config.trajectories, config.splits)
    split_by_id: dict[str, str] = {}
    cursor = 0
    for split in ("train", "validation", "calibration", "test"):
        for trajectory_id in shuffled_ids[cursor : cursor + counts[split]]:
            split_by_id[trajectory_id] = split
        cursor += counts[split]

    trajectories: list[ControlledTrajectory] = []
    for index, trajectory_id in enumerate(ids):
        generator = random.Random(seed + 104_729 * (index + 1))
        challenge = generator.random() < config.challenge_fraction
        if challenge:
            direction = generator.choice((-1.0, 1.0))
            initial_coordinate = direction * generator.uniform(
                0.72 * config.position_limit,
                min(1.02 * config.position_limit, 0.95 * config.render_extent),
            )
            initial_velocity = direction * generator.uniform(0.12, 0.52)
            corrective_probability = config.challenge_corrective_probability
        else:
            initial_coordinate = generator.uniform(
                -0.72 * config.position_limit, 0.72 * config.position_limit
            )
            initial_velocity = generator.uniform(-0.45, 0.45)
            corrective_probability = config.corrective_policy_probability
        nuisance_phase = generator.uniform(0.0, 2.0 * math.pi)
        if config.task == "controlled_cart_video":
            state: PhysicalState = CartState(
                position=initial_coordinate,
                velocity=initial_velocity,
                nuisance_phase=nuisance_phase,
            )
        else:
            state = PendulumState(
                angle=initial_coordinate,
                angular_velocity=initial_velocity,
                nuisance_phase=nuisance_phase,
            )
        states = [state]
        actions: list[float] = []
        margins: list[float] = []
        profiles: list[tuple[float, ...]] = []
        for _ in range(config.horizon):
            margins.append(safety_margin(state, config))
            profiles.append(action_safety_profile(state, config))
            action = _policy(
                state,
                config,
                generator,
                corrective_probability=corrective_probability,
            )
            actions.append(action)
            noise = (
                generator.gauss(0.0, config.process_noise)
                if config.process_noise > 0.0
                else 0.0
            )
            state = _next_state(state, action, config, noise=noise)
            states.append(state)
        trajectories.append(
            ControlledTrajectory(
                trajectory_id=trajectory_id,
                split=split_by_id[trajectory_id],
                scenario="boundary_challenge" if challenge else "nominal",
                states=tuple(states),
                actions=tuple(actions),
                safety_margins=tuple(margins),
                action_safety_margins=tuple(profiles),
            )
        )
    return tuple(trajectories)


def _render_cart_state(state: CartState, config: DataConfig) -> list[float]:
    """Render one cart state without exposing velocity in the pixels."""

    size = config.image_size
    channels = config.channels
    plane = size * size
    image = [0.0] * (channels * plane)
    wall_left = round(
        (config.render_extent - config.position_limit)
        / (2.0 * config.render_extent)
        * (size - 1)
    )
    wall_right = round(
        (config.render_extent + config.position_limit)
        / (2.0 * config.render_extent)
        * (size - 1)
    )
    cart_x = round(
        (state.position + config.render_extent)
        / (2.0 * config.render_extent)
        * (size - 1)
    )
    cart_x = max(0, min(size - 1, cart_x))
    cart_y = size // 2
    radius = max(1, size // 16)

    def assign(channel: int, row: int, column: int, value: float) -> None:
        offset = channel * plane + row * size + column
        image[offset] = max(image[offset], max(0.0, min(1.0, value)))

    for row in range(size):
        for column in range(size):
            nuisance = config.nuisance_strength * 0.16 * (
                0.5
                + 0.5
                * math.sin(0.31 * column + 0.19 * row + state.nuisance_phase)
            )
            if channels == 1:
                assign(0, row, column, nuisance)
            else:
                assign(2, row, column, nuisance)
    for row in range(size):
        for column in (wall_left, wall_right):
            if channels == 1:
                assign(0, row, column, 0.55)
            else:
                assign(0, row, column, 0.75)
    track_row = min(size - 1, cart_y + radius + 1)
    for column in range(size):
        for channel in range(channels):
            assign(channel, track_row, column, 0.22)
    for row in range(max(0, cart_y - radius), min(size, cart_y + radius + 1)):
        for column in range(max(0, cart_x - radius), min(size, cart_x + radius + 1)):
            if (row - cart_y) ** 2 + (column - cart_x) ** 2 <= radius**2:
                if channels == 1:
                    assign(0, row, column, 1.0)
                else:
                    assign(1, row, column, 1.0)
                    assign(2, row, column, 0.35)
    return image


def _segment_distance_squared(
    x: float,
    y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 0.0:
        return (x - start_x) ** 2 + (y - start_y) ** 2
    fraction = ((x - start_x) * delta_x + (y - start_y) * delta_y) / denominator
    fraction = max(0.0, min(1.0, fraction))
    closest_x = start_x + fraction * delta_x
    closest_y = start_y + fraction * delta_y
    return (x - closest_x) ** 2 + (y - closest_y) ** 2


def _render_pendulum_state(state: PendulumState, config: DataConfig) -> list[float]:
    """Render an inverted pendulum; angular velocity is intentionally unobserved."""

    size = config.image_size
    channels = config.channels
    plane = size * size
    image = [0.0] * (channels * plane)
    pivot_x = 0.5 * (size - 1)
    pivot_y = 0.78 * (size - 1)
    rod_length = 0.58 * (size - 1)
    angle = _wrap_angle(state.angle)
    bob_x = pivot_x + rod_length * math.sin(angle)
    bob_y = pivot_y - rod_length * math.cos(angle)
    line_width_squared = max(0.65, size / 48.0) ** 2
    pivot_radius_squared = max(1.0, size / 18.0) ** 2
    bob_radius_squared = max(1.0, size / 12.0) ** 2
    boundary_ends = tuple(
        (
            pivot_x + rod_length * math.sin(boundary_angle),
            pivot_y - rod_length * math.cos(boundary_angle),
        )
        for boundary_angle in (-config.position_limit, config.position_limit)
    )

    def assign(channel: int, row: int, column: int, value: float) -> None:
        offset = channel * plane + row * size + column
        image[offset] = max(image[offset], max(0.0, min(1.0, value)))

    for row in range(size):
        for column in range(size):
            nuisance = config.nuisance_strength * 0.16 * (
                0.5
                + 0.5
                * math.sin(0.31 * column + 0.19 * row + state.nuisance_phase)
            )
            assign(0 if channels == 1 else 2, row, column, nuisance)
            if any(
                _segment_distance_squared(
                    column,
                    row,
                    pivot_x,
                    pivot_y,
                    end_x,
                    end_y,
                )
                <= line_width_squared
                for end_x, end_y in boundary_ends
            ):
                assign(0, row, column, 0.36 if channels == 3 else 0.34)
            if (
                _segment_distance_squared(
                    column,
                    row,
                    pivot_x,
                    pivot_y,
                    bob_x,
                    bob_y,
                )
                <= line_width_squared
            ):
                assign(0 if channels == 1 else 1, row, column, 0.72)
            if (column - pivot_x) ** 2 + (row - pivot_y) ** 2 <= pivot_radius_squared:
                for channel in range(channels):
                    assign(channel, row, column, 0.82)
            if (column - bob_x) ** 2 + (row - bob_y) ** 2 <= bob_radius_squared:
                if channels == 1:
                    assign(0, row, column, 1.0)
                else:
                    assign(1, row, column, 1.0)
                    assign(2, row, column, 0.30)
    return image


def render_state(state: PhysicalState, config: DataConfig) -> list[float]:
    """Render a channel-first image as a flat list without third-party dependencies."""

    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        return _render_cart_state(state, config)
    if config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        return _render_pendulum_state(state, config)
    raise ValueError(f"unsupported controlled-video task: {config.task}")


def _render_cart_trajectory_tensor(
    trajectory: ControlledTrajectory, config: DataConfig, torch: Any
) -> Any:
    """Vectorize rendering over a trajectory once, avoiding an epoch-scale Python bottleneck."""

    size = config.image_size
    positions = torch.tensor(
        [state.position for state in trajectory.states], dtype=torch.float32
    )
    phases = torch.tensor(
        [state.nuisance_phase for state in trajectory.states], dtype=torch.float32
    )
    rows = torch.arange(size, dtype=torch.float32).view(1, size, 1)
    columns = torch.arange(size, dtype=torch.float32).view(1, 1, size)
    nuisance = config.nuisance_strength * 0.16 * (
        0.5 + 0.5 * torch.sin(0.31 * columns + 0.19 * rows + phases[:, None, None])
    )
    frames = torch.zeros(
        len(trajectory.states),
        config.channels,
        size,
        size,
        dtype=torch.float32,
    )
    nuisance_channel = 0 if config.channels == 1 else 2
    frames[:, nuisance_channel] = nuisance
    wall_left = round(
        (config.render_extent - config.position_limit)
        / (2.0 * config.render_extent)
        * (size - 1)
    )
    wall_right = round(
        (config.render_extent + config.position_limit)
        / (2.0 * config.render_extent)
        * (size - 1)
    )
    wall_channel = 0
    wall_value = 0.55 if config.channels == 1 else 0.75
    frames[:, wall_channel, :, wall_left] = torch.maximum(
        frames[:, wall_channel, :, wall_left],
        frames.new_full((len(trajectory.states), size), wall_value),
    )
    frames[:, wall_channel, :, wall_right] = torch.maximum(
        frames[:, wall_channel, :, wall_right],
        frames.new_full((len(trajectory.states), size), wall_value),
    )
    radius = max(1, size // 16)
    cart_y = size // 2
    track_row = min(size - 1, cart_y + radius + 1)
    frames[:, :, track_row, :] = torch.maximum(
        frames[:, :, track_row, :],
        frames.new_full((len(trajectory.states), config.channels, size), 0.22),
    )
    cart_x = torch.round(
        (positions + config.render_extent)
        / (2.0 * config.render_extent)
        * (size - 1)
    ).clamp(0, size - 1)
    cart_mask = (
        (columns - cart_x[:, None, None]).square() + (rows - cart_y) ** 2 <= radius**2
    )
    cart_channel = 0 if config.channels == 1 else 1
    frames[:, cart_channel] = torch.maximum(
        frames[:, cart_channel], cart_mask.to(dtype=frames.dtype)
    )
    if config.channels == 3:
        frames[:, 2] = torch.maximum(frames[:, 2], 0.35 * cart_mask)
    return frames


def _render_pendulum_trajectory_tensor(
    trajectory: ControlledTrajectory, config: DataConfig, torch: Any
) -> Any:
    """Vectorized renderer matching the observation content of the scalar renderer."""

    if not all(isinstance(state, PendulumState) for state in trajectory.states):
        raise TypeError("controlled_pendulum_video trajectory contains a non-pendulum state")
    size = config.image_size
    angles = torch.tensor(
        [state.angle for state in trajectory.states], dtype=torch.float32
    )
    phases = torch.tensor(
        [state.nuisance_phase for state in trajectory.states], dtype=torch.float32
    )
    rows = torch.arange(size, dtype=torch.float32).view(1, size, 1)
    columns = torch.arange(size, dtype=torch.float32).view(1, 1, size)
    nuisance = config.nuisance_strength * 0.16 * (
        0.5 + 0.5 * torch.sin(0.31 * columns + 0.19 * rows + phases[:, None, None])
    )
    frames = torch.zeros(
        len(trajectory.states),
        config.channels,
        size,
        size,
        dtype=torch.float32,
    )
    nuisance_channel = 0 if config.channels == 1 else 2
    frames[:, nuisance_channel] = nuisance

    pivot_x = 0.5 * (size - 1)
    pivot_y = 0.78 * (size - 1)
    rod_length = 0.58 * (size - 1)
    bob_x = pivot_x + rod_length * torch.sin(angles)
    bob_y = pivot_y - rod_length * torch.cos(angles)
    line_width_squared = max(0.65, size / 48.0) ** 2

    def segment_mask(end_x: Any, end_y: Any) -> Any:
        delta_x = end_x - pivot_x
        delta_y = end_y - pivot_y
        if hasattr(delta_x, "ndim") and delta_x.ndim == 1:
            delta_x = delta_x[:, None, None]
            delta_y = delta_y[:, None, None]
        denominator = delta_x * delta_x + delta_y * delta_y
        fraction = (
            (columns - pivot_x) * delta_x + (rows - pivot_y) * delta_y
        ) / denominator
        fraction = fraction.clamp(0.0, 1.0)
        closest_x = pivot_x + fraction * delta_x
        closest_y = pivot_y + fraction * delta_y
        return (
            (columns - closest_x).square() + (rows - closest_y).square()
            <= line_width_squared
        )

    boundary_mask = torch.zeros((1, size, size), dtype=torch.bool)
    for boundary_angle in (-config.position_limit, config.position_limit):
        endpoint_x = pivot_x + rod_length * math.sin(boundary_angle)
        endpoint_y = pivot_y - rod_length * math.cos(boundary_angle)
        boundary_mask |= segment_mask(endpoint_x, endpoint_y)
    frames[:, 0] = torch.maximum(
        frames[:, 0],
        (0.36 if config.channels == 3 else 0.34) * boundary_mask,
    )
    rod_mask = segment_mask(bob_x, bob_y)
    rod_channel = 0 if config.channels == 1 else 1
    frames[:, rod_channel] = torch.maximum(frames[:, rod_channel], 0.72 * rod_mask)
    pivot_mask = (
        (columns - pivot_x).square() + (rows - pivot_y).square()
        <= max(1.0, size / 18.0) ** 2
    )
    frames = torch.maximum(frames, 0.82 * pivot_mask[:, None])
    bob_mask = (
        (columns - bob_x[:, None, None]).square()
        + (rows - bob_y[:, None, None]).square()
        <= max(1.0, size / 12.0) ** 2
    )
    bob_channel = 0 if config.channels == 1 else 1
    frames[:, bob_channel] = torch.maximum(
        frames[:, bob_channel], bob_mask.to(dtype=frames.dtype)
    )
    if config.channels == 3:
        frames[:, 2] = torch.maximum(frames[:, 2], 0.30 * bob_mask)
    return frames


def _render_trajectory_tensor(
    trajectory: ControlledTrajectory, config: DataConfig, torch: Any
) -> Any:
    if config.task == "controlled_cart_video":
        if not all(isinstance(state, CartState) for state in trajectory.states):
            raise TypeError("controlled_cart_video trajectory contains a non-cart state")
        return _render_cart_trajectory_tensor(trajectory, config, torch)
    if config.task == "controlled_pendulum_video":
        return _render_pendulum_trajectory_tensor(trajectory, config, torch)
    raise ValueError(f"unsupported controlled-video task: {config.task}")


def _history_indices(timestep: int, history_length: int) -> tuple[int, ...]:
    return tuple(
        max(0, timestep - history_length + 1 + offset)
        for offset in range(history_length)
    )


def observation_feature_vector(
    states: tuple[PhysicalState, ...], timestep: int, config: DataConfig
) -> tuple[float, ...]:
    """Return a controlled observation-oracle feature map over the deployed history.

    These are not raw pixels or privileged state.  They retain only the safety-relevant geometry
    visible in each frame, use the same left-padding rule as image histories, and deliberately
    omit velocity and nuisance.  Pendulum direction is represented continuously by sine/cosine.
    """

    if timestep < 0 or timestep >= len(states):
        raise IndexError("observation feature timestep is outside the trajectory")
    features: list[float] = []
    for state_index in _history_indices(timestep, config.history_length):
        state = states[state_index]
        if config.task == "controlled_cart_video":
            if not isinstance(state, CartState):
                raise TypeError("controlled_cart_video requires CartState")
            features.append(state.position / config.render_extent)
        elif config.task == "controlled_pendulum_video":
            if not isinstance(state, PendulumState):
                raise TypeError("controlled_pendulum_video requires PendulumState")
            angle = _wrap_angle(state.angle)
            features.extend((math.sin(angle), math.cos(angle)))
        else:
            raise ValueError(f"unsupported controlled-video task: {config.task}")
    return tuple(features)


def state_feature_vector(
    state: PhysicalState, config: DataConfig
) -> tuple[float, ...]:
    """Return a privileged ground-truth kinematic control with nuisance removed.

    Unlike :func:`observation_feature_vector`, this oracle includes hidden velocity and therefore
    must never be described as a deployable pixel representation.  Sine/cosine avoids introducing
    an artificial discontinuity at the pendulum angle wrap.
    """

    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        return (state.position, state.velocity)
    if config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        angle = _wrap_angle(state.angle)
        return (math.sin(angle), math.cos(angle), state.angular_velocity)
    raise ValueError(f"unsupported controlled-video task: {config.task}")


class ControlledVideoDataset:
    """Map-style dataset whose split unit is always a complete trajectory."""

    def __init__(
        self,
        trajectories: tuple[ControlledTrajectory, ...],
        trajectory_indices: tuple[int, ...],
        config: DataConfig,
        rendered_frames: tuple[Any, ...],
    ) -> None:
        self.trajectories = trajectories
        self.trajectory_indices = trajectory_indices
        self.config = config
        self.rendered_frames = rendered_frames
        self.refs = tuple(
            SampleRef(trajectory_index=index, timestep=timestep)
            for index in trajectory_indices
            for timestep in range(config.horizon)
        )

    def __len__(self) -> int:
        return len(self.refs)

    def _history(self, trajectory_index: int, timestep: int) -> Any:
        frames = self.rendered_frames[trajectory_index]
        indices = _history_indices(timestep, self.config.history_length)
        # A tuple is interpreted as multidimensional indexing by PyTorch; a list selects frames.
        return frames[list(indices)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        trajectory = self.trajectories[ref.trajectory_index]
        timestep = ref.timestep
        reference = self.rendered_frames[ref.trajectory_index]
        return {
            "sample_id": f"{trajectory.trajectory_id}:{timestep:04d}",
            "trajectory_id": trajectory.trajectory_id,
            "timestep": timestep,
            "history": self._history(ref.trajectory_index, timestep),
            "next_history": self._history(ref.trajectory_index, timestep + 1),
            "observation_features": reference.new_tensor(
                observation_feature_vector(
                    trajectory.states,
                    timestep,
                    self.config,
                )
            ),
            "state_features": reference.new_tensor(
                state_feature_vector(trajectory.states[timestep], self.config)
            ),
            "action": reference.new_tensor([trajectory.actions[timestep]]),
            "safety_margin": reference.new_tensor(trajectory.safety_margins[timestep]),
            "action_safety_margins": reference.new_tensor(
                trajectory.action_safety_margins[timestep]
            ),
        }

    def rollout_refs(self, max_horizon: int) -> tuple[SampleRef, ...]:
        return tuple(
            ref for ref in self.refs if ref.timestep + max_horizon <= self.config.horizon
        )

    def rollout_case(
        self, ref: SampleRef, horizons: tuple[int, ...]
    ) -> tuple[Any, Any, dict[int, Any]]:
        trajectory = self.trajectories[ref.trajectory_index]
        reference = self.rendered_frames[ref.trajectory_index]
        actions = reference.new_tensor(
            [
                [trajectory.actions[ref.timestep + offset]]
                for offset in range(max(horizons))
            ],
        )
        targets = {
            horizon: self._history(ref.trajectory_index, ref.timestep + horizon)
            for horizon in horizons
        }
        return self._history(ref.trajectory_index, ref.timestep), actions, targets


# Compatibility name for downstream code written before the second domain was introduced.
ControlledCartDataset = ControlledVideoDataset


def _serialized_state(state: PhysicalState, config: DataConfig) -> list[float]:
    if config.task == "controlled_cart_video":
        if not isinstance(state, CartState):
            raise TypeError("controlled_cart_video requires CartState")
        return [state.position, state.velocity, state.nuisance_phase]
    if config.task == "controlled_pendulum_video":
        if not isinstance(state, PendulumState):
            raise TypeError("controlled_pendulum_video requires PendulumState")
        return [state.angle, state.angular_velocity, state.nuisance_phase]
    raise ValueError(f"unsupported controlled-video task: {config.task}")


def build_dataset_manifest(
    trajectories: tuple[ControlledTrajectory, ...], config: DataConfig, seed: int
) -> tuple[dict[str, Any], str]:
    """Build a canonical, task-explicit manifest without importing PyTorch."""

    pendulum = config.task == "controlled_pendulum_video"
    payload = {
        "schema_version": 1,
        "generator": (
            "controlled_pendulum_video_v1" if pendulum else "controlled_cart_video_v1"
        ),
        "task": config.task,
        "domain_kind": "synthetic_known_dynamics_not_gym",
        "seed": seed,
        "config": {
            "trajectories": config.trajectories,
            "horizon": config.horizon,
            "history_length": config.history_length,
            "actions": config.actions,
            "action_profile_horizon": config.action_profile_horizon,
            "split_fractions": dict(config.splits.items()),
            "corrective_policy_probability": config.corrective_policy_probability,
            "challenge_fraction": config.challenge_fraction,
            "challenge_corrective_probability": config.challenge_corrective_probability,
            "rendering": {
                "image_size": config.image_size,
                "channels": config.channels,
                "nuisance_strength": config.nuisance_strength,
                "nuisance_phase_increment": 0.17,
                "velocity_visible": False,
            },
            "dynamics": {
                "dt": config.dt,
                "damping": config.damping,
                "acceleration": config.acceleration,
                "control_gain": config.acceleration,
                "gravity_gain": _PENDULUM_GRAVITY_GAIN if pendulum else None,
                "process_noise": config.process_noise,
                "safe_coordinate": "wrapped_angle_radians" if pendulum else "position",
                "safe_coordinate_limit": config.position_limit,
                "render_extent": config.render_extent,
            },
        },
        "state_fields": (
            ["angle", "angular_velocity", "nuisance_phase"]
            if pendulum
            else ["position", "velocity", "nuisance_phase"]
        ),
        "label_semantics": {
            "state_margin": "safe_coordinate_limit - abs(safe_coordinate)",
            "action_profile": {
                "meaning": "minimum state margin under each constant discrete action",
                "horizon": config.action_profile_horizon,
                "process_noise": 0.0,
                "uses_hidden_velocity": True,
                "scope": "finite_horizon_known_dynamics_oracle_not_policy_certificate",
            },
            "observation_features": {
                "meaning": (
                    "padded history of pendulum sine/cosine directions"
                    if pendulum
                    else "padded history of normalized cart positions"
                ),
                "uses_hidden_velocity": False,
                "includes_visual_nuisance": False,
                "scope": "controlled_observation_oracle_not_raw_pixels",
            },
            "state_features": {
                "meaning": (
                    "pendulum sine/cosine direction and angular velocity"
                    if pendulum
                    else "cart position and velocity"
                ),
                "uses_hidden_velocity": True,
                "includes_visual_nuisance": False,
                "scope": "privileged_ground_truth_state_control_not_deployable",
            },
        },
        "trajectories": [
            {
                "id": trajectory.trajectory_id,
                "split": trajectory.split,
                "scenario": trajectory.scenario,
                "states": [
                    _serialized_state(state, config) for state in trajectory.states
                ],
                "actions": trajectory.actions,
                "safety_margins": trajectory.safety_margins,
                "action_safety_margins": trajectory.action_safety_margins,
            }
            for trajectory in trajectories
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_datasets(config: DataConfig, *, seed: int, torch: Any) -> DatasetBundle:
    trajectories = generate_trajectories(config, seed=seed)
    rendered_frames = tuple(
        _render_trajectory_tensor(trajectory, config, torch) for trajectory in trajectories
    )
    indices_by_split: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    for index, trajectory in enumerate(trajectories):
        indices_by_split[trajectory.split].append(index)
    datasets = {
        split: ControlledCartDataset(
            trajectories,
            tuple(indices),
            config,
            rendered_frames,
        )
        for split, indices in indices_by_split.items()
    }
    split_ids = {
        split: tuple(trajectories[index].trajectory_id for index in indices)
        for split, indices in indices_by_split.items()
    }
    manifest_payload, manifest_sha256 = build_dataset_manifest(
        trajectories, config, seed
    )
    return DatasetBundle(
        trajectories=trajectories,
        datasets=datasets,
        manifest_payload=manifest_payload,
        manifest_sha256=manifest_sha256,
        split_trajectory_ids=split_ids,
    )
