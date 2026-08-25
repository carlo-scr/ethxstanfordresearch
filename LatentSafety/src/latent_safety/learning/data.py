"""Deterministic controlled-state video data for representation audits.

Three deliberately small, dependency-free domains are supported: a translating cart, an inverted
pendulum, and a Dubins-style vehicle sharing a workspace with a moving obstacle.  Their pixels omit
at least one safety-relevant velocity, they use finite controls, and they retain the true state
only to construct known-dynamics oracle labels.  These are synthetic research fixtures, not Gym
wrappers or claims of realistic control fidelity.
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


@dataclass(frozen=True)
class DubinsState:
    """Planar constant-speed vehicle and the phase of one moving circular obstacle."""

    x: float
    y: float
    heading: float
    obstacle_phase: float
    obstacle_direction: int
    nuisance_phase: float


PhysicalState = CartState | PendulumState | DubinsState


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
_DUBINS_OBSTACLE_ORBIT_FRACTION = 0.42
_DUBINS_OBSTACLE_RADIUS_FRACTION = 0.16
_DUBINS_AGENT_RADIUS_FRACTION = 0.055
_DUBINS_OBSTACLE_PHASE_INCREMENT = 0.21
_TASK_PREFIXES = {
    "controlled_cart_video": "cart",
    "controlled_pendulum_video": "pendulum",
    "controlled_dubins_navigation_pixels": "dubins",
}


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


def _dubins_obstacle_center(
    obstacle_phase: float, config: DataConfig
) -> tuple[float, float]:
    radius = _DUBINS_OBSTACLE_ORBIT_FRACTION * config.position_limit
    return radius * math.cos(obstacle_phase), radius * math.sin(obstacle_phase)


def _next_dubins_state(
    state: DubinsState,
    action: float,
    config: DataConfig,
    *,
    noise: float,
) -> DubinsState:
    """Advance a constant-speed vehicle; ``damping`` is its frozen forward speed."""

    if state.obstacle_direction not in {-1, 1}:
        raise ValueError("Dubins obstacle_direction must be -1 or 1")
    heading = _wrap_angle(
        state.heading + config.dt * (config.acceleration * action + noise)
    )
    x = state.x + config.dt * config.damping * math.cos(heading)
    y = state.y + config.dt * config.damping * math.sin(heading)
    mechanical_limit = 0.95 * config.render_extent
    x = max(-mechanical_limit, min(mechanical_limit, x))
    y = max(-mechanical_limit, min(mechanical_limit, y))
    return DubinsState(
        x=x,
        y=y,
        heading=heading,
        obstacle_phase=_wrap_angle(
            state.obstacle_phase
            + state.obstacle_direction * _DUBINS_OBSTACLE_PHASE_INCREMENT
        ),
        obstacle_direction=state.obstacle_direction,
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
    if config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        return _next_dubins_state(state, action, config, noise=noise)
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
    elif config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        obstacle_x, obstacle_y = _dubins_obstacle_center(
            state.obstacle_phase, config
        )
        boundary_margin = min(
            config.position_limit - abs(state.x),
            config.position_limit - abs(state.y),
        )
        collision_margin = math.hypot(
            state.x - obstacle_x, state.y - obstacle_y
        ) - (
            _DUBINS_OBSTACLE_RADIUS_FRACTION
            + _DUBINS_AGENT_RADIUS_FRACTION
        ) * config.position_limit
        return min(boundary_margin, collision_margin)
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


def observed_constant_action_profile(
    state: PhysicalState, config: DataConfig
) -> tuple[float, ...]:
    """Execute explicit noise-free branches and retain every observed state margin.

    This evaluation helper intentionally stays distinct from ``action_safety_profile`` so strict
    validation protocols can prove that their targets came from enumerated physical rollouts rather
    than a precomputed training-label path.
    """

    observed: list[float] = []
    for action in config.actions:
        branch = state
        margins = [safety_margin(branch, config)]
        for _ in range(config.action_profile_horizon):
            branch = deterministic_step(branch, float(action), config)
            margins.append(safety_margin(branch, config))
        observed.append(min(margins))
    return tuple(observed)


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
    elif config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        scored_actions = [
            (
                safety_margin(
                    _next_dubins_state(state, action, config, noise=0.0), config
                ),
                -abs(action),
                action,
            )
            for action in config.actions
        ]
        corrective = max(scored_actions)[2]
        if generator.random() < corrective_probability:
            return float(corrective)
        return float(generator.choice(config.actions))
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


def trajectory_split_assignment(config: DataConfig, *, seed: int) -> dict[str, str]:
    """Return the deterministic whole-trajectory split without generating rollouts."""

    try:
        prefix = _TASK_PREFIXES[config.task]
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
    return split_by_id


def generate_trajectories(
    config: DataConfig,
    *,
    seed: int,
    include_splits: tuple[str, ...] | None = None,
    include_action_profiles: bool = True,
) -> tuple[ControlledTrajectory, ...]:
    """Generate independently seeded trajectories after assigning whole-trajectory splits.

    ``include_splits`` is an access-control boundary for protocols that must not materialize
    calibration or final-test trajectories.  Skipped trajectories retain their would-be index in
    the full ID ordering, so independently seeded included trajectories are byte-for-byte identical
    to the corresponding members of a full generation.  ``include_action_profiles=False`` avoids
    constructing known-dynamics counterfactual labels for observed-margin-only teacher training;
    the resulting trajectories deliberately carry an empty ``action_safety_margins`` tuple, and
    the dataset omits that key from every sample.
    """

    split_by_id = trajectory_split_assignment(config, seed=seed)
    ids = sorted(split_by_id)
    allowed_splits = {"train", "validation", "calibration", "test"}
    if include_splits is None:
        selected_splits = allowed_splits
    else:
        if (
            not include_splits
            or len(set(include_splits)) != len(include_splits)
            or any(split not in allowed_splits for split in include_splits)
        ):
            raise ValueError(
                "include_splits must contain unique registered trajectory splits"
            )
        selected_splits = set(include_splits)
    if not isinstance(include_action_profiles, bool):
        raise TypeError("include_action_profiles must be a boolean")

    trajectories: list[ControlledTrajectory] = []
    for index, trajectory_id in enumerate(ids):
        if split_by_id[trajectory_id] not in selected_splits:
            continue
        generator = random.Random(seed + 104_729 * (index + 1))
        challenge = generator.random() < config.challenge_fraction
        nuisance_phase = generator.uniform(0.0, 2.0 * math.pi)
        if config.task == "controlled_dubins_navigation_pixels":
            obstacle_phase = generator.uniform(-math.pi, math.pi)
            obstacle_direction = generator.choice((-1, 1))
            obstacle_x, obstacle_y = _dubins_obstacle_center(
                obstacle_phase, config
            )
            if challenge and generator.random() < 0.5:
                wall_axis = generator.choice(("x", "y"))
                wall_sign = generator.choice((-1.0, 1.0))
                tangent = generator.uniform(
                    -0.65 * config.position_limit,
                    0.65 * config.position_limit,
                )
                normal = wall_sign * generator.uniform(
                    0.76 * config.position_limit,
                    1.01 * config.position_limit,
                )
                if wall_axis == "x":
                    x, y = normal, tangent
                    heading = 0.0 if wall_sign > 0.0 else math.pi
                else:
                    x, y = tangent, normal
                    heading = math.pi / 2.0 if wall_sign > 0.0 else -math.pi / 2.0
                scenario = "boundary_challenge"
            elif challenge:
                bearing = generator.uniform(-math.pi, math.pi)
                clearance = (
                    _DUBINS_OBSTACLE_RADIUS_FRACTION
                    + _DUBINS_AGENT_RADIUS_FRACTION
                ) * config.position_limit
                separation = generator.uniform(0.90 * clearance, 1.65 * clearance)
                x = obstacle_x + separation * math.cos(bearing)
                y = obstacle_y + separation * math.sin(bearing)
                heading = _wrap_angle(bearing + math.pi + generator.uniform(-0.25, 0.25))
                scenario = "moving_obstacle_challenge"
            else:
                for _ in range(100):
                    x = generator.uniform(
                        -0.68 * config.position_limit,
                        0.68 * config.position_limit,
                    )
                    y = generator.uniform(
                        -0.68 * config.position_limit,
                        0.68 * config.position_limit,
                    )
                    if math.hypot(x - obstacle_x, y - obstacle_y) > (
                        0.40 * config.position_limit
                    ):
                        break
                heading = generator.uniform(-math.pi, math.pi)
                scenario = "nominal"
            state: PhysicalState = DubinsState(
                x=x,
                y=y,
                heading=heading,
                obstacle_phase=obstacle_phase,
                obstacle_direction=obstacle_direction,
                nuisance_phase=nuisance_phase,
            )
            corrective_probability = (
                config.challenge_corrective_probability
                if challenge
                else config.corrective_policy_probability
            )
        else:
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
            if config.task == "controlled_cart_video":
                state = CartState(
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
            scenario = "boundary_challenge" if challenge else "nominal"
        states = [state]
        actions: list[float] = []
        margins: list[float] = []
        profiles: list[tuple[float, ...]] = []
        for _ in range(config.horizon):
            margins.append(safety_margin(state, config))
            if include_action_profiles:
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
                scenario=scenario,
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


def _render_dubins_state(state: DubinsState, config: DataConfig) -> list[float]:
    """Render a top-down vehicle and moving obstacle without obstacle velocity."""

    size = config.image_size
    channels = config.channels
    plane = size * size
    image = [0.0] * (channels * plane)
    obstacle_x, obstacle_y = _dubins_obstacle_center(state.obstacle_phase, config)
    obstacle_radius = _DUBINS_OBSTACLE_RADIUS_FRACTION * config.position_limit
    agent_radius = _DUBINS_AGENT_RADIUS_FRACTION * config.position_limit
    pixel_width = 2.0 * config.render_extent / max(1, size - 1)
    heading_length = max(2.2 * agent_radius, 1.5 * pixel_width)
    nose_x = state.x + heading_length * math.cos(state.heading)
    nose_y = state.y + heading_length * math.sin(state.heading)

    def assign(channel: int, row: int, column: int, value: float) -> None:
        offset = channel * plane + row * size + column
        image[offset] = max(image[offset], max(0.0, min(1.0, value)))

    for row in range(size):
        world_y = config.render_extent - 2.0 * config.render_extent * row / (size - 1)
        for column in range(size):
            world_x = -config.render_extent + 2.0 * config.render_extent * column / (
                size - 1
            )
            nuisance = config.nuisance_strength * 0.14 * (
                0.5
                + 0.5
                * math.sin(0.31 * column + 0.19 * row + state.nuisance_phase)
            )
            assign(0 if channels == 1 else 2, row, column, nuisance)
            if (
                abs(abs(world_x) - config.position_limit) <= 0.55 * pixel_width
                and abs(world_y) <= config.position_limit
            ) or (
                abs(abs(world_y) - config.position_limit) <= 0.55 * pixel_width
                and abs(world_x) <= config.position_limit
            ):
                assign(0, row, column, 0.46 if channels == 1 else 0.72)
            if math.hypot(world_x - obstacle_x, world_y - obstacle_y) <= obstacle_radius:
                assign(0, row, column, 0.86)
                if channels == 3:
                    assign(1, row, column, 0.18)
            if math.hypot(world_x - state.x, world_y - state.y) <= agent_radius:
                assign(0 if channels == 1 else 1, row, column, 1.0)
                if channels == 3:
                    assign(2, row, column, 0.28)
            if _segment_distance_squared(
                world_x,
                world_y,
                state.x,
                state.y,
                nose_x,
                nose_y,
            ) <= (0.52 * pixel_width) ** 2:
                for channel in range(channels):
                    assign(channel, row, column, 0.96)
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
    if config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        return _render_dubins_state(state, config)
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


def _render_dubins_trajectory_tensor(
    trajectory: ControlledTrajectory, config: DataConfig, torch: Any
) -> Any:
    """Vectorized top-down renderer for the moving-obstacle navigation domain."""

    if not all(isinstance(state, DubinsState) for state in trajectory.states):
        raise TypeError(
            "controlled_dubins_navigation_pixels trajectory contains a non-Dubins state"
        )
    size = config.image_size
    states = trajectory.states
    x = torch.tensor([state.x for state in states], dtype=torch.float32)
    y = torch.tensor([state.y for state in states], dtype=torch.float32)
    heading = torch.tensor([state.heading for state in states], dtype=torch.float32)
    obstacle_phase = torch.tensor(
        [state.obstacle_phase for state in states], dtype=torch.float32
    )
    nuisance_phase = torch.tensor(
        [state.nuisance_phase for state in states], dtype=torch.float32
    )
    coordinates = torch.linspace(-config.render_extent, config.render_extent, size)
    world_x = coordinates.view(1, 1, size)
    world_y = coordinates.flip(0).view(1, size, 1)
    row_index = torch.arange(size, dtype=torch.float32).view(1, size, 1)
    column_index = torch.arange(size, dtype=torch.float32).view(1, 1, size)
    frames = torch.zeros(
        len(states), config.channels, size, size, dtype=torch.float32
    )
    nuisance = config.nuisance_strength * 0.14 * (
        0.5
        + 0.5
        * torch.sin(
            0.31 * column_index
            + 0.19 * row_index
            + nuisance_phase[:, None, None]
        )
    )
    nuisance_channel = 0 if config.channels == 1 else 2
    frames[:, nuisance_channel] = nuisance

    pixel_width = 2.0 * config.render_extent / max(1, size - 1)
    boundary_mask = (
        (
            (world_x.abs() - config.position_limit).abs() <= 0.55 * pixel_width
        )
        & (world_y.abs() <= config.position_limit)
    ) | (
        (
            (world_y.abs() - config.position_limit).abs() <= 0.55 * pixel_width
        )
        & (world_x.abs() <= config.position_limit)
    )
    frames[:, 0] = torch.maximum(
        frames[:, 0],
        (0.46 if config.channels == 1 else 0.72) * boundary_mask,
    )

    orbit = _DUBINS_OBSTACLE_ORBIT_FRACTION * config.position_limit
    obstacle_x = orbit * torch.cos(obstacle_phase)
    obstacle_y = orbit * torch.sin(obstacle_phase)
    obstacle_mask = (
        (world_x - obstacle_x[:, None, None]).square()
        + (world_y - obstacle_y[:, None, None]).square()
        <= (_DUBINS_OBSTACLE_RADIUS_FRACTION * config.position_limit) ** 2
    )
    frames[:, 0] = torch.maximum(frames[:, 0], 0.86 * obstacle_mask)
    if config.channels == 3:
        frames[:, 1] = torch.maximum(frames[:, 1], 0.18 * obstacle_mask)

    agent_mask = (
        (world_x - x[:, None, None]).square()
        + (world_y - y[:, None, None]).square()
        <= (_DUBINS_AGENT_RADIUS_FRACTION * config.position_limit) ** 2
    )
    agent_channel = 0 if config.channels == 1 else 1
    frames[:, agent_channel] = torch.maximum(
        frames[:, agent_channel], agent_mask.to(dtype=frames.dtype)
    )
    if config.channels == 3:
        frames[:, 2] = torch.maximum(frames[:, 2], 0.28 * agent_mask)

    heading_length = max(
        2.2 * _DUBINS_AGENT_RADIUS_FRACTION * config.position_limit,
        1.5 * pixel_width,
    )
    nose_x = x + heading_length * torch.cos(heading)
    nose_y = y + heading_length * torch.sin(heading)
    delta_x = nose_x - x
    delta_y = nose_y - y
    denominator = (delta_x.square() + delta_y.square()).clamp_min(1e-12)
    fraction = (
        (world_x - x[:, None, None]) * delta_x[:, None, None]
        + (world_y - y[:, None, None]) * delta_y[:, None, None]
    ) / denominator[:, None, None]
    fraction = fraction.clamp(0.0, 1.0)
    closest_x = x[:, None, None] + fraction * delta_x[:, None, None]
    closest_y = y[:, None, None] + fraction * delta_y[:, None, None]
    heading_mask = (
        (world_x - closest_x).square() + (world_y - closest_y).square()
        <= (0.52 * pixel_width) ** 2
    )
    frames = torch.maximum(frames, 0.96 * heading_mask[:, None])
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
    if config.task == "controlled_dubins_navigation_pixels":
        return _render_dubins_trajectory_tensor(trajectory, config, torch)
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
    omit velocity and nuisance.  Directions are represented continuously by sine/cosine.
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
        elif config.task == "controlled_dubins_navigation_pixels":
            if not isinstance(state, DubinsState):
                raise TypeError(
                    "controlled_dubins_navigation_pixels requires DubinsState"
                )
            obstacle_x, obstacle_y = _dubins_obstacle_center(
                state.obstacle_phase, config
            )
            features.extend(
                (
                    state.x / config.render_extent,
                    state.y / config.render_extent,
                    math.sin(state.heading),
                    math.cos(state.heading),
                    obstacle_x / config.render_extent,
                    obstacle_y / config.render_extent,
                )
            )
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
    if config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        obstacle_x, obstacle_y = _dubins_obstacle_center(
            state.obstacle_phase, config
        )
        obstacle_speed = (
            _DUBINS_OBSTACLE_ORBIT_FRACTION
            * config.position_limit
            * _DUBINS_OBSTACLE_PHASE_INCREMENT
            / config.dt
        )
        obstacle_velocity_x = (
            -state.obstacle_direction
            * obstacle_speed
            * math.sin(state.obstacle_phase)
        )
        obstacle_velocity_y = (
            state.obstacle_direction
            * obstacle_speed
            * math.cos(state.obstacle_phase)
        )
        return (
            state.x,
            state.y,
            math.sin(state.heading),
            math.cos(state.heading),
            obstacle_x,
            obstacle_y,
            obstacle_velocity_x,
            obstacle_velocity_y,
        )
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
        sample = {
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
        }
        if trajectory.action_safety_margins:
            sample["action_safety_margins"] = reference.new_tensor(
                trajectory.action_safety_margins[timestep]
            )
        return sample

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
    if config.task == "controlled_dubins_navigation_pixels":
        if not isinstance(state, DubinsState):
            raise TypeError("controlled_dubins_navigation_pixels requires DubinsState")
        return [
            state.x,
            state.y,
            state.heading,
            state.obstacle_phase,
            state.obstacle_direction,
            state.nuisance_phase,
        ]
    raise ValueError(f"unsupported controlled-video task: {config.task}")


def build_dataset_manifest(
    trajectories: tuple[ControlledTrajectory, ...],
    config: DataConfig,
    seed: int,
    *,
    materialized_splits: tuple[str, ...] | None = None,
    oracle_action_profiles_materialized: bool | None = None,
) -> tuple[dict[str, Any], str]:
    """Build a canonical, task-explicit manifest without importing PyTorch."""

    pendulum = config.task == "controlled_pendulum_video"
    dubins = config.task == "controlled_dubins_navigation_pixels"
    if dubins:
        generator_name = "controlled_dubins_navigation_pixels_v1"
    elif pendulum:
        generator_name = "controlled_pendulum_video_v1"
    else:
        generator_name = "controlled_cart_video_v1"
    split_order = ("train", "validation", "calibration", "test")
    if materialized_splits is None:
        materialized_splits = tuple(
            split for split in split_order if any(item.split == split for item in trajectories)
        )
    if oracle_action_profiles_materialized is None:
        oracle_action_profiles_materialized = bool(trajectories) and all(
            len(trajectory.action_safety_margins) == config.horizon
            for trajectory in trajectories
        )
    payload = {
        "schema_version": 1,
        "generator": generator_name,
        "task": config.task,
        "domain_kind": "synthetic_known_dynamics_not_gym",
        "seed": seed,
        "access_scope": {
            "materialized_splits": list(materialized_splits),
            "not_materialized_splits": [
                split for split in split_order if split not in materialized_splits
            ],
            "oracle_action_profiles_materialized": oracle_action_profiles_materialized,
        },
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
                "moving_obstacle_velocity_visible": False if dubins else None,
            },
            "dynamics": {
                "dt": config.dt,
                "damping": config.damping,
                "acceleration": config.acceleration,
                "control_gain": config.acceleration,
                "gravity_gain": _PENDULUM_GRAVITY_GAIN if pendulum else None,
                "dubins_forward_speed": config.damping if dubins else None,
                "dubins_turn_rate_gain": config.acceleration if dubins else None,
                "moving_obstacle_phase_increment": (
                    _DUBINS_OBSTACLE_PHASE_INCREMENT if dubins else None
                ),
                "moving_obstacle_direction_values": [-1, 1] if dubins else None,
                "moving_obstacle_orbit_fraction": (
                    _DUBINS_OBSTACLE_ORBIT_FRACTION if dubins else None
                ),
                "moving_obstacle_radius_fraction": (
                    _DUBINS_OBSTACLE_RADIUS_FRACTION if dubins else None
                ),
                "agent_radius_fraction": (
                    _DUBINS_AGENT_RADIUS_FRACTION if dubins else None
                ),
                "process_noise": config.process_noise,
                "safe_coordinate": (
                    "arena_and_moving_obstacle_clearance"
                    if dubins
                    else "wrapped_angle_radians" if pendulum else "position"
                ),
                "safe_coordinate_limit": config.position_limit,
                "render_extent": config.render_extent,
            },
        },
        "state_fields": (
            [
                "x",
                "y",
                "heading",
                "obstacle_phase",
                "obstacle_direction",
                "nuisance_phase",
            ]
            if dubins
            else ["angle", "angular_velocity", "nuisance_phase"]
            if pendulum
            else ["position", "velocity", "nuisance_phase"]
        ),
        "label_semantics": {
            "state_margin": (
                "minimum arena-boundary and moving-disc collision clearance"
                if dubins
                else "safe_coordinate_limit - abs(safe_coordinate)"
            ),
            "action_profile": {
                "meaning": "minimum state margin under each constant discrete action",
                "horizon": config.action_profile_horizon,
                "process_noise": 0.0,
                "uses_hidden_velocity": True,
                "uses_hidden_obstacle_velocity": dubins,
                "scope": "finite_horizon_known_dynamics_oracle_not_policy_certificate",
            },
            "observation_features": {
                "meaning": (
                    "padded history of vehicle pose and visible obstacle position"
                    if dubins
                    else "padded history of pendulum sine/cosine directions"
                    if pendulum
                    else "padded history of normalized cart positions"
                ),
                "uses_hidden_velocity": False,
                "uses_hidden_obstacle_velocity": False,
                "includes_visual_nuisance": False,
                "scope": "controlled_observation_oracle_not_raw_pixels",
            },
            "state_features": {
                "meaning": (
                    "vehicle pose, obstacle position, and obstacle velocity"
                    if dubins
                    else "pendulum sine/cosine direction and angular velocity"
                    if pendulum
                    else "cart position and velocity"
                ),
                "uses_hidden_velocity": True,
                "uses_hidden_obstacle_velocity": dubins,
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


def build_datasets(
    config: DataConfig,
    *,
    seed: int,
    torch: Any,
    include_splits: tuple[str, ...] | None = None,
    include_action_profiles: bool = True,
) -> DatasetBundle:
    split_order = ("train", "validation", "calibration", "test")
    selected_splits = include_splits or split_order
    trajectories = generate_trajectories(
        config,
        seed=seed,
        include_splits=selected_splits,
        include_action_profiles=include_action_profiles,
    )
    rendered_frames = tuple(
        _render_trajectory_tensor(trajectory, config, torch) for trajectory in trajectories
    )
    indices_by_split: dict[str, list[int]] = {split: [] for split in selected_splits}
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
        trajectories,
        config,
        seed,
        materialized_splits=selected_splits,
        oracle_action_profiles_materialized=include_action_profiles,
    )
    return DatasetBundle(
        trajectories=trajectories,
        datasets=datasets,
        manifest_payload=manifest_payload,
        manifest_sha256=manifest_sha256,
        split_trajectory_ids=split_ids,
    )
