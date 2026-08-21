"""Finite-tree dynamic-safety oracles for the two controlled visual domains.

This module deliberately has no third-party dependencies.  It enumerates complete action trees
under the repository's *noise-free nominal* cart and pendulum dynamics, then applies the exact
finite-history Bellman audit from :mod:`latent_safety.metrics.dynamic`.

The returned checks exhaust the enumerated roots, finite action grid, declared horizon, rendered
observations, and nominal transition up to a recorded floating-point tolerance.  They are not
certificates for the continuous state population, learned representations, unenumerated actions,
or the Gaussian-noise training process.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from latent_safety.learning.config import DataConfig
from latent_safety.learning.data import (
    CartState,
    PendulumState,
    PhysicalState,
    deterministic_step,
    render_state,
    safety_margin,
)
from latent_safety.metrics.dynamic import (
    DynamicSafetyAudit,
    audit_finite_dynamic_sufficiency,
)

PRIVILEGED_STATE = "privileged_kinematic_state_exact"
MEMORYLESS_PIXELS = "memoryless_rendered_frame_exact"
TWO_FRAME_HISTORY = "two_frame_rendered_history_exact"
REPRESENTATIONS = (PRIVILEGED_STATE, MEMORYLESS_PIXELS, TWO_FRAME_HISTORY)

_REGISTERED_PARAMETERS: Mapping[str, Mapping[str, object]] = {
    "controlled_cart_video": {
        "actions": (-1.0, 0.0, 1.0),
        "position_limit": 0.82,
        "render_extent": 1.60,
        "dt": 0.12,
        "damping": 0.92,
        "acceleration": 0.90,
        "image_size": 32,
        "channels": 3,
        "nuisance_strength": 0.65,
    },
    "controlled_pendulum_video": {
        "actions": (-1.0, 0.0, 1.0),
        "position_limit": 0.70,
        "render_extent": math.pi,
        "dt": 0.08,
        "damping": 0.985,
        "acceleration": 1.60,
        "image_size": 32,
        "channels": 3,
        "nuisance_strength": 0.65,
    },
}


@dataclass(frozen=True, slots=True)
class DynamicTreeNode:
    """One path-specific history in a complete deterministic action tree."""

    history_id: str
    state: PhysicalState
    previous_state: PhysicalState
    previous_action: float
    action_path: tuple[float, ...]


@dataclass(frozen=True)
class ControlledDynamicTree:
    """Inputs required by the exhaustive finite dynamic-sufficiency oracle."""

    domain: str
    horizon: int
    actions: tuple[float, ...]
    histories_by_remaining: tuple[tuple[str, ...], ...]
    margins: Mapping[tuple[int, str], float]
    successors: Mapping[tuple[int, str, float], tuple[str, ...]]
    codes_by_representation: Mapping[str, Mapping[tuple[int, str], object]]
    nodes_by_history: Mapping[str, DynamicTreeNode]
    root_histories: tuple[str, ...]
    code_semantics: Mapping[str, str]
    node_count_by_remaining: tuple[int, ...]
    transition_semantics: str
    config_sha256: str
    config_parameters: Mapping[str, object]
    fixture_version: str
    population_certificate: bool


@dataclass(frozen=True, slots=True)
class DynamicStageSummary:
    """Auditable quantities at one remaining-horizon index."""

    remaining_steps: int
    fiber_count: int
    rho_s: float
    kappa_s: float
    delta_s_star: float
    max_realized_policy_loss: float
    max_path_bound_B: float
    global_sum_bound: float
    full_history_viable_count: int
    selected_policy_retained_viable_count: int
    selected_policy_retained_viable_fraction: float
    first_action_safety_policy_retained_viable_count: int
    first_action_safety_policy_retained_viable_fraction: float
    fibers_with_viable_histories: int
    fibers_without_common_safe_action: int
    factor_two_bound_holds: bool


@dataclass(frozen=True, slots=True)
class DynamicRootOutcome:
    """Optimal and representation-policy values at one declared root history."""

    history_id: str
    optimal_value: float
    representation_policy_value: float
    first_action_safety_policy_value: float
    realized_policy_loss: float
    path_bound_B: float
    finite_horizon_viable_under_optimal_policy: bool
    finite_horizon_viable_under_representation_policy: bool
    finite_horizon_viable_under_first_action_safety_policy: bool


@dataclass(frozen=True, slots=True)
class DynamicFixtureRoot:
    """Compact provenance for one constructed root and its predecessor."""

    history_id: str
    state_kind: str
    current_kinematics: tuple[float, float]
    previous_kinematics: tuple[float, float]
    previous_action: float
    current_margin: float
    previous_margin: float


@dataclass(frozen=True, slots=True)
class DynamicRootFiberSafety:
    """Sign-specific first-action feasibility for one root representation fiber."""

    fiber_index: int
    histories: tuple[str, ...]
    viable_histories: tuple[str, ...]
    optimal_margin_action: float
    least_violating_safety_action: float
    common_safe_actions: tuple[float, ...]
    first_action_obstruction: float


@dataclass(frozen=True, slots=True)
class RepresentationDynamicSummary:
    """Exhaustive registered finite-tree results for one representation."""

    representation: str
    code_semantics: str
    uses_privileged_state: bool
    root_code_count: int
    stages: tuple[DynamicStageSummary, ...]
    root_outcomes: tuple[DynamicRootOutcome, ...]
    root_fiber_safety: tuple[DynamicRootFiberSafety, ...]
    stagewise_first_action_feasible_within_tolerance: bool
    exact_viability_preserving_code_policy_exists: bool
    global_sign_characterization_holds: bool
    sign_feasibility_method: str
    theorem_bound_holds: bool
    closed_endpoint_holds: bool


@dataclass(frozen=True, slots=True)
class ControlledDynamicOracleReport:
    """A bounded theorem-oracle report for one controlled domain."""

    domain: str
    horizon: int
    actions: tuple[float, ...]
    node_count_by_remaining: tuple[int, ...]
    fixture_roots: tuple[DynamicFixtureRoot, ...]
    representations: tuple[RepresentationDynamicSummary, ...]
    transition_semantics: str
    floating_point_semantics: str
    certificate_scope: str
    config_sha256: str
    config_parameters: Mapping[str, object]
    fixture_version: str
    tolerance: float
    training_support_status: str
    population_certificate: bool


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _fixture_roots(
    config: DataConfig,
) -> tuple[tuple[str, PhysicalState, PhysicalState, float], ...]:
    """Return two reachable histories with an identical current rendered frame.

    These are hand-selected finite witnesses. Both roots have nonnegative three-step full-state
    robust value, while their safe first-action sets are disjoint. The predecessor is the algebraic
    inverse for previous action zero.
    """

    previous_action = 0.0
    if previous_action not in config.actions:
        raise ValueError("the controlled dynamic fixture requires action 0.0")
    if config.damping <= 0.0:
        raise ValueError("the controlled dynamic fixture requires positive damping")

    if config.task == "controlled_cart_video":
        current_states: tuple[PhysicalState, ...] = (
            CartState(position=-0.30, velocity=-1.85, nuisance_phase=0.20),
            CartState(position=-0.30, velocity=3.85, nuisance_phase=0.20),
        )
        names = ("cart_incoming_left", "cart_incoming_right")
        predecessors: list[PhysicalState] = []
        for state in current_states:
            assert isinstance(state, CartState)
            previous_velocity = (
                state.velocity
                - config.dt * config.acceleration * previous_action
            ) / config.damping
            previous_position = state.position - config.dt * state.velocity
            predecessors.append(
                CartState(
                    position=previous_position,
                    velocity=previous_velocity,
                    nuisance_phase=state.nuisance_phase - 0.17,
                )
            )
    elif config.task == "controlled_pendulum_video":
        current_states = (
            PendulumState(angle=-0.04, angular_velocity=-3.00, nuisance_phase=0.20),
            PendulumState(angle=-0.04, angular_velocity=3.25, nuisance_phase=0.20),
        )
        names = (
            "pendulum_fast_clockwise",
            "pendulum_fast_counterclockwise",
        )
        predecessors = []
        for state in current_states:
            assert isinstance(state, PendulumState)
            previous_angle = _wrap_angle(
                state.angle - config.dt * state.angular_velocity
            )
            previous_velocity = (
                state.angular_velocity
                - config.dt
                * (
                    math.sin(previous_angle)
                    + config.acceleration * previous_action
                )
            ) / config.damping
            predecessors.append(
                PendulumState(
                    angle=previous_angle,
                    angular_velocity=previous_velocity,
                    nuisance_phase=state.nuisance_phase - 0.17,
                )
            )
    else:
        raise ValueError(f"unsupported controlled-video task: {config.task}")

    return tuple(
        (name, state, previous, previous_action)
        for name, state, previous in zip(
            names, current_states, predecessors, strict=True
        )
    )


def _kinematic_code(state: PhysicalState) -> tuple[object, ...]:
    """Use exact Python-float equality on the Markov kinematic state."""

    if isinstance(state, CartState):
        return ("cart", state.position, state.velocity)
    if isinstance(state, PendulumState):
        return ("pendulum", state.angle, state.angular_velocity)
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _kinematics(state: PhysicalState) -> tuple[float, float]:
    if isinstance(state, CartState):
        return (state.position, state.velocity)
    if isinstance(state, PendulumState):
        return (state.angle, state.angular_velocity)
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _validate_registered_problem(config: DataConfig, horizon: int) -> None:
    if horizon != 3:
        raise ValueError("the controlled dynamic oracle is registered only for horizon = 3")
    try:
        expected = _REGISTERED_PARAMETERS[config.task]
    except KeyError as error:
        raise ValueError(f"unsupported controlled-video task: {config.task}") from error
    mismatches = {
        name: (getattr(config, name), expected_value)
        for name, expected_value in expected.items()
        if getattr(config, name) != expected_value
    }
    if mismatches:
        raise ValueError(
            "controlled dynamic fixture parameters differ from the registered problem: "
            f"{mismatches!r}"
        )


def _state_max_error(left: PhysicalState, right: PhysicalState) -> float:
    if type(left) is not type(right):
        return math.inf
    if isinstance(left, CartState) and isinstance(right, CartState):
        return max(
            abs(left.position - right.position),
            abs(left.velocity - right.velocity),
            abs(left.nuisance_phase - right.nuisance_phase),
        )
    if isinstance(left, PendulumState) and isinstance(right, PendulumState):
        return max(
            abs(math.remainder(left.angle - right.angle, 2.0 * math.pi)),
            abs(left.angular_velocity - right.angular_velocity),
            abs(left.nuisance_phase - right.nuisance_phase),
        )
    return math.inf


def build_controlled_dynamic_tree(
    config: DataConfig, *, horizon: int = 3
) -> ControlledDynamicTree:
    """Enumerate every action sequence from two observation-aliased root histories.

    ``horizon=3`` and the shipped dynamics/rendering parameters define the registered problem.
    The builder fails closed on any drift because the constructed roots and sign obstruction were
    selected for that exact problem.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    _validate_registered_problem(config, horizon)
    actions = tuple(float(action) for action in config.actions)
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("config.actions must be a nonempty unique finite grid")
    if any(not math.isfinite(action) for action in actions):
        raise ValueError("config.actions must be finite")

    roots = tuple(
        DynamicTreeNode(
            history_id=name,
            state=state,
            previous_state=previous,
            previous_action=previous_action,
            action_path=(),
        )
        for name, state, previous, previous_action in _fixture_roots(config)
    )
    for root in roots:
        reached = deterministic_step(
            root.previous_state, root.previous_action, config
        )
        if _state_max_error(reached, root.state) > 1e-12:
            raise ValueError(
                f"registered predecessor does not reach root {root.history_id!r}"
            )
        if render_state(reached, config) != render_state(root.state, config):
            raise ValueError(
                f"registered predecessor does not reproduce root frame {root.history_id!r}"
            )
        if safety_margin(root.previous_state, config) < 0.0:
            raise ValueError(
                f"registered predecessor is outside the safe envelope: {root.history_id!r}"
            )
        if safety_margin(root.state, config) < 0.0:
            raise ValueError(
                f"registered root is outside the safe envelope: {root.history_id!r}"
            )
    nodes_by_depth: list[tuple[DynamicTreeNode, ...]] = [roots]
    successor_by_depth: dict[tuple[int, str, float], tuple[str, ...]] = {}

    for depth in range(horizon):
        next_nodes: list[DynamicTreeNode] = []
        remaining = horizon - depth
        for node in nodes_by_depth[depth]:
            for action_index, action in enumerate(actions):
                next_id = f"{node.history_id}|a{action_index}"
                next_node = DynamicTreeNode(
                    history_id=next_id,
                    state=deterministic_step(node.state, action, config),
                    previous_state=node.state,
                    previous_action=action,
                    action_path=node.action_path + (action,),
                )
                next_nodes.append(next_node)
                successor_by_depth[(remaining, node.history_id, action)] = (
                    next_id,
                )
        nodes_by_depth.append(tuple(next_nodes))

    histories_by_remaining = tuple(
        tuple(node.history_id for node in nodes_by_depth[horizon - remaining])
        for remaining in range(horizon + 1)
    )
    nodes_by_history = {
        node.history_id: node for layer in nodes_by_depth for node in layer
    }
    margins = {
        (remaining, history): safety_margin(
            nodes_by_history[history].state, config
        )
        for remaining, layer in enumerate(histories_by_remaining)
        for history in layer
    }
    successors = {
        (remaining, history, action): successor_by_depth[
            (remaining, history, action)
        ]
        for remaining in range(1, horizon + 1)
        for history in histories_by_remaining[remaining]
        for action in actions
    }

    frame_cache: dict[PhysicalState, tuple[float, ...]] = {}

    def frame(state: PhysicalState) -> tuple[float, ...]:
        if state not in frame_cache:
            frame_cache[state] = tuple(render_state(state, config))
        return frame_cache[state]

    codes: dict[str, dict[tuple[int, str], object]] = {
        representation: {} for representation in REPRESENTATIONS
    }
    for remaining, layer in enumerate(histories_by_remaining):
        for history in layer:
            node = nodes_by_history[history]
            key = (remaining, history)
            codes[PRIVILEGED_STATE][key] = _kinematic_code(node.state)
            codes[MEMORYLESS_PIXELS][key] = frame(node.state)
            codes[TWO_FRAME_HISTORY][key] = (
                frame(node.previous_state),
                frame(node.state),
            )

    root_layer = histories_by_remaining[horizon]
    if len({codes[MEMORYLESS_PIXELS][(horizon, root)] for root in root_layer}) != 1:
        raise AssertionError("registered roots must have one exact current-frame code")
    if len({codes[TWO_FRAME_HISTORY][(horizon, root)] for root in root_layer}) != len(
        root_layer
    ):
        raise AssertionError("registered two-frame histories must separate the roots")

    return ControlledDynamicTree(
        domain=config.task,
        horizon=horizon,
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes_by_representation=codes,
        nodes_by_history=nodes_by_history,
        root_histories=root_layer,
        code_semantics={
            PRIVILEGED_STATE: (
                "exact Python-float equality of the privileged Markov kinematic state; "
                "renderer nuisance is omitted because it affects neither kinematic dynamics "
                "nor safety"
            ),
            MEMORYLESS_PIXELS: (
                "exact equality of the complete rendered frame tuple, without hashing; scene "
                "geometry is deliberately quantized by the configured finite raster"
            ),
            TWO_FRAME_HISTORY: (
                "exact equality of (previous rendered frame, current rendered frame), with no "
                "privileged state or action; geometry retains the configured raster quantization"
            ),
        },
        node_count_by_remaining=tuple(
            len(layer) for layer in histories_by_remaining
        ),
        transition_semantics=(
            "complete deterministic nominal action tree with singleton successors; process "
            f"noise is fixed to zero although the training config declares {config.process_noise}"
        ),
        config_sha256=hashlib.sha256(
            json.dumps(
                asdict(config), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        config_parameters=asdict(config),
        fixture_version="two_domain_empty_common_safe_action_v1",
        population_certificate=False,
    )


def _summarize_audit(
    tree: ControlledDynamicTree,
    representation: str,
    audit: DynamicSafetyAudit,
    *,
    tolerance: float,
) -> RepresentationDynamicSummary:
    stage_summaries: list[DynamicStageSummary] = []
    for stage in audit.stage_audits:
        remaining = stage.remaining_steps
        full_viable_count = audit.full_history_viable_counts[remaining]
        retained_count = audit.optimal_margin_policy_retained_viable_counts[
            remaining
        ]
        safety_retained_count = (
            audit.first_action_safety_policy_retained_viable_counts[remaining]
        )
        viable_counts = dict(stage.viable_history_counts)
        common_actions = dict(stage.common_safe_actions)
        fibers_with_viable = sum(count > 0 for count in viable_counts.values())
        fibers_without_common = sum(
            count > 0 and not common_actions[code]
            for code, count in viable_counts.items()
        )
        stage_summaries.append(
            DynamicStageSummary(
                remaining_steps=remaining,
                fiber_count=stage.fiber_count,
                rho_s=stage.max_optimal_margin_regret,
                kappa_s=stage.max_first_action_obstruction,
                delta_s_star=stage.best_uniform_q_error,
                max_realized_policy_loss=audit.max_actual_loss[remaining],
                max_path_bound_B=max(
                    audit.certificate_bounds[remaining].values()
                ),
                global_sum_bound=audit.cumulative_global_regret[remaining],
                full_history_viable_count=full_viable_count,
                selected_policy_retained_viable_count=retained_count,
                selected_policy_retained_viable_fraction=(
                    1.0
                    if full_viable_count == 0
                    else retained_count / full_viable_count
                ),
                first_action_safety_policy_retained_viable_count=(
                    safety_retained_count
                ),
                first_action_safety_policy_retained_viable_fraction=(
                    1.0
                    if full_viable_count == 0
                    else safety_retained_count / full_viable_count
                ),
                fibers_with_viable_histories=fibers_with_viable,
                fibers_without_common_safe_action=fibers_without_common,
                factor_two_bound_holds=stage.factor_two_bound_holds,
            )
        )
    stages = tuple(stage_summaries)
    root_outcomes = tuple(
        DynamicRootOutcome(
            history_id=history,
            optimal_value=audit.optimal_values[tree.horizon][history],
            representation_policy_value=audit.policy_values[tree.horizon][history],
            first_action_safety_policy_value=(
                audit.first_action_safety_policy_values[tree.horizon][history]
            ),
            realized_policy_loss=(
                audit.optimal_values[tree.horizon][history]
                - audit.policy_values[tree.horizon][history]
            ),
            path_bound_B=audit.certificate_bounds[tree.horizon][history],
            finite_horizon_viable_under_optimal_policy=(
                audit.optimal_values[tree.horizon][history] >= 0.0
            ),
            finite_horizon_viable_under_representation_policy=(
                audit.policy_values[tree.horizon][history] >= 0.0
            ),
            finite_horizon_viable_under_first_action_safety_policy=(
                audit.first_action_safety_policy_values[tree.horizon][history]
                >= 0.0
            ),
        )
        for history in tree.root_histories
    )
    root_codes = {
        tree.codes_by_representation[representation][(tree.horizon, history)]
        for history in tree.root_histories
    }
    root_stage = audit.stage_audits[-1]
    root_fibers: dict[object, list[str]] = {}
    for history in tree.root_histories:
        code = tree.codes_by_representation[representation][
            (tree.horizon, history)
        ]
        root_fibers.setdefault(code, []).append(history)
    optimal_margin_actions = dict(root_stage.optimal_margin_actions)
    safety_actions = dict(root_stage.first_action_safety_actions)
    common_safe_actions = dict(root_stage.common_safe_actions)
    obstructions = dict(root_stage.fiber_first_action_obstructions)
    root_fiber_safety = tuple(
        DynamicRootFiberSafety(
            fiber_index=index,
            histories=tuple(histories),
            viable_histories=tuple(
                history
                for history in histories
                if audit.optimal_values[tree.horizon][history] >= 0.0
            ),
            optimal_margin_action=float(optimal_margin_actions[code]),
            least_violating_safety_action=float(safety_actions[code]),
            common_safe_actions=tuple(
                float(action) for action in common_safe_actions[code]
            ),
            first_action_obstruction=obstructions[code],
        )
        for index, (code, histories) in enumerate(root_fibers.items())
    )
    return RepresentationDynamicSummary(
        representation=representation,
        code_semantics=tree.code_semantics[representation],
        uses_privileged_state=representation == PRIVILEGED_STATE,
        root_code_count=len(root_codes),
        stages=stages,
        root_outcomes=root_outcomes,
        root_fiber_safety=root_fiber_safety,
        stagewise_first_action_feasible_within_tolerance=all(
            stage.kappa_s <= tolerance for stage in stages
        ),
        exact_viability_preserving_code_policy_exists=(
            audit.preserves_all_full_history_viability
        ),
        global_sign_characterization_holds=(
            audit.global_sign_characterization_holds
        ),
        sign_feasibility_method=(
            "exact global Boolean from finite common-safe-action Bellman induction, plus a "
            "separately labeled tolerance-qualified stagewise diagnostic; both are distinct "
            "from the selected minimax optimal-margin and least-violating safety policies"
        ),
        theorem_bound_holds=audit.theorem_bound_holds,
        closed_endpoint_holds=audit.closed_endpoint_holds,
    )


def audit_controlled_dynamic_oracle(
    config: DataConfig, *, horizon: int = 3, tolerance: float = 1e-12
) -> ControlledDynamicOracleReport:
    """Build and audit all registered representations on one controlled domain."""

    tree = build_controlled_dynamic_tree(config, horizon=horizon)
    if tree.domain == "controlled_cart_video":
        training_support_status = (
            "constructed algebraically reachable nominal-state controls, not sampled from the "
            "behavior-policy dataset and not claimed to lie in its empirical support; the root "
            "velocities exceed the bounded noise-free cart generator envelope"
        )
    else:
        training_support_status = (
            "constructed algebraically reachable nominal-state controls, not sampled from the "
            "behavior-policy dataset; no empirical or analytical behavior-policy support "
            "coverage is claimed for these pendulum roots"
        )
    summaries: list[RepresentationDynamicSummary] = []
    for representation in REPRESENTATIONS:
        audit = audit_finite_dynamic_sufficiency(
            actions=tree.actions,
            histories_by_remaining=tree.histories_by_remaining,
            margins=tree.margins,
            successors=tree.successors,
            codes=tree.codes_by_representation[representation],
            tolerance=tolerance,
        )
        summaries.append(
            _summarize_audit(
                tree,
                representation,
                audit,
                tolerance=tolerance,
            )
        )
    return ControlledDynamicOracleReport(
        domain=tree.domain,
        horizon=tree.horizon,
        actions=tree.actions,
        node_count_by_remaining=tree.node_count_by_remaining,
        fixture_roots=tuple(
            DynamicFixtureRoot(
                history_id=history,
                state_kind=type(tree.nodes_by_history[history].state).__name__,
                current_kinematics=_kinematics(
                    tree.nodes_by_history[history].state
                ),
                previous_kinematics=_kinematics(
                    tree.nodes_by_history[history].previous_state
                ),
                previous_action=tree.nodes_by_history[history].previous_action,
                current_margin=tree.margins[(tree.horizon, history)],
                previous_margin=safety_margin(
                    tree.nodes_by_history[history].previous_state, config
                ),
            )
            for history in tree.root_histories
        ),
        representations=tuple(summaries),
        transition_semantics=tree.transition_semantics,
        floating_point_semantics=(
            "exhaustive finite enumeration and exact Python tuple equality, computed with "
            "binary floating-point dynamics and libm trigonometry; theorem inequalities are "
            "checked up to the recorded tolerance, not by symbolic exact arithmetic"
        ),
        certificate_scope=(
            "exhaustive only for the two registered root histories, finite configured action "
            "grid, declared horizon, exact code equality, and fully enumerated noise-free "
            "nominal tree, with inequalities checked at the recorded floating-point tolerance; "
            "not a learned-model, continuous-population, disturbance-robust, or deployment "
            "certificate"
        ),
        config_sha256=tree.config_sha256,
        config_parameters=tree.config_parameters,
        fixture_version=tree.fixture_version,
        tolerance=tolerance,
        training_support_status=training_support_status,
        population_certificate=False,
    )
