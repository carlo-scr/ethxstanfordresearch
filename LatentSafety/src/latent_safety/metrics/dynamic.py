"""Finite-history optimal-margin regret and representation-policy viability checks.

The routines in this module solve a finite, support-robust safety game on histories.  They are
theorem oracles for enumerated models, not statistical certificates for an unobserved population.
The safety value is the best worst-case minimum physical margin over the remaining horizon.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Real

History = Hashable
Action = Hashable
Code = Hashable
MarginKey = tuple[int, History]
SuccessorKey = tuple[int, History, Action]
CodeKey = tuple[int, History]
Vector = Sequence[float]
PolicyKey = tuple[int, Code]
LatentQKey = tuple[int, Code, Action]


@dataclass(frozen=True)
class StageDynamicAudit:
    """Exact margin-optimality and first-action safety audit at one horizon index."""

    remaining_steps: int
    fiber_count: int
    fiber_optimal_margin_regrets: tuple[tuple[Code, float], ...]
    optimal_margin_actions: tuple[tuple[Code, Action], ...]
    fiber_first_action_obstructions: tuple[tuple[Code, float], ...]
    first_action_safety_actions: tuple[tuple[Code, Action], ...]
    common_safe_actions: tuple[tuple[Code, tuple[Action, ...]], ...]
    viable_history_counts: tuple[tuple[Code, int], ...]
    fiber_q_errors: tuple[tuple[Code, float], ...]
    max_optimal_margin_regret: float
    max_first_action_obstruction: float
    best_uniform_q_error: float
    max_q_oscillation: float
    factor_two_bound_holds: bool

    @property
    def fiber_regrets(self) -> tuple[tuple[Code, float], ...]:
        """Compatibility alias; these are optimal-margin regrets, not safety defects."""

        return self.fiber_optimal_margin_regrets

    @property
    def selected_actions(self) -> tuple[tuple[Code, Action], ...]:
        """Compatibility alias for the minimax optimal-margin actions."""

        return self.optimal_margin_actions

    @property
    def max_common_regret(self) -> float:
        """Compatibility alias for :attr:`max_optimal_margin_regret`."""

        return self.max_optimal_margin_regret


@dataclass(frozen=True)
class CodePolicyCompositionAudit:
    """Exact composition check for one arbitrary deterministic code policy."""

    selected_actions: tuple[Mapping[Code, Action], ...]
    stage_fiber_regrets: tuple[tuple[tuple[Code, float], ...], ...]
    stage_max_regrets: tuple[float, ...]
    policy_values: tuple[Mapping[History, float], ...]
    certificate_bounds: tuple[Mapping[History, float], ...]
    cumulative_global_regret: tuple[float, ...]
    max_actual_loss: tuple[float, ...]
    min_actual_loss: tuple[float, ...]
    optimal_dominates_policy: bool
    local_bound_holds: bool
    certificate_within_global_bound: bool
    theorem_bound_holds: bool
    closed_endpoint_holds: bool


@dataclass(frozen=True)
class QGreedyPolicyAudit:
    """Composition audit for a policy greedy in a declared latent-Q table."""

    composition: CodePolicyCompositionAudit
    uniform_q_errors: tuple[float, ...]
    cumulative_factor_two_bounds: tuple[float, ...]
    stage_factor_two_bounds_hold: tuple[bool, ...]
    composed_factor_two_bound_holds: bool


@dataclass(frozen=True)
class ExactRepresentationViabilityAudit:
    """Exhaustive finite audit of recursively evaluated representation policies."""

    horizon: int
    policy_count: int
    fiber_latent_values: tuple[tuple[tuple[Code, float], ...], ...]
    fiber_latent_viable: tuple[tuple[tuple[Code, bool], ...], ...]
    fiber_safe_initial_actions: tuple[
        tuple[tuple[Code, tuple[Action, ...]], ...], ...
    ]
    full_history_viable_counts: tuple[int, ...]
    max_retained_viable_counts: tuple[int, ...]
    max_retained_viable_fractions: tuple[float, ...]
    preserves_all_viable_histories: tuple[bool, ...]
    all_policy_values_below_optimal: bool


@dataclass(frozen=True)
class DynamicSafetyAudit:
    """Bellman values and minimax optimal-margin code-policy audit."""

    horizon: int
    stage_audits: tuple[StageDynamicAudit, ...]
    optimal_values: tuple[Mapping[History, float], ...]
    q_values: tuple[Mapping[tuple[History, Action], float], ...]
    policy_values: tuple[Mapping[History, float], ...]
    first_action_safety_policy_values: tuple[Mapping[History, float], ...]
    certificate_bounds: tuple[Mapping[History, float], ...]
    cumulative_global_regret: tuple[float, ...]
    max_actual_loss: tuple[float, ...]
    min_actual_loss: tuple[float, ...]
    optimal_dominates_policy: bool
    local_bound_holds: bool
    certificate_within_global_bound: bool
    theorem_bound_holds: bool
    closed_endpoint_holds: bool
    full_history_viable_counts: tuple[int, ...]
    optimal_margin_policy_retained_viable_counts: tuple[int, ...]
    first_action_safety_policy_retained_viable_counts: tuple[int, ...]
    all_stage_first_action_feasible: bool
    preserves_all_full_history_viability: bool
    global_sign_characterization_holds: bool


@dataclass(frozen=True)
class DynamicDataProcessingCheck:
    """Check stagewise optimal-margin-regret monotonicity under coarsening."""

    is_deterministic_postprocessing: bool
    stage_postprocessing: tuple[bool, ...]
    terminal_postprocessing: bool
    fine_regrets: tuple[float, ...]
    coarse_regrets: tuple[float, ...]
    stage_monotone: tuple[bool, ...]
    monotone: bool


@dataclass(frozen=True)
class ConditionalEuclideanQBound:
    """Conditional Euclidean pair-search arithmetic, not a verified certificate."""

    sample_count: int
    action_count: int
    latent_metric: str
    status: str
    cover_radius: float
    representation_lipschitz: float
    q_lipschitz: float
    q_estimation_error: float
    operational_radius: float
    nominal_search_radius: float
    radius_comparison_tolerance: float
    effective_search_radius: float
    sampled_q_oscillation: float
    q_oscillation_upper_bound: float
    optimal_margin_regret_upper_bound: float
    premise_provenance: tuple[tuple[str, str], ...]
    premise: str

    @property
    def expanded_search_radius(self) -> float:
        """Compatibility alias for the nominal theorem radius."""

        return self.nominal_search_radius

    @property
    def dynamic_regret_upper_bound(self) -> float:
        """Compatibility alias for the optimal-margin-regret upper bound."""

        return self.optimal_margin_regret_upper_bound


# Compatibility type alias.  The canonical name deliberately avoids "certificate".
CoverQRegretCertificate = ConditionalEuclideanQBound


def _require_hashable(value: Hashable, *, label: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{label} values must be hashable") from error


def _require_finite_real(
    value: object, *, label: str, nonnegative: bool = False
) -> float:
    """Reject booleans and values accepted only through permissive string coercion."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return numeric


def _validated_problem(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    codes: Mapping[CodeKey, Code],
) -> tuple[
    tuple[Action, ...],
    tuple[tuple[History, ...], ...],
    dict[MarginKey, float],
    dict[SuccessorKey, tuple[History, ...]],
    dict[CodeKey, Code],
]:
    action_tuple = tuple(actions)
    if not action_tuple:
        raise ValueError("at least one action is required")
    for action in action_tuple:
        _require_hashable(action, label="action")
    if len(set(action_tuple)) != len(action_tuple):
        raise ValueError("actions must be unique")

    layers = tuple(tuple(layer) for layer in histories_by_remaining)
    if not layers:
        raise ValueError("histories_by_remaining must contain the terminal layer")

    numeric_margins: dict[MarginKey, float] = {}
    validated_codes: dict[CodeKey, Code] = {}
    layer_sets: list[set[History]] = []
    for remaining, layer in enumerate(layers):
        if not layer:
            raise ValueError(f"history layer {remaining} must be non-empty")
        for history in layer:
            _require_hashable(history, label="history")
        if len(set(layer)) != len(layer):
            raise ValueError(f"history layer {remaining} contains duplicates")
        layer_sets.append(set(layer))
        for history in layer:
            key = (remaining, history)
            if key not in margins:
                raise ValueError(f"missing margin for history {key!r}")
            margin = _require_finite_real(margins[key], label=f"margin for {key!r}")
            numeric_margins[key] = margin
            if key not in codes:
                raise ValueError(f"missing representation code for history {key!r}")
            code = codes[key]
            _require_hashable(code, label="code")
            validated_codes[key] = code

    validated_successors: dict[SuccessorKey, tuple[History, ...]] = {}
    for remaining in range(1, len(layers)):
        for history in layers[remaining]:
            for action in action_tuple:
                key = (remaining, history, action)
                if key not in successors:
                    raise ValueError(f"missing successor set for {key!r}")
                next_histories = tuple(successors[key])
                if not next_histories:
                    raise ValueError(f"successor set for {key!r} must be non-empty")
                if any(
                    next_history not in layer_sets[remaining - 1]
                    for next_history in next_histories
                ):
                    raise ValueError(
                        f"successors for {key!r} must belong to layer {remaining - 1}"
                    )
                validated_successors[key] = next_histories

    return (
        action_tuple,
        layers,
        numeric_margins,
        validated_successors,
        validated_codes,
    )


def _solve_bellman(
    *,
    actions: tuple[Action, ...],
    layers: tuple[tuple[History, ...], ...],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, tuple[History, ...]],
) -> tuple[
    list[dict[History, float]],
    list[dict[tuple[History, Action], float]],
]:
    values: list[dict[History, float]] = [dict() for _ in layers]
    q_values: list[dict[tuple[History, Action], float]] = [dict() for _ in layers]
    values[0] = {history: margins[(0, history)] for history in layers[0]}
    for remaining in range(1, len(layers)):
        for history in layers[remaining]:
            current_margin = margins[(remaining, history)]
            for action in actions:
                future_margin = min(
                    values[remaining - 1][next_history]
                    for next_history in successors[(remaining, history, action)]
                )
                q_values[remaining][(history, action)] = min(
                    current_margin, future_margin
                )
            values[remaining][history] = max(
                q_values[remaining][(history, action)] for action in actions
            )
    return values, q_values


def _stage_fibers(
    *,
    layers: tuple[tuple[History, ...], ...],
    codes: Mapping[CodeKey, Code],
) -> list[dict[Code, list[History]]]:
    fibers: list[dict[Code, list[History]]] = [dict() for _ in layers]
    for remaining, layer in enumerate(layers):
        for history in layer:
            fibers[remaining].setdefault(codes[(remaining, history)], []).append(history)
    return fibers


def _validated_policy_actions(
    *,
    actions: tuple[Action, ...],
    fibers: Sequence[Mapping[Code, Sequence[History]]],
    policy_actions: Mapping[PolicyKey, Action],
) -> list[dict[Code, Action]]:
    selected: list[dict[Code, Action]] = [dict() for _ in fibers]
    action_set = set(actions)
    for remaining in range(1, len(fibers)):
        for code in fibers[remaining]:
            key = (remaining, code)
            if key not in policy_actions:
                raise ValueError(f"missing code-policy action for {key!r}")
            action = policy_actions[key]
            if action not in action_set:
                raise ValueError(f"code-policy action for {key!r} is not declared")
            selected[remaining][code] = action
    return selected


def _evaluate_code_policy(
    *,
    actions: tuple[Action, ...],
    layers: tuple[tuple[History, ...], ...],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, tuple[History, ...]],
    codes: Mapping[CodeKey, Code],
    values: Sequence[Mapping[History, float]],
    q_values: Sequence[Mapping[tuple[History, Action], float]],
    fibers: Sequence[Mapping[Code, Sequence[History]]],
    policy_actions: Mapping[PolicyKey, Action],
    tolerance: float,
) -> CodePolicyCompositionAudit:
    """Evaluate the arbitrary-code-policy composition lemma on one finite game."""

    selected = _validated_policy_actions(
        actions=actions,
        fibers=fibers,
        policy_actions=policy_actions,
    )
    stage_fiber_regrets: list[tuple[tuple[Code, float], ...]] = []
    stage_max_regrets: list[float] = []
    regret_maps: list[dict[Code, float]] = [dict() for _ in layers]
    for remaining in range(1, len(layers)):
        fiber_regrets: list[tuple[Code, float]] = []
        for code, fiber in fibers[remaining].items():
            action = selected[remaining][code]
            regret = max(
                values[remaining][history]
                - q_values[remaining][(history, action)]
                for history in fiber
            )
            regret_maps[remaining][code] = regret
            fiber_regrets.append((code, regret))
        stage_fiber_regrets.append(tuple(fiber_regrets))
        stage_max_regrets.append(max(regret_maps[remaining].values()))

    policy_values: list[dict[History, float]] = [dict() for _ in layers]
    certificate_bounds: list[dict[History, float]] = [dict() for _ in layers]
    policy_values[0] = dict(values[0])
    certificate_bounds[0] = {history: 0.0 for history in layers[0]}
    cumulative_global_regret = [0.0]
    max_actual_loss = [0.0]
    min_actual_loss = [0.0]
    optimal_dominates_policy = True
    local_bound_holds = True
    certificate_within_global_bound = True
    global_loss_bound_holds = True
    closed_endpoint_holds = True

    for remaining in range(1, len(layers)):
        cumulative_global_regret.append(
            cumulative_global_regret[-1] + stage_max_regrets[remaining - 1]
        )
        layer_losses: list[float] = []
        for history in layers[remaining]:
            code = codes[(remaining, history)]
            action = selected[remaining][code]
            next_histories = successors[(remaining, history, action)]
            policy_values[remaining][history] = min(
                margins[(remaining, history)],
                min(
                    policy_values[remaining - 1][next_history]
                    for next_history in next_histories
                ),
            )
            certificate_bounds[remaining][history] = (
                regret_maps[remaining][code]
                + max(
                    certificate_bounds[remaining - 1][next_history]
                    for next_history in next_histories
                )
            )
            loss = values[remaining][history] - policy_values[remaining][history]
            layer_losses.append(loss)
            local_bound = certificate_bounds[remaining][history]
            global_bound = cumulative_global_regret[remaining]
            if loss < -tolerance:
                optimal_dominates_policy = False
            if local_bound < -tolerance or loss > local_bound + tolerance:
                local_bound_holds = False
            if local_bound > global_bound + tolerance:
                certificate_within_global_bound = False
            if loss > global_bound + tolerance:
                global_loss_bound_holds = False
            if (
                values[remaining][history] + tolerance >= local_bound
                and policy_values[remaining][history] < -tolerance
            ):
                closed_endpoint_holds = False
        max_actual_loss.append(max(layer_losses))
        min_actual_loss.append(min(layer_losses))

    theorem_bound_holds = (
        optimal_dominates_policy
        and local_bound_holds
        and certificate_within_global_bound
        and global_loss_bound_holds
    )
    return CodePolicyCompositionAudit(
        selected_actions=tuple(selected),
        stage_fiber_regrets=tuple(stage_fiber_regrets),
        stage_max_regrets=tuple(stage_max_regrets),
        policy_values=tuple(policy_values),
        certificate_bounds=tuple(certificate_bounds),
        cumulative_global_regret=tuple(cumulative_global_regret),
        max_actual_loss=tuple(max_actual_loss),
        min_actual_loss=tuple(min_actual_loss),
        optimal_dominates_policy=optimal_dominates_policy,
        local_bound_holds=local_bound_holds,
        certificate_within_global_bound=certificate_within_global_bound,
        theorem_bound_holds=theorem_bound_holds,
        closed_endpoint_holds=closed_endpoint_holds,
    )


def audit_finite_dynamic_sufficiency(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    codes: Mapping[CodeKey, Code],
    tolerance: float = 1e-12,
) -> DynamicSafetyAudit:
    """Solve and audit an enumerated finite-horizon robust safety problem.

    Layer ``s`` contains histories with ``s`` controls remaining.  ``successors[(s, h, a)]``
    contains every admissible next history in layer ``s - 1``.  The same finite action set is
    available everywhere.  Safety uses the closed convention: a margin of zero is safe.

    For each representation fiber, the routine separately computes (i) robust optimal-margin
    regret and its minimax action, and (ii) the sign-specific first-action obstruction over viable
    histories.  The former policy is then evaluated using the arbitrary-code-policy composition
    lemma.  It need not maximize retained viability; see ``fiber_first_action_obstructions``.
    """

    numeric_tolerance = _require_finite_real(
        tolerance, label="tolerance", nonnegative=True
    )
    (
        action_tuple,
        layers,
        numeric_margins,
        next_map,
        code_map,
    ) = _validated_problem(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=codes,
    )

    horizon = len(layers) - 1
    values, q_values = _solve_bellman(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
    )
    fibers = _stage_fibers(layers=layers, codes=code_map)

    stage_audits: list[StageDynamicAudit] = []
    optimal_margin_policy: dict[PolicyKey, Action] = {}
    first_action_safety_policy: dict[PolicyKey, Action] = {}
    for remaining in range(1, horizon + 1):
        fiber_regrets: list[tuple[Code, float]] = []
        optimal_margin_actions: list[tuple[Code, Action]] = []
        fiber_obstructions: list[tuple[Code, float]] = []
        first_action_safety_actions: list[tuple[Code, Action]] = []
        common_safe_actions: list[tuple[Code, tuple[Action, ...]]] = []
        viable_history_counts: list[tuple[Code, int]] = []
        fiber_q_errors: list[tuple[Code, float]] = []
        stage_q_oscillation = 0.0
        for code, fiber in fibers[remaining].items():
            action_regrets: list[tuple[float, int, Action]] = []
            viable_histories = [
                history for history in fiber if values[remaining][history] >= 0.0
            ]
            action_obstructions: list[tuple[float, int, Action]] = []
            fiber_max_oscillation = 0.0
            for action_index, action in enumerate(action_tuple):
                worst_regret = max(
                    values[remaining][history]
                    - q_values[remaining][(history, action)]
                    for history in fiber
                )
                action_regrets.append((worst_regret, action_index, action))
                obstruction = max(
                    (
                        max(0.0, -q_values[remaining][(history, action)])
                        for history in viable_histories
                    ),
                    default=0.0,
                )
                action_obstructions.append((obstruction, action_index, action))
                action_q_values = [
                    q_values[remaining][(history, action)] for history in fiber
                ]
                oscillation = max(action_q_values) - min(action_q_values)
                fiber_max_oscillation = max(fiber_max_oscillation, oscillation)

            best_regret, _, best_action = min(action_regrets)
            best_obstruction, _, safety_action = min(action_obstructions)
            optimal_margin_policy[(remaining, code)] = best_action
            first_action_safety_policy[(remaining, code)] = safety_action
            fiber_regrets.append((code, best_regret))
            optimal_margin_actions.append((code, best_action))
            fiber_obstructions.append((code, best_obstruction))
            first_action_safety_actions.append((code, safety_action))
            common_safe_actions.append(
                (
                    code,
                    tuple(
                        action
                        for obstruction, _, action in action_obstructions
                        if obstruction == 0.0
                    ),
                )
            )
            viable_history_counts.append((code, len(viable_histories)))
            fiber_q_errors.append((code, 0.5 * fiber_max_oscillation))
            stage_q_oscillation = max(stage_q_oscillation, fiber_max_oscillation)

        max_regret = max((regret for _, regret in fiber_regrets), default=0.0)
        max_obstruction = max(
            (obstruction for _, obstruction in fiber_obstructions), default=0.0
        )
        best_uniform_q_error = max(
            (error for _, error in fiber_q_errors), default=0.0
        )
        stage_audits.append(
            StageDynamicAudit(
                remaining_steps=remaining,
                fiber_count=len(fibers[remaining]),
                fiber_optimal_margin_regrets=tuple(fiber_regrets),
                optimal_margin_actions=tuple(optimal_margin_actions),
                fiber_first_action_obstructions=tuple(fiber_obstructions),
                first_action_safety_actions=tuple(first_action_safety_actions),
                common_safe_actions=tuple(common_safe_actions),
                viable_history_counts=tuple(viable_history_counts),
                fiber_q_errors=tuple(fiber_q_errors),
                max_optimal_margin_regret=max_regret,
                max_first_action_obstruction=max_obstruction,
                best_uniform_q_error=best_uniform_q_error,
                max_q_oscillation=stage_q_oscillation,
                factor_two_bound_holds=max_regret
                <= 2.0 * best_uniform_q_error + numeric_tolerance,
            )
        )

    composition = _evaluate_code_policy(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
        codes=code_map,
        values=values,
        q_values=q_values,
        fibers=fibers,
        policy_actions=optimal_margin_policy,
        tolerance=numeric_tolerance,
    )
    safety_composition = _evaluate_code_policy(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
        codes=code_map,
        values=values,
        q_values=q_values,
        fibers=fibers,
        policy_actions=first_action_safety_policy,
        tolerance=numeric_tolerance,
    )
    full_history_viable_counts = tuple(
        sum(value >= 0.0 for value in stage_values.values())
        for stage_values in values
    )

    def retained_counts(
        policy_values: Sequence[Mapping[History, float]],
    ) -> tuple[int, ...]:
        return tuple(
            sum(
                values[remaining][history] >= 0.0
                and policy_values[remaining][history] >= 0.0
                for history in layers[remaining]
            )
            for remaining in range(len(layers))
        )

    optimal_margin_retained = retained_counts(composition.policy_values)
    first_action_safety_retained = retained_counts(safety_composition.policy_values)
    all_stage_first_action_feasible = all(
        stage.max_first_action_obstruction == 0.0 for stage in stage_audits
    )
    preserves_all_viability = all(
        retained == viable
        for retained, viable in zip(
            first_action_safety_retained,
            full_history_viable_counts,
            strict=True,
        )
    )

    return DynamicSafetyAudit(
        horizon=horizon,
        stage_audits=tuple(stage_audits),
        optimal_values=tuple(values),
        q_values=tuple(q_values),
        policy_values=composition.policy_values,
        first_action_safety_policy_values=safety_composition.policy_values,
        certificate_bounds=composition.certificate_bounds,
        cumulative_global_regret=composition.cumulative_global_regret,
        max_actual_loss=composition.max_actual_loss,
        min_actual_loss=composition.min_actual_loss,
        optimal_dominates_policy=composition.optimal_dominates_policy,
        local_bound_holds=composition.local_bound_holds,
        certificate_within_global_bound=(
            composition.certificate_within_global_bound
        ),
        theorem_bound_holds=composition.theorem_bound_holds,
        closed_endpoint_holds=composition.closed_endpoint_holds,
        full_history_viable_counts=full_history_viable_counts,
        optimal_margin_policy_retained_viable_counts=optimal_margin_retained,
        first_action_safety_policy_retained_viable_counts=(
            first_action_safety_retained
        ),
        all_stage_first_action_feasible=all_stage_first_action_feasible,
        preserves_all_full_history_viability=preserves_all_viability,
        global_sign_characterization_holds=(
            all_stage_first_action_feasible == preserves_all_viability
        ),
    )


def audit_finite_code_policy_composition(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    codes: Mapping[CodeKey, Code],
    policy_actions: Mapping[PolicyKey, Action],
    tolerance: float = 1e-12,
) -> CodePolicyCompositionAudit:
    """Check the finite-horizon loss composition lemma for any code policy."""

    numeric_tolerance = _require_finite_real(
        tolerance, label="tolerance", nonnegative=True
    )
    action_tuple, layers, numeric_margins, next_map, code_map = _validated_problem(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=codes,
    )
    values, q_values = _solve_bellman(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
    )
    fibers = _stage_fibers(layers=layers, codes=code_map)
    return _evaluate_code_policy(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
        codes=code_map,
        values=values,
        q_values=q_values,
        fibers=fibers,
        policy_actions=policy_actions,
        tolerance=numeric_tolerance,
    )


def audit_finite_q_greedy_policy(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    codes: Mapping[CodeKey, Code],
    latent_q_estimates: Mapping[LatentQKey, float],
    tolerance: float = 1e-12,
) -> QGreedyPolicyAudit:
    """Evaluate a latent-Q-greedy policy and its independent factor-two composition bound."""

    numeric_tolerance = _require_finite_real(
        tolerance, label="tolerance", nonnegative=True
    )
    action_tuple, layers, numeric_margins, next_map, code_map = _validated_problem(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=codes,
    )
    values, q_values = _solve_bellman(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
    )
    fibers = _stage_fibers(layers=layers, codes=code_map)
    q_policy: dict[PolicyKey, Action] = {}
    uniform_q_errors: list[float] = []
    for remaining in range(1, len(layers)):
        stage_error = 0.0
        for code, fiber in fibers[remaining].items():
            estimates: list[tuple[float, int, Action]] = []
            for action_index, action in enumerate(action_tuple):
                key = (remaining, code, action)
                if key not in latent_q_estimates:
                    raise ValueError(f"missing latent-Q estimate for {key!r}")
                estimate = _require_finite_real(
                    latent_q_estimates[key], label=f"latent-Q estimate for {key!r}"
                )
                estimates.append((estimate, -action_index, action))
                stage_error = max(
                    stage_error,
                    max(
                        abs(q_values[remaining][(history, action)] - estimate)
                        for history in fiber
                    ),
                )
            _, _, greedy_action = max(estimates)
            q_policy[(remaining, code)] = greedy_action
        uniform_q_errors.append(stage_error)

    composition = _evaluate_code_policy(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
        codes=code_map,
        values=values,
        q_values=q_values,
        fibers=fibers,
        policy_actions=q_policy,
        tolerance=numeric_tolerance,
    )
    stage_factor_two_bounds_hold = tuple(
        regret <= 2.0 * error + numeric_tolerance
        for regret, error in zip(
            composition.stage_max_regrets, uniform_q_errors, strict=True
        )
    )
    cumulative_factor_two_bounds = [0.0]
    for error in uniform_q_errors:
        cumulative_factor_two_bounds.append(
            cumulative_factor_two_bounds[-1] + 2.0 * error
        )
    composed_factor_two_bound_holds = composition.optimal_dominates_policy and all(
        max_loss <= bound + numeric_tolerance
        for max_loss, bound in zip(
            composition.max_actual_loss,
            cumulative_factor_two_bounds,
            strict=True,
        )
    )
    return QGreedyPolicyAudit(
        composition=composition,
        uniform_q_errors=tuple(uniform_q_errors),
        cumulative_factor_two_bounds=tuple(cumulative_factor_two_bounds),
        stage_factor_two_bounds_hold=stage_factor_two_bounds_hold,
        composed_factor_two_bound_holds=(
            all(stage_factor_two_bounds_hold) and composed_factor_two_bound_holds
        ),
    )


def audit_exact_representation_viability(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    codes: Mapping[CodeKey, Code],
    max_policy_count: int = 100_000,
    tolerance: float = 1e-12,
) -> ExactRepresentationViabilityAudit:
    """Exhaustively audit exact safety under recursively evaluated code policies.

    This finite oracle enumerates every time-indexed map from representation codes to actions.
    It is exponential and intentionally guarded.  A fiber value is the best, over such policies,
    of the worst recursively evaluated policy margin among histories in that entire fiber.
    """

    if isinstance(max_policy_count, bool) or not isinstance(max_policy_count, int):
        raise ValueError("max_policy_count must be a positive integer")
    if max_policy_count <= 0:
        raise ValueError("max_policy_count must be a positive integer")
    numeric_tolerance = _require_finite_real(
        tolerance, label="tolerance", nonnegative=True
    )
    action_tuple, layers, numeric_margins, next_map, code_map = _validated_problem(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=codes,
    )
    values, _ = _solve_bellman(
        actions=action_tuple,
        layers=layers,
        margins=numeric_margins,
        successors=next_map,
    )
    fibers = _stage_fibers(layers=layers, codes=code_map)
    policy_keys = tuple(
        (remaining, code)
        for remaining in range(1, len(layers))
        for code in fibers[remaining]
    )
    policy_count = len(action_tuple) ** len(policy_keys)
    if policy_count > max_policy_count:
        raise ValueError(
            f"exact representation audit requires {policy_count} policies, "
            f"above max_policy_count={max_policy_count}"
        )

    fiber_values: list[dict[Code, float]] = [dict() for _ in layers]
    fiber_values[0] = {
        code: min(values[0][history] for history in fiber)
        for code, fiber in fibers[0].items()
    }
    for remaining in range(1, len(layers)):
        fiber_values[remaining] = {code: -math.inf for code in fibers[remaining]}
    safe_initial_action_sets: list[dict[Code, set[Action]]] = [
        {code: set() for code in stage_fibers} for stage_fibers in fibers
    ]
    full_history_viable_counts = tuple(
        sum(value >= 0.0 for value in values[remaining].values())
        for remaining in range(len(layers))
    )
    max_retained_viable_counts = [full_history_viable_counts[0]] + [
        0 for _ in range(1, len(layers))
    ]
    preserves_all_viable_histories = [True] + [False] * (len(layers) - 1)
    all_policy_values_below_optimal = True

    for assignment in product(action_tuple, repeat=len(policy_keys)):
        policy = dict(zip(policy_keys, assignment, strict=True))
        selected = _validated_policy_actions(
            actions=action_tuple,
            fibers=fibers,
            policy_actions=policy,
        )
        policy_values: list[dict[History, float]] = [dict() for _ in layers]
        policy_values[0] = dict(values[0])
        for remaining in range(1, len(layers)):
            for history in layers[remaining]:
                action = selected[remaining][code_map[(remaining, history)]]
                policy_values[remaining][history] = min(
                    numeric_margins[(remaining, history)],
                    min(
                        policy_values[remaining - 1][next_history]
                        for next_history in next_map[(remaining, history, action)]
                    ),
                )

        for remaining in range(1, len(layers)):
            retained_count = 0
            for history in layers[remaining]:
                loss = values[remaining][history] - policy_values[remaining][history]
                if loss < -numeric_tolerance:
                    all_policy_values_below_optimal = False
                if (
                    values[remaining][history] >= 0.0
                    and policy_values[remaining][history] >= 0.0
                ):
                    retained_count += 1
            max_retained_viable_counts[remaining] = max(
                max_retained_viable_counts[remaining], retained_count
            )
            if retained_count == full_history_viable_counts[remaining]:
                preserves_all_viable_histories[remaining] = True

            for code, fiber in fibers[remaining].items():
                worst_policy_margin = min(
                    policy_values[remaining][history] for history in fiber
                )
                fiber_values[remaining][code] = max(
                    fiber_values[remaining][code], worst_policy_margin
                )
                if worst_policy_margin >= 0.0:
                    safe_initial_action_sets[remaining][code].add(
                        selected[remaining][code]
                    )

    latent_values = tuple(
        tuple(stage_values.items()) for stage_values in fiber_values
    )
    latent_viable = tuple(
        tuple((code, value >= 0.0) for code, value in stage_values.items())
        for stage_values in fiber_values
    )
    safe_initial_actions = tuple(
        tuple(
            (
                code,
                tuple(action for action in action_tuple if action in action_set),
            )
            for code, action_set in stage_sets.items()
        )
        for stage_sets in safe_initial_action_sets
    )
    retained_fractions = tuple(
        1.0 if viable_count == 0 else retained_count / viable_count
        for retained_count, viable_count in zip(
            max_retained_viable_counts,
            full_history_viable_counts,
            strict=True,
        )
    )
    return ExactRepresentationViabilityAudit(
        horizon=len(layers) - 1,
        policy_count=policy_count,
        fiber_latent_values=latent_values,
        fiber_latent_viable=latent_viable,
        fiber_safe_initial_actions=safe_initial_actions,
        full_history_viable_counts=full_history_viable_counts,
        max_retained_viable_counts=tuple(max_retained_viable_counts),
        max_retained_viable_fractions=retained_fractions,
        preserves_all_viable_histories=tuple(preserves_all_viable_histories),
        all_policy_values_below_optimal=all_policy_values_below_optimal,
    )


def check_dynamic_data_processing(
    *,
    actions: Sequence[Action],
    histories_by_remaining: Sequence[Sequence[History]],
    margins: Mapping[MarginKey, float],
    successors: Mapping[SuccessorKey, Sequence[History]],
    fine_codes: Mapping[CodeKey, Code],
    coarse_codes: Mapping[CodeKey, Code],
    tolerance: float = 1e-12,
) -> DynamicDataProcessingCheck:
    """Verify optimal-margin-regret data processing on each controlled layer."""

    numeric_tolerance = _require_finite_real(
        tolerance, label="tolerance", nonnegative=True
    )

    fine = audit_finite_dynamic_sufficiency(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=fine_codes,
        tolerance=numeric_tolerance,
    )
    coarse = audit_finite_dynamic_sufficiency(
        actions=actions,
        histories_by_remaining=histories_by_remaining,
        margins=margins,
        successors=successors,
        codes=coarse_codes,
        tolerance=numeric_tolerance,
    )

    layer_postprocessing: list[bool] = []
    for remaining, layer in enumerate(histories_by_remaining):
        deterministic = True
        postprocessing: dict[Code, Code] = {}
        for history in layer:
            fine_code = fine_codes[(remaining, history)]
            coarse_code = coarse_codes[(remaining, history)]
            previous = postprocessing.setdefault(fine_code, coarse_code)
            if previous != coarse_code:
                deterministic = False
        layer_postprocessing.append(deterministic)

    stage_postprocessing = tuple(layer_postprocessing[1:])
    fine_regrets = tuple(
        stage.max_optimal_margin_regret for stage in fine.stage_audits
    )
    coarse_regrets = tuple(
        stage.max_optimal_margin_regret for stage in coarse.stage_audits
    )
    stage_monotone = tuple(
        postprocessing
        and fine_regret <= coarse_regret + numeric_tolerance
        for postprocessing, fine_regret, coarse_regret in zip(
            stage_postprocessing, fine_regrets, coarse_regrets, strict=True
        )
    )
    deterministic = all(stage_postprocessing)
    numerically_monotone = all(
        fine_regret <= coarse_regret + numeric_tolerance
        for fine_regret, coarse_regret in zip(fine_regrets, coarse_regrets, strict=True)
    )
    return DynamicDataProcessingCheck(
        is_deterministic_postprocessing=deterministic,
        stage_postprocessing=stage_postprocessing,
        terminal_postprocessing=layer_postprocessing[0],
        fine_regrets=fine_regrets,
        coarse_regrets=coarse_regrets,
        stage_monotone=stage_monotone,
        monotone=deterministic and numerically_monotone,
    )


def calculate_conditional_euclidean_q_bound(
    *,
    sample_latents: Sequence[Vector],
    sample_q_values: Sequence[Vector],
    cover_radius: float,
    representation_lipschitz: float,
    q_lipschitz: float,
    premise_provenance: Mapping[str, str],
    q_estimation_error: float = 0.0,
    operational_radius: float = 0.0,
    radius_comparison_tolerance: float = 0.0,
) -> ConditionalEuclideanQBound:
    """Calculate a conditional Euclidean cover bound on Q oscillation.

    The caller must establish that the sampled histories form a ``cover_radius``-net in a declared
    history metric, that the representation and every action-value coordinate have the supplied
    Lipschitz constants, and that each tabulated Q value has uniform error at most
    ``q_estimation_error``.  This routine checks the finite pair search and arithmetic; it cannot
    verify those population premises from the sample itself.  Nonempty provenance strings are
    required so a serialized result cannot silently omit where each premise came from.

    The latent metric is always Euclidean L2, and ``representation_lipschitz`` must use that same
    codomain norm.  ``radius_comparison_tolerance`` is added to the finite pair-search radius and
    recorded explicitly.  At operational radius zero the result conditionally upper-bounds exact-
    fiber optimal-margin regret.
    """

    parameters = {
        "cover_radius": cover_radius,
        "representation_lipschitz": representation_lipschitz,
        "q_lipschitz": q_lipschitz,
        "q_estimation_error": q_estimation_error,
        "operational_radius": operational_radius,
        "radius_comparison_tolerance": radius_comparison_tolerance,
    }
    numeric_parameters = {
        name: _require_finite_real(value, label=name, nonnegative=True)
        for name, value in parameters.items()
    }
    if not isinstance(premise_provenance, Mapping):
        raise ValueError("premise_provenance must be a mapping")
    required_provenance = (
        "history_cover",
        "representation_lipschitz",
        "q_lipschitz",
        "q_estimation_error",
        "latent_metric",
    )
    validated_provenance: dict[str, str] = {}
    for name in required_provenance:
        if name not in premise_provenance:
            raise ValueError(f"missing premise provenance for {name}")
        source = premise_provenance[name]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"premise provenance for {name} must be nonempty text")
        validated_provenance[name] = source.strip()

    if len(sample_latents) != len(sample_q_values):
        raise ValueError("sample_latents and sample_q_values must have the same length")
    if not sample_latents:
        raise ValueError("at least one sampled history is required")
    latent_dimension = len(sample_latents[0])
    action_count = len(sample_q_values[0])
    if latent_dimension == 0:
        raise ValueError("latent vectors must be non-empty")
    if action_count == 0:
        raise ValueError("Q profiles must contain at least one action")

    latents: list[tuple[float, ...]] = []
    q_profiles: list[tuple[float, ...]] = []
    for latent, q_profile in zip(sample_latents, sample_q_values, strict=True):
        if len(latent) != latent_dimension:
            raise ValueError("latent vectors must share one dimension")
        if len(q_profile) != action_count:
            raise ValueError("Q profiles must share one action dimension")
        numeric_latent = tuple(
            _require_finite_real(value, label="each latent value") for value in latent
        )
        numeric_q = tuple(
            _require_finite_real(value, label="each Q value") for value in q_profile
        )
        latents.append(numeric_latent)
        q_profiles.append(numeric_q)

    nominal_radius = (
        numeric_parameters["operational_radius"]
        + 2.0
        * numeric_parameters["representation_lipschitz"]
        * numeric_parameters["cover_radius"]
    )
    effective_radius = (
        nominal_radius + numeric_parameters["radius_comparison_tolerance"]
    )
    sampled_oscillation = 0.0
    for first in range(len(latents)):
        for second in range(first, len(latents)):
            distance = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        latents[first], latents[second], strict=True
                    )
                )
            )
            if distance <= effective_radius:
                sampled_oscillation = max(
                    sampled_oscillation,
                    max(
                        abs(left - right)
                        for left, right in zip(
                            q_profiles[first], q_profiles[second], strict=True
                        )
                    ),
                )

    upper_bound = sampled_oscillation + 2.0 * (
        numeric_parameters["q_lipschitz"] * numeric_parameters["cover_radius"]
        + numeric_parameters["q_estimation_error"]
    )
    return ConditionalEuclideanQBound(
        sample_count=len(latents),
        action_count=action_count,
        latent_metric="euclidean_l2",
        status="conditional_arithmetic_only_not_verified_certificate",
        cover_radius=numeric_parameters["cover_radius"],
        representation_lipschitz=numeric_parameters["representation_lipschitz"],
        q_lipschitz=numeric_parameters["q_lipschitz"],
        q_estimation_error=numeric_parameters["q_estimation_error"],
        operational_radius=numeric_parameters["operational_radius"],
        nominal_search_radius=nominal_radius,
        radius_comparison_tolerance=numeric_parameters[
            "radius_comparison_tolerance"
        ],
        effective_search_radius=effective_radius,
        sampled_q_oscillation=sampled_oscillation,
        q_oscillation_upper_bound=upper_bound,
        optimal_margin_regret_upper_bound=upper_bound,
        premise_provenance=tuple(validated_provenance.items()),
        premise=(
            "conditional Euclidean-L2 arithmetic only: the declared history cover, "
            "representation/Q Lipschitz constants, and uniform sampled-Q error are not "
            "inferred or verified from the sample"
        ),
    )


def cover_based_q_regret_certificate(
    *,
    sample_latents: Sequence[Vector],
    sample_q_values: Sequence[Vector],
    cover_radius: float,
    representation_lipschitz: float,
    q_lipschitz: float,
    premise_provenance: Mapping[str, str],
    q_estimation_error: float = 0.0,
    operational_radius: float = 0.0,
    radius_comparison_tolerance: float = 0.0,
) -> ConditionalEuclideanQBound:
    """Compatibility wrapper for :func:`calculate_conditional_euclidean_q_bound`."""

    return calculate_conditional_euclidean_q_bound(
        sample_latents=sample_latents,
        sample_q_values=sample_q_values,
        cover_radius=cover_radius,
        representation_lipschitz=representation_lipschitz,
        q_lipschitz=q_lipschitz,
        premise_provenance=premise_provenance,
        q_estimation_error=q_estimation_error,
        operational_radius=operational_radius,
        radius_comparison_tolerance=radius_comparison_tolerance,
    )
