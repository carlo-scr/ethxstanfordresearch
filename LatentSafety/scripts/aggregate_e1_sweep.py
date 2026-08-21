#!/usr/bin/env python3
"""Aggregate an E1 task plan without silently dropping runs or opening test results."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import itertools
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis import (  # noqa: E402
    BLOCK_INDEPENDENCE_CAVEAT,
    VALIDATION_SELECTION_CAVEAT,
    PairedBlock,
    ValidationScore,
    paired_block_bootstrap_ci,
    select_under_safety_budget,
    standardized_paired_effect,
    validation_pareto_frontier,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402


ROW_METRICS = (
    "utility_loss",
    "reconstruction_mse",
    "rollout_pixel_mse_at_max_horizon",
    "static_defect",
    "action_conflict_fraction",
    "action_tail_required_violation",
    "action_worst_required_violation",
)

LATENT_AUDIT_VIEW = "latent"
BASELINE_AUDIT_VIEWS = frozenset(
    {
        LATENT_AUDIT_VIEW,
        "observation_oracle",
        "state_oracle",
        "latent_plus_margin",
        "latent_plus_action_profile",
    }
)


class AggregationError(ValueError):
    """An input violates the frozen sweep-analysis contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AggregationError(f"{label} must contain a JSON object: {path}")
    return payload


def _finite_number(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AggregationError(f"{field} must be finite")
    if nonnegative and numeric < 0.0:
        raise AggregationError(f"{field} must be non-negative")
    return numeric


def _integer(value: object, *, field: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregationError(f"{field} must be an integer")
    if nonnegative and value < 0:
        raise AggregationError(f"{field} must be non-negative")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggregationError(f"{field} must be a non-empty string")
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AggregationError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AggregationError(f"{field} must be a list")
    return value


def _declared_plan_hash(plan: dict[str, Any]) -> str:
    declared = _string(plan.get("plan_sha256"), field="plan.plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(declared, actual):
        raise AggregationError(
            f"plan_sha256 mismatch: declared {declared}, recomputed {actual}"
        )
    return declared


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise AggregationError("only plan schema_version = 1 is supported")
    plan_hash = _declared_plan_hash(plan)
    tasks = _sequence(plan.get("tasks"), field="plan.tasks")
    task_count = _integer(
        plan.get("task_count"), field="plan.task_count", nonnegative=True
    )
    if task_count != len(tasks) or task_count == 0:
        raise AggregationError("plan.task_count must equal a non-empty tasks list")
    analysis = _mapping(plan.get("analysis"), field="plan.analysis")
    if analysis.get("selection_split") != "validation":
        raise AggregationError("plan.analysis.selection_split must be exactly 'validation'")
    primary_radius = _finite_number(
        analysis.get("primary_relative_radius"),
        field="plan.analysis.primary_relative_radius",
        nonnegative=True,
    )
    radii = _sequence(
        analysis.get("relative_radii"), field="plan.analysis.relative_radii"
    )
    normalized_radii = [
        _finite_number(value, field="plan.analysis.relative_radii", nonnegative=True)
        for value in radii
    ]
    if normalized_radii != sorted(set(normalized_radii)):
        raise AggregationError("plan.analysis.relative_radii must be sorted and unique")
    if not any(
        math.isclose(value, primary_radius, rel_tol=0.0, abs_tol=1e-12)
        for value in normalized_radii
    ):
        raise AggregationError("primary_relative_radius is absent from relative_radii")
    bootstrap_resamples = _integer(
        analysis.get("bootstrap_resamples"),
        field="plan.analysis.bootstrap_resamples",
        nonnegative=True,
    )
    if bootstrap_resamples < 100:
        raise AggregationError("plan.analysis.bootstrap_resamples must be at least 100")
    bootstrap_seed = _integer(
        analysis.get("bootstrap_seed"), field="plan.analysis.bootstrap_seed"
    )
    safety_budget = _finite_number(
        analysis.get("max_validation_static_defect"),
        field="plan.analysis.max_validation_static_defect",
        nonnegative=True,
    )

    axes = _mapping(plan.get("axes"), field="plan.axes")

    def string_axis(name: str) -> list[str]:
        values = _sequence(axes.get(name), field=f"plan.axes.{name}")
        normalized = [
            _string(value, field=f"plan.axes.{name}") for value in values
        ]
        if not normalized or len(normalized) != len(set(normalized)):
            raise AggregationError(f"plan.axes.{name} must be non-empty and unique")
        return normalized

    def integer_axis(name: str, *, positive: bool = False) -> list[int]:
        values = _sequence(axes.get(name), field=f"plan.axes.{name}")
        normalized = [
            _integer(value, field=f"plan.axes.{name}", nonnegative=True)
            for value in values
        ]
        if not normalized or len(normalized) != len(set(normalized)):
            raise AggregationError(f"plan.axes.{name} must be non-empty and unique")
        if positive and any(value == 0 for value in normalized):
            raise AggregationError(f"plan.axes.{name} values must be positive")
        return normalized

    families = string_axis("model_families")
    history_modes = string_axis("history_modes")
    dimensions = integer_axis("latent_dimensions", positive=True)
    arms = string_axis("safety_arms")
    seeds = integer_axis("seeds")
    if "none" not in arms:
        raise AggregationError("plan.axes.safety_arms must include baseline arm 'none'")
    definitions = _mapping(
        plan.get("history_mode_definitions"), field="plan.history_mode_definitions"
    )
    if set(definitions) != set(history_modes):
        raise AggregationError(
            "history_mode_definitions must define exactly the declared history modes"
        )

    task_ids: set[int] = set()
    output_dirs: set[str] = set()
    model_seed_keys: set[tuple[str, str, int, str, int]] = set()
    for index, value in enumerate(tasks):
        task = _mapping(value, field=f"plan.tasks[{index}]")
        task_id = _integer(
            task.get("task_id"),
            field=f"tasks[{index}].task_id",
            nonnegative=True,
        )
        if task_id in task_ids:
            raise AggregationError(f"duplicate task_id {task_id}")
        task_ids.add(task_id)
        output_dir = _string(task.get("output_dir"), field=f"tasks[{index}].output_dir")
        if output_dir in output_dirs:
            raise AggregationError(f"duplicate task output_dir {output_dir!r}")
        output_dirs.add(output_dir)
        family = _string(
            task.get("model_family"), field=f"tasks[{index}].model_family"
        )
        history_mode = _string(
            task.get("history_mode"), field=f"tasks[{index}].history_mode"
        )
        dimension = _integer(
            task.get("latent_dim"), field=f"tasks[{index}].latent_dim", nonnegative=True
        )
        if dimension == 0:
            raise AggregationError(f"tasks[{index}].latent_dim must be positive")
        arm = _string(task.get("safety_arm"), field=f"tasks[{index}].safety_arm")
        seed = _integer(
            task.get("seed"), field=f"tasks[{index}].seed", nonnegative=True
        )
        definition = _mapping(
            definitions.get(history_mode),
            field=f"plan.history_mode_definitions.{history_mode}",
        )
        expected_encoder = _string(
            definition.get("history_encoder"),
            field=f"history_mode_definitions.{history_mode}.history_encoder",
        )
        expected_length = _integer(
            definition.get("history_length"),
            field=f"history_mode_definitions.{history_mode}.history_length",
            nonnegative=True,
        )
        if expected_length == 0:
            raise AggregationError("history-mode lengths must be positive")
        if (
            task.get("history_encoder") != expected_encoder
            or task.get("history_length") != expected_length
        ):
            raise AggregationError(
                f"task {task_id} disagrees with history-mode definition {history_mode!r}"
            )
        key = (family, history_mode, dimension, arm, seed)
        if key in model_seed_keys:
            raise AggregationError(f"duplicate model/seed task cell {key!r}")
        model_seed_keys.add(key)

    if task_ids != set(range(task_count)):
        raise AggregationError("task IDs must be exactly 0..task_count-1")
    expected_cells = set(
        itertools.product(families, history_modes, dimensions, arms, seeds)
    )
    if model_seed_keys != expected_cells or task_count != len(expected_cells):
        missing = sorted(expected_cells - model_seed_keys)
        extra = sorted(model_seed_keys - expected_cells)
        raise AggregationError(
            "tasks must exactly cover the declared factorial axes; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    return {
        "plan_sha256": plan_hash,
        "tasks": tasks,
        "primary_radius": primary_radius,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "safety_budget": safety_budget,
        "planned_seeds": seeds,
    }


def _resolve_run_dir(task: dict[str, Any]) -> Path:
    output = Path(_string(task.get("output_dir"), field="task.output_dir"))
    return output.resolve() if output.is_absolute() else (ROOT / output).resolve()


def _reference_path(value: object, *, field: str) -> Path:
    reference = Path(_string(value, field=field))
    return reference.resolve() if reference.is_absolute() else (ROOT / reference).resolve()


def _model_config_id(task: dict[str, Any]) -> str:
    return (
        f"family-{task['model_family']}__history-{task['history_mode']}"
        f"__dim-{task['latent_dim']}__arm-{task['safety_arm']}"
    )


def _checkpoint_config_id(task: dict[str, Any]) -> str:
    return f"task-{int(task['task_id']):04d}__{_model_config_id(task)}"


def _check_resolved_config(
    task: dict[str, Any], run: dict[str, Any]
) -> tuple[str, tuple[int, ...]]:
    resolved = _mapping(run.get("resolved_config"), field="run.resolved_config")
    resolved_hash = _string(resolved.get("sha256"), field="run.resolved_config.sha256")
    values = _mapping(resolved.get("values"), field="run.resolved_config.values")
    checks = {
        "model.family": ("model", "family", task["model_family"]),
        "model.history_encoder": ("model", "history_encoder", task["history_encoder"]),
        "model.latent_dim": ("model", "latent_dim", task["latent_dim"]),
        "data.history_length": ("data", "history_length", task["history_length"]),
        "objective.safety_arm": ("objective", "safety_arm", task["safety_arm"]),
        "run.seed": ("run", "seed", task["seed"]),
    }
    for field, (section, key, expected) in checks.items():
        observed = _mapping(
            values.get(section), field=f"resolved_config.values.{section}"
        ).get(key)
        if observed != expected:
            raise AggregationError(
                f"resolved config {field}={observed!r} disagrees with plan value {expected!r}"
            )
    if run.get("seed") != task["seed"] or run.get("training_arm") != task["safety_arm"]:
        raise AggregationError("run seed/training_arm disagrees with the task plan")
    evaluation = _mapping(
        values.get("evaluation"), field="resolved_config.values.evaluation"
    )
    raw_horizons = _sequence(
        evaluation.get("rollout_horizons"),
        field="resolved_config.values.evaluation.rollout_horizons",
    )
    horizons = tuple(
        _integer(
            value,
            field="resolved_config.values.evaluation.rollout_horizons",
            nonnegative=True,
        )
        for value in raw_horizons
    )
    if (
        not horizons
        or any(horizon == 0 for horizon in horizons)
        or list(horizons) != sorted(set(horizons))
    ):
        raise AggregationError(
            "resolved_config.values.evaluation.rollout_horizons must be "
            "sorted unique positive integers"
        )
    return resolved_hash, horizons


def _check_orchestration(
    task: dict[str, Any],
    run: dict[str, Any],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    orchestration = _mapping(
        run.get("orchestration"), field="run.orchestration"
    )
    observed_plan_hash = _string(
        orchestration.get("plan_sha256"),
        field="run.orchestration.plan_sha256",
    )
    if not hmac.compare_digest(observed_plan_hash, plan_sha256):
        raise AggregationError(
            "run.orchestration.plan_sha256 disagrees with the source plan"
        )
    observed_task_id = _integer(
        orchestration.get("task_id"),
        field="run.orchestration.task_id",
        nonnegative=True,
    )
    if observed_task_id != task["task_id"]:
        raise AggregationError(
            "run.orchestration.task_id disagrees with the source plan task"
        )
    completed_values = _sequence(
        orchestration.get("completed_audits"),
        field="run.orchestration.completed_audits",
    )
    completed_audits = [
        _string(value, field="run.orchestration.completed_audits")
        for value in completed_values
    ]
    if len(completed_audits) != len(set(completed_audits)):
        raise AggregationError(
            "run.orchestration.completed_audits must not contain duplicates"
        )
    completed_set = set(completed_audits)
    expected_set = (
        BASELINE_AUDIT_VIEWS
        if task["safety_arm"] == "none"
        else frozenset({LATENT_AUDIT_VIEW})
    )
    if completed_set != expected_set:
        missing = sorted(expected_set - completed_set)
        extra = sorted(completed_set - expected_set)
        raise AggregationError(
            "run.orchestration.completed_audits does not match the runner's "
            f"required views for safety arm {task['safety_arm']!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )
    if orchestration.get("failed_audit") is not None:
        raise AggregationError(
            "run.orchestration.failed_audit must be null for an aggregate-eligible run"
        )
    updated_at = _string(
        orchestration.get("updated_at_utc"),
        field="run.orchestration.updated_at_utc",
    )
    return {
        "plan_sha256": observed_plan_hash,
        "task_id": observed_task_id,
        "completed_audits": completed_audits,
        "failed_audit": None,
        "updated_at_utc": updated_at,
    }


def _rollout_pixel_metrics(
    evaluation: dict[str, Any],
    *,
    split: str,
    configured_horizons: tuple[int, ...],
) -> dict[str, Any]:
    rollout_metrics = _mapping(
        evaluation.get("rollout_metrics"), field="evaluation.rollout_metrics"
    )
    split_rollouts = _mapping(
        rollout_metrics.get(split), field=f"evaluation.rollout_metrics.{split}"
    )
    expected_keys = {str(horizon) for horizon in configured_horizons}
    observed_keys = set(split_rollouts)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AggregationError(
            f"evaluation.rollout_metrics.{split} must exactly cover the resolved "
            f"configured horizons; missing={missing!r}, extra={extra!r}"
        )

    by_horizon: dict[str, dict[str, float | int]] = {}
    for horizon in configured_horizons:
        key = str(horizon)
        point = _mapping(
            split_rollouts.get(key),
            field=f"evaluation.rollout_metrics.{split}.{key}",
        )
        pixel_mse = _finite_number(
            point.get("pixel_mse"),
            field=f"evaluation.rollout_metrics.{split}.{key}.pixel_mse",
            nonnegative=True,
        )
        cases = _integer(
            point.get("cases"),
            field=f"evaluation.rollout_metrics.{split}.{key}.cases",
            nonnegative=True,
        )
        if cases == 0:
            raise AggregationError(
                f"evaluation.rollout_metrics.{split}.{key}.cases must be positive"
            )
        by_horizon[key] = {"pixel_mse": pixel_mse, "cases": cases}

    summary_horizon = max(configured_horizons)
    return {
        "summary_rule": "maximum_predeclared_configured_horizon",
        "summary_horizon": summary_horizon,
        "pixel_mse_at_summary_horizon": by_horizon[str(summary_horizon)]["pixel_mse"],
        "pixel_mse_by_horizon": by_horizon,
    }


def _radius_point(curve: object, *, radius: float, field: str) -> dict[str, Any]:
    points = _sequence(curve, field=field)
    matches: list[dict[str, Any]] = []
    for index, value in enumerate(points):
        point = _mapping(value, field=f"{field}[{index}]")
        observed = _finite_number(
            point.get("relative_radius"),
            field=f"{field}[{index}].relative_radius",
            nonnegative=True,
        )
        if math.isclose(observed, radius, rel_tol=0.0, abs_tol=1e-12):
            matches.append(point)
    if len(matches) != 1:
        raise AggregationError(
            f"{field} must contain exactly one point at primary relative radius {radius}"
        )
    return matches[0]


def _split_metrics(
    evaluation: dict[str, Any],
    audit: dict[str, Any],
    *,
    split: str,
    radius: float,
    configured_rollout_horizons: tuple[int, ...],
) -> dict[str, Any]:
    metrics = _mapping(
        _mapping(evaluation.get("split_metrics"), field="evaluation.split_metrics").get(split),
        field=f"evaluation.split_metrics.{split}",
    )
    utility = _finite_number(
        metrics.get("world_model_utility"),
        field=f"evaluation.split_metrics.{split}.world_model_utility",
        nonnegative=True,
    )
    reconstruction = _finite_number(
        metrics.get("reconstruction"),
        field=f"evaluation.split_metrics.{split}.reconstruction",
        nonnegative=True,
    )
    rollout = _rollout_pixel_metrics(
        evaluation,
        split=split,
        configured_horizons=configured_rollout_horizons,
    )
    split_audit = _mapping(
        _mapping(audit.get("split_audits"), field="audit.split_audits").get(split),
        field=f"audit.split_audits.{split}",
    )
    static = _radius_point(
        split_audit.get("static_curve"), radius=radius, field=f"audit.{split}.static_curve"
    )
    action = _radius_point(
        split_audit.get("action_curve"), radius=radius, field=f"audit.{split}.action_curve"
    )
    estimate = _mapping(static.get("estimate"), field=f"audit.{split}.static.estimate")
    absolute_radius = _finite_number(
        static.get("absolute_radius"),
        field=f"audit.{split}.static.absolute_radius",
        nonnegative=True,
    )
    static_delta = _finite_number(
        estimate.get("delta"),
        field=f"audit.{split}.static.delta",
        nonnegative=True,
    )
    action_delta = _finite_number(
        action.get("delta"),
        field=f"audit.{split}.action.delta",
        nonnegative=True,
    )
    if not (
        math.isclose(absolute_radius, static_delta, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(absolute_radius, action_delta, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise AggregationError("static and action audits disagree on the absolute radius")
    static_defect = _finite_number(
        estimate.get("witness_margin"),
        field=f"audit.{split}.static.witness_margin",
        nonnegative=True,
    )
    static_fraction = _finite_number(
        estimate.get("confounded_fraction"),
        field=f"audit.{split}.static.confounded_fraction",
        nonnegative=True,
    )
    action_fraction = _finite_number(
        action.get("conflict_fraction"),
        field=f"audit.{split}.action.conflict_fraction",
        nonnegative=True,
    )
    if static_fraction > 1.0 or action_fraction > 1.0:
        raise AggregationError("static/action fractions must lie in [0, 1]")
    record_count = _integer(
        split_audit.get("record_count"),
        field=f"audit.{split}.record_count",
        nonnegative=True,
    )
    safe_count = _integer(
        estimate.get("safe_count"),
        field=f"audit.{split}.static.safe_count",
        nonnegative=True,
    )
    confounded_count = _integer(
        estimate.get("confounded_safe_count"),
        field=f"audit.{split}.static.confounded_safe_count",
        nonnegative=True,
    )
    if confounded_count > safe_count or safe_count > record_count:
        raise AggregationError("static defect counts have inconsistent denominators")
    expected_static_fraction = confounded_count / safe_count if safe_count else 0.0
    if not math.isclose(
        static_fraction, expected_static_fraction, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AggregationError("static confounded fraction disagrees with its counts")
    if confounded_count == 0 and static_defect != 0.0:
        raise AggregationError("static witness margin must be zero without a witness")
    action_metrics = {
        "conflict_fraction": action_fraction,
        "worst_required_violation": _finite_number(
            action.get("worst_required_violation"),
            field=f"audit.{split}.action.worst_required_violation",
            nonnegative=True,
        ),
        "conflicting_neighborhood_count": _integer(
            action.get("conflicting_neighborhood_count"),
            field=f"audit.{split}.action.conflicting_neighborhood_count",
            nonnegative=True,
        ),
        "individually_viable_neighborhood_count": _integer(
            action.get("individually_viable_neighborhood_count"),
            field=f"audit.{split}.action.individually_viable_neighborhood_count",
            nonnegative=True,
        ),
        "nontrivial_neighborhood_count": _integer(
            action.get("nontrivial_neighborhood_count"),
            field=f"audit.{split}.action.nontrivial_neighborhood_count",
            nonnegative=True,
        ),
        "state_count": _integer(
            action.get("state_count"),
            field=f"audit.{split}.action.state_count",
            nonnegative=True,
        ),
    }
    trajectory_tail = _mapping(
        action.get("trajectory_tail_summary"),
        field=f"audit.{split}.action.trajectory_tail_summary",
    )
    trajectory_count = _integer(
        trajectory_tail.get("all_trajectory_count"),
        field=f"audit.{split}.action.trajectory_tail_summary.all_trajectory_count",
        nonnegative=True,
    )
    evaluable_trajectory_count = _integer(
        trajectory_tail.get("evaluable_trajectory_count"),
        field=(
            f"audit.{split}.action.trajectory_tail_summary."
            "evaluable_trajectory_count"
        ),
        nonnegative=True,
    )
    tail_required_violation = _finite_number(
        trajectory_tail.get("mean_trajectory_tail_required_violation"),
        field=(
            f"audit.{split}.action.trajectory_tail_summary."
            "mean_trajectory_tail_required_violation"
        ),
        nonnegative=True,
    )
    if evaluable_trajectory_count > trajectory_count:
        raise AggregationError("action trajectory-tail counts have inconsistent denominators")
    if evaluable_trajectory_count == 0 and tail_required_violation != 0.0:
        raise AggregationError("action trajectory-tail mean must be zero without evaluable data")
    if tail_required_violation > action_metrics["worst_required_violation"] + 1e-12:
        raise AggregationError("action trajectory-tail mean exceeds the global witnessed maximum")
    if (
        action_metrics["state_count"] != record_count
        or action_metrics["nontrivial_neighborhood_count"]
        > action_metrics["state_count"]
        or action_metrics["conflicting_neighborhood_count"]
        > action_metrics["individually_viable_neighborhood_count"]
        or action_metrics["individually_viable_neighborhood_count"]
        > action_metrics["nontrivial_neighborhood_count"]
    ):
        raise AggregationError("action conflict counts have inconsistent denominators")
    expected_fraction = (
        action_metrics["conflicting_neighborhood_count"]
        / action_metrics["individually_viable_neighborhood_count"]
        if action_metrics["individually_viable_neighborhood_count"]
        else 0.0
    )
    if not math.isclose(
        action_fraction, expected_fraction, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AggregationError("action conflict fraction disagrees with its counts")
    if (
        action_metrics["conflicting_neighborhood_count"] == 0
        and action_metrics["worst_required_violation"] != 0.0
    ):
        raise AggregationError("action violation must be zero without a conflict")
    if (
        action_metrics["conflicting_neighborhood_count"] == 0
        and tail_required_violation != 0.0
    ):
        raise AggregationError("action trajectory-tail mean must be zero without a conflict")
    if (
        action_metrics["conflicting_neighborhood_count"] > 0
        and action_metrics["worst_required_violation"] <= 0.0
    ):
        raise AggregationError("a conflicting action neighborhood requires positive violation")
    return {
        "utility_loss": utility,
        "reconstruction_mse": reconstruction,
        "rollout_pixel_mse_at_max_horizon": rollout[
            "pixel_mse_at_summary_horizon"
        ],
        "rollout_configured_horizons": list(configured_rollout_horizons),
        "rollout_summary_horizon": rollout["summary_horizon"],
        "rollout_summary_rule": rollout["summary_rule"],
        "rollout_pixel_mse_by_horizon": rollout["pixel_mse_by_horizon"],
        "static_defect": static_defect,
        "static_confounded_fraction": static_fraction,
        "action_conflict_fraction": action_fraction,
        "action_tail_required_violation": tail_required_violation,
        "action_worst_required_violation": action_metrics["worst_required_violation"],
        "action_conflicting_neighborhood_count": action_metrics[
            "conflicting_neighborhood_count"
        ],
        "action_individually_viable_neighborhood_count": action_metrics[
            "individually_viable_neighborhood_count"
        ],
        "action_nontrivial_neighborhood_count": action_metrics[
            "nontrivial_neighborhood_count"
        ],
    }


def _extract_row(
    task: dict[str, Any],
    *,
    plan_sha256: str,
    primary_radius: float,
    unblind_test: bool,
) -> dict[str, Any]:
    run_dir = _resolve_run_dir(task)
    paths = {
        "run_manifest": run_dir / "run_manifest.json",
        "evaluation_manifest": run_dir / "evaluation_manifest.json",
        "latent_safety_audit": run_dir / "latent_safety_audit.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    run = _load_object(paths["run_manifest"], label="run manifest")
    if run.get("status") != "success":
        failure = run.get("failure")
        raise RuntimeError(f"run status is {run.get('status')!r}; failure={failure!r}")
    orchestration = _check_orchestration(
        task,
        run,
        plan_sha256=plan_sha256,
    )
    evaluation = _load_object(paths["evaluation_manifest"], label="evaluation manifest")
    audit = _load_object(paths["latent_safety_audit"], label="latent safety audit")
    if evaluation.get("status") != "success":
        raise AggregationError(f"evaluation status is {evaluation.get('status')!r}")
    if audit.get("status") != "success":
        raise AggregationError(f"audit status is {audit.get('status')!r}")
    if run.get("experiment") != evaluation.get("experiment"):
        raise AggregationError("run and evaluation experiment identifiers disagree")
    run_config_hash = _string(
        _mapping(run.get("config"), field="run.config").get("sha256"),
        field="run.config.sha256",
    )
    evaluation_config_hash = _string(
        _mapping(evaluation.get("config"), field="evaluation.config").get("sha256"),
        field="evaluation.config.sha256",
    )
    audit_config_hash = _string(
        _mapping(audit.get("config"), field="audit.config").get("sha256"),
        field="audit.config.sha256",
    )
    if len({run_config_hash, evaluation_config_hash, audit_config_hash}) != 1:
        raise AggregationError("run, evaluation, and audit base-config hashes disagree")
    run_dataset_hash = _string(
        _mapping(run.get("dataset"), field="run.dataset").get("manifest_sha256"),
        field="run.dataset.manifest_sha256",
    )
    if evaluation.get("dataset_manifest_sha256") != run_dataset_hash:
        raise AggregationError("run and evaluation dataset-manifest hashes disagree")
    if _reference_path(
        run.get("evaluation_manifest"), field="run.evaluation_manifest"
    ) != paths["evaluation_manifest"].resolve():
        raise AggregationError("run points to a different evaluation manifest")
    if _reference_path(
        run.get("checkpoint_manifest"), field="run.checkpoint_manifest"
    ) != _reference_path(
        evaluation.get("checkpoint_manifest"),
        field="evaluation.checkpoint_manifest",
    ):
        raise AggregationError("run and evaluation checkpoint-manifest paths disagree")
    view = _mapping(audit.get("representation_view"), field="audit.representation_view")
    if view.get("name") != "latent" or view.get("oracle_control") is not False:
        raise AggregationError("primary aggregate requires the non-oracle 'latent' audit")
    calibration = _mapping(audit.get("radius_calibration"), field="audit.radius_calibration")
    if calibration.get("uses_safety_labels") is not False:
        raise AggregationError("radius calibration must explicitly report uses_safety_labels=false")
    calibration_scale = _finite_number(
        calibration.get("calibration_scale"),
        field="audit.radius_calibration.calibration_scale",
        nonnegative=True,
    )
    audit_inputs = _mapping(audit.get("inputs"), field="audit.inputs")
    audit_records = _mapping(
        evaluation.get("audit_records"), field="evaluation.audit_records"
    )
    for split in ("calibration", "validation"):
        if _reference_path(
            audit_inputs.get(f"{split}_records"),
            field=f"audit.inputs.{split}_records",
        ) != _reference_path(
            audit_records.get(split), field=f"evaluation.audit_records.{split}"
        ):
            raise AggregationError(
                f"audit and evaluation disagree on {split} record provenance"
            )

    resolved_hash, configured_rollout_horizons = _check_resolved_config(task, run)
    if evaluation.get("resolved_config_sha256") != resolved_hash:
        raise AggregationError("evaluation and run resolved-config hashes disagree")
    checkpoint_hash = _string(
        evaluation.get("checkpoint_sha256"), field="evaluation.checkpoint_sha256"
    )
    validation = _split_metrics(
        evaluation,
        audit,
        split="validation",
        radius=primary_radius,
        configured_rollout_horizons=configured_rollout_horizons,
    )
    expected_absolute_radius = primary_radius * calibration_scale
    validation_audit = _mapping(
        _mapping(audit.get("split_audits"), field="audit.split_audits").get(
            "validation"
        ),
        field="audit.split_audits.validation",
    )
    primary_static = _radius_point(
        validation_audit.get("static_curve"),
        radius=primary_radius,
        field="audit.validation.static_curve",
    )
    observed_absolute_radius = _finite_number(
        primary_static.get("absolute_radius"),
        field="audit.validation.static.absolute_radius",
        nonnegative=True,
    )
    if not math.isclose(
        observed_absolute_radius,
        expected_absolute_radius,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AggregationError(
            "primary absolute radius disagrees with the calibration scale"
        )
    validation_count = _integer(
        audit_inputs.get("validation_count"),
        field="audit.inputs.validation_count",
        nonnegative=True,
    )
    if validation_count != validation_audit.get("record_count"):
        raise AggregationError("validation input count disagrees with the audit")
    row = {
        "config_id": _checkpoint_config_id(task),
        "model_config_id": _model_config_id(task),
        "task_id": int(task["task_id"]),
        "model_family": str(task["model_family"]),
        "history_mode": str(task["history_mode"]),
        "history_encoder": str(task["history_encoder"]),
        "history_length": int(task["history_length"]),
        "latent_dim": int(task["latent_dim"]),
        "safety_arm": str(task["safety_arm"]),
        "seed": int(task["seed"]),
        "checkpoint_sha256": checkpoint_hash,
        "resolved_config_sha256": resolved_hash,
        "orchestration": orchestration,
        "validation": validation,
        "provenance": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    # This is the sole code path that accesses final-test keys.
    if unblind_test:
        row["test"] = _split_metrics(
            evaluation,
            audit,
            split="test",
            radius=primary_radius,
            configured_rollout_horizons=configured_rollout_horizons,
        )
    return row


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row["validation"][metric]) for row in rows)
        for metric in ROW_METRICS
    }


def _median_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: float(
            statistics.median(float(row["validation"][metric]) for row in rows)
        )
        for metric in ROW_METRICS
    }


def _model_rows(
    rows: list[dict[str, Any]], *, planned_seeds: list[int]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_config_id"]), []).append(row)
    result = []
    for config_id, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: int(row["seed"]))
        first = ordered[0]
        rollout_signatures = {
            (
                str(row["validation"]["rollout_summary_rule"]),
                tuple(row["validation"]["rollout_configured_horizons"]),
                int(row["validation"]["rollout_summary_horizon"]),
            )
            for row in ordered
        }
        if len(rollout_signatures) != 1:
            raise AggregationError(
                f"model config {config_id} mixes incompatible rollout-horizon summaries"
            )
        observed_seeds = [int(row["seed"]) for row in ordered]
        complete_seed_set = observed_seeds == sorted(planned_seeds)
        result.append(
            {
                "config_id": config_id,
                "model_family": first["model_family"],
                "history_mode": first["history_mode"],
                "history_encoder": first["history_encoder"],
                "history_length": first["history_length"],
                "latent_dim": first["latent_dim"],
                "safety_arm": first["safety_arm"],
                "seed_count": len(ordered),
                "seeds": observed_seeds,
                "planned_seeds": sorted(planned_seeds),
                "seed_set_status": (
                    "complete" if complete_seed_set else "partial_exploratory"
                ),
                "checkpoint_config_ids": [row["config_id"] for row in ordered],
                "rollout_summary_rule": first["validation"]["rollout_summary_rule"],
                "rollout_configured_horizons": first["validation"][
                    "rollout_configured_horizons"
                ],
                "rollout_summary_horizon": first["validation"][
                    "rollout_summary_horizon"
                ],
                "validation_mean": _mean_metrics(ordered),
                "validation_median": _median_metrics(ordered),
            }
        )
    return result


def _paired_metric(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    blocks = tuple(
        PairedBlock(
            block_id=f"seed-{baseline['seed']}",
            baseline=float(baseline["validation"][metric]),
            candidate=float(candidate["validation"][metric]),
        )
        for baseline, candidate in pairs
    )
    return {
        "paired_bootstrap_ci": dataclasses.asdict(
            paired_block_bootstrap_ci(
                blocks,
                resamples=resamples,
                seed=seed,
            )
        ),
        "standardized_paired_effect": dataclasses.asdict(
            standardized_paired_effect(blocks)
        ),
        "seed_values": [
            {
                "seed": baseline["seed"],
                "baseline": baseline["validation"][metric],
                "candidate": candidate["validation"][metric],
                "difference": (
                    float(candidate["validation"][metric])
                    - float(baseline["validation"][metric])
                ),
            }
            for baseline, candidate in pairs
        ],
    }


def _paired_comparisons(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, int, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_family"]),
            str(row["history_mode"]),
            int(row["latent_dim"]),
            str(row["safety_arm"]),
        )
        by_cell.setdefault(key, {})[int(row["seed"])] = row
    axes = _mapping(plan.get("axes"), field="plan.axes")
    families = _sequence(axes.get("model_families"), field="plan.axes.model_families")
    histories = _sequence(axes.get("history_modes"), field="plan.axes.history_modes")
    dimensions = _sequence(
        axes.get("latent_dimensions"), field="plan.axes.latent_dimensions"
    )
    arms = _sequence(axes.get("safety_arms"), field="plan.axes.safety_arms")
    seeds = {
        _integer(value, field="plan.axes.seeds")
        for value in _sequence(axes.get("seeds"), field="plan.axes.seeds")
    }
    if "none" not in arms:
        raise AggregationError("plan.axes.safety_arms must include baseline arm 'none'")

    comparisons = []
    for family in families:
        for history in histories:
            for dimension in dimensions:
                baseline = by_cell.get(
                    (str(family), str(history), int(dimension), "none"), {}
                )
                for arm in arms:
                    if arm == "none":
                        continue
                    candidate = by_cell.get(
                        (str(family), str(history), int(dimension), str(arm)), {}
                    )
                    paired_seeds = sorted(set(baseline) & set(candidate))
                    missing_seeds = sorted(seeds - set(paired_seeds))
                    comparison_id = (
                        f"family-{family}__history-{history}__dim-{dimension}"
                        f"__candidate-{arm}__baseline-none"
                    )
                    result: dict[str, Any] = {
                        "comparison_id": comparison_id,
                        "model_family": family,
                        "history_mode": history,
                        "latent_dim": int(dimension),
                        "baseline_arm": "none",
                        "candidate_arm": arm,
                        "paired_seeds": paired_seeds,
                        "missing_paired_seeds": missing_seeds,
                        "difference_convention": (
                            "candidate - baseline; all reported metrics are minimized"
                        ),
                    }
                    if len(paired_seeds) < 2:
                        result.update(
                            {
                                "status": "insufficient_pairs",
                                "metrics": {},
                            }
                        )
                    else:
                        pairs = [(baseline[seed], candidate[seed]) for seed in paired_seeds]
                        result.update(
                            {
                                "status": (
                                    "complete" if not missing_seeds else "partial_exploratory"
                                ),
                                "metrics": {
                                    metric: _paired_metric(
                                        pairs,
                                        metric=metric,
                                        resamples=resamples,
                                        seed=bootstrap_seed,
                                    )
                                    for metric in ROW_METRICS
                                },
                            }
                        )
                    comparisons.append(result)
    return comparisons


def _frontier_payload(
    model_rows: list[dict[str, Any]], *, safety_budget: float
) -> dict[str, Any] | None:
    eligible_rows = [
        row for row in model_rows if row["seed_set_status"] == "complete"
    ]
    if not eligible_rows:
        return None
    scores = tuple(
        ValidationScore(
            config_id=str(row["config_id"]),
            safety_defect=float(row["validation_mean"]["static_defect"]),
            utility_loss=float(row["validation_mean"]["utility_loss"]),
        )
        for row in eligible_rows
    )
    frontier = validation_pareto_frontier(scores)
    try:
        selected = dataclasses.asdict(
            select_under_safety_budget(
                frontier,
                max_safety_defect=safety_budget,
            )
        )
        selection_status = "selected"
    except ValueError as error:
        selected = None
        selection_status = f"no_candidate_within_budget: {error}"
    return {
        "selection_split": "validation",
        "objectives": ["static_defect", "utility_loss"],
        "seed_aggregation": "arithmetic_mean_across_complete_planned_seed_set",
        "included_config_ids": [row["config_id"] for row in eligible_rows],
        "excluded_incomplete_config_ids": [
            row["config_id"]
            for row in model_rows
            if row["seed_set_status"] != "complete"
        ],
        "points": [dataclasses.asdict(point) for point in frontier.points],
        "dominated_config_ids": list(frontier.dominated_config_ids),
        "candidate_count": frontier.candidate_count,
        "max_validation_static_defect": safety_budget,
        "budget_selection_status": selection_status,
        "selected": selected,
    }


def aggregate(
    *,
    plan_path: Path,
    allow_partial: bool,
    unblind_test: bool,
) -> tuple[dict[str, Any], int]:
    plan = _load_object(plan_path, label="sweep plan")
    validated = _validate_plan(plan)
    rows: list[dict[str, Any]] = []
    missing_tasks: list[dict[str, Any]] = []
    failed_tasks: list[dict[str, Any]] = []
    for task_value in validated["tasks"]:
        task = _mapping(task_value, field="task")
        identity = {
            "task_id": task["task_id"],
            "config_id": _checkpoint_config_id(task),
            "model_config_id": _model_config_id(task),
            "output_dir": str(_resolve_run_dir(task)),
        }
        try:
            rows.append(
                _extract_row(
                    task,
                    plan_sha256=validated["plan_sha256"],
                    primary_radius=validated["primary_radius"],
                    unblind_test=unblind_test,
                )
            )
        except FileNotFoundError as error:
            missing_tasks.append({**identity, "reason": str(error)})
        except (AggregationError, RuntimeError) as error:
            failed_tasks.append(
                {
                    **identity,
                    "reason": str(error),
                    "failure_kind": type(error).__name__,
                }
            )

    rows.sort(key=lambda row: int(row["task_id"]))
    rollout_horizon_sets = {
        tuple(row["validation"]["rollout_configured_horizons"]) for row in rows
    }
    if len(rollout_horizon_sets) > 1:
        raise AggregationError(
            "successful task rows disagree on configured rollout horizons; "
            "the maximum-horizon utility endpoint would not be comparable"
        )
    checkpoint_hashes = [str(row["checkpoint_sha256"]) for row in rows]
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise AggregationError("multiple task rows resolve to the same checkpoint SHA-256")
    incomplete = bool(missing_tasks or failed_tasks or len(rows) != len(validated["tasks"]))
    rejected = incomplete and not allow_partial
    status = (
        "incomplete_rejected"
        if rejected
        else ("partial_exploratory" if incomplete else "complete")
    )
    model_rows = (
        []
        if rejected
        else _model_rows(rows, planned_seeds=validated["planned_seeds"])
    )
    frontier = (
        None
        if rejected or not model_rows
        else _frontier_payload(model_rows, safety_budget=validated["safety_budget"])
    )
    comparisons = (
        []
        if rejected
        else _paired_comparisons(
            rows,
            plan,
            resamples=validated["bootstrap_resamples"],
            bootstrap_seed=validated["bootstrap_seed"],
        )
    )
    report = {
        "schema_version": 1,
        "experiment": "e1_sweep_validation_aggregate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "analysis_ready": not rejected and frontier is not None,
        "exploratory_partial_mode": bool(allow_partial),
        "test_unblinded": bool(unblind_test),
        "selection_uses_test_data": False,
        "source_plan": {
            "path": str(plan_path),
            "file_sha256": _sha256(plan_path),
            "plan_sha256": validated["plan_sha256"],
            "experiment": plan.get("experiment"),
            "grid": plan.get("grid"),
            "base_config": plan.get("base_config"),
        },
        "analysis_contract": {
            "selection_split": "validation",
            "primary_relative_radius": validated["primary_radius"],
            "max_validation_static_defect": validated["safety_budget"],
            "bootstrap_resamples": validated["bootstrap_resamples"],
            "bootstrap_seed": validated["bootstrap_seed"],
            "inferential_unit": "paired training seed",
            "block_independence_caveat": BLOCK_INDEPENDENCE_CAVEAT,
            "selection_caveat": VALIDATION_SELECTION_CAVEAT,
            "run_eligibility": (
                "run.status=success with exact source-plan/task orchestration, "
                "failed_audit=null, latent audit complete, and all five canonical "
                "audit views complete for baseline safety-arm runs"
            ),
            "selection_metric": "validation.world_model_utility",
            "utility_reporting": {
                "selection_semantics_changed": False,
                "separate_metrics": [
                    "reconstruction_mse",
                    "rollout_pixel_mse_at_max_horizon",
                ],
                "rollout_summary_rule": (
                    "maximum_predeclared_configured_horizon"
                ),
                "rollout_horizon_provenance": (
                    "run.resolved_config.values.evaluation.rollout_horizons; "
                    "each row retains pixel MSE and case count for every horizon"
                ),
            },
            "test_access": (
                "explicitly_unblinded_for_reporting_only"
                if unblind_test
                else "not_accessed"
            ),
        },
        "metric_definitions": {
            "utility_loss": (
                "split-specific world_model_utility composite; smaller is better; "
                "retained as the unchanged checkpoint/frontier selection metric and "
                "not valid for global AE versus beta-VAE utility matching"
            ),
            "reconstruction_mse": (
                "split-specific mean squared current-frame reconstruction error; "
                "smaller is better"
            ),
            "rollout_pixel_mse_at_max_horizon": (
                "split-specific pixel-space rollout mean squared error at the maximum "
                "predeclared resolved-config horizon; smaller is better; each row "
                "retains all configured horizon values and case counts"
            ),
            "static_defect": (
                "validation finite-sample witness_margin at the primary calibrated "
                "radius; smaller is better"
            ),
            "action_conflict_fraction": (
                "validation conflicting/individually-viable centered neighborhoods "
                "at the primary radius"
            ),
            "action_tail_required_violation": (
                "validation mean across trajectories of the within-trajectory 95th "
                "percentile finite-action common violation at the primary radius"
            ),
            "action_worst_required_violation": (
                "validation worst finite-action common-action violation at the "
                "primary radius"
            ),
        },
        "completeness": {
            "planned_task_count": len(validated["tasks"]),
            "successful_task_count": len(rows),
            "missing_task_count": len(missing_tasks),
            "failed_task_count": len(failed_tasks),
            "missing_tasks": missing_tasks,
            "failed_tasks": failed_tasks,
        },
        "rows": rows,
        "model_rows": model_rows,
        "validation_frontier": frontier,
        "paired_seed_comparisons": comparisons,
        "warnings": [
            (
                "Finite validation audits are sample witnesses, not population or "
                "deployment certificates."
            ),
            (
                "Three-seed pilot intervals are exploratory and do not replace the "
                "confirmatory seed plan."
            ),
            *(
                [
                    "PARTIAL MODE: missing/failed tasks are retained above; model means, "
                    "frontiers, and pairs may use unequal or incomplete seed sets."
                ]
                if incomplete and allow_partial
                else []
            ),
            *(
                [
                    "INCOMPLETE REJECTED: no frontier or paired comparisons were "
                    "produced; rerun only after all planned tasks succeed or opt in "
                    "with --allow-partial."
                ]
                if rejected
                else []
            ),
            *(
                [
                    "TEST UNBLINDED: test values are descriptive reporting fields "
                    "only and did not enter grouping, frontier construction, pairing, "
                    "or selection."
                ]
                if unblind_test
                else []
            ),
        ],
    }
    return report, 2 if rejected else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="emit an explicitly exploratory aggregate despite missing or failed planned tasks",
    )
    parser.add_argument(
        "--unblind-test",
        action="store_true",
        help="include test metrics for reporting; validation still exclusively controls selection",
    )
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else (ROOT / args.plan)
    output_path = args.output if args.output.is_absolute() else (ROOT / args.output)
    try:
        report, return_code = aggregate(
            plan_path=plan_path.resolve(),
            allow_partial=args.allow_partial,
            unblind_test=args.unblind_test,
        )
    except AggregationError as error:
        parser.error(str(error))
    write_json_atomic(output_path.resolve(), report)
    print(
        f"wrote {report['status']} aggregate with {len(report['rows'])}/"
        f"{report['completeness']['planned_task_count']} tasks to {output_path.resolve()}"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
