"""Deterministic building blocks for the nonprivileged predicted-profile protocol.

These functions freeze fold assignment, teacher seeds, coverage routing, the error statistic, and
the stratum manifest.  The single-fold trainer lives in ``profile_teacher.py`` so these dependency-
free protocol checks remain usable before PyTorch is imported or final-test data exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

PROFILE_PROTOCOL_VERSION = "predicted_profile_v1"
FOLD_COUNT = 5
FOLD_SEED = 20260822
COVERAGE_BUNDLES_PER_DOMAIN_SEED = 200
REQUIRED_NORMALIZED_P95 = 0.10


class ProfileProtocolError(ValueError):
    """Raised when a profile artifact violates the frozen protocol."""


@dataclass(frozen=True)
class ProfileGateResult:
    bundle_count: int
    action_count: int
    normalized_bundle_errors: tuple[float, ...]
    normalized_p95_error: float
    normalized_component_mae: float
    overall_sign_disagreement: float
    per_action_sign_disagreement: tuple[float, ...]
    false_safe_count: int
    unsafe_target_count: int
    false_safe_rate: float
    required_max: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assign_trajectory_folds(
    trajectory_ids: Sequence[str],
    *,
    fold_count: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    """Apply the frozen sort, shuffle, and modulo trajectory-fold rule."""

    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ProfileProtocolError("fold_count must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProfileProtocolError("fold seed must be an integer")
    normalized = list(trajectory_ids)
    if not normalized or any(not isinstance(item, str) or not item for item in normalized):
        raise ProfileProtocolError("trajectory IDs must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ProfileProtocolError("trajectory IDs must be unique")
    shuffled = sorted(normalized)
    random.Random(seed).shuffle(shuffled)
    return {
        trajectory_id: index % fold_count
        for index, trajectory_id in enumerate(shuffled)
    }


def teacher_seed(data_seed: int, fold_index: int) -> int:
    """Return ``10000 + 10 * data_seed + fold_index`` with strict input checks."""

    if (
        isinstance(data_seed, bool)
        or not isinstance(data_seed, int)
        or data_seed < 0
    ):
        raise ProfileProtocolError("data_seed must be a non-negative integer")
    if (
        isinstance(fold_index, bool)
        or not isinstance(fold_index, int)
        or not 0 <= fold_index < FOLD_COUNT
    ):
        raise ProfileProtocolError(f"fold_index must lie in [0, {FOLD_COUNT - 1}]")
    return 10_000 + 10 * data_seed + fold_index


def assign_coverage_teachers(
    bundle_ids: Sequence[str], *, fold_count: int = FOLD_COUNT
) -> dict[str, int]:
    """Route sorted coverage bundles to fold teachers without ensembling."""

    normalized = list(bundle_ids)
    if not normalized or any(not isinstance(item, str) or not item for item in normalized):
        raise ProfileProtocolError("bundle IDs must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ProfileProtocolError("bundle IDs must be unique")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ProfileProtocolError("fold_count must be an integer of at least two")
    return {
        bundle_id: index % fold_count
        for index, bundle_id in enumerate(sorted(normalized))
    }


def nearest_rank(values: Sequence[float], probability: float) -> float:
    """Return the registered nearest-rank empirical quantile."""

    converted = tuple(float(value) for value in values)
    if not converted or any(not math.isfinite(value) for value in converted):
        raise ProfileProtocolError("quantile values must be non-empty and finite")
    if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ProfileProtocolError("probability must lie in (0, 1]")
    ordered = sorted(converted)
    return ordered[math.ceil(probability * len(ordered)) - 1]


def evaluate_profile_gate(
    predictions: Mapping[str, Sequence[float]],
    targets: Mapping[str, Sequence[float]],
    *,
    margin_scale: float,
    required_max: float = REQUIRED_NORMALIZED_P95,
) -> ProfileGateResult:
    """Evaluate the exact bundle-wise maximum-error and sign-error contract."""

    if not math.isfinite(margin_scale) or margin_scale <= 0.0:
        raise ProfileProtocolError("margin_scale must be finite and positive")
    if not math.isfinite(required_max) or required_max < 0.0:
        raise ProfileProtocolError("required_max must be finite and non-negative")
    prediction_ids = set(predictions)
    target_ids = set(targets)
    if not prediction_ids or prediction_ids != target_ids:
        missing_predictions = sorted(target_ids - prediction_ids)
        missing_targets = sorted(prediction_ids - target_ids)
        raise ProfileProtocolError(
            "prediction and target bundle IDs must match exactly; "
            f"missing_predictions={missing_predictions[:3]!r}, "
            f"missing_targets={missing_targets[:3]!r}"
        )

    normalized_bundle_errors: list[float] = []
    component_absolute_errors: list[float] = []
    sign_disagreements: list[int] = []
    per_action_disagreements: list[list[int]] | None = None
    false_safe_count = 0
    unsafe_target_count = 0
    action_count: int | None = None
    for bundle_id in sorted(prediction_ids):
        predicted = tuple(float(value) for value in predictions[bundle_id])
        observed = tuple(float(value) for value in targets[bundle_id])
        if not predicted or len(predicted) != len(observed):
            raise ProfileProtocolError(
                f"bundle {bundle_id!r} must have matching non-empty action profiles"
            )
        if action_count is None:
            action_count = len(predicted)
            per_action_disagreements = [[] for _ in range(action_count)]
        elif len(predicted) != action_count:
            raise ProfileProtocolError("every bundle must use the same action count")
        if any(not math.isfinite(value) for value in predicted + observed):
            raise ProfileProtocolError(f"bundle {bundle_id!r} contains a non-finite value")
        absolute_errors = [
            abs(prediction - target)
            for prediction, target in zip(predicted, observed, strict=True)
        ]
        normalized_bundle_errors.append(max(absolute_errors) / margin_scale)
        component_absolute_errors.extend(error / margin_scale for error in absolute_errors)
        assert per_action_disagreements is not None
        for action_index, (prediction, target) in enumerate(
            zip(predicted, observed, strict=True)
        ):
            disagreement = int((prediction >= 0.0) != (target >= 0.0))
            sign_disagreements.append(disagreement)
            per_action_disagreements[action_index].append(disagreement)
            if target < 0.0:
                unsafe_target_count += 1
                if prediction >= 0.0:
                    false_safe_count += 1

    assert action_count is not None and per_action_disagreements is not None
    p95 = nearest_rank(normalized_bundle_errors, 0.95)
    component_count = len(component_absolute_errors)
    false_safe_rate = (
        false_safe_count / unsafe_target_count if unsafe_target_count else 0.0
    )
    return ProfileGateResult(
        bundle_count=len(normalized_bundle_errors),
        action_count=action_count,
        normalized_bundle_errors=tuple(normalized_bundle_errors),
        normalized_p95_error=p95,
        normalized_component_mae=sum(component_absolute_errors) / component_count,
        overall_sign_disagreement=sum(sign_disagreements) / component_count,
        per_action_sign_disagreement=tuple(
            sum(values) / len(values) for values in per_action_disagreements
        ),
        false_safe_count=false_safe_count,
        unsafe_target_count=unsafe_target_count,
        false_safe_rate=false_safe_rate,
        required_max=required_max,
        passed=p95 <= required_max,
    )


def build_crossfit_manifest(
    *,
    task: str,
    model_family: str,
    data_seed: int,
    training_trajectory_ids: Sequence[str],
    action_grid: Sequence[float],
    horizon: int,
    margin_scale: float,
    inherited_config_sha256: str,
) -> tuple[dict[str, object], str]:
    """Build the immutable pre-training portion of a cross-fit label manifest."""

    if not task or not model_family:
        raise ProfileProtocolError("task and model_family must be non-empty")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ProfileProtocolError("horizon must be a positive integer")
    actions = tuple(float(action) for action in action_grid)
    if (
        len(actions) < 2
        or len(set(actions)) != len(actions)
        or any(not math.isfinite(action) for action in actions)
    ):
        raise ProfileProtocolError("action_grid must contain unique finite actions")
    if not math.isfinite(margin_scale) or margin_scale <= 0.0:
        raise ProfileProtocolError("margin_scale must be finite and positive")
    if (
        len(inherited_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in inherited_config_sha256)
    ):
        raise ProfileProtocolError("inherited_config_sha256 must be lowercase SHA-256")
    assignments = assign_trajectory_folds(training_trajectory_ids)
    folds = {
        str(fold): sorted(
            trajectory_id
            for trajectory_id, assigned_fold in assignments.items()
            if assigned_fold == fold
        )
        for fold in range(FOLD_COUNT)
    }
    if any(not ids for ids in folds.values()):
        raise ProfileProtocolError("every cross-fit fold must contain a trajectory")
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "task": task,
        "model_family": model_family,
        "data_seed": data_seed,
        "fold_seed": FOLD_SEED,
        "fold_count": FOLD_COUNT,
        "folds": folds,
        "teacher_seeds": {
            str(fold): teacher_seed(data_seed, fold) for fold in range(FOLD_COUNT)
        },
        "action_grid": actions,
        "horizon": horizon,
        "margin_scale": margin_scale,
        "coverage_bundles_required": COVERAGE_BUNDLES_PER_DOMAIN_SEED,
        "required_normalized_p95": REQUIRED_NORMALIZED_P95,
        "inherited_config_sha256": inherited_config_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(canonical.encode("utf-8")).hexdigest()
