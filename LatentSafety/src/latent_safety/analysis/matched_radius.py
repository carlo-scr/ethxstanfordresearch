"""Validation-only matched-neighborhood-radius reduction for E2.

The reducer compares a learned representation with its paired ``none`` control without treating
raw latent radii as comparable.  A prospectively fixed relative-radius point is used for the
control.  The learned radius is selected *only* by proximity to the control's median normalized
nonself neighborhood mass; safety values never enter radius selection.

For ``N`` paired validation records, the nonself mass of an eligible center is
``(|B_delta(z_i) intersect I_0| - 1) / (N - 1)``.  The reference population is first restricted to
individually viable samples; a viable center is eligible when its resulting ball contains at least
one other viable audited sample.  Eligible-center coverage is conditioned on this viable
population, while the viable fraction is reported separately.  The reported mass is the median
over eligible centers.  The primary safety endpoint is the arithmetic mean, over trajectories, of
the linearly interpolated 95th percentile of eligible-center required common-action violations
within each trajectory. Empty trajectories are reported explicitly and fail downstream eligibility.

All scale and radius work is validation-only.  The per-arm scale is the median pairwise latent
distance over the same deterministic, sample-ID-selected validation subset.  This scale uses no
safety labels.  The action margins are privileged validation evaluation labels, never fitting
targets.  These finite-grid quantities do not certify unobserved states or continuous actions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from latent_safety.metrics.defect import median_pairwise_distance
from latent_safety.records import AuditRecord, validate_records


RADIUS_AUDIT_SCHEMA_VERSION = 2
MATCHED_RADIUS_SCHEMA_VERSION = 2
RADIUS_AUDIT_PROTOCOL = "e2_validation_radius_curve_v2"
MATCHED_RADIUS_PROTOCOL = "e2_validation_matched_radius_v2"
RADIUS_AUDIT_SEMANTICS_VERSION = (
    "viable_reference_population_all_sample_mass_conditional_coverage_v2"
)
MATCHED_RADIUS_SEMANTICS_VERSION = (
    "viable_reference_mass_matching_conditional_coverage_v2"
)
RELATIVE_RADIUS_GRID = (0.0, 0.01, 0.02, 0.05, 0.10)
CONTROL_REFERENCE_RELATIVE_RADIUS = 0.05
TAIL_QUANTILE = 0.95
MAX_SCALE_POINTS = 1024
MAX_RELATIVE_MASS_MISMATCH = 0.05
MINIMUM_ELIGIBLE_CENTER_COVERAGE = 0.80
MAX_COVERAGE_FRACTION_LOSS_VS_CONTROL = 0.05

MASS_SEMANTICS = (
    "median over eligible centers of (number of viable nonself validation samples inside "
    "the closed latent ball) / (paired validation sample count - 1)"
)
ELIGIBLE_CENTER_SEMANTICS = (
    "the center is individually viable and its closed ball, formed only over individually "
    "viable reference samples, contains at least one nonself sample"
)
COVERAGE_SEMANTICS = (
    "eligible-center coverage is the fraction of individually viable validation centers with "
    "at least one viable nonself neighbor; viable fraction is reported separately"
)
SAFETY_SEMANTICS = (
    "arithmetic mean across all validation trajectories of each trajectory's linearly "
    "interpolated p95 eligible-center required common-action violation"
)
RADIUS_SELECTION_SEMANTICS = (
    "select the learned radius minimizing relative mismatch to the fixed control radius's "
    "median normalized viable-reference nonself mass; break ties toward the smaller learned "
    "relative radius; do not use safety values"
)
MASS_MISMATCH_SEMANTICS = (
    "absolute learned-minus-control median normalized viable-reference nonself mass divided "
    "by the control mass; undefined or nonpositive control mass fails closed"
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _jsonable(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return numeric


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _radius_grid(values: Iterable[float]) -> tuple[float, ...]:
    radii = tuple(_finite(value, field="relative radius", nonnegative=True) for value in values)
    if not radii or tuple(sorted(set(radii))) != radii:
        raise ValueError("relative radii must be non-empty, unique, and sorted")
    return radii


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("a quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _pairing_rows(records: Sequence[AuditRecord]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record.sample_id,
            "trajectory_id": record.trajectory_id,
            "safety_margin": float(record.safety_margin),
            "action_safety_margins": [
                float(value) for value in (record.action_safety_margins or ())
            ],
        }
        for record in records
    ]


def _prepare_validation_records(
    records: Iterable[AuditRecord],
) -> tuple[AuditRecord, ...]:
    materialized = tuple(sorted(validate_records(records), key=lambda row: row.sample_id))
    if len(materialized) < 2:
        raise ValueError("matched-radius reduction requires at least two validation records")
    observed_splits = {record.split for record in materialized}
    if observed_splits != {"validation"}:
        raise ValueError(
            "matched-radius inputs must contain validation records only; found "
            f"{sorted(observed_splits)!r}"
        )
    if any(record.action_safety_margins is None for record in materialized):
        raise ValueError(
            "every validation record must contain evaluation-only action_safety_margins"
        )
    action_counts = {
        len(record.action_safety_margins or ()) for record in materialized
    }
    if len(action_counts) != 1:
        raise ValueError("every validation record must use the same finite action grid")
    return materialized


def validate_paired_records(
    control_records: Iterable[AuditRecord],
    learned_records: Iterable[AuditRecord],
) -> tuple[tuple[AuditRecord, ...], tuple[AuditRecord, ...]]:
    """Require exactly paired physical validation samples while allowing different latents."""

    control = _prepare_validation_records(control_records)
    learned = _prepare_validation_records(learned_records)
    if _pairing_rows(control) != _pairing_rows(learned):
        raise ValueError(
            "control and learned records must have identical sample IDs, trajectory IDs, "
            "physical margins, and evaluation action profiles"
        )
    return control, learned


@dataclass(frozen=True)
class RadiusPoint:
    """One arm's validation endpoint at one prospectively fixed relative radius."""

    relative_radius: float
    absolute_radius: float
    viable_sample_count: int
    viable_fraction: float
    eligible_center_count: int
    eligible_center_coverage: float
    eligible_given_viable: float | None
    median_nonself_neighborhood_mass: float | None
    trajectory_count: int
    evaluable_trajectory_count: int
    empty_trajectory_count: int
    trajectory_balanced_p95_required_violation: float | None

    def validate(self, *, record_count: int) -> None:
        relative = _finite(
            self.relative_radius, field="relative_radius", nonnegative=True
        )
        del relative
        _finite(self.absolute_radius, field="absolute_radius", nonnegative=True)
        viable = _integer(self.viable_sample_count, field="viable_sample_count")
        if viable > record_count:
            raise ValueError("viable_sample_count exceeds record_count")
        viable_fraction = _finite(
            self.viable_fraction,
            field="viable_fraction",
            nonnegative=True,
        )
        if viable_fraction > 1.0 or not math.isclose(
            viable_fraction,
            viable / record_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("viable_fraction must equal viable_sample_count / record_count")
        eligible = _integer(
            self.eligible_center_count, field="eligible_center_count"
        )
        if eligible > viable:
            raise ValueError("eligible_center_count exceeds viable_sample_count")
        coverage = _finite(
            self.eligible_center_coverage,
            field="eligible_center_coverage",
            nonnegative=True,
        )
        expected_given_viable = eligible / viable if viable else None
        if expected_given_viable is None:
            if coverage != 0.0 or self.eligible_given_viable is not None:
                raise ValueError(
                    "zero viable samples require zero eligible_center_coverage and null "
                    "eligible_given_viable"
                )
        else:
            explicit_given_viable = _finite(
                self.eligible_given_viable,
                field="eligible_given_viable",
                nonnegative=True,
            )
            if (
                coverage > 1.0
                or explicit_given_viable > 1.0
                or not math.isclose(
                    coverage,
                    expected_given_viable,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    explicit_given_viable,
                    expected_given_viable,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "coverage fields must equal eligible_center_count / viable_sample_count"
                )
        trajectories = _integer(
            self.trajectory_count, field="trajectory_count", minimum=1
        )
        evaluable = _integer(
            self.evaluable_trajectory_count, field="evaluable_trajectory_count"
        )
        empty = _integer(self.empty_trajectory_count, field="empty_trajectory_count")
        if evaluable + empty != trajectories:
            raise ValueError(
                "evaluable_trajectory_count + empty_trajectory_count must equal "
                "trajectory_count"
            )
        mass = self.median_nonself_neighborhood_mass
        safety = self.trajectory_balanced_p95_required_violation
        if eligible == 0:
            if mass is not None or safety is not None or evaluable != 0:
                raise ValueError(
                    "zero eligible centers require null mass, null safety, and zero "
                    "evaluable trajectories"
                )
            return
        if mass is None:
            raise ValueError("positive eligible-center count requires a defined mass")
        normalized_mass = _finite(
            mass, field="median_nonself_neighborhood_mass", nonnegative=True
        )
        # An eligible center has at least one nonself neighbor.  A zero mass is therefore not a
        # meaningful 0/0 case; treating it as matched would hide an invalid upstream audit.
        if normalized_mass < 1.0 / (record_count - 1) or normalized_mass > 1.0:
            raise ValueError(
                "median_nonself_neighborhood_mass is inconsistent with eligible-center "
                "semantics"
            )
        if safety is None:
            raise ValueError("positive eligible-center count requires a defined safety endpoint")
        _finite(
            safety,
            field="trajectory_balanced_p95_required_violation",
            nonnegative=True,
        )


@dataclass(frozen=True)
class ValidationRadiusAudit:
    """Integrity-checked validation-only radius curve for one representation arm."""

    schema_version: int
    protocol: str
    semantics_version: str
    split: str
    calibration_access: bool
    final_test_access: bool
    safety_used_for_radius_scale: bool
    safety_used_for_radius_selection: bool
    record_count: int
    trajectory_count: int
    action_count: int
    pairing_sha256: str
    validation_scale: float
    scale_method: str
    scale_available_points: int
    scale_used_points: int
    scale_selection_sha256: str
    curve: tuple[RadiusPoint, ...]

    def validate(self) -> None:
        if (
            self.schema_version != RADIUS_AUDIT_SCHEMA_VERSION
            or self.protocol != RADIUS_AUDIT_PROTOCOL
            or self.semantics_version != RADIUS_AUDIT_SEMANTICS_VERSION
        ):
            raise ValueError("unsupported validation radius audit protocol")
        if (
            self.split != "validation_only"
            or self.calibration_access is not False
            or self.final_test_access is not False
        ):
            raise ValueError(
                "radius audit must declare validation-only input with no calibration or "
                "final-test access"
            )
        if self.safety_used_for_radius_scale is not False:
            raise ValueError("safety labels must not be used to set the radius scale")
        if self.safety_used_for_radius_selection is not False:
            raise ValueError("safety values must not be used to select the matched radius")
        records = _integer(self.record_count, field="record_count", minimum=2)
        trajectories = _integer(
            self.trajectory_count, field="trajectory_count", minimum=1
        )
        actions = _integer(self.action_count, field="action_count", minimum=1)
        del trajectories, actions
        _lower_sha256(self.pairing_sha256, field="pairing_sha256")
        scale = _finite(
            self.validation_scale,
            field="validation_scale",
            nonnegative=True,
        )
        if self.scale_method != "median_pairwise_latent_distance":
            raise ValueError("unexpected validation radius scale method")
        available = _integer(
            self.scale_available_points, field="scale_available_points", minimum=2
        )
        used = _integer(self.scale_used_points, field="scale_used_points", minimum=2)
        if available != records or used > min(records, MAX_SCALE_POINTS):
            raise ValueError("invalid validation scale sample counts")
        _lower_sha256(self.scale_selection_sha256, field="scale_selection_sha256")
        if not self.curve:
            raise ValueError("validation radius audit curve must not be empty")
        relative_radii = tuple(point.relative_radius for point in self.curve)
        if _radius_grid(relative_radii) != relative_radii:
            raise ValueError("validation radius audit curve is not canonically ordered")
        if any(point.trajectory_count != self.trajectory_count for point in self.curve):
            raise ValueError("curve trajectory count does not match audit metadata")
        for point in self.curve:
            point.validate(record_count=records)
            expected_absolute_radius = point.relative_radius * scale
            if point.absolute_radius != expected_absolute_radius:
                raise ValueError(
                    "absolute_radius must equal relative_radius * validation_scale"
                )

    def unsigned_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @property
    def audit_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["audit_sha256"] = self.audit_sha256
        payload["semantics"] = _radius_audit_semantics()
        return payload


def _radius_audit_semantics() -> dict[str, object]:
    """Return the exact required prose contract for the v2 numerical payload."""

    return {
        "version": RADIUS_AUDIT_SEMANTICS_VERSION,
        "neighborhood_mass": MASS_SEMANTICS,
        "eligible_center": ELIGIBLE_CENTER_SEMANTICS,
        "coverage": COVERAGE_SEMANTICS,
        "primary_safety": SAFETY_SEMANTICS,
        "finite_evaluation_only": True,
    }


def _radius_point(
    records: tuple[AuditRecord, ...],
    vectors: tuple[tuple[float, ...], ...],
    *,
    relative_radius: float,
    absolute_radius: float,
) -> RadiusPoint:
    record_count = len(records)
    trajectory_ids = tuple(sorted({record.trajectory_id for record in records}))
    action_rows = tuple(
        tuple(float(value) for value in (record.action_safety_margins or ()))
        for record in records
    )
    individually_viable = tuple(max(row) >= 0.0 for row in action_rows)
    closed_radius = (
        absolute_radius
        if absolute_radius == 0.0
        else math.nextafter(absolute_radius, math.inf)
    )
    masses: list[float] = []
    by_trajectory: dict[str, list[float]] = {}
    viable_indices = tuple(
        index for index, is_viable in enumerate(individually_viable) if is_viable
    )
    for center_index in viable_indices:
        center = vectors[center_index]
        members = tuple(
            index
            for index in viable_indices
            if _distance(center, vectors[index]) <= closed_radius
        )
        if len(members) < 2:
            continue
        masses.append((len(members) - 1) / (record_count - 1))
        common_action_margins = tuple(
            min(action_rows[index][action] for index in members)
            for action in range(len(action_rows[0]))
        )
        required_violation = max(0.0, -max(common_action_margins))
        by_trajectory.setdefault(records[center_index].trajectory_id, []).append(
            required_violation
        )

    trajectory_p95 = tuple(
        _linear_quantile(by_trajectory[trajectory_id], TAIL_QUANTILE)
        for trajectory_id in trajectory_ids
        if trajectory_id in by_trajectory
    )
    eligible_count = len(masses)
    evaluable_trajectory_count = len(trajectory_p95)
    point = RadiusPoint(
        relative_radius=relative_radius,
        absolute_radius=absolute_radius,
        viable_sample_count=len(viable_indices),
        viable_fraction=len(viable_indices) / record_count,
        eligible_center_count=eligible_count,
        eligible_center_coverage=(
            eligible_count / len(viable_indices) if viable_indices else 0.0
        ),
        eligible_given_viable=(
            eligible_count / len(viable_indices) if viable_indices else None
        ),
        median_nonself_neighborhood_mass=(
            float(statistics.median(masses)) if masses else None
        ),
        trajectory_count=len(trajectory_ids),
        evaluable_trajectory_count=evaluable_trajectory_count,
        empty_trajectory_count=len(trajectory_ids) - evaluable_trajectory_count,
        trajectory_balanced_p95_required_violation=(
            statistics.fmean(trajectory_p95) if trajectory_p95 else None
        ),
    )
    point.validate(record_count=record_count)
    return point


def build_validation_radius_audit(
    records: Iterable[AuditRecord],
    *,
    relative_radii: Iterable[float] = RELATIVE_RADIUS_GRID,
    max_scale_points: int = MAX_SCALE_POINTS,
) -> ValidationRadiusAudit:
    """Build one deterministic validation-only curve from evaluation audit records."""

    materialized = _prepare_validation_records(records)
    radii = _radius_grid(relative_radii)
    if isinstance(max_scale_points, bool) or not isinstance(max_scale_points, Integral):
        raise ValueError("max_scale_points must be an integer")
    max_points = int(max_scale_points)
    if max_points < 2 or max_points > MAX_SCALE_POINTS:
        raise ValueError(f"max_scale_points must lie in [2, {MAX_SCALE_POINTS}]")
    selected = tuple(
        sorted(
            materialized,
            key=lambda record: hashlib.sha256(record.sample_id.encode("utf-8")).digest(),
        )[:max_points]
    )
    scale = float(median_pairwise_distance([record.latent for record in selected]))
    vectors = tuple(tuple(float(value) for value in record.latent) for record in materialized)
    curve = tuple(
        _radius_point(
            materialized,
            vectors,
            relative_radius=relative,
            absolute_radius=relative * scale,
        )
        for relative in radii
    )
    pairing_sha256 = _canonical_sha256({"paired_validation_records": _pairing_rows(materialized)})
    selection_sha256 = _canonical_sha256(
        {"scale_sample_ids": [record.sample_id for record in selected]}
    )
    audit = ValidationRadiusAudit(
        schema_version=RADIUS_AUDIT_SCHEMA_VERSION,
        protocol=RADIUS_AUDIT_PROTOCOL,
        semantics_version=RADIUS_AUDIT_SEMANTICS_VERSION,
        split="validation_only",
        calibration_access=False,
        final_test_access=False,
        safety_used_for_radius_scale=False,
        safety_used_for_radius_selection=False,
        record_count=len(materialized),
        trajectory_count=len({record.trajectory_id for record in materialized}),
        action_count=len(materialized[0].action_safety_margins or ()),
        pairing_sha256=pairing_sha256,
        validation_scale=scale,
        scale_method="median_pairwise_latent_distance",
        scale_available_points=len(materialized),
        scale_used_points=len(selected),
        scale_selection_sha256=selection_sha256,
        curve=curve,
    )
    audit.validate()
    return audit


def validation_radius_audit_from_dict(payload: object) -> ValidationRadiusAudit:
    """Parse and verify the checksum of an arm-level radius audit artifact."""

    if not isinstance(payload, dict):
        raise ValueError("validation radius audit must be a JSON object")
    declared_sha256 = _lower_sha256(
        payload.get("audit_sha256"), field="audit_sha256"
    )
    allowed = {field.name for field in dataclasses.fields(ValidationRadiusAudit)} | {
        "audit_sha256",
        "semantics",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"unexpected validation radius audit fields: {unexpected!r}")
    raw_curve = payload.get("curve")
    if not isinstance(raw_curve, list):
        raise ValueError("validation radius audit curve must be a list")
    try:
        curve = tuple(
            RadiusPoint(**row) if isinstance(row, dict) else None for row in raw_curve
        )
        if any(point is None for point in curve):
            raise TypeError("curve entries must be objects")
        audit = ValidationRadiusAudit(
            **{
                field.name: payload[field.name]
                for field in dataclasses.fields(ValidationRadiusAudit)
                if field.name != "curve"
            },
            curve=curve,  # type: ignore[arg-type]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid validation radius audit fields: {error}") from error
    audit.validate()
    actual_sha256 = audit.audit_sha256
    if not hmac.compare_digest(declared_sha256, actual_sha256):
        raise ValueError(
            f"validation radius audit SHA-256 mismatch: declared {declared_sha256}, "
            f"recomputed {actual_sha256}"
        )
    semantics = payload.get("semantics")
    if semantics is None:
        raise ValueError("validation radius audit must include exact v2 semantics")
    if semantics != _radius_audit_semantics():
        raise ValueError("validation radius audit semantics do not match the implementation")
    return audit


def _find_point(audit: ValidationRadiusAudit, relative_radius: float) -> RadiusPoint:
    matches = tuple(
        point
        for point in audit.curve
        if math.isclose(
            point.relative_radius, relative_radius, rel_tol=0.0, abs_tol=1e-15
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"control reference relative radius {relative_radius} must occur exactly once"
        )
    return matches[0]


def _relative_mass_mismatch(
    learned_mass: float | None, control_mass: float | None
) -> float | None:
    # Zero cannot be generated by a valid eligible-center audit.  Rejecting it explicitly avoids
    # silently treating undefined 0/0 neighborhood support as a perfect match.
    if learned_mass is None or control_mass is None or control_mass <= 0.0:
        return None
    mismatch = abs(learned_mass - control_mass) / control_mass
    return mismatch if math.isfinite(mismatch) else None


def _selector_fields(point: RadiusPoint) -> dict[str, Any] | None:
    if (
        point.median_nonself_neighborhood_mass is None
        or point.trajectory_balanced_p95_required_violation is None
    ):
        return None
    return {
        "safety": point.trajectory_balanced_p95_required_violation,
        "eligible_center_coverage": point.eligible_center_coverage,
        "viable_fraction": point.viable_fraction,
        "eligible_given_viable": point.eligible_given_viable,
        "neighborhood_mass": point.median_nonself_neighborhood_mass,
        "empty_trajectory_count": point.empty_trajectory_count,
        "coverage_complete": point.empty_trajectory_count == 0,
    }


def _matched_radius_semantics() -> dict[str, object]:
    """Return the exact v2 contract for matching two checksum-verified radius curves."""

    return {
        "version": MATCHED_RADIUS_SEMANTICS_VERSION,
        "neighborhood_mass": MASS_SEMANTICS,
        "eligible_center": ELIGIBLE_CENTER_SEMANTICS,
        "coverage": COVERAGE_SEMANTICS,
        "radius_selection": RADIUS_SELECTION_SEMANTICS,
        "relative_mass_mismatch": MASS_MISMATCH_SEMANTICS,
        "primary_safety": SAFETY_SEMANTICS,
        "finite_evaluation_only": True,
    }


@dataclass(frozen=True)
class MatchedRadiusResult:
    """One deterministic mass-matched radius choice and downstream gate diagnostics."""

    schema_version: int
    protocol: str
    semantics_version: str
    control_audit_sha256: str
    learned_audit_sha256: str
    pairing_sha256: str
    control_reference_relative_radius: float
    learned_relative_radius: float | None
    relative_neighborhood_mass_mismatch: float | None
    maximum_relative_neighborhood_mass_mismatch: float
    mass_match_passed: bool
    minimum_eligible_center_coverage: float
    maximum_coverage_fraction_loss_vs_control: float
    coverage_fraction_loss_vs_control: float | None
    coverage_floor_passed: bool
    coverage_drop_passed: bool
    strict_safety_improvement: bool
    radius_selection_used_safety: bool
    control_point: RadiusPoint
    learned_point: RadiusPoint | None
    failure_reasons: tuple[str, ...]

    @property
    def selector_metrics_ready(self) -> bool:
        control_fields = _selector_fields(self.control_point)
        learned_fields = (
            _selector_fields(self.learned_point)
            if self.learned_point is not None
            else None
        )
        return bool(
            control_fields is not None
            and learned_fields is not None
            and control_fields["coverage_complete"]
            and learned_fields["coverage_complete"]
        )

    @property
    def all_matched_radius_gates_passed(self) -> bool:
        return (
            self.selector_metrics_ready
            and self.mass_match_passed
            and self.coverage_floor_passed
            and self.coverage_drop_passed
            and self.strict_safety_improvement
            and self.control_point.empty_trajectory_count == 0
            and self.learned_point is not None
            and self.learned_point.empty_trajectory_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        learned_absolute_radius = (
            self.learned_point.absolute_radius
            if self.learned_point is not None
            else None
        )
        payload.update(
            {
                "semantics": _matched_radius_semantics(),
                "selector_metrics_ready": self.selector_metrics_ready,
                "all_matched_radius_gates_passed": self.all_matched_radius_gates_passed,
                "selector_metric_fields": {
                    "control": _selector_fields(self.control_point),
                    "learned": (
                        _selector_fields(self.learned_point)
                        if self.learned_point is not None
                        else None
                    ),
                },
                "radius_selection_rule": (
                    "minimize relative median normalized nonself-mass mismatch; ties use "
                    "smaller learned relative radius; safety is evaluated only afterward"
                ),
                "selector_radius_provenance": {
                    "control_reference_relative_radius": (
                        self.control_reference_relative_radius
                    ),
                    "control_absolute_radius": self.control_point.absolute_radius,
                    "learned_relative_radius": self.learned_relative_radius,
                    "learned_absolute_radius": learned_absolute_radius,
                    "control_audit_sha256": self.control_audit_sha256,
                    "learned_audit_sha256": self.learned_audit_sha256,
                    "pairing_sha256": self.pairing_sha256,
                    "relative_neighborhood_mass_mismatch": (
                        self.relative_neighborhood_mass_mismatch
                    ),
                    "mass_match_passed": self.mass_match_passed,
                    "minimum_eligible_center_coverage": (
                        self.minimum_eligible_center_coverage
                    ),
                    "maximum_coverage_fraction_loss_vs_control": (
                        self.maximum_coverage_fraction_loss_vs_control
                    ),
                    "coverage_fraction_loss_vs_control": (
                        self.coverage_fraction_loss_vs_control
                    ),
                    "coverage_floor_passed": self.coverage_floor_passed,
                    "coverage_drop_passed": self.coverage_drop_passed,
                    "strict_safety_improvement": self.strict_safety_improvement,
                },
            }
        )
        return payload


def match_validation_radius_audits(
    control: ValidationRadiusAudit,
    learned: ValidationRadiusAudit,
    *,
    control_reference_relative_radius: float = CONTROL_REFERENCE_RELATIVE_RADIUS,
    max_relative_mass_mismatch: float = MAX_RELATIVE_MASS_MISMATCH,
    minimum_eligible_center_coverage: float = MINIMUM_ELIGIBLE_CENTER_COVERAGE,
    max_coverage_fraction_loss_vs_control: float = (
        MAX_COVERAGE_FRACTION_LOSS_VS_CONTROL
    ),
) -> MatchedRadiusResult:
    """Choose the learned radius by mass alone and report the frozen safety sign."""

    control.validate()
    learned.validate()
    reference = _finite(
        control_reference_relative_radius,
        field="control_reference_relative_radius",
        nonnegative=True,
    )
    tolerance = _finite(
        max_relative_mass_mismatch,
        field="max_relative_mass_mismatch",
        nonnegative=True,
    )
    if tolerance > 1.0:
        raise ValueError("max_relative_mass_mismatch must lie in [0, 1]")
    minimum_coverage = _finite(
        minimum_eligible_center_coverage,
        field="minimum_eligible_center_coverage",
        nonnegative=True,
    )
    maximum_coverage_loss = _finite(
        max_coverage_fraction_loss_vs_control,
        field="max_coverage_fraction_loss_vs_control",
        nonnegative=True,
    )
    if minimum_coverage > 1.0 or maximum_coverage_loss > 1.0:
        raise ValueError("coverage thresholds must lie in [0, 1]")
    if (
        control.record_count != learned.record_count
        or control.trajectory_count != learned.trajectory_count
        or control.action_count != learned.action_count
        or not hmac.compare_digest(control.pairing_sha256, learned.pairing_sha256)
        or not hmac.compare_digest(
            control.scale_selection_sha256, learned.scale_selection_sha256
        )
    ):
        raise ValueError(
            "control and learned radius audits are not the same paired physical validation "
            "set and deterministic scale subset"
        )
    control_radii = tuple(point.relative_radius for point in control.curve)
    learned_radii = tuple(point.relative_radius for point in learned.curve)
    if control_radii != learned_radii:
        raise ValueError("control and learned audits must use the same frozen radius grid")

    control_point = _find_point(control, reference)
    candidates = tuple(
        (mismatch, point)
        for point in learned.curve
        if (
            mismatch := _relative_mass_mismatch(
                point.median_nonself_neighborhood_mass,
                control_point.median_nonself_neighborhood_mass,
            )
        )
        is not None
    )
    selected_mismatch: float | None = None
    selected_point: RadiusPoint | None = None
    if candidates:
        selected_mismatch, selected_point = min(
            candidates,
            key=lambda item: (item[0], item[1].relative_radius),
        )

    selected_coverage = (
        selected_point.eligible_given_viable if selected_point is not None else None
    )
    control_coverage = control_point.eligible_given_viable
    coverage_loss = (
        control_coverage - selected_coverage
        if control_coverage is not None and selected_coverage is not None
        else None
    )
    coverage_floor_passed = bool(
        selected_coverage is not None and selected_coverage >= minimum_coverage
    )
    coverage_drop_passed = bool(
        coverage_loss is not None and coverage_loss <= maximum_coverage_loss
    )

    reasons: list[str] = []
    if control_point.median_nonself_neighborhood_mass is None:
        reasons.append("control_reference_has_no_eligible_centers")
    elif control_point.median_nonself_neighborhood_mass <= 0.0:
        reasons.append("control_reference_mass_is_zero_or_undefined")
    if control_point.empty_trajectory_count > 0:
        reasons.append("control_reference_has_empty_trajectory")
    if selected_point is None or selected_mismatch is None:
        reasons.append("no_learned_radius_has_defined_nonzero_mass_match")
    else:
        if selected_mismatch > tolerance:
            reasons.append("relative_neighborhood_mass_mismatch_above_threshold")
        if selected_point.empty_trajectory_count > 0:
            reasons.append("learned_matched_radius_has_empty_trajectory")
        if not coverage_floor_passed:
            reasons.append("eligible_center_coverage_below_minimum")
        if not coverage_drop_passed:
            reasons.append("coverage_loss_vs_control_above_maximum")

    strict_improvement = bool(
        selected_point is not None
        and selected_point.trajectory_balanced_p95_required_violation is not None
        and control_point.trajectory_balanced_p95_required_violation is not None
        and selected_point.trajectory_balanced_p95_required_violation
        < control_point.trajectory_balanced_p95_required_violation
    )
    if not strict_improvement:
        reasons.append("matched_mass_safety_improvement_is_not_strict")

    result = MatchedRadiusResult(
        schema_version=MATCHED_RADIUS_SCHEMA_VERSION,
        protocol=MATCHED_RADIUS_PROTOCOL,
        semantics_version=MATCHED_RADIUS_SEMANTICS_VERSION,
        control_audit_sha256=control.audit_sha256,
        learned_audit_sha256=learned.audit_sha256,
        pairing_sha256=control.pairing_sha256,
        control_reference_relative_radius=reference,
        learned_relative_radius=(
            selected_point.relative_radius if selected_point is not None else None
        ),
        relative_neighborhood_mass_mismatch=selected_mismatch,
        maximum_relative_neighborhood_mass_mismatch=tolerance,
        mass_match_passed=(
            selected_mismatch is not None and selected_mismatch <= tolerance
        ),
        minimum_eligible_center_coverage=minimum_coverage,
        maximum_coverage_fraction_loss_vs_control=maximum_coverage_loss,
        coverage_fraction_loss_vs_control=coverage_loss,
        coverage_floor_passed=coverage_floor_passed,
        coverage_drop_passed=coverage_drop_passed,
        strict_safety_improvement=strict_improvement,
        radius_selection_used_safety=False,
        control_point=control_point,
        learned_point=selected_point,
        failure_reasons=tuple(reasons),
    )
    return result
