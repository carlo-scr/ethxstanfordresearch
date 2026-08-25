"""Fail-closed validation-only weight selection for the frozen E2 learned arms.

The selector consumes one aggregate validation row for every planned pilot cell.  It requires the
complete 288-row factorial (18 paired ``none`` controls and 270 positive-weight learned-arm rows),
applies the prospectively frozen utility, coverage, matched-neighborhood-mass, and predicted-profile
gates, and returns one selected positive weight per domain/family/arm stratum.  It never consumes
calibration or final-test metrics.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Any

from latent_safety.analysis.matched_radius import (
    MAX_COVERAGE_FRACTION_LOSS_VS_CONTROL,
    MINIMUM_ELIGIBLE_CENTER_COVERAGE,
)


DOMAINS = (
    "controlled_cart_video",
    "controlled_pendulum_video",
    "controlled_dubins_navigation_pixels",
)
MODEL_FAMILIES = ("ae", "beta_vae")
CONTROL_ARM = "none"
LEARNED_ARMS = (
    "h_prediction",
    "nonprivileged_predicted_action_profile",
    "fcsrl_feasibility_loss_adaptation",
)
PROFILE_ARM = "nonprivileged_predicted_action_profile"
PILOT_SEEDS = (0, 1, 2)
POSITIVE_WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0)
EXPECTED_OBSERVATIONS = 288
MAX_UTILITY_RATIO = 1.05
MAX_RELATIVE_UTILITY_DEGRADATION = 0.05
MINIMUM_COVERAGE = MINIMUM_ELIGIBLE_CENTER_COVERAGE
MAX_COVERAGE_LOSS = MAX_COVERAGE_FRACTION_LOSS_VS_CONTROL
MAX_NEIGHBORHOOD_MASS_MISMATCH = 0.05
MAX_PROFILE_NORMALIZED_P95_ERROR = 0.10
REQUIRED_PROFILE_TEACHERS = 5


def frozen_e2_eligibility_contract() -> dict[str, object]:
    """Return the exact declared validation-eligibility contract used by selection."""

    return {
        "required_strata": "every required domain-family-data-seed validation cell",
        "max_relative_reconstruction_degradation": MAX_RELATIVE_UTILITY_DEGRADATION,
        "max_relative_rollout_degradation": MAX_RELATIVE_UTILITY_DEGRADATION,
        "minimum_eligible_center_coverage": MINIMUM_COVERAGE,
        "max_coverage_fraction_loss_vs_none": MAX_COVERAGE_LOSS,
        "eligible_center_coverage_denominator": "individually viable validation centers",
        "minimum_eligible_centers_per_trajectory": 1,
        "neighborhood_mass_statistic": "median nonself neighborhood mass",
        "max_relative_neighborhood_mass_mismatch": (
            MAX_NEIGHBORHOOD_MASS_MISMATCH
        ),
        "required_matched_mass_effect": "primary p95 improvement retains its sign",
        "matched_radius_freeze": "before final-test access",
        "predicted_profile_normalized_p95_error_max": (
            MAX_PROFILE_NORMALIZED_P95_ERROR
        ),
    }


def frozen_e2_fail_closed_contract() -> dict[str, object]:
    """Return the exact failure policy shared by planning and weight selection."""

    return {
        "all_required_core_learned_arms_must_pass_all_required_cells": True,
        "no_eligible_weight": "arm fails the stratum",
        "empty_trajectory_allowed": False,
        "post_hoc_arm_substitution_allowed": False,
        "any_confirmation_teacher_failure": (
            "cancel confirmation before final-test access"
        ),
        "on_any_required_failure": (
            "cancel or prospectively reframe before final-test access"
        ),
    }


def frozen_e2_utility_selection_contract() -> dict[str, object]:
    """Return selection declarations whose thresholds are implemented below."""

    return {
        "primary_utility": "paired_none_normalized_reconstruction_and_rollout_ratios",
        "utility_noninferiority": (
            f"each utility ratio at most {MAX_UTILITY_RATIO:.2f}"
        ),
        "zero_control_error": (
            "only zero learned error is eligible; define the zero-over-zero ratio as 1"
        ),
        "tie_break": (
            "lowest primary safety endpoint, then lower maximum utility ratio, "
            "then smaller weight"
        ),
        "safety_selection_aggregate": (
            "arithmetic mean across all three complete paired preconfirmation seeds"
        ),
        "reconstruction_ratio": (
            "mean arm validation reconstruction MSE divided by mean paired-none "
            "validation reconstruction MSE"
        ),
        "rollout_ratio": (
            "mean arm validation maximum-horizon rollout pixel MSE divided by mean "
            "paired-none validation rollout MSE"
        ),
    }


def _frozen_e2_selection_contract() -> dict[str, object]:
    return {
        "rule": "validation_pareto_frontier_only",
        **frozen_e2_utility_selection_contract(),
        "primary_safety": (
            "trajectory_balanced_p95_required_common_action_violation_at_frozen_radius"
        ),
        "secondary_safety": "pre_registered_operational_defect_auc",
        "tuning_scope": (
            "required_core_learned_arms only; jobs are outside the 192 fresh-seed core"
        ),
        "primary_aggregate": "equal weight per domain-by-model-family stratum",
        "joint_success": (
            "safety superiority and utility noninferiority in every domain"
        ),
    }


def frozen_e2_matched_radius_contract() -> dict[str, object]:
    """Return declarations tied directly to the matched-radius implementation constants."""

    from latent_safety.analysis.matched_radius import (
        CONTROL_REFERENCE_RELATIVE_RADIUS,
        COVERAGE_SEMANTICS,
        ELIGIBLE_CENTER_SEMANTICS,
        MASS_SEMANTICS,
        MAX_RELATIVE_MASS_MISMATCH,
        MAX_SCALE_POINTS,
        RELATIVE_RADIUS_GRID,
        TAIL_QUANTILE,
    )

    return {
        "entrypoint": "scripts/reduce_preconfirmation_radius.py",
        "input_split": "validation_only",
        "relative_radius_grid": list(RELATIVE_RADIUS_GRID),
        "control_reference_relative_radius": CONTROL_REFERENCE_RELATIVE_RADIUS,
        "scale": (
            "per-arm median pairwise latent distance over at most "
            f"{MAX_SCALE_POINTS} validation samples selected by SHA-256 of sample ID"
        ),
        "max_scale_points": MAX_SCALE_POINTS,
        "scale_uses_safety_labels": False,
        "mass": MASS_SEMANTICS,
        "eligible_center": ELIGIBLE_CENTER_SEMANTICS,
        "coverage": COVERAGE_SEMANTICS,
        "primary_safety": (
            "arithmetic mean across validation trajectories of the within-trajectory "
            "linearly interpolated p95 eligible-center required common-action violation"
        ),
        "trajectory_tail_quantile": TAIL_QUANTILE,
        "radius_selection": (
            "minimize relative median normalized nonself-mass mismatch; ties use smaller "
            "learned relative radius"
        ),
        "radius_selection_uses_safety": False,
        "zero_mass": (
            "undefined support; fail closed rather than treating zero-over-zero as a match"
        ),
        "max_relative_mass_mismatch": MAX_RELATIVE_MASS_MISMATCH,
        "paired_identity": (
            "identical sample IDs, trajectory IDs, physical margins, and evaluation "
            "action profiles"
        ),
        "observation_provenance": (
            "every learned selector row carries control relative radius "
            f"{CONTROL_REFERENCE_RELATIVE_RADIUS:.2f}, selected learned relative radius, "
            "both absolute radii, control and learned audit SHA-256, pairing SHA-256, "
            "mass mismatch, and strict sign"
        ),
        "freeze_provenance": (
            "the selected-weight artifact retains the selected candidate's complete "
            "matched-radius provenance"
        ),
        "calibration_access": False,
        "final_test_access": False,
    }


def _exact_protocol_table(
    payload: Mapping[str, object], name: str, expected: Mapping[str, object]
) -> None:
    value = payload.get(name)
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ValueError(
            f"intervention [{name}] must exactly match the frozen E2 protocol"
        )


def validate_frozen_e2_intervention_protocol(
    payload: Mapping[str, object],
) -> None:
    """Fail closed when TOML declarations disagree with executable E2 constants."""

    if not isinstance(payload, Mapping):
        raise ValueError("E2 intervention protocol must be a mapping")
    _exact_protocol_table(payload, "selection", _frozen_e2_selection_contract())
    _exact_protocol_table(
        payload, "eligibility", frozen_e2_eligibility_contract()
    )
    _exact_protocol_table(
        payload, "matched_radius_reduction", frozen_e2_matched_radius_contract()
    )
    _exact_protocol_table(
        payload, "fail_closed", frozen_e2_fail_closed_contract()
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return numeric


def _fraction(value: object, *, field: str) -> float:
    numeric = _finite_nonnegative(value, field=field)
    if numeric > 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return numeric


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class PreconfirmationObservation:
    """One validation-only aggregate for a control or positive-weight learned arm."""

    domain: str
    model_family: str
    arm: str
    seed: int
    weight: float | None
    safety: float
    reconstruction: float
    rollout: float
    eligible_center_coverage: float
    neighborhood_mass: float
    empty_trajectory_count: int
    run_complete: bool
    coverage_complete: bool
    profile_normalized_p95_error: float | None = None
    profile_teacher_count: int | None = None
    profile_teacher_failure_count: int | None = None
    profile_label_manifest_sha256: str | None = None

    def validate(self) -> None:
        identity = f"{self.domain}/{self.model_family}/{self.arm}/seed-{self.seed}"
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown preconfirmation domain: {self.domain!r}")
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(
                f"unknown preconfirmation model family: {self.model_family!r}"
            )
        if self.arm not in (CONTROL_ARM, *LEARNED_ARMS):
            raise ValueError(f"unknown preconfirmation arm: {self.arm!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("preconfirmation seed must be an integer")
        if int(self.seed) not in PILOT_SEEDS:
            raise ValueError(f"unexpected preconfirmation seed: {self.seed!r}")
        if self.arm == CONTROL_ARM:
            if self.weight is not None:
                raise ValueError(f"none control weight must be null for {identity}")
        else:
            weight = _finite_nonnegative(self.weight, field=f"weight for {identity}")
            if weight not in POSITIVE_WEIGHTS:
                raise ValueError(
                    f"weight for {identity} must be one of {POSITIVE_WEIGHTS}"
                )
        for field in ("safety", "reconstruction", "rollout", "neighborhood_mass"):
            _finite_nonnegative(getattr(self, field), field=f"{field} for {identity}")
        _fraction(
            self.eligible_center_coverage,
            field=f"eligible_center_coverage for {identity}",
        )
        _nonnegative_integer(
            self.empty_trajectory_count,
            field=f"empty_trajectory_count for {identity}",
        )
        if self.run_complete is not True:
            raise ValueError(f"run is not complete for {identity}")
        if self.coverage_complete is not True:
            raise ValueError(f"metric coverage is incomplete for {identity}")

        profile_fields = (
            self.profile_normalized_p95_error,
            self.profile_teacher_count,
            self.profile_teacher_failure_count,
            self.profile_label_manifest_sha256,
        )
        if self.arm == PROFILE_ARM:
            _finite_nonnegative(
                self.profile_normalized_p95_error,
                field=f"profile_normalized_p95_error for {identity}",
            )
            _nonnegative_integer(
                self.profile_teacher_count,
                field=f"profile_teacher_count for {identity}",
            )
            _nonnegative_integer(
                self.profile_teacher_failure_count,
                field=f"profile_teacher_failure_count for {identity}",
            )
            if (
                not isinstance(self.profile_label_manifest_sha256, str)
                or len(self.profile_label_manifest_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.profile_label_manifest_sha256
                )
            ):
                raise ValueError(
                    f"profile_label_manifest_sha256 for {identity} must be lowercase SHA-256"
                )
        elif any(value is not None for value in profile_fields):
            raise ValueError(f"profile-only fields must be null for {identity}")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class CandidateWeightEvaluation:
    weight: float
    seed_count: int
    mean_safety: float
    reconstruction_ratio: float | None
    rollout_ratio: float | None
    maximum_utility_ratio: float | None
    minimum_eligible_center_coverage: float
    maximum_coverage_loss_vs_none: float
    maximum_neighborhood_mass_mismatch: float | None
    all_seed_safety_improvements: bool
    maximum_profile_normalized_p95_error: float | None
    minimum_profile_teacher_count: int | None
    total_profile_teacher_failures: int | None
    eligible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StratumWeightSelection:
    domain: str
    model_family: str
    arm: str
    candidates: tuple[CandidateWeightEvaluation, ...]
    selected_weight: float | None
    selected_mean_safety: float | None
    selected_maximum_utility_ratio: float | None
    passed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class ProfileLabelManifestCell:
    domain: str
    model_family: str
    seed: int
    sha256: str


@dataclass(frozen=True)
class PreconfirmationSelectionResult:
    schema_version: int
    protocol: str
    input_observation_count: int
    expected_observation_count: int
    selections: tuple[StratumWeightSelection, ...]
    profile_label_manifests: tuple[ProfileLabelManifestCell, ...]
    required_selection_count: int
    passed_selection_count: int
    ready_for_confirmation: bool

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _expected_keys() -> set[tuple[str, str, str, int, float | None]]:
    controls = {
        (domain, family, CONTROL_ARM, seed, None)
        for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS)
    }
    learned = {
        (domain, family, arm, seed, weight)
        for domain, family, arm, seed, weight in product(
            DOMAINS,
            MODEL_FAMILIES,
            LEARNED_ARMS,
            PILOT_SEEDS,
            POSITIVE_WEIGHTS,
        )
    }
    return controls | learned


def _validated_observations(
    observations: Iterable[PreconfirmationObservation],
) -> dict[tuple[str, str, str, int, float | None], PreconfirmationObservation]:
    lookup: dict[
        tuple[str, str, str, int, float | None], PreconfirmationObservation
    ] = {}
    for observation in tuple(observations):
        if not isinstance(observation, PreconfirmationObservation):
            raise ValueError(
                "observations must contain PreconfirmationObservation instances"
            )
        observation.validate()
        key = (
            observation.domain,
            observation.model_family,
            observation.arm,
            int(observation.seed),
            None if observation.weight is None else float(observation.weight),
        )
        if key in lookup:
            raise ValueError(f"duplicate preconfirmation observation: {key!r}")
        lookup[key] = observation

    expected = _expected_keys()
    observed = set(lookup)
    missing = sorted(expected - observed, key=repr)
    unexpected = sorted(observed - expected, key=repr)
    if missing or unexpected:
        fragments: list[str] = []
        if missing:
            fragments.append(f"missing {len(missing)} cells; first={missing[0]!r}")
        if unexpected:
            fragments.append(
                f"unexpected {len(unexpected)} cells; first={unexpected[0]!r}"
            )
        raise ValueError("incomplete preconfirmation factorial: " + "; ".join(fragments))
    if len(lookup) != EXPECTED_OBSERVATIONS:
        raise AssertionError("preconfirmation observation-count invariant drifted")
    return lookup


def _profile_label_manifests(
    lookup: dict[
        tuple[str, str, str, int, float | None], PreconfirmationObservation
    ],
) -> tuple[ProfileLabelManifestCell, ...]:
    cells: list[ProfileLabelManifestCell] = []
    for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS):
        hashes = {
            lookup[(domain, family, PROFILE_ARM, seed, weight)].profile_label_manifest_sha256
            for weight in POSITIVE_WEIGHTS
        }
        if len(hashes) != 1:
            raise ValueError(
                "predicted-profile weights must share one label manifest in "
                f"{domain}/{family}/seed-{seed}"
            )
        sha256 = hashes.pop()
        if not isinstance(sha256, str):
            raise AssertionError("validated profile manifest SHA unexpectedly missing")
        cells.append(
            ProfileLabelManifestCell(
                domain=domain,
                model_family=family,
                seed=seed,
                sha256=sha256,
            )
        )
    return tuple(cells)


def _ratio(numerators: tuple[float, ...], denominators: tuple[float, ...]) -> float | None:
    numerator = math.fsum(numerators)
    denominator = math.fsum(denominators)
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def _relative_mismatch(value: float, reference: float) -> float | None:
    if reference == 0.0:
        # The frozen matched-radius protocol treats a zero control mass as an
        # undefined relative comparison, including the nominal 0/0 case.  It
        # must therefore fail closed instead of being interpreted as a perfect
        # mass match by the direct in-memory selector.
        return None
    mismatch = abs(value - reference) / reference
    return mismatch if math.isfinite(mismatch) else None


def _evaluate_candidate(
    rows: tuple[PreconfirmationObservation, ...],
    controls: tuple[PreconfirmationObservation, ...],
) -> CandidateWeightEvaluation:
    if len(rows) != len(controls) or len(rows) != len(PILOT_SEEDS):
        raise AssertionError("candidate evaluation requires exactly three paired seeds")
    reconstruction_ratio = _ratio(
        tuple(row.reconstruction for row in rows),
        tuple(row.reconstruction for row in controls),
    )
    rollout_ratio = _ratio(
        tuple(row.rollout for row in rows),
        tuple(row.rollout for row in controls),
    )
    maximum_utility_ratio = (
        max(reconstruction_ratio, rollout_ratio)
        if reconstruction_ratio is not None and rollout_ratio is not None
        else None
    )
    coverage_losses = tuple(
        control.eligible_center_coverage - row.eligible_center_coverage
        for row, control in zip(rows, controls, strict=True)
    )
    mismatches = tuple(
        _relative_mismatch(row.neighborhood_mass, control.neighborhood_mass)
        for row, control in zip(rows, controls, strict=True)
    )
    maximum_mismatch = (
        max(float(value) for value in mismatches if value is not None)
        if all(value is not None for value in mismatches)
        else None
    )
    all_seed_safety_improvements = all(
        row.safety < control.safety
        for row, control in zip(rows, controls, strict=True)
    )

    profile_rows = rows if rows[0].arm == PROFILE_ARM else ()
    maximum_profile_error = (
        max(float(row.profile_normalized_p95_error) for row in profile_rows)
        if profile_rows
        else None
    )
    minimum_teacher_count = (
        min(int(row.profile_teacher_count) for row in profile_rows)
        if profile_rows
        else None
    )
    total_teacher_failures = (
        sum(int(row.profile_teacher_failure_count) for row in profile_rows)
        if profile_rows
        else None
    )

    reasons: list[str] = []
    if reconstruction_ratio is None:
        reasons.append("inadmissible_reconstruction_zero_denominator")
    elif reconstruction_ratio > MAX_UTILITY_RATIO:
        reasons.append("reconstruction_ratio_above_1.05")
    if rollout_ratio is None:
        reasons.append("inadmissible_rollout_zero_denominator")
    elif rollout_ratio > MAX_UTILITY_RATIO:
        reasons.append("rollout_ratio_above_1.05")
    if min(row.eligible_center_coverage for row in rows) < MINIMUM_COVERAGE:
        reasons.append("eligible_center_coverage_below_0.80")
    if max(coverage_losses) > MAX_COVERAGE_LOSS:
        reasons.append("coverage_loss_vs_none_above_0.05")
    if any(row.empty_trajectory_count > 0 for row in rows):
        reasons.append("empty_trajectory")
    if any(control.empty_trajectory_count > 0 for control in controls):
        reasons.append("paired_none_empty_trajectory")
    if maximum_mismatch is None:
        reasons.append("inadmissible_neighborhood_mass_zero_denominator")
    elif maximum_mismatch > MAX_NEIGHBORHOOD_MASS_MISMATCH:
        reasons.append("neighborhood_mass_mismatch_above_0.05")
    if not all_seed_safety_improvements:
        reasons.append("matched_mass_safety_improvement_not_strict_in_every_seed")
    if profile_rows:
        if maximum_profile_error > MAX_PROFILE_NORMALIZED_P95_ERROR:
            reasons.append("profile_normalized_p95_error_above_0.10")
        if minimum_teacher_count != REQUIRED_PROFILE_TEACHERS:
            reasons.append("profile_teacher_count_not_five")
        if total_teacher_failures != 0:
            reasons.append("profile_teacher_failure")

    return CandidateWeightEvaluation(
        weight=float(rows[0].weight),
        seed_count=len(rows),
        mean_safety=math.fsum(row.safety for row in rows) / len(rows),
        reconstruction_ratio=reconstruction_ratio,
        rollout_ratio=rollout_ratio,
        maximum_utility_ratio=maximum_utility_ratio,
        minimum_eligible_center_coverage=min(
            row.eligible_center_coverage for row in rows
        ),
        maximum_coverage_loss_vs_none=max(coverage_losses),
        maximum_neighborhood_mass_mismatch=maximum_mismatch,
        all_seed_safety_improvements=all_seed_safety_improvements,
        maximum_profile_normalized_p95_error=maximum_profile_error,
        minimum_profile_teacher_count=minimum_teacher_count,
        total_profile_teacher_failures=total_teacher_failures,
        eligible=not reasons,
        failure_reasons=tuple(reasons),
    )


def run_preconfirmation_selection(
    observations: Iterable[PreconfirmationObservation],
) -> PreconfirmationSelectionResult:
    """Apply every frozen gate and select one weight in each required stratum."""

    lookup = _validated_observations(observations)
    profile_label_manifests = _profile_label_manifests(lookup)
    selections: list[StratumWeightSelection] = []
    for domain, family, arm in product(DOMAINS, MODEL_FAMILIES, LEARNED_ARMS):
        controls = tuple(
            lookup[(domain, family, CONTROL_ARM, seed, None)] for seed in PILOT_SEEDS
        )
        candidates = tuple(
            _evaluate_candidate(
                tuple(
                    lookup[(domain, family, arm, seed, weight)]
                    for seed in PILOT_SEEDS
                ),
                controls,
            )
            for weight in POSITIVE_WEIGHTS
        )
        eligible = tuple(candidate for candidate in candidates if candidate.eligible)
        selected = (
            min(
                eligible,
                key=lambda candidate: (
                    candidate.mean_safety,
                    float(candidate.maximum_utility_ratio),
                    candidate.weight,
                ),
            )
            if eligible
            else None
        )
        selections.append(
            StratumWeightSelection(
                domain=domain,
                model_family=family,
                arm=arm,
                candidates=candidates,
                selected_weight=selected.weight if selected is not None else None,
                selected_mean_safety=(
                    selected.mean_safety if selected is not None else None
                ),
                selected_maximum_utility_ratio=(
                    selected.maximum_utility_ratio if selected is not None else None
                ),
                passed=selected is not None,
                failure_reason=(
                    None
                    if selected is not None
                    else "no_positive_weight_passed_every_frozen_gate"
                ),
            )
        )

    result = PreconfirmationSelectionResult(
        schema_version=1,
        protocol="e2_preconfirmation_weight_freeze_v1",
        input_observation_count=len(lookup),
        expected_observation_count=EXPECTED_OBSERVATIONS,
        selections=tuple(selections),
        profile_label_manifests=profile_label_manifests,
        required_selection_count=len(DOMAINS) * len(MODEL_FAMILIES) * len(LEARNED_ARMS),
        passed_selection_count=sum(selection.passed for selection in selections),
        ready_for_confirmation=all(selection.passed for selection in selections),
    )
    if len(result.selections) != result.required_selection_count:
        raise AssertionError("preconfirmation selection-count invariant drifted")
    return result
