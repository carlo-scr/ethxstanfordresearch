"""Frozen confirmatory inference for the E2 safety/utility frontier.

The engine in this module implements the exact design declared in
``configs/e2_frontier/confirmatory_core.toml``.  Its independent unit is one paired training
seed after equal averaging of the AE and beta-VAE within a domain.  It deliberately rejects
partial factorials, failed runs, and incomplete metric coverage instead of silently reducing the
confirmatory family.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Integral, Real
from typing import Any, Literal, cast


DOMAINS = (
    "controlled_cart_video",
    "controlled_pendulum_video",
    "controlled_dubins_navigation_pixels",
)
MODEL_FAMILIES = ("ae", "beta_vae")
PROPOSED_ARM = "nonprivileged_predicted_action_profile"
COMPARATORS = (
    "none",
    "h_prediction",
    "fcsrl_feasibility_loss_adaptation",
    "append_true_h_none_view",
)
ENDPOINTS = ("safety", "reconstruction", "rollout")
PAIRED_SEEDS = tuple(range(100, 108))
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_822
FAMILY_ALPHA = 0.05
ELEMENTARY_BOUND_COUNT = 36
UCB_PROBABILITY = 1.0 - FAMILY_ALPHA / ELEMENTARY_BOUND_COUNT
SAFETY_POINT_MEAN_MAX = -0.10
SAFETY_UCB_MAX = 0.0
UTILITY_UCB_MAX = 0.05

Endpoint = Literal["safety", "reconstruction", "rollout"]
DifferenceStatus = Literal[
    "finite",
    "tied_zero_denominator",
    "positive_over_zero_denominator",
    "relative_degradation_overflow",
]


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return numeric


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


@dataclass(frozen=True)
class ConfirmatoryObservation:
    """One complete final-test metric row for a frozen arm checkpoint or derived view."""

    domain: str
    model_family: str
    arm: str
    seed: int
    safety: float
    reconstruction: float
    rollout: float
    run_complete: bool
    coverage_complete: bool

    def validate(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown confirmatory domain: {self.domain!r}")
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"unknown confirmatory model family: {self.model_family!r}")
        if self.arm not in (PROPOSED_ARM, *COMPARATORS):
            raise ValueError(f"unknown confirmatory arm or derived view: {self.arm!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("confirmatory seed must be an integer")
        if int(self.seed) not in PAIRED_SEEDS:
            raise ValueError(f"unexpected confirmatory seed: {self.seed!r}")
        for endpoint in ENDPOINTS:
            _finite_nonnegative(
                getattr(self, endpoint),
                field=(
                    f"{endpoint} for {self.domain}/{self.model_family}/"
                    f"{self.arm}/seed-{self.seed}"
                ),
            )
        if self.run_complete is not True:
            raise ValueError(
                f"run is not complete for {self.domain}/{self.model_family}/"
                f"{self.arm}/seed-{self.seed}"
            )
        if self.coverage_complete is not True:
            raise ValueError(
                f"metric coverage is incomplete for {self.domain}/{self.model_family}/"
                f"{self.arm}/seed-{self.seed}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return _jsonable(self)


@dataclass(frozen=True)
class ConfirmatoryInferenceSpec:
    """Machine-readable copy of the frozen inferential contract."""

    domains: tuple[str, ...] = DOMAINS
    model_families: tuple[str, ...] = MODEL_FAMILIES
    proposed_arm: str = PROPOSED_ARM
    comparators: tuple[str, ...] = COMPARATORS
    endpoints: tuple[str, ...] = ENDPOINTS
    paired_seeds: tuple[int, ...] = PAIRED_SEEDS
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    family_alpha: float = FAMILY_ALPHA
    elementary_bound_count: int = ELEMENTARY_BOUND_COUNT
    ucb_probability: float = UCB_PROBABILITY
    family_aggregation: str = "equal AE/beta-VAE mean within seed and domain"
    bootstrap_unit: str = "paired training seed"
    bootstrap_method: str = "one-sided nonstudentized percentile UCB"
    multiplicity_method: str = "Bonferroni over all 36 elementary bounds"
    safety_point_mean_max: float = SAFETY_POINT_MEAN_MAX
    safety_ucb_max: float = SAFETY_UCB_MAX
    utility_ucb_max: float = UTILITY_UCB_MAX

    def validate(self) -> None:
        if len(self.domains) * len(self.comparators) * len(self.endpoints) != 36:
            raise ValueError("confirmatory spec must define exactly 36 elementary bounds")
        if self.elementary_bound_count != 36:
            raise ValueError("elementary_bound_count must be exactly 36")
        if self.model_families != ("ae", "beta_vae"):
            raise ValueError("model families must be exactly AE and beta-VAE")
        if len(self.paired_seeds) != 8 or len(set(self.paired_seeds)) != 8:
            raise ValueError("confirmatory spec must contain exactly eight unique paired seeds")
        expected_probability = 1.0 - self.family_alpha / self.elementary_bound_count
        if not math.isclose(
            self.ucb_probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("UCB probability must equal 1 - family_alpha / 36")
        if self.bootstrap_resamples != 100_000 or self.bootstrap_seed != 20_260_822:
            raise ValueError("bootstrap must use 100000 resamples and seed 20260822")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


FROZEN_CONFIRMATORY_SPEC = ConfirmatoryInferenceSpec()


@dataclass(frozen=True)
class FamilyDifference:
    """One family-specific paired difference before the equal-family mean."""

    model_family: str
    value: float | None
    status: DifferenceStatus


@dataclass(frozen=True)
class PairedSeedDifference:
    """The two family differences and their equally weighted seed-level aggregate."""

    seed: int
    family_differences: tuple[FamilyDifference, FamilyDifference]
    equal_family_mean: float | None
    valid: bool


@dataclass(frozen=True)
class ElementaryBoundResult:
    """One of the 36 prospectively declared elementary confidence bounds."""

    hypothesis_id: str
    domain: str
    comparator: str
    endpoint: Endpoint
    difference_definition: str
    seed_differences: tuple[PairedSeedDifference, ...]
    point_mean: float | None
    upper_confidence_bound: float | None
    ucb_probability: float
    bootstrap_resamples: int
    bootstrap_seed: int
    point_gate_passed: bool | None
    ucb_gate_passed: bool
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DomainGateResult:
    """Joint safety/utility decision for one required domain."""

    domain: str
    safety_bounds_passed: bool
    utility_bounds_passed: bool
    all_bounds_passed: bool


@dataclass(frozen=True)
class ConfirmatoryInferenceResult:
    """Complete result artifact for all 36 elementary bounds."""

    schema_version: int
    spec: ConfirmatoryInferenceSpec
    input_observation_count: int
    expected_observation_count: int
    bounds: tuple[ElementaryBoundResult, ...]
    domain_gates: tuple[DomainGateResult, ...]
    safety_bound_count: int
    utility_bound_count: int
    passed_bound_count: int
    all_elementary_bounds_passed: bool
    confirmatory_success: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible nested dictionary with no NaN or infinity sentinels."""

        return _jsonable(self)


def _expected_keys(spec: ConfirmatoryInferenceSpec) -> set[tuple[str, str, str, int]]:
    return {
        (domain, family, arm, seed)
        for domain, family, arm, seed in product(
            spec.domains,
            spec.model_families,
            (spec.proposed_arm, *spec.comparators),
            spec.paired_seeds,
        )
    }


def _validated_observations(
    observations: Iterable[ConfirmatoryObservation],
    spec: ConfirmatoryInferenceSpec,
) -> dict[tuple[str, str, str, int], ConfirmatoryObservation]:
    materialized = tuple(observations)
    lookup: dict[tuple[str, str, str, int], ConfirmatoryObservation] = {}
    for observation in materialized:
        if not isinstance(observation, ConfirmatoryObservation):
            raise ValueError("observations must contain ConfirmatoryObservation instances")
        observation.validate()
        key = (
            observation.domain,
            observation.model_family,
            observation.arm,
            int(observation.seed),
        )
        if key in lookup:
            raise ValueError(f"duplicate confirmatory observation: {key!r}")
        lookup[key] = observation

    expected = _expected_keys(spec)
    observed = set(lookup)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        fragments: list[str] = []
        if missing:
            fragments.append(f"missing {len(missing)} cells; first={missing[0]!r}")
        if unexpected:
            fragments.append(f"unexpected {len(unexpected)} cells; first={unexpected[0]!r}")
        raise ValueError("incomplete confirmatory factorial: " + "; ".join(fragments))
    return lookup


def _relative_degradation(
    proposed: float, comparator: float
) -> tuple[float | None, DifferenceStatus]:
    if comparator == 0.0:
        if proposed == 0.0:
            return 0.0, "tied_zero_denominator"
        return None, "positive_over_zero_denominator"
    value = (proposed - comparator) / comparator
    if not math.isfinite(value):
        return None, "relative_degradation_overflow"
    return value, "finite"


@lru_cache(maxsize=8)
def _bootstrap_count_distribution(
    sample_size: int, resamples: int, seed: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Compress a fixed bootstrap index stream into multinomial count frequencies."""

    generator = random.Random(seed)
    frequencies: Counter[tuple[int, ...]] = Counter()
    for _ in range(resamples):
        counts = [0] * sample_size
        for _ in range(sample_size):
            counts[generator.randrange(sample_size)] += 1
        frequencies[tuple(counts)] += 1
    return tuple(frequencies.items())


def _weighted_order_statistic(
    ordered: list[tuple[float, int]], index: int
) -> float:
    cumulative = 0
    for value, frequency in ordered:
        cumulative += frequency
        if index < cumulative:
            return value
    raise AssertionError("weighted order-statistic index exceeds bootstrap sample")


def _percentile_upper_bound(
    values: tuple[float, ...],
    *,
    probability: float,
    resamples: int,
    seed: int,
) -> float:
    """Return the Type-7 percentile UCB from paired-seed bootstrap means."""

    distribution = _bootstrap_count_distribution(len(values), resamples, seed)
    weighted_statistics: Counter[float] = Counter()
    sample_size = len(values)
    for counts, frequency in distribution:
        statistic = math.fsum(
            value * (count / sample_size)
            for value, count in zip(values, counts, strict=True)
        )
        if not math.isfinite(statistic):
            raise ValueError("bootstrap mean is not finite")
        weighted_statistics[statistic] += frequency
    ordered = sorted(weighted_statistics.items())
    position = (resamples - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = _weighted_order_statistic(ordered, lower_index)
    if lower_index == upper_index:
        return lower
    upper = _weighted_order_statistic(ordered, upper_index)
    weight = position - lower_index
    result = lower * (1.0 - weight) + upper * weight
    if not math.isfinite(result):
        raise ValueError("bootstrap upper confidence bound is not finite")
    return result


def _family_difference(
    proposed: ConfirmatoryObservation,
    comparator: ConfirmatoryObservation,
    endpoint: Endpoint,
) -> FamilyDifference:
    proposed_value = float(getattr(proposed, endpoint))
    comparator_value = float(getattr(comparator, endpoint))
    if endpoint == "safety":
        return FamilyDifference(
            model_family=proposed.model_family,
            value=proposed_value - comparator_value,
            status="finite",
        )
    value, status = _relative_degradation(proposed_value, comparator_value)
    return FamilyDifference(
        model_family=proposed.model_family,
        value=value,
        status=status,
    )


def _elementary_bound(
    lookup: dict[tuple[str, str, str, int], ConfirmatoryObservation],
    *,
    domain: str,
    comparator: str,
    endpoint: Endpoint,
    spec: ConfirmatoryInferenceSpec,
) -> ElementaryBoundResult:
    seeds: list[PairedSeedDifference] = []
    invalid_reasons: list[str] = []
    numeric_seed_differences: list[float] = []
    for seed in spec.paired_seeds:
        family_results = tuple(
            _family_difference(
                lookup[(domain, family, spec.proposed_arm, seed)],
                lookup[(domain, family, comparator, seed)],
                endpoint,
            )
            for family in spec.model_families
        )
        if len(family_results) != 2:
            raise AssertionError("frozen inference requires exactly two model families")
        invalid = [result for result in family_results if result.value is None]
        if invalid:
            averaged = None
            invalid_reasons.extend(
                f"seed-{seed}/{result.model_family}: {result.status}"
                for result in invalid
            )
        else:
            averaged = statistics.fmean(
                result.value for result in family_results if result.value is not None
            )
            if not math.isfinite(averaged):
                raise ValueError(
                    f"equal-family mean is not finite for {domain}/{comparator}/{endpoint}/"
                    f"seed-{seed}"
                )
            numeric_seed_differences.append(averaged)
        seeds.append(
            PairedSeedDifference(
                seed=seed,
                family_differences=(family_results[0], family_results[1]),
                equal_family_mean=averaged,
                valid=averaged is not None,
            )
        )

    difference_definition = (
        f"{spec.proposed_arm} - comparator"
        if endpoint == "safety"
        else f"({spec.proposed_arm} - comparator) / comparator"
    )
    point_mean: float | None
    upper: float | None
    point_gate: bool | None
    ucb_gate: bool
    if invalid_reasons:
        point_mean = None
        upper = None
        point_gate = False if endpoint == "safety" else None
        ucb_gate = False
    else:
        if len(numeric_seed_differences) != len(spec.paired_seeds):
            raise AssertionError("validated bound lost a paired seed")
        differences = tuple(numeric_seed_differences)
        point_mean = statistics.fmean(differences)
        upper = _percentile_upper_bound(
            differences,
            probability=spec.ucb_probability,
            resamples=spec.bootstrap_resamples,
            seed=spec.bootstrap_seed,
        )
        if endpoint == "safety":
            point_gate = point_mean <= spec.safety_point_mean_max
            ucb_gate = upper < spec.safety_ucb_max
        else:
            point_gate = None
            ucb_gate = upper <= spec.utility_ucb_max

    passed = ucb_gate and (point_gate is not False)
    reasons = list(invalid_reasons)
    if point_gate is False and not invalid_reasons:
        reasons.append(
            f"point mean exceeds safety threshold {spec.safety_point_mean_max}"
        )
    if not ucb_gate and not invalid_reasons:
        threshold = spec.safety_ucb_max if endpoint == "safety" else spec.utility_ucb_max
        operator = "<" if endpoint == "safety" else "<="
        reasons.append(f"upper confidence bound does not satisfy {operator} {threshold}")

    return ElementaryBoundResult(
        hypothesis_id=f"{domain}::{comparator}::{endpoint}",
        domain=domain,
        comparator=comparator,
        endpoint=endpoint,
        difference_definition=difference_definition,
        seed_differences=tuple(seeds),
        point_mean=point_mean,
        upper_confidence_bound=upper,
        ucb_probability=spec.ucb_probability,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_seed=spec.bootstrap_seed,
        point_gate_passed=point_gate,
        ucb_gate_passed=ucb_gate,
        passed=passed,
        failure_reasons=tuple(reasons),
    )


def run_confirmatory_inference(
    observations: Iterable[ConfirmatoryObservation],
) -> ConfirmatoryInferenceResult:
    """Run the exact frozen 36-bound E2 confirmatory analysis.

    Inputs must cover every domain/family/arm/seed cell for the proposed arm and all four
    comparators.  The append-true-h comparator is a derived view, but its 48 cells are still
    mandatory.  Utility zero denominators are handled by the frozen rule: zero over zero produces
    zero relative degradation, while a positive proposed error over a zero comparator error makes
    that elementary bound fail without emitting JSON-incompatible infinity.
    """

    spec = FROZEN_CONFIRMATORY_SPEC
    spec.validate()
    lookup = _validated_observations(observations, spec)
    bounds = tuple(
        _elementary_bound(
            lookup,
            domain=domain,
            comparator=comparator,
            endpoint=cast(Endpoint, endpoint),
            spec=spec,
        )
        for domain, comparator, endpoint in product(
            spec.domains, spec.comparators, spec.endpoints
        )
    )
    if len(bounds) != spec.elementary_bound_count:
        raise AssertionError("confirmatory inference did not produce exactly 36 bounds")

    domain_gates = tuple(
        DomainGateResult(
            domain=domain,
            safety_bounds_passed=all(
                bound.passed
                for bound in bounds
                if bound.domain == domain and bound.endpoint == "safety"
            ),
            utility_bounds_passed=all(
                bound.passed
                for bound in bounds
                if bound.domain == domain and bound.endpoint != "safety"
            ),
            all_bounds_passed=all(bound.passed for bound in bounds if bound.domain == domain),
        )
        for domain in spec.domains
    )
    all_passed = all(bound.passed for bound in bounds)
    return ConfirmatoryInferenceResult(
        schema_version=1,
        spec=spec,
        input_observation_count=len(lookup),
        expected_observation_count=len(_expected_keys(spec)),
        bounds=bounds,
        domain_gates=domain_gates,
        safety_bound_count=sum(bound.endpoint == "safety" for bound in bounds),
        utility_bound_count=sum(bound.endpoint != "safety" for bound in bounds),
        passed_bound_count=sum(bound.passed for bound in bounds),
        all_elementary_bounds_passed=all_passed,
        confirmatory_success=all_passed and all(gate.all_bounds_passed for gate in domain_gates),
    )


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "COMPARATORS",
    "DOMAINS",
    "ELEMENTARY_BOUND_COUNT",
    "ENDPOINTS",
    "FAMILY_ALPHA",
    "FROZEN_CONFIRMATORY_SPEC",
    "MODEL_FAMILIES",
    "PAIRED_SEEDS",
    "PROPOSED_ARM",
    "UCB_PROBABILITY",
    "ConfirmatoryInferenceResult",
    "ConfirmatoryInferenceSpec",
    "ConfirmatoryObservation",
    "DomainGateResult",
    "ElementaryBoundResult",
    "FamilyDifference",
    "PairedSeedDifference",
    "run_confirmatory_inference",
]
