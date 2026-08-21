"""Small-sample analysis primitives with explicit experimental units.

This module has no third-party dependencies.  It is intentionally narrower than a general
statistics package: its purpose is to encode the confirmatory analysis rules in the LatentSafety
protocol and make common leakage and pseudo-replication errors harder to commit.

The routines do not manufacture independence.  A :class:`PairedBlock` must be one aggregate per
independent training seed or per predeclared environment/trajectory block.  If a quantity is
measured at every video frame, aggregate within the declared block first and resample blocks, not
frames.  Nested or strongly dependent block designs may require a hierarchical analysis beyond
this module.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Literal

StatisticName = Literal["mean", "median"]
SignFlipAlternative = Literal["two-sided", "candidate_less", "candidate_greater"]
EffectStatus = Literal[
    "defined",
    "zero_variance_zero_difference",
    "zero_variance_constant_difference",
]

BLOCK_INDEPENDENCE_CAVEAT = (
    "Each PairedBlock must represent one independent seed or one pre-aggregated, predeclared "
    "environment/trajectory block. Never enter correlated frames as separate blocks."
)

VALIDATION_SELECTION_CAVEAT = (
    "Frontier construction and budget selection may use validation metrics only. Final-test "
    "safety labels must not select configurations, checkpoints, budgets, or stopping times."
)


def _finite_real(value: object, *, field: str) -> float:
    """Return a finite float while rejecting booleans and non-real values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _nonempty_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class PairedBlock:
    """One matched baseline/candidate observation at the independent block level.

    ``candidate - baseline`` is the difference convention used throughout.  Thus, for metrics
    where smaller is better (such as defect or utility loss), a negative difference favors the
    candidate.

    The class cannot determine whether a caller has mislabeled frames as blocks.  The block ID
    should therefore name an auditable seed, trajectory, or environment block from the split
    manifest.
    """

    block_id: str
    baseline: float
    candidate: float

    def validate(self) -> None:
        _nonempty_identifier(self.block_id, field="block_id")
        _finite_real(self.baseline, field=f"baseline for block {self.block_id!r}")
        _finite_real(self.candidate, field=f"candidate for block {self.block_id!r}")


@dataclass(frozen=True)
class PairedBootstrapCI:
    """Percentile bootstrap interval for a paired block-level difference."""

    statistic: StatisticName
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    rng_seed: int
    n_blocks: int
    difference_convention: str = "candidate - baseline"
    resampling_unit: str = "paired seed/pre-aggregated block"
    method: str = "paired percentile bootstrap"


@dataclass(frozen=True)
class StandardizedPairedEffect:
    """Cohen's ``d_z`` based on the sample SD of paired differences.

    ``value`` is ``None`` when all paired differences are identical.  For an identically zero
    difference this means there is no observed effect; for an identical nonzero difference the
    standardized effect is unbounded/undefined because no empirical difference variance exists.
    Returning ``None`` prevents either case from being silently presented as a finite estimate.
    """

    value: float | None
    mean_difference: float
    sample_sd_difference: float
    n_blocks: int
    status: EffectStatus
    definition: str = "Cohen d_z = mean(candidate - baseline) / sample SD(candidate - baseline)"

    @property
    def is_defined(self) -> bool:
        return self.status == "defined"


@dataclass(frozen=True)
class ExactPairedSignFlipTest:
    """Exact randomization test over every sign assignment of paired differences.

    Validity requires joint sign-exchangeability under the sharp null. The caller must provide a
    study-specific justification; pairing alone does not establish that assumption.
    """

    estimate: float
    p_value: float
    alternative: SignFlipAlternative
    n_blocks: int
    permutation_count: int
    exchangeability_justification: str
    difference_convention: str = "candidate - baseline"
    method: str = "exact paired sign-flip randomization test"


def _validated_paired_differences(blocks: Iterable[PairedBlock]) -> tuple[float, ...]:
    materialized = tuple(blocks)
    if not materialized:
        raise ValueError("at least two paired seed/block aggregates are required; received none")
    if len(materialized) < 2:
        raise ValueError("at least two paired seed/block aggregates are required")

    seen_ids: set[str] = set()
    differences: list[float] = []
    for block in materialized:
        if not isinstance(block, PairedBlock):
            raise ValueError("blocks must contain PairedBlock instances")
        block.validate()
        if block.block_id in seen_ids:
            raise ValueError(f"duplicate block_id: {block.block_id!r}")
        seen_ids.add(block.block_id)
        baseline = _finite_real(block.baseline, field="baseline")
        candidate = _finite_real(block.candidate, field="candidate")
        difference = candidate - baseline
        if not math.isfinite(difference):
            raise ValueError(f"paired difference overflows for block {block.block_id!r}")
        differences.append(difference)
    return tuple(differences)


def _checked_summary(values: tuple[float, ...] | list[float], statistic: StatisticName) -> float:
    try:
        if statistic == "mean":
            result = statistics.fmean(values)
        else:
            result = float(statistics.median(values))
    except (OverflowError, statistics.StatisticsError) as error:
        raise ValueError("the requested statistic is not finite for these values") from error
    if not math.isfinite(result):
        raise ValueError("the requested statistic is not finite for these values")
    return result


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    """Type-7 linear sample quantile, including exact endpoints."""

    if probability <= 0.0:
        return sorted_values[0]
    if probability >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def paired_block_bootstrap_ci(
    blocks: Iterable[PairedBlock],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    statistic: StatisticName = "mean",
) -> PairedBootstrapCI:
    """Compute a deterministic paired percentile bootstrap confidence interval.

    Resampling is over whole matched blocks and preserves the baseline/candidate pairing.  The
    default deterministic seed makes analysis artifacts exactly reproducible; sensitivity to the
    Monte Carlo seed and interval method should still be checked for final claims.

    This is a percentile interval, not a guarantee of nominal coverage with very few seeds.  The
    experiment protocol's minimum seed count and a pre-registered precision analysis remain
    necessary.  Individual frames must never be supplied as if they were independent blocks.
    """

    confidence = _finite_real(confidence_level, field="confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be either 'mean' or 'median'")

    differences = _validated_paired_differences(blocks)
    estimate = _checked_summary(differences, statistic)
    generator = random.Random(int(seed))
    sample_size = len(differences)
    bootstrap_statistics: list[float] = []
    for _ in range(int(resamples)):
        resampled = [differences[generator.randrange(sample_size)] for _ in range(sample_size)]
        bootstrap_statistics.append(_checked_summary(resampled, statistic))
    bootstrap_statistics.sort()

    tail_probability = (1.0 - confidence) / 2.0
    lower = _linear_quantile(bootstrap_statistics, tail_probability)
    upper = _linear_quantile(bootstrap_statistics, 1.0 - tail_probability)
    return PairedBootstrapCI(
        statistic=statistic,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        resamples=int(resamples),
        rng_seed=int(seed),
        n_blocks=sample_size,
    )


def standardized_paired_effect(blocks: Iterable[PairedBlock]) -> StandardizedPairedEffect:
    """Return Cohen's ``d_z`` for independent paired seed/block aggregates.

    The effect uses the between-block sample standard deviation of the paired differences.  It is
    not appropriate to inflate ``n`` with frames from the same seed or trajectory.  No small-sample
    bias correction is applied, so the definition should be named explicitly in paper tables.
    """

    differences = _validated_paired_differences(blocks)
    mean_difference = _checked_summary(differences, "mean")
    try:
        sample_sd = float(statistics.stdev(differences))
    except (OverflowError, statistics.StatisticsError) as error:
        raise ValueError("sample SD is not finite for these paired differences") from error
    if not math.isfinite(sample_sd):
        raise ValueError("sample SD is not finite for these paired differences")

    if sample_sd == 0.0:
        status: EffectStatus
        if mean_difference == 0.0:
            status = "zero_variance_zero_difference"
        else:
            status = "zero_variance_constant_difference"
        return StandardizedPairedEffect(
            value=None,
            mean_difference=mean_difference,
            sample_sd_difference=0.0,
            n_blocks=len(differences),
            status=status,
        )

    value = mean_difference / sample_sd
    if not math.isfinite(value):
        raise ValueError("standardized effect is not finite for these paired differences")
    return StandardizedPairedEffect(
        value=value,
        mean_difference=mean_difference,
        sample_sd_difference=sample_sd,
        n_blocks=len(differences),
        status="defined",
    )


def exact_paired_sign_flip_test(
    blocks: Iterable[PairedBlock],
    *,
    alternative: SignFlipAlternative = "two-sided",
    exchangeability_justification: str,
) -> ExactPairedSignFlipTest:
    """Enumerate the exact paired sign-flip distribution of the mean difference.

    This is appropriate only when swapping the paired method labels is justified under a sharp
    null. It is intentionally capped at 20 blocks so a supposedly exact call cannot silently turn
    into an impractical computation. Use ``candidate_less`` for a preregistered smaller-is-better
    alternative and ``candidate_greater`` for a larger-is-better alternative.
    """

    if alternative not in {"two-sided", "candidate_less", "candidate_greater"}:
        raise ValueError(
            "alternative must be 'two-sided', 'candidate_less', or 'candidate_greater'"
        )
    justification = _nonempty_identifier(
        exchangeability_justification,
        field="exchangeability_justification",
    )
    differences = _validated_paired_differences(blocks)
    if len(differences) > 20:
        raise ValueError("exact sign-flip enumeration is limited to at most 20 blocks")
    observed = _checked_summary(differences, "mean")
    threshold = abs(observed) if alternative == "two-sided" else observed
    extreme = 0
    permutation_count = 1 << len(differences)
    for signs in product((-1.0, 1.0), repeat=len(differences)):
        permuted = _checked_summary(
            [sign * difference for sign, difference in zip(signs, differences, strict=True)],
            "mean",
        )
        if alternative == "two-sided":
            is_extreme = abs(permuted) >= threshold
        elif alternative == "candidate_less":
            is_extreme = permuted <= threshold
        else:
            is_extreme = permuted >= threshold
        extreme += int(is_extreme)

    return ExactPairedSignFlipTest(
        estimate=observed,
        p_value=extreme / permutation_count,
        alternative=alternative,
        n_blocks=len(differences),
        permutation_count=permutation_count,
        exchangeability_justification=justification,
    )


@dataclass(frozen=True)
class HypothesisPValue:
    """One named raw p-value in a predeclared confirmatory family."""

    hypothesis_id: str
    p_value: float

    def validate(self) -> None:
        _nonempty_identifier(self.hypothesis_id, field="hypothesis_id")
        numeric = _finite_real(self.p_value, field=f"p_value for {self.hypothesis_id!r}")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"p_value for {self.hypothesis_id!r} must lie in [0, 1]")


@dataclass(frozen=True)
class HolmAdjustedPValue:
    """One Holm step-down family-wise-error adjustment result."""

    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    rank: int
    step_down_threshold: float
    rejected: bool
    alpha: float
    family_size: int


def holm_adjust(
    hypotheses: Iterable[HypothesisPValue], *, alpha: float = 0.05
) -> tuple[HolmAdjustedPValue, ...]:
    """Apply Holm's step-down correction and return results in input order.

    Tied p-values are ranked by ``hypothesis_id`` to keep output deterministic.  Holm controls the
    family-wise error rate under arbitrary dependence when the supplied raw p-values are valid;
    the procedure cannot repair invalid tests, post-hoc family definitions, or selective reporting.
    """

    family = tuple(hypotheses)
    if not family:
        raise ValueError("at least one hypothesis is required")
    numeric_alpha = _finite_real(alpha, field="alpha")
    if not 0.0 < numeric_alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    identifiers: set[str] = set()
    indexed: list[tuple[int, str, float]] = []
    for input_index, hypothesis in enumerate(family):
        if not isinstance(hypothesis, HypothesisPValue):
            raise ValueError("hypotheses must contain HypothesisPValue instances")
        hypothesis.validate()
        identifier = _nonempty_identifier(hypothesis.hypothesis_id, field="hypothesis_id")
        if identifier in identifiers:
            raise ValueError(f"duplicate hypothesis_id: {identifier!r}")
        identifiers.add(identifier)
        p_value = _finite_real(hypothesis.p_value, field="p_value")
        indexed.append((input_index, identifier, p_value))

    ordered = sorted(indexed, key=lambda item: (item[2], item[1]))
    family_size = len(ordered)
    running_adjusted = 0.0
    rejection_open = True
    by_input_index: dict[int, HolmAdjustedPValue] = {}
    for zero_based_rank, (input_index, identifier, p_value) in enumerate(ordered):
        remaining = family_size - zero_based_rank
        threshold = numeric_alpha / remaining
        adjusted = min(1.0, max(running_adjusted, remaining * p_value))
        running_adjusted = adjusted
        rejected = rejection_open and p_value <= threshold
        if not rejected:
            rejection_open = False
        by_input_index[input_index] = HolmAdjustedPValue(
            hypothesis_id=identifier,
            raw_p_value=p_value,
            adjusted_p_value=adjusted,
            rank=zero_based_rank + 1,
            step_down_threshold=threshold,
            rejected=rejected,
            alpha=numeric_alpha,
            family_size=family_size,
        )
    return tuple(by_input_index[index] for index in range(family_size))


@dataclass(frozen=True)
class ValidationScore:
    """Validation-only safety/utility score for one frozen candidate.

    Both objectives are losses and therefore nonnegative with smaller values preferred.  The
    configuration ID should resolve to a frozen checkpoint and manifest.  Do not construct these
    records from final-test labels.
    """

    config_id: str
    safety_defect: float
    utility_loss: float

    def validate(self) -> None:
        _nonempty_identifier(self.config_id, field="config_id")
        safety = _finite_real(
            self.safety_defect, field=f"safety_defect for {self.config_id!r}"
        )
        utility = _finite_real(
            self.utility_loss, field=f"utility_loss for {self.config_id!r}"
        )
        if safety < 0.0:
            raise ValueError(f"safety_defect for {self.config_id!r} must be nonnegative")
        if utility < 0.0:
            raise ValueError(f"utility_loss for {self.config_id!r} must be nonnegative")


@dataclass(frozen=True)
class ValidationParetoFrontier:
    """Nondominated validation candidates for two objectives being minimized."""

    points: tuple[ValidationScore, ...]
    dominated_config_ids: tuple[str, ...]
    candidate_count: int
    selection_split: str = "validation"

    def validate(self) -> None:
        if not self.points:
            raise ValueError("validation Pareto frontier must contain at least one point")
        if self.selection_split != "validation":
            raise ValueError("selection_split must be exactly 'validation'")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, Integral)
            or self.candidate_count < 1
        ):
            raise ValueError("candidate_count must be a positive integer")

        point_ids: set[str] = set()
        for point in self.points:
            if not isinstance(point, ValidationScore):
                raise ValueError("frontier points must contain ValidationScore instances")
            point.validate()
            if point.config_id in point_ids:
                raise ValueError(f"duplicate frontier config_id: {point.config_id!r}")
            point_ids.add(point.config_id)

        dominated_ids: set[str] = set()
        for config_id in self.dominated_config_ids:
            identifier = _nonempty_identifier(config_id, field="dominated config_id")
            if identifier in dominated_ids:
                raise ValueError(f"duplicate dominated config_id: {identifier!r}")
            if identifier in point_ids:
                raise ValueError(f"config_id appears as both frontier and dominated: {identifier!r}")
            dominated_ids.add(identifier)
        if self.candidate_count != len(point_ids) + len(dominated_ids):
            raise ValueError("candidate_count does not match frontier and dominated IDs")


def _dominates(left: ValidationScore, right: ValidationScore) -> bool:
    no_worse = (
        left.safety_defect <= right.safety_defect
        and left.utility_loss <= right.utility_loss
    )
    strictly_better = (
        left.safety_defect < right.safety_defect
        or left.utility_loss < right.utility_loss
    )
    return no_worse and strictly_better


def validation_pareto_frontier(
    scores: Iterable[ValidationScore],
) -> ValidationParetoFrontier:
    """Return the validation-selected Pareto frontier for two minimized losses.

    A candidate is dominated when another candidate is no worse on both validation objectives and
    strictly better on at least one.  Exact metric ties are retained, since discarding one would
    require an undeclared tie-breaker.  Frontier points are sorted by safety defect, utility loss,
    and configuration ID for stable artifacts.

    This function cannot detect test leakage: callers must ensure the inputs were computed solely
    on the frozen validation split and must evaluate the chosen rule only once on final test data.
    """

    materialized = tuple(scores)
    if not materialized:
        raise ValueError("at least one validation score is required")
    identifiers: set[str] = set()
    normalized: list[ValidationScore] = []
    for score in materialized:
        if not isinstance(score, ValidationScore):
            raise ValueError("scores must contain ValidationScore instances")
        score.validate()
        if score.config_id in identifiers:
            raise ValueError(f"duplicate config_id: {score.config_id!r}")
        identifiers.add(score.config_id)
        normalized.append(
            ValidationScore(
                config_id=score.config_id,
                safety_defect=_finite_real(score.safety_defect, field="safety_defect"),
                utility_loss=_finite_real(score.utility_loss, field="utility_loss"),
            )
        )

    points = [
        candidate
        for candidate in normalized
        if not any(
            _dominates(other, candidate)
            for other in normalized
            if other.config_id != candidate.config_id
        )
    ]
    points.sort(key=lambda point: (point.safety_defect, point.utility_loss, point.config_id))
    frontier_ids = {point.config_id for point in points}
    dominated_ids = tuple(sorted(identifiers - frontier_ids))
    return ValidationParetoFrontier(
        points=tuple(points),
        dominated_config_ids=dominated_ids,
        candidate_count=len(normalized),
    )


def select_under_safety_budget(
    frontier: ValidationParetoFrontier, *, max_safety_defect: float
) -> ValidationScore:
    """Select minimum validation utility loss under a predeclared safety-defect budget.

    Ties are resolved by lower safety defect and then configuration ID.  The budget and tie-breaker
    must be declared before opening final-test labels; this routine must never be rerun to optimize
    a reported test result.
    """

    if not isinstance(frontier, ValidationParetoFrontier):
        raise ValueError("frontier must be a non-empty ValidationParetoFrontier")
    frontier.validate()
    budget = _finite_real(max_safety_defect, field="max_safety_defect")
    if budget < 0.0:
        raise ValueError("max_safety_defect must be nonnegative")
    eligible = [point for point in frontier.points if point.safety_defect <= budget]
    if not eligible:
        raise ValueError("no validation candidate satisfies max_safety_defect")
    return min(
        eligible,
        key=lambda point: (point.utility_loss, point.safety_defect, point.config_id),
    )
