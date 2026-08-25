"""Exact finite-state oracles for the static safety-sufficiency theory.

These routines are deliberately combinatorial.  They are useful for theorem unit tests and
synthetic controls where every state in the audited region is enumerated.  They are not estimators
of a continuous population from an i.i.d. sample.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

Vector = Sequence[float]
Code = tuple[float, ...]


@dataclass(frozen=True)
class FiniteStaticAudit:
    """Exact static-fiber summary on an enumerated state space."""

    state_count: int
    fiber_count: int
    mixed_fiber_count: int
    collided_safe_count: int
    exact_defect: float
    collision_margins: tuple[float, ...]
    maximal_sound_codes: tuple[Code, ...]


@dataclass(frozen=True)
class CompletenessCheck:
    """Existence and construction check for a sound, margin-complete certificate."""

    margin: float
    exists: bool
    required_codes: tuple[Code, ...]
    conflicting_required_codes: tuple[Code, ...]


@dataclass(frozen=True)
class DataProcessingCheck:
    """Finite verification of exact-defect monotonicity under a proposed coarsening."""

    is_deterministic_postprocessing: bool
    fine_defect: float
    coarse_defect: float
    monotone: bool


@dataclass(frozen=True)
class StochasticPairFrontierPoint:
    """One safe/unsafe pair under a finite stochastic encoder.

    ``mutually_singular`` is an exact statement about the supports of the validated binary64
    probability laws: zero means zero, with no numerical threshold.  The optimal separator accepts
    outputs where the safe mass strictly exceeds the unsafe mass; ties are rejected without
    changing the optimum.  Its total error is safe rejection plus unsafe acceptance, while Bayes
    error is half that sum under equal priors.  TV and all errors are first computed as exact
    rationals and then exposed as guarded floating summaries.  If only the ancillary half-error
    underflows, ``equal_prior_bayes_error`` and its residual are ``None`` while the explicit positivity and
    underflow flags preserve its semantics.  None of these are worst-state or pathwise
    probabilities.
    """

    safe_index: int
    unsafe_index: int
    safe_margin: float
    mutually_singular: bool
    shared_support_indices: tuple[int, ...]
    optimal_separator_output_indices: tuple[int, ...]
    total_variation: float
    overlap_mass: float
    safe_rejection_error: float
    unsafe_acceptance_error: float
    minimum_total_separator_error: float
    equal_prior_bayes_error: float | None
    equal_prior_bayes_error_positive: bool
    equal_prior_bayes_error_underflowed: bool
    tv_separator_identity_residual: float
    tv_bayes_identity_residual: float | None


@dataclass(frozen=True)
class FiniteStochasticAudit:
    """Exact-support audit for a finite state/output stochastic encoder.

    A measurable certificate is simply a subset of the finite output alphabet.  It is almost-surely
    sound when every unsafe row assigns it probability zero.  A safe state conflicts exactly when
    its positive-probability output support intersects the union of unsafe supports.  This support
    condition, rather than a rounded ``TV == 1`` comparison, defines ``exact_defect``.
    """

    state_count: int
    output_count: int
    safe_count: int
    unsafe_count: int
    conflicted_safe_count: int
    exact_defect: float
    conflict_margins: tuple[float, ...]
    maximal_sound_output_indices: tuple[int, ...]
    pair_frontier: tuple[StochasticPairFrontierPoint, ...]
    probability_tolerance: float


@dataclass(frozen=True)
class StochasticCompletenessCheck:
    """Exact finite existence check for an almost-sure stochastic certificate."""

    margin: float
    strict: bool
    exists: bool
    required_safe_indices: tuple[int, ...]
    conflicting_pairs: tuple[tuple[int, int], ...]
    maximal_sound_output_indices: tuple[int, ...]


@dataclass(frozen=True)
class StochasticDataProcessingCheck:
    """Support and TV checks after a declared finite Markov post-processing kernel."""

    state_count: int
    fine_output_count: int
    coarse_output_count: int
    safe_unsafe_pair_count: int
    fine_exact_defect: float
    coarse_exact_defect: float
    exact_conflicts_preserved: bool
    exact_defect_monotone: bool
    total_variation_contracted: bool
    maximum_total_variation_increase: float
    probability_tolerance: float
    monotone: bool


@dataclass(frozen=True)
class StochasticActionTradeoffCheck:
    """Exact finite instance of the TV randomized-action lower bound.

    The inequality booleans are decided exactly.  If only the ancillary induced
    action-TV display value underflows, it is ``None`` and the adjacent flags
    preserve positivity and underflow status.
    """

    first_state_index: int
    second_state_index: int
    first_safe_action_indices: tuple[int, ...]
    second_safe_action_indices: tuple[int, ...]
    code_total_variation: float
    induced_action_total_variation: float | None
    induced_action_total_variation_positive: bool
    induced_action_total_variation_underflowed: bool
    first_violation_probability: float
    second_violation_probability: float
    summed_violation_probability: float
    tv_lower_bound: float
    tradeoff_slack: float
    action_tv_contracted: bool
    tradeoff_holds: bool


def _finite_real(value: object, *, label: str, nonnegative: bool = False) -> float:
    """Return an exactly represented finite binary64 input or fail closed.

    Exact support, equality, and sign decisions are statements about the stored
    finite game.  A broader ``Real`` that changes during float conversion is
    rejected rather than allowed to fabricate a zero probability or margin.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be finite and exactly representable as binary64"
        ) from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if value != numeric:
        raise ValueError(f"{label} must be exactly representable as binary64")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return numeric


def _exact_float_fraction(value: float) -> Fraction:
    """Interpret one finite binary64 value as its exact rational number."""

    return Fraction.from_float(value)


def _fraction_to_float(
    value: Fraction,
    *,
    label: str,
    reject_nonzero_underflow: bool = False,
) -> float:
    """Round an exact finite rational summary to binary64 with an underflow guard."""

    numeric = float(value)
    if not math.isfinite(numeric):  # pragma: no cover - bounded probability invariant
        raise ValueError(f"{label} overflowed binary64")
    if reject_nonzero_underflow and value != 0 and numeric == 0.0:
        raise ValueError(f"{label} underflowed binary64")
    return numeric


def _canonical_binary64_probability_row(
    exact_row: Sequence[Fraction],
    *,
    label: str,
) -> tuple[float, ...]:
    """Project a positive rational row to an exact-sum binary64 row or fail closed.

    Non-pivot entries receive their nearest binary64 values.  Candidate pivots are
    tried from largest mass to smallest, with the pivot set to the exact residual
    needed for total mass one.  A projection is returned only when that residual
    is itself binary64-representable and the zero/positive support mask is
    unchanged.  The deterministic pivot rule makes any correction reproducible.
    """

    fractions = tuple(exact_row)
    if not fractions or any(value < 0 for value in fractions):
        raise ValueError(f"{label} must be a non-empty nonnegative probability row")
    support = tuple(index for index, value in enumerate(fractions) if value > 0)
    if not support:
        raise ValueError(f"{label} must contain positive probability mass")

    pivot_order = sorted(support, key=lambda index: (-fractions[index], index))
    for pivot in pivot_order:
        candidate = [0.0 for _ in fractions]
        valid = True
        for index, exact_value in enumerate(fractions):
            if index == pivot or exact_value == 0:
                continue
            numeric = _fraction_to_float(
                exact_value,
                label=f"{label}[{index}]",
                reject_nonzero_underflow=True,
            )
            if numeric <= 0.0 or numeric > 1.0:
                valid = False
                break
            candidate[index] = numeric
        if not valid:
            continue

        residual = Fraction(1) - sum(
            (_exact_float_fraction(value) for value in candidate),
            Fraction(),
        )
        if residual <= 0 or residual > 1:
            continue
        pivot_numeric = _fraction_to_float(
            residual,
            label=f"{label}[{pivot}] residual",
            reject_nonzero_underflow=True,
        )
        if _exact_float_fraction(pivot_numeric) != residual:
            continue
        candidate[pivot] = pivot_numeric
        projected = tuple(candidate)
        projected_support = tuple(
            index for index, value in enumerate(projected) if value > 0.0
        )
        projected_total = sum(
            (_exact_float_fraction(value) for value in projected),
            Fraction(),
        )
        if projected_support == support and projected_total == 1:
            return projected

    raise ValueError(
        f"{label} cannot be represented as an exact-sum binary64 probability "
        "row without changing positive support"
    )


def _exact_total_variation(
    left: Sequence[float | Fraction],
    right: Sequence[float | Fraction],
) -> Fraction:
    """Return exact total variation for two aligned finite probability rows."""

    if len(left) != len(right):  # pragma: no cover - internal shape invariant
        raise ValueError("probability rows must have the same dimension")

    def as_fraction(value: float | Fraction) -> Fraction:
        return value if isinstance(value, Fraction) else _exact_float_fraction(value)

    return sum(
        (
            abs(as_fraction(first) - as_fraction(second))
            for first, second in zip(left, right, strict=True)
        ),
        Fraction(),
    ) / 2


def _exact_postprocessed_probability_kernel(
    encoder: Sequence[Sequence[float]],
    transition: Sequence[Sequence[float]],
) -> tuple[tuple[Fraction, ...], ...]:
    """Compose two exact-sum binary64 kernels using exact rational arithmetic."""

    coarse_output_count = len(transition[0])
    result: list[tuple[Fraction, ...]] = []
    for state_row in encoder:
        exact_row = tuple(
            sum(
                (
                    _exact_float_fraction(state_probability)
                    * _exact_float_fraction(transition[fine_output][coarse_output])
                    for fine_output, state_probability in enumerate(state_row)
                ),
                Fraction(),
            )
            for coarse_output in range(coarse_output_count)
        )
        if sum(exact_row, Fraction()) != 1:  # pragma: no cover - stochastic invariant
            raise AssertionError("exact Markov composition lost probability mass")
        result.append(exact_row)
    return tuple(result)


def _project_exact_probability_kernel(
    kernel: Sequence[Sequence[Fraction]],
    *,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    """Return a support-preserving exact-sum binary64 projection of a rational kernel."""

    return tuple(
        _canonical_binary64_probability_row(
            row,
            label=f"{label} row {row_index}",
        )
        for row_index, row in enumerate(kernel)
    )


def _validated_probability_kernel(
    kernel: Sequence[Sequence[float]],
    *,
    label: str,
    probability_tolerance: float,
    expected_row_count: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Validate and canonically close finite rows to exact probability laws.

    Admissibility is decided from the exact rational values represented by the
    supplied binary64 entries, not from a rounded floating sum.  A row within
    ``probability_tolerance`` of unit mass is corrected at one deterministic
    pivot so that its binary64 entries sum to one as exact reals.  The correction
    must preserve every exact zero and positive support entry or validation fails.
    Set the tolerance to zero to forbid any correction.
    """

    tolerance = _finite_real(
        probability_tolerance,
        label="probability_tolerance",
        nonnegative=True,
    )
    if tolerance >= 1.0:
        raise ValueError("probability_tolerance must be less than one")
    rows = tuple(tuple(row) for row in kernel)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    if expected_row_count is not None and len(rows) != expected_row_count:
        raise ValueError(
            f"{label} must have exactly {expected_row_count} rows; found {len(rows)}"
        )
    output_count = len(rows[0])
    if output_count == 0 or any(len(row) != output_count for row in rows):
        raise ValueError(f"{label} rows must be non-empty and share one output dimension")

    validated: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        numeric_row = tuple(
            _finite_real(
                value,
                label=f"{label}[{row_index}][{column_index}]",
                nonnegative=True,
            )
            for column_index, value in enumerate(row)
        )
        if any(value > 1.0 for value in numeric_row):
            raise ValueError(f"{label} probabilities must not exceed one")
        exact_row = tuple(_exact_float_fraction(value) for value in numeric_row)
        exact_total = sum(exact_row, Fraction())
        if abs(exact_total - 1) > _exact_float_fraction(tolerance):
            raise ValueError(
                f"{label} row {row_index} must sum to one within "
                f"probability_tolerance={tolerance}; found {float(exact_total)}"
            )
        if exact_total == 1:
            validated.append(numeric_row)
        else:
            validated.append(
                _canonical_binary64_probability_row(
                    exact_row,
                    label=f"{label} row {row_index}",
                )
            )
    return tuple(validated)


def _validated_stochastic_problem(
    kernel: Sequence[Sequence[float]],
    margins: Sequence[float],
    *,
    probability_tolerance: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], float]:
    tolerance = _finite_real(
        probability_tolerance,
        label="probability_tolerance",
        nonnegative=True,
    )
    rows = _validated_probability_kernel(
        kernel,
        label="encoder_kernel",
        probability_tolerance=tolerance,
    )
    if len(rows) != len(margins):
        raise ValueError("encoder_kernel and margins must have the same number of states")
    numeric_margins = tuple(
        _finite_real(value, label=f"margins[{index}]")
        for index, value in enumerate(margins)
    )
    return rows, numeric_margins, tolerance


def _stochastic_pair_frontier_from_validated(
    rows: tuple[tuple[float, ...], ...],
    margins: tuple[float, ...],
) -> tuple[StochasticPairFrontierPoint, ...]:
    safe_indices = tuple(index for index, margin in enumerate(margins) if margin >= 0.0)
    unsafe_indices = tuple(index for index, margin in enumerate(margins) if margin < 0.0)
    points: list[StochasticPairFrontierPoint] = []
    for safe_index in safe_indices:
        for unsafe_index in unsafe_indices:
            safe_row = rows[safe_index]
            unsafe_row = rows[unsafe_index]
            shared = tuple(
                output
                for output, (safe_probability, unsafe_probability) in enumerate(
                    zip(safe_row, unsafe_row, strict=True)
                )
                if safe_probability > 0.0 and unsafe_probability > 0.0
            )
            optimal_separator = tuple(
                output
                for output, (safe_probability, unsafe_probability) in enumerate(
                    zip(safe_row, unsafe_row, strict=True)
                )
                if safe_probability > unsafe_probability
            )
            optimal_separator_set = frozenset(optimal_separator)
            safe_fractions = tuple(
                _exact_float_fraction(probability) for probability in safe_row
            )
            unsafe_fractions = tuple(
                _exact_float_fraction(probability) for probability in unsafe_row
            )
            safe_rejection_fraction = sum(
                (safe_fractions[output]
                for output in range(len(safe_row))
                if output not in optimal_separator_set),
                Fraction(),
            )
            unsafe_acceptance_fraction = sum(
                (unsafe_fractions[output] for output in optimal_separator),
                Fraction(),
            )
            total_separator_fraction = (
                safe_rejection_fraction + unsafe_acceptance_fraction
            )
            overlap_fraction = sum(
                (min(first, second) for first, second in zip(
                    safe_fractions, unsafe_fractions, strict=True
                )),
                Fraction(),
            )
            total_variation_fraction = _exact_total_variation(
                safe_fractions, unsafe_fractions
            )
            if (
                total_separator_fraction != overlap_fraction
                or total_separator_fraction != 1 - total_variation_fraction
            ):  # pragma: no cover - finite testing identity invariant
                raise AssertionError("exact finite TV/separator identity failed")

            total_variation = _fraction_to_float(
                total_variation_fraction,
                label="total variation",
            )
            overlap_mass = _fraction_to_float(
                overlap_fraction,
                label="overlap mass",
                reject_nonzero_underflow=True,
            )
            safe_rejection_error = _fraction_to_float(
                safe_rejection_fraction,
                label="safe rejection error",
                reject_nonzero_underflow=True,
            )
            unsafe_acceptance_error = _fraction_to_float(
                unsafe_acceptance_fraction,
                label="unsafe acceptance error",
                reject_nonzero_underflow=True,
            )
            total_separator_error = _fraction_to_float(
                total_separator_fraction,
                label="minimum total separator error",
                reject_nonzero_underflow=True,
            )
            bayes_fraction = total_separator_fraction / 2
            rounded_bayes_error = _fraction_to_float(
                bayes_fraction,
                label="equal-prior Bayes error",
            )
            bayes_underflowed = bayes_fraction > 0 and rounded_bayes_error == 0.0
            bayes_error = None if bayes_underflowed else rounded_bayes_error
            separator_residual_fraction = abs(
                _exact_float_fraction(total_separator_error)
                - (1 - _exact_float_fraction(total_variation))
            )
            bayes_residual_fraction = (
                None
                if bayes_error is None
                else abs(
                    _exact_float_fraction(bayes_error)
                    - (1 - _exact_float_fraction(total_variation)) / 2
                )
            )
            points.append(
                StochasticPairFrontierPoint(
                    safe_index=safe_index,
                    unsafe_index=unsafe_index,
                    safe_margin=margins[safe_index],
                    mutually_singular=not shared,
                    shared_support_indices=shared,
                    optimal_separator_output_indices=optimal_separator,
                    total_variation=total_variation,
                    overlap_mass=overlap_mass,
                    safe_rejection_error=safe_rejection_error,
                    unsafe_acceptance_error=unsafe_acceptance_error,
                    minimum_total_separator_error=total_separator_error,
                    equal_prior_bayes_error=bayes_error,
                    equal_prior_bayes_error_positive=bayes_fraction > 0,
                    equal_prior_bayes_error_underflowed=bayes_underflowed,
                    tv_separator_identity_residual=_fraction_to_float(
                        separator_residual_fraction,
                        label="TV/separator identity residual",
                        reject_nonzero_underflow=True,
                    ),
                    tv_bayes_identity_residual=(
                        None
                        if bayes_residual_fraction is None
                        else _fraction_to_float(
                            bayes_residual_fraction,
                            label="TV/Bayes identity residual",
                        )
                    ),
                )
            )
    return tuple(points)


def _validated_codes(
    latents: Sequence[Vector], margins: Sequence[float], *, rounding_digits: int | None
) -> tuple[tuple[Code, ...], tuple[float, ...]]:
    if len(latents) != len(margins):
        raise ValueError("latents and margins must have the same length")
    if not latents:
        raise ValueError("at least one state is required")
    dimension = len(latents[0])
    if dimension == 0 or any(len(latent) != dimension for latent in latents):
        raise ValueError("latent codes must be non-empty and share one dimension")

    codes: list[Code] = []
    for latent_index, latent in enumerate(latents):
        code = tuple(
            _finite_real(
                value,
                label=f"latents[{latent_index}][{coordinate_index}]",
            )
            for coordinate_index, value in enumerate(latent)
        )
        if rounding_digits is not None:
            code = tuple(round(value, rounding_digits) for value in code)
        codes.append(code)

    numeric_margins = tuple(
        _finite_real(margin, label=f"margins[{index}]")
        for index, margin in enumerate(margins)
    )
    return tuple(codes), numeric_margins


def _fiber_indices(codes: Sequence[Code]) -> dict[Code, list[int]]:
    fibers: dict[Code, list[int]] = {}
    for index, code in enumerate(codes):
        fibers.setdefault(code, []).append(index)
    return fibers


def audit_finite_static_fibers(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    rounding_digits: int | None = None,
) -> FiniteStaticAudit:
    """Enumerate exact mixed fibers and the maximal sound latent certificate.

    A state is safe at ``h >= 0`` and unsafe at ``h < 0``.  A latent code is accepted by the
    maximal sound certificate exactly when every enumerated state in that fiber is safe.  The
    returned defect is ``max({h(x): x safe and its fiber contains an unsafe state} union {0})``.

    ``rounding_digits`` intentionally changes the audited representation by quantizing it.  It must
    not be described as proof of exact collisions for the unrounded encoder.
    """

    codes, numeric_margins = _validated_codes(
        latents, margins, rounding_digits=rounding_digits
    )
    fibers = _fiber_indices(codes)
    collision_margins: list[float] = []
    mixed_fiber_count = 0
    maximal_sound_codes: list[Code] = []

    for code, indices in fibers.items():
        has_unsafe = any(numeric_margins[index] < 0.0 for index in indices)
        safe_margins = [
            numeric_margins[index] for index in indices if numeric_margins[index] >= 0.0
        ]
        if has_unsafe and safe_margins:
            mixed_fiber_count += 1
            collision_margins.extend(safe_margins)
        if not has_unsafe:
            maximal_sound_codes.append(code)

    ordered_collision_margins = tuple(sorted(collision_margins))
    return FiniteStaticAudit(
        state_count=len(codes),
        fiber_count=len(fibers),
        mixed_fiber_count=mixed_fiber_count,
        collided_safe_count=len(collision_margins),
        exact_defect=max(ordered_collision_margins, default=0.0),
        collision_margins=ordered_collision_margins,
        maximal_sound_codes=tuple(sorted(maximal_sound_codes)),
    )


def check_sound_completeness(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    completeness_margin: float,
    rounding_digits: int | None = None,
) -> CompletenessCheck:
    """Check whether any sound certificate can include all states with ``h >= margin``.

    On a finite enumerated space the answer is exact: such a certificate exists iff no required
    code also contains an unsafe state.  This explicitly resolves the endpoint case that a bare
    supremum value cannot distinguish when the defect equals zero.
    """

    margin = _finite_real(completeness_margin, label="completeness_margin")
    codes, numeric_margins = _validated_codes(
        latents, margins, rounding_digits=rounding_digits
    )
    fibers = _fiber_indices(codes)
    required_codes = {
        code for code, value in zip(codes, numeric_margins, strict=True) if value >= margin
    }
    conflicting = {
        code
        for code in required_codes
        if any(numeric_margins[index] < 0.0 for index in fibers[code])
    }
    return CompletenessCheck(
        margin=margin,
        exists=not conflicting,
        required_codes=tuple(sorted(required_codes)),
        conflicting_required_codes=tuple(sorted(conflicting)),
    )


def check_finite_data_processing(
    fine_latents: Sequence[Vector],
    coarse_latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    rounding_digits: int | None = None,
) -> DataProcessingCheck:
    """Check exact-defect monotonicity when coarse codes are a function of fine codes.

    ``is_deterministic_postprocessing`` is false when one fine code maps to multiple coarse codes on
    the enumerated states.  In that case the data-processing premise is absent and ``monotone`` is
    reported as false, even if the two numerical defects happen to be ordered.
    """

    fine_codes, numeric_margins = _validated_codes(
        fine_latents, margins, rounding_digits=rounding_digits
    )
    coarse_codes, _ = _validated_codes(
        coarse_latents, margins, rounding_digits=rounding_digits
    )
    mapping: dict[Code, Code] = {}
    deterministic = True
    for fine, coarse in zip(fine_codes, coarse_codes, strict=True):
        previous = mapping.setdefault(fine, coarse)
        if previous != coarse:
            deterministic = False

    fine_audit = audit_finite_static_fibers(
        fine_codes, numeric_margins, rounding_digits=None
    )
    coarse_audit = audit_finite_static_fibers(
        coarse_codes, numeric_margins, rounding_digits=None
    )
    numerically_monotone = coarse_audit.exact_defect + 1e-12 >= fine_audit.exact_defect
    return DataProcessingCheck(
        is_deterministic_postprocessing=deterministic,
        fine_defect=fine_audit.exact_defect,
        coarse_defect=coarse_audit.exact_defect,
        monotone=deterministic and numerically_monotone,
    )


def finite_stochastic_pair_frontier(
    encoder_kernel: Sequence[Sequence[float]],
    margins: Sequence[float],
    *,
    probability_tolerance: float = 1e-12,
) -> tuple[StochasticPairFrontierPoint, ...]:
    """Return the exact finite TV/separator-error summary for each safe/unsafe pair.

    Mutual singularity is decided by exact positive-support intersection, never by rounding total
    variation to one.  Admissible near-normalized inputs are first canonically closed to exact-sum
    binary64 laws without changing support.  The minimum total separator error is the sum of safe
    rejection and unsafe acceptance errors and equals one minus TV in exact rational arithmetic.
    The Bayes quantity is half that sum under equal priors.  Neither is the minimax worst-state
    error, and neither is a population claim beyond the enumerated states.
    """

    rows, numeric_margins, _ = _validated_stochastic_problem(
        encoder_kernel,
        margins,
        probability_tolerance=probability_tolerance,
    )
    return _stochastic_pair_frontier_from_validated(rows, numeric_margins)


def check_finite_stochastic_action_tradeoff(
    encoder_kernel: Sequence[Sequence[float]],
    policy_kernel: Sequence[Sequence[float]],
    *,
    first_state_index: int,
    second_state_index: int,
    first_safe_action_indices: Sequence[int],
    second_safe_action_indices: Sequence[int],
    probability_tolerance: float = 1e-12,
) -> StochasticActionTradeoffCheck:
    """Check the exact finite randomized-action TV tradeoff for two states.

    ``policy_kernel`` maps each encoder output to a distribution on a common
    finite action alphabet.  The two declared safe-action index sets must be
    disjoint.  All probability rows receive the same exact-rational validation
    and support-preserving canonical closure used by the stochastic separator
    oracle.  The induced action laws and inequality are then computed exactly.
    """

    tolerance = _finite_real(
        probability_tolerance,
        label="probability_tolerance",
        nonnegative=True,
    )
    encoder = _validated_probability_kernel(
        encoder_kernel,
        label="encoder_kernel",
        probability_tolerance=tolerance,
    )
    policy = _validated_probability_kernel(
        policy_kernel,
        label="policy_kernel",
        probability_tolerance=tolerance,
        expected_row_count=len(encoder[0]),
    )

    def state_index(value: object, *, label: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= len(encoder)
        ):
            raise ValueError(f"{label} must index an encoder state")
        return value

    first_state = state_index(first_state_index, label="first_state_index")
    second_state = state_index(second_state_index, label="second_state_index")
    action_count = len(policy[0])

    def action_indices(values: Sequence[int], *, label: str) -> tuple[int, ...]:
        indices = tuple(values)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= action_count
            for index in indices
        ):
            raise ValueError(f"{label} must contain valid action indices")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{label} must not contain duplicates")
        return tuple(sorted(indices))

    first_safe = action_indices(
        first_safe_action_indices,
        label="first_safe_action_indices",
    )
    second_safe = action_indices(
        second_safe_action_indices,
        label="second_safe_action_indices",
    )
    if set(first_safe) & set(second_safe):
        raise ValueError("the two safe-action index sets must be disjoint")

    induced = _exact_postprocessed_probability_kernel(encoder, policy)
    code_tv = _exact_total_variation(
        encoder[first_state], encoder[second_state]
    )
    action_tv = _exact_total_variation(
        induced[first_state], induced[second_state]
    )
    first_safe_mass = sum(
        (induced[first_state][action] for action in first_safe),
        Fraction(),
    )
    second_safe_mass = sum(
        (induced[second_state][action] for action in second_safe),
        Fraction(),
    )
    first_violation = 1 - first_safe_mass
    second_violation = 1 - second_safe_mass
    violation_sum = first_violation + second_violation
    lower_bound = 1 - code_tv
    slack = violation_sum - lower_bound
    action_tv_contracted = action_tv <= code_tv
    tradeoff_holds = slack >= 0
    rounded_action_tv = _fraction_to_float(
        action_tv,
        label="induced action total variation",
    )
    action_tv_underflowed = action_tv > 0 and rounded_action_tv == 0.0

    return StochasticActionTradeoffCheck(
        first_state_index=first_state,
        second_state_index=second_state,
        first_safe_action_indices=first_safe,
        second_safe_action_indices=second_safe,
        code_total_variation=_fraction_to_float(
            code_tv,
            label="code total variation",
        ),
        induced_action_total_variation=(
            None if action_tv_underflowed else rounded_action_tv
        ),
        induced_action_total_variation_positive=action_tv > 0,
        induced_action_total_variation_underflowed=action_tv_underflowed,
        first_violation_probability=_fraction_to_float(
            first_violation,
            label="first violation probability",
            reject_nonzero_underflow=True,
        ),
        second_violation_probability=_fraction_to_float(
            second_violation,
            label="second violation probability",
            reject_nonzero_underflow=True,
        ),
        summed_violation_probability=_fraction_to_float(
            violation_sum,
            label="summed violation probability",
            reject_nonzero_underflow=True,
        ),
        tv_lower_bound=_fraction_to_float(
            lower_bound,
            label="TV action lower bound",
            reject_nonzero_underflow=True,
        ),
        tradeoff_slack=_fraction_to_float(
            slack,
            label="TV action tradeoff slack",
            reject_nonzero_underflow=True,
        ),
        action_tv_contracted=action_tv_contracted,
        tradeoff_holds=tradeoff_holds,
    )


def audit_finite_stochastic_kernel(
    encoder_kernel: Sequence[Sequence[float]],
    margins: Sequence[float],
    *,
    probability_tolerance: float = 1e-12,
) -> FiniteStochasticAudit:
    """Audit exact almost-sure conflicts for a finite stochastic representation.

    The maximal sound output set contains precisely the symbols assigned zero probability by every
    unsafe state.  A safe state is conflicted when it puts positive probability outside that set,
    equivalently when its law is not mutually singular with at least one unsafe law.  Because both
    state families are finite, pairwise singularity is also sufficient for one common measurable
    output certificate.  This equivalence does not extend to arbitrary uncountable families.
    """

    rows, numeric_margins, tolerance = _validated_stochastic_problem(
        encoder_kernel,
        margins,
        probability_tolerance=probability_tolerance,
    )
    safe_indices = tuple(
        index for index, margin in enumerate(numeric_margins) if margin >= 0.0
    )
    unsafe_indices = tuple(
        index for index, margin in enumerate(numeric_margins) if margin < 0.0
    )
    unsafe_support = {
        output
        for unsafe_index in unsafe_indices
        for output, probability in enumerate(rows[unsafe_index])
        if probability > 0.0
    }
    maximal_sound = tuple(
        output for output in range(len(rows[0])) if output not in unsafe_support
    )
    conflict_margins = tuple(
        sorted(
            numeric_margins[safe_index]
            for safe_index in safe_indices
            if any(
                rows[safe_index][output] > 0.0 for output in unsafe_support
            )
        )
    )
    return FiniteStochasticAudit(
        state_count=len(rows),
        output_count=len(rows[0]),
        safe_count=len(safe_indices),
        unsafe_count=len(unsafe_indices),
        conflicted_safe_count=len(conflict_margins),
        exact_defect=max(conflict_margins, default=0.0),
        conflict_margins=conflict_margins,
        maximal_sound_output_indices=maximal_sound,
        pair_frontier=_stochastic_pair_frontier_from_validated(rows, numeric_margins),
        probability_tolerance=tolerance,
    )


def check_finite_stochastic_completeness(
    encoder_kernel: Sequence[Sequence[float]],
    margins: Sequence[float],
    *,
    completeness_margin: float,
    strict: bool = False,
    probability_tolerance: float = 1e-12,
) -> StochasticCompletenessCheck:
    """Check existence of an a.s.-sound finite completeness separator.

    On this finite domain the answer is exact in support semantics.  The default
    closed convention requires every state with ``h >= completeness_margin``;
    ``strict=True`` implements the open requirement ``h > completeness_margin``.
    Every positive-probability output of a required state must lie in the maximal
    sound output set.  A probability smaller than the normalization tolerance is
    still positive support and can therefore block a certificate.
    """

    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")

    threshold = _finite_real(
        completeness_margin,
        label="completeness_margin",
        nonnegative=True,
    )
    rows, numeric_margins, _ = _validated_stochastic_problem(
        encoder_kernel,
        margins,
        probability_tolerance=probability_tolerance,
    )
    required = tuple(
        index
        for index, margin in enumerate(numeric_margins)
        if (margin > threshold if strict else margin >= threshold)
    )
    unsafe = tuple(
        index for index, margin in enumerate(numeric_margins) if margin < 0.0
    )
    unsafe_support = {
        output
        for unsafe_index in unsafe
        for output, probability in enumerate(rows[unsafe_index])
        if probability > 0.0
    }
    maximal_sound = tuple(
        output for output in range(len(rows[0])) if output not in unsafe_support
    )
    conflicts = tuple(
        (required_index, unsafe_index)
        for required_index in required
        for unsafe_index in unsafe
        if any(
            safe_probability > 0.0 and unsafe_probability > 0.0
            for safe_probability, unsafe_probability in zip(
                rows[required_index], rows[unsafe_index], strict=True
            )
        )
    )
    return StochasticCompletenessCheck(
        margin=threshold,
        strict=strict,
        exists=not conflicts,
        required_safe_indices=required,
        conflicting_pairs=conflicts,
        maximal_sound_output_indices=maximal_sound,
    )


def postprocess_finite_stochastic_kernel(
    encoder_kernel: Sequence[Sequence[float]],
    markov_kernel: Sequence[Sequence[float]],
    *,
    probability_tolerance: float = 1e-12,
) -> tuple[tuple[float, ...], ...]:
    """Compose a finite stochastic encoder with a finite Markov kernel.

    Composition is performed exactly on the rational values represented by the
    validated binary64 inputs.  The public binary64 result is a deterministic
    exact-sum projection that must preserve the exact positive support of every
    summed output mass.  Failure to represent that support is rejected rather
    than allowed to fabricate singularity.  Exact TV data-processing checks use
    the unprojected rational product, not this display-oriented projection.
    """

    tolerance = _finite_real(
        probability_tolerance,
        label="probability_tolerance",
        nonnegative=True,
    )
    encoder = _validated_probability_kernel(
        encoder_kernel,
        label="encoder_kernel",
        probability_tolerance=tolerance,
    )
    transition = _validated_probability_kernel(
        markov_kernel,
        label="markov_kernel",
        probability_tolerance=tolerance,
        expected_row_count=len(encoder[0]),
    )
    exact_result = _exact_postprocessed_probability_kernel(encoder, transition)
    return _project_exact_probability_kernel(
        exact_result,
        label="postprocessed_kernel",
    )


def check_finite_stochastic_data_processing(
    encoder_kernel: Sequence[Sequence[float]],
    markov_kernel: Sequence[Sequence[float]],
    margins: Sequence[float],
    *,
    probability_tolerance: float = 1e-12,
) -> StochasticDataProcessingCheck:
    """Check exact-conflict persistence and exact-rational TV contraction.

    ``probability_tolerance`` governs only input row normalization.  It is never
    reused as an arithmetic or theorem tolerance: TV contraction is decided on
    the exact rational Markov product.  This checker does not require that product
    to have a support-preserving binary64 projection; only the separate public
    postprocessing helper has that display-oriented limitation.
    """

    tolerance = _finite_real(
        probability_tolerance,
        label="probability_tolerance",
        nonnegative=True,
    )
    encoder, numeric_margins, _ = _validated_stochastic_problem(
        encoder_kernel,
        margins,
        probability_tolerance=tolerance,
    )
    transition = _validated_probability_kernel(
        markov_kernel,
        label="markov_kernel",
        probability_tolerance=tolerance,
        expected_row_count=len(encoder[0]),
    )
    exact_coarse_kernel = _exact_postprocessed_probability_kernel(
        encoder, transition
    )
    fine = audit_finite_stochastic_kernel(
        encoder,
        numeric_margins,
        probability_tolerance=0.0,
    )
    fine_pairs = {
        (point.safe_index, point.unsafe_index): point for point in fine.pair_frontier
    }
    exact_conflicts_preserved = all(
        point.mutually_singular
        or any(
            exact_coarse_kernel[safe_index][output] > 0
            and exact_coarse_kernel[unsafe_index][output] > 0
            for output in range(len(exact_coarse_kernel[0]))
        )
        for (safe_index, unsafe_index), point in fine_pairs.items()
    )
    unsafe_indices = tuple(
        index for index, margin in enumerate(numeric_margins) if margin < 0.0
    )
    unsafe_support = {
        output
        for unsafe_index in unsafe_indices
        for output, probability in enumerate(exact_coarse_kernel[unsafe_index])
        if probability > 0
    }
    coarse_conflict_margins = tuple(
        numeric_margins[safe_index]
        for safe_index, margin in enumerate(numeric_margins)
        if margin >= 0.0
        and any(
            exact_coarse_kernel[safe_index][output] > 0
            for output in unsafe_support
        )
    )
    coarse_exact_defect = max(coarse_conflict_margins, default=0.0)
    tv_increases_exact = tuple(
        _exact_total_variation(
            exact_coarse_kernel[safe_index],
            exact_coarse_kernel[unsafe_index],
        )
        - _exact_total_variation(
            encoder[safe_index],
            encoder[unsafe_index],
        )
        for safe_index, unsafe_index in fine_pairs
    )
    maximum_increase_exact = max(tv_increases_exact, default=Fraction())
    maximum_increase = _fraction_to_float(
        maximum_increase_exact,
        label="maximum total-variation increase",
    )
    tv_contracted = maximum_increase_exact <= 0
    defect_monotone = coarse_exact_defect >= fine.exact_defect
    monotone = exact_conflicts_preserved and defect_monotone and tv_contracted
    return StochasticDataProcessingCheck(
        state_count=fine.state_count,
        fine_output_count=fine.output_count,
        coarse_output_count=len(exact_coarse_kernel[0]),
        safe_unsafe_pair_count=len(fine_pairs),
        fine_exact_defect=fine.exact_defect,
        coarse_exact_defect=coarse_exact_defect,
        exact_conflicts_preserved=exact_conflicts_preserved,
        exact_defect_monotone=defect_monotone,
        total_variation_contracted=tv_contracted,
        maximum_total_variation_increase=maximum_increase,
        probability_tolerance=tolerance,
        monotone=monotone,
    )


def enumerate_binary_certificates(codes: Iterable[Code]) -> tuple[frozenset[Code], ...]:
    """Enumerate all latent subsets for tiny theorem checks.

    This helper is exponential and intentionally refuses more than 20 codes.
    """

    unique_codes = tuple(sorted(set(codes)))
    if len(unique_codes) > 20:
        raise ValueError("certificate enumeration is restricted to at most 20 unique codes")
    return tuple(
        frozenset(
            unique_codes[index]
            for index in range(len(unique_codes))
            if mask & (1 << index)
        )
        for mask in range(1 << len(unique_codes))
    )
