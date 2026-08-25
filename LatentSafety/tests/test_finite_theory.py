import math
import unittest
from fractions import Fraction
from itertools import product

from latent_safety.metrics.finite import (
    audit_finite_stochastic_kernel,
    audit_finite_static_fibers,
    check_finite_data_processing,
    check_finite_stochastic_action_tradeoff,
    check_finite_stochastic_completeness,
    check_finite_stochastic_data_processing,
    check_sound_completeness,
    enumerate_binary_certificates,
    finite_stochastic_pair_frontier,
    postprocess_finite_stochastic_kernel,
)


class FiniteStaticTheoryTests(unittest.TestCase):
    def test_exact_audit_and_maximal_sound_certificate(self) -> None:
        latents = [(0.0,), (0.0,), (1.0,), (2.0,)]
        margins = [0.4, -0.1, 0.8, -0.2]
        audit = audit_finite_static_fibers(latents, margins)

        self.assertEqual(audit.mixed_fiber_count, 1)
        self.assertEqual(audit.collided_safe_count, 1)
        self.assertAlmostEqual(audit.exact_defect, 0.4)
        self.assertEqual(audit.maximal_sound_codes, ((1.0,),))

    def test_finite_endpoint_is_explicit(self) -> None:
        latents = [(0.0,), (0.0,), (1.0,)]
        margins = [0.4, -0.1, 0.8]

        self.assertFalse(
            check_sound_completeness(
                latents, margins, completeness_margin=0.4
            ).exists
        )
        self.assertTrue(
            check_sound_completeness(
                latents, margins, completeness_margin=0.400001
            ).exists
        )

    def test_zero_defect_does_not_encode_endpoint_attainment(self) -> None:
        no_collision = check_sound_completeness(
            [(0.0,), (1.0,)], [0.0, -0.1], completeness_margin=0.0
        )
        boundary_collision = check_sound_completeness(
            [(0.0,), (0.0,)], [0.0, -0.1], completeness_margin=0.0
        )

        self.assertTrue(no_collision.exists)
        self.assertFalse(boundary_collision.exists)
        self.assertEqual(
            audit_finite_static_fibers([(0.0,), (1.0,)], [0.0, -0.1]).exact_defect,
            0.0,
        )
        self.assertEqual(
            audit_finite_static_fibers([(0.0,), (0.0,)], [0.0, -0.1]).exact_defect,
            0.0,
        )

    def test_data_processing_under_coarsening(self) -> None:
        margins = [0.8, -0.1, 0.3]
        fine = [(0.0,), (1.0,), (2.0,)]
        coarse = [(0.0,), (0.0,), (2.0,)]
        result = check_finite_data_processing(fine, coarse, margins)

        self.assertTrue(result.is_deterministic_postprocessing)
        self.assertTrue(result.monotone)
        self.assertEqual(result.fine_defect, 0.0)
        self.assertEqual(result.coarse_defect, 0.8)

    def test_detects_missing_postprocessing_premise(self) -> None:
        result = check_finite_data_processing(
            [(0.0,), (0.0,)], [(0.0,), (1.0,)], [0.2, -0.1]
        )
        self.assertFalse(result.is_deterministic_postprocessing)
        self.assertFalse(result.monotone)

    def test_certificate_enumeration_guard(self) -> None:
        certificates = enumerate_binary_certificates([(0.0,), (1.0,)])
        self.assertEqual(len(certificates), 4)
        with self.assertRaises(ValueError):
            enumerate_binary_certificates((float(index),) for index in range(21))

    def test_static_oracle_rejects_inexact_binary64_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            audit_finite_static_fibers(
                ((0.0,), (0.0,)),
                (0.5, Fraction(-1, 10**400)),
            )
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            audit_finite_static_fibers(
                ((Fraction(1, 10),), (0.0,)),
                (0.5, -0.5),
            )
        with self.assertRaisesRegex(ValueError, "finite and exactly representable"):
            audit_finite_static_fibers(
                ((0.0,), (0.0,)),
                (0.5, 10**5000),
            )
        accepted = audit_finite_static_fibers(
            ((Fraction(1, 2),), (0.0,)),
            (Fraction(1, 2), -0.5),
        )
        self.assertEqual(accepted.exact_defect, 0.0)


class FiniteStochasticTheoryTests(unittest.TestCase):
    def test_exact_support_audit_and_closed_endpoint(self) -> None:
        margins = (0.8, 0.4, -0.1)
        kernel = (
            (1.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 1.0, 0.0),
        )
        audit = audit_finite_stochastic_kernel(kernel, margins)

        self.assertEqual(audit.state_count, 3)
        self.assertEqual(audit.output_count, 3)
        self.assertEqual(audit.safe_count, 2)
        self.assertEqual(audit.unsafe_count, 1)
        self.assertEqual(audit.conflicted_safe_count, 1)
        self.assertEqual(audit.conflict_margins, (0.4,))
        self.assertEqual(audit.exact_defect, 0.4)
        self.assertEqual(audit.maximal_sound_output_indices, (0, 2))

        endpoint = check_finite_stochastic_completeness(
            kernel, margins, completeness_margin=0.4
        )
        above = check_finite_stochastic_completeness(
            kernel, margins, completeness_margin=0.400001
        )
        open_endpoint = check_finite_stochastic_completeness(
            kernel,
            margins,
            completeness_margin=0.4,
            strict=True,
        )
        self.assertFalse(endpoint.exists)
        self.assertFalse(endpoint.strict)
        self.assertEqual(endpoint.required_safe_indices, (0, 1))
        self.assertEqual(endpoint.conflicting_pairs, ((1, 2),))
        self.assertTrue(above.exists)
        self.assertEqual(above.required_safe_indices, (0,))
        self.assertTrue(open_endpoint.exists)
        self.assertTrue(open_endpoint.strict)
        self.assertEqual(open_endpoint.required_safe_indices, (0,))

    def test_pair_tv_and_equal_prior_bayes_error_frontier(self) -> None:
        points = finite_stochastic_pair_frontier(
            ((0.9, 0.1), (0.6, 0.4), (0.0, 1.0)),
            (1.0, -0.5, -0.25),
        )
        first, second = points
        self.assertEqual((first.safe_index, first.unsafe_index), (0, 1))
        self.assertFalse(first.mutually_singular)
        self.assertEqual(first.shared_support_indices, (0, 1))
        self.assertEqual(first.optimal_separator_output_indices, (0,))
        self.assertAlmostEqual(first.total_variation, 0.3)
        self.assertAlmostEqual(first.overlap_mass, 0.7)
        self.assertAlmostEqual(first.safe_rejection_error, 0.1)
        self.assertAlmostEqual(first.unsafe_acceptance_error, 0.6)
        self.assertAlmostEqual(first.minimum_total_separator_error, 0.7)
        self.assertAlmostEqual(first.equal_prior_bayes_error, 0.35)
        self.assertLessEqual(first.tv_separator_identity_residual, 1e-16)
        self.assertLessEqual(first.tv_bayes_identity_residual, 1e-16)

        self.assertEqual((second.safe_index, second.unsafe_index), (0, 2))
        self.assertFalse(second.mutually_singular)
        self.assertAlmostEqual(second.total_variation, 0.9)
        self.assertAlmostEqual(second.minimum_total_separator_error, 0.1)
        self.assertAlmostEqual(second.equal_prior_bayes_error, 0.05)

    def test_exhaustive_pair_frontier_matches_all_output_subsets(self) -> None:
        row_options = tuple(
            (first / 2.0, second / 2.0, third / 2.0)
            for first in range(3)
            for second in range(3)
            for third in range(3)
            if first + second + third == 2
        )
        checked = 0
        for safe_row in row_options:
            for unsafe_row in row_options:
                point = finite_stochastic_pair_frontier(
                    (safe_row, unsafe_row),
                    (1.0, -1.0),
                )[0]
                errors = []
                for mask in range(1 << len(safe_row)):
                    accepted = {
                        output
                        for output in range(len(safe_row))
                        if mask & (1 << output)
                    }
                    errors.append(
                        math.fsum(
                            probability
                            for output, probability in enumerate(safe_row)
                            if output not in accepted
                        )
                        + math.fsum(
                            probability
                            for output, probability in enumerate(unsafe_row)
                            if output in accepted
                        )
                    )
                self.assertEqual(point.minimum_total_separator_error, min(errors))
                self.assertEqual(
                    point.minimum_total_separator_error,
                    1.0 - point.total_variation,
                )
                self.assertEqual(
                    point.overlap_mass,
                    1.0 - point.total_variation,
                )
                self.assertEqual(
                    point.equal_prior_bayes_error,
                    0.5 * point.minimum_total_separator_error,
                )
                checked += 1
        self.assertEqual(checked, 36)

    def test_tiny_positive_support_is_not_inferred_from_near_one_tv(self) -> None:
        tiny = 2.0**-50
        point = finite_stochastic_pair_frontier(
            ((1.0 - tiny, tiny), (0.0, 1.0)),
            (0.7, -0.1),
        )[0]
        audit = audit_finite_stochastic_kernel(
            ((1.0 - tiny, tiny), (0.0, 1.0)),
            (0.7, -0.1),
        )

        self.assertLess(point.total_variation, 1.0)
        self.assertTrue(math.isclose(point.total_variation, 1.0, abs_tol=1e-12))
        self.assertGreater(point.equal_prior_bayes_error, 0.0)
        self.assertTrue(point.equal_prior_bayes_error_positive)
        self.assertFalse(point.equal_prior_bayes_error_underflowed)
        self.assertFalse(point.mutually_singular)
        self.assertEqual(point.shared_support_indices, (1,))
        self.assertEqual(audit.exact_defect, 0.7)

    def test_near_normalized_rows_are_canonically_closed_before_tv(self) -> None:
        excess = 2.0**-41
        point = finite_stochastic_pair_frontier(
            (
                (0.5 + excess, 0.5, 0.0, 0.0),
                (0.0, 0.0, 0.5 + excess, 0.5),
            ),
            (1.0, -1.0),
        )[0]

        self.assertTrue(point.mutually_singular)
        self.assertEqual(point.total_variation, 1.0)
        self.assertEqual(point.minimum_total_separator_error, 0.0)
        self.assertEqual(point.tv_separator_identity_residual, 0.0)

    def test_bayes_half_error_underflow_is_explicit_not_false_zero(self) -> None:
        smallest = math.ulp(0.0)
        geometric_tail = tuple(2.0**-power for power in range(1, 1075))
        zeros = (0.0,) * len(geometric_tail)
        safe_row = (smallest,) + geometric_tail + zeros
        unsafe_row = (smallest,) + zeros + geometric_tail

        point = finite_stochastic_pair_frontier(
            (safe_row, unsafe_row),
            (1.0, -1.0),
        )[0]
        self.assertFalse(point.mutually_singular)
        self.assertEqual(point.overlap_mass, smallest)
        self.assertEqual(point.minimum_total_separator_error, smallest)
        self.assertEqual(point.total_variation, 1.0)
        self.assertIsNone(point.equal_prior_bayes_error)
        self.assertTrue(point.equal_prior_bayes_error_positive)
        self.assertTrue(point.equal_prior_bayes_error_underflowed)
        self.assertIsNone(point.tv_bayes_identity_residual)

    def test_exhaustive_output_subsets_match_completeness_oracle(self) -> None:
        row_options = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.0, 0.5),
        )
        margins = (0.8, 0.2, -0.1)
        thresholds = (0.0, 0.2, 0.200001, 0.8, 0.800001)
        for row_assignment in product(row_options, repeat=len(margins)):
            for threshold in thresholds:
                oracle = check_finite_stochastic_completeness(
                    row_assignment,
                    margins,
                    completeness_margin=threshold,
                ).exists
                brute = False
                for mask in range(1 << len(row_options[0])):
                    accepted = {
                        output
                        for output in range(len(row_options[0]))
                        if mask & (1 << output)
                    }
                    sound = all(
                        probability == 0.0
                        for index, margin in enumerate(margins)
                        if margin < 0.0
                        for output, probability in enumerate(row_assignment[index])
                        if output in accepted
                    )
                    complete = all(
                        probability == 0.0 or output in accepted
                        for index, margin in enumerate(margins)
                        if margin >= threshold
                        for output, probability in enumerate(row_assignment[index])
                    )
                    if sound and complete:
                        brute = True
                        break
                self.assertEqual(
                    oracle,
                    brute,
                    msg=(row_assignment, threshold, oracle, brute),
                )

    def test_deterministic_one_hot_reduction_matches_static_oracle(self) -> None:
        margins = (-0.4, 0.0, 0.25, 0.9)
        thresholds = (0.0, 0.1, 0.25, 0.250001, 0.9, 0.900001)
        for assignment in product(range(3), repeat=len(margins)):
            latents = tuple((float(code),) for code in assignment)
            kernel = tuple(
                tuple(1.0 if output == code else 0.0 for output in range(3))
                for code in assignment
            )
            deterministic = audit_finite_static_fibers(latents, margins)
            stochastic = audit_finite_stochastic_kernel(kernel, margins)
            self.assertEqual(stochastic.exact_defect, deterministic.exact_defect)
            self.assertEqual(
                stochastic.conflict_margins,
                deterministic.collision_margins,
            )
            for threshold in thresholds:
                self.assertEqual(
                    check_finite_stochastic_completeness(
                        kernel,
                        margins,
                        completeness_margin=threshold,
                    ).exists,
                    check_sound_completeness(
                        latents,
                        margins,
                        completeness_margin=threshold,
                    ).exists,
                )

    def test_markov_postprocessing_creates_conflict_and_contracts_tv(self) -> None:
        encoder = ((1.0, 0.0), (0.0, 1.0))
        postprocessor = ((0.75, 0.25), (0.25, 0.75))
        margins = (0.8, -0.1)
        coarse = postprocess_finite_stochastic_kernel(encoder, postprocessor)
        result = check_finite_stochastic_data_processing(
            encoder, postprocessor, margins
        )

        self.assertEqual(coarse, postprocessor)
        self.assertEqual(result.fine_exact_defect, 0.0)
        self.assertEqual(result.coarse_exact_defect, 0.8)
        self.assertTrue(result.exact_conflicts_preserved)
        self.assertTrue(result.exact_defect_monotone)
        self.assertTrue(result.total_variation_contracted)
        self.assertLessEqual(result.maximum_total_variation_increase, 0.0)
        self.assertTrue(result.monotone)
        coarse_point = finite_stochastic_pair_frontier(coarse, margins)[0]
        self.assertAlmostEqual(coarse_point.total_variation, 0.5)
        self.assertAlmostEqual(coarse_point.equal_prior_bayes_error, 0.25)

    def test_exhaustive_small_markov_kernels_obey_data_processing(self) -> None:
        row_options = ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))
        margins = (0.8, 0.2, -0.1)
        checked = 0
        for encoder in product(row_options, repeat=len(margins)):
            for postprocessor in product(row_options, repeat=2):
                result = check_finite_stochastic_data_processing(
                    encoder,
                    postprocessor,
                    margins,
                )
                self.assertTrue(result.monotone)
                checked += 1
        self.assertEqual(checked, 243)

    def test_normalization_tolerance_is_not_a_tv_contraction_tolerance(self) -> None:
        excess = 2.0**-41
        result = check_finite_stochastic_data_processing(
            ((0.75, 0.25), (0.25, 0.75)),
            (
                (1.0, excess, 0.0, 0.0),
                (0.0, 0.0, 1.0, excess),
            ),
            (1.0, -1.0),
        )

        self.assertTrue(result.total_variation_contracted)
        self.assertLessEqual(result.maximum_total_variation_increase, 0.0)
        self.assertTrue(result.monotone)

    def test_exhaustive_randomized_action_tradeoff(self) -> None:
        row_options = ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))
        checked = 0
        for encoder in product(row_options, repeat=2):
            for policy in product(row_options, repeat=2):
                result = check_finite_stochastic_action_tradeoff(
                    encoder,
                    policy,
                    first_state_index=0,
                    second_state_index=1,
                    first_safe_action_indices=(0,),
                    second_safe_action_indices=(1,),
                )
                self.assertTrue(result.action_tv_contracted)
                self.assertTrue(result.tradeoff_holds)
                self.assertGreaterEqual(
                    result.summed_violation_probability,
                    result.tv_lower_bound,
                )
                checked += 1
        self.assertEqual(checked, 81)

        tight = check_finite_stochastic_action_tradeoff(
            ((0.75, 0.25), (0.25, 0.75)),
            ((1.0, 0.0), (0.0, 1.0)),
            first_state_index=0,
            second_state_index=1,
            first_safe_action_indices=(0,),
            second_safe_action_indices=(1,),
        )
        self.assertEqual(tight.code_total_variation, 0.5)
        self.assertEqual(tight.induced_action_total_variation, 0.5)
        self.assertEqual(tight.first_violation_probability, 0.25)
        self.assertEqual(tight.second_violation_probability, 0.25)
        self.assertEqual(tight.summed_violation_probability, 0.5)
        self.assertEqual(tight.tv_lower_bound, 0.5)
        self.assertEqual(tight.tradeoff_slack, 0.0)

        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            check_finite_stochastic_action_tradeoff(
                row_options[:2],
                row_options[:2],
                first_state_index=0,
                second_state_index=1,
                first_safe_action_indices=(0,),
                second_safe_action_indices=(0,),
            )

    def test_induced_action_tv_underflow_is_explicit(self) -> None:
        smallest = math.ulp(0.0)
        base = (
            (smallest,)
            + tuple(2.0**-power for power in range(1, 1075))
            + (0.0,)
        )
        moved = (0.0,) + base[1:-1] + (smallest,)
        policy = ((0.75, 0.25),) + ((0.5, 0.5),) * (len(base) - 1)
        result = check_finite_stochastic_action_tradeoff(
            (base, moved),
            policy,
            first_state_index=0,
            second_state_index=1,
            first_safe_action_indices=(0,),
            second_safe_action_indices=(1,),
        )

        self.assertIsNone(result.induced_action_total_variation)
        self.assertTrue(result.induced_action_total_variation_positive)
        self.assertTrue(result.induced_action_total_variation_underflowed)
        self.assertTrue(result.action_tv_contracted)
        self.assertTrue(result.tradeoff_holds)

    def test_stochastic_kernel_validation_and_underflow_are_fail_closed(self) -> None:
        invalid_calls = (
            lambda: audit_finite_stochastic_kernel((), ()),
            lambda: audit_finite_stochastic_kernel(((1.0,),), (0.1, -0.1)),
            lambda: audit_finite_stochastic_kernel(((0.5, 0.4),), (0.1,)),
            lambda: audit_finite_stochastic_kernel(((1.1, -0.1),), (0.1,)),
            lambda: audit_finite_stochastic_kernel(((True, 0.0),), (0.1,)),
            lambda: audit_finite_stochastic_kernel((("1", 0.0),), (0.1,)),
            lambda: audit_finite_stochastic_kernel(((math.nan, 0.0),), (0.1,)),
            lambda: audit_finite_stochastic_kernel(
                ((1.0, 0.0),), (0.1,), probability_tolerance=True
            ),
            lambda: audit_finite_stochastic_kernel(
                ((0.0, 0.0),), (0.1,), probability_tolerance=1.0
            ),
            lambda: check_finite_stochastic_completeness(
                ((1.0,),),
                (0.1,),
                completeness_margin=-0.1,
            ),
            lambda: check_finite_stochastic_completeness(
                ((1.0,),),
                (0.1,),
                completeness_margin=0.0,
                strict=1,  # type: ignore[arg-type]
            ),
            lambda: postprocess_finite_stochastic_kernel(
                ((1.0, 0.0),), ((1.0, 0.0),)
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

        with self.assertRaisesRegex(ValueError, "without changing positive support"):
            postprocess_finite_stochastic_kernel(
                ((math.ulp(0.0), 1.0),),
                ((math.ulp(0.0), 1.0), (0.0, 1.0)),
            )

        exponent = 538
        small = 2.0**-exponent
        exact_row = (small,) + tuple(2.0**-power for power in range(1, exponent + 1))
        other_transition_row = (0.0, 1.0) + (0.0,) * (len(exact_row) - 2)
        transition = (exact_row,) + (other_transition_row,) * (len(exact_row) - 1)
        with self.assertRaisesRegex(ValueError, "underflowed"):
            postprocess_finite_stochastic_kernel((exact_row,), transition)
        exact_check = check_finite_stochastic_data_processing(
            (exact_row, exact_row),
            transition,
            (0.5, -0.5),
        )
        self.assertTrue(exact_check.exact_conflicts_preserved)
        self.assertTrue(exact_check.total_variation_contracted)
        self.assertTrue(exact_check.monotone)

        with self.assertRaisesRegex(ValueError, "exactly representable"):
            audit_finite_stochastic_kernel(
                ((Fraction(1, 10**400), 1.0), (0.0, 1.0)),
                (0.5, -0.5),
            )
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            audit_finite_stochastic_kernel(
                ((1.0,), (1.0,)),
                (0.5, Fraction(-1, 10**400)),
            )


if __name__ == "__main__":
    unittest.main()
