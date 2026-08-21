import math
import unittest

from latent_safety.analysis import (
    BLOCK_INDEPENDENCE_CAVEAT,
    VALIDATION_SELECTION_CAVEAT,
    HypothesisPValue,
    PairedBlock,
    ValidationParetoFrontier,
    ValidationScore,
    exact_paired_sign_flip_test,
    holm_adjust,
    paired_block_bootstrap_ci,
    select_under_safety_budget,
    standardized_paired_effect,
    validation_pareto_frontier,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_pairing_is_preserved_and_constant_difference_collapses_interval(self) -> None:
        blocks = [
            PairedBlock("seed-0", baseline=100.0, candidate=101.0),
            PairedBlock("seed-1", baseline=-50.0, candidate=-49.0),
            PairedBlock("seed-2", baseline=0.25, candidate=1.25),
        ]
        result = paired_block_bootstrap_ci(blocks, resamples=500, seed=17)

        self.assertEqual(result.estimate, 1.0)
        self.assertEqual(result.lower, 1.0)
        self.assertEqual(result.upper, 1.0)
        self.assertEqual(result.n_blocks, 3)
        self.assertEqual(result.difference_convention, "candidate - baseline")

    def test_bootstrap_is_exactly_reproducible_for_a_fixed_seed(self) -> None:
        blocks = [
            PairedBlock(f"seed-{index}", 0.0, value)
            for index, value in enumerate((0.0, 0.5, 1.5, 4.0, 9.0))
        ]
        first = paired_block_bootstrap_ci(blocks, resamples=1_000, seed=90210)
        second = paired_block_bootstrap_ci(blocks, resamples=1_000, seed=90210)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, 3.0)
        self.assertLessEqual(min(block.candidate for block in blocks), first.lower)
        self.assertLessEqual(first.lower, first.estimate)
        self.assertLessEqual(first.estimate, first.upper)
        self.assertLessEqual(first.upper, max(block.candidate for block in blocks))

    def test_median_statistic_is_supported_and_labeled(self) -> None:
        blocks = [
            PairedBlock("a", 0.0, 0.0),
            PairedBlock("b", 0.0, 1.0),
            PairedBlock("c", 0.0, 100.0),
        ]
        result = paired_block_bootstrap_ci(
            blocks, statistic="median", resamples=500, seed=4
        )
        self.assertEqual(result.statistic, "median")
        self.assertEqual(result.estimate, 1.0)

    def test_invalid_units_and_parameters_fail_closed(self) -> None:
        valid = [PairedBlock("a", 0.0, 1.0), PairedBlock("b", 0.0, 2.0)]
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci([])
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci([valid[0]])
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci([valid[0], PairedBlock("a", 2.0, 3.0)])
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(
                [valid[0], PairedBlock("bad", math.nan, 0.0)]
            )
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(valid, confidence_level=1.0)
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(valid, resamples=99)
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(valid, seed=True)
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(valid, statistic="trimmed_mean")  # type: ignore[arg-type]

    def test_difference_overflow_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_block_bootstrap_ci(
                [
                    PairedBlock("a", -1.0e308, 1.0e308),
                    PairedBlock("b", 0.0, 0.0),
                ]
            )

    def test_public_caveat_rules_out_frame_pseudoreplication(self) -> None:
        self.assertIn("Never enter correlated frames", BLOCK_INDEPENDENCE_CAVEAT)


class StandardizedPairedEffectTests(unittest.TestCase):
    def test_cohen_dz_uses_sample_sd_of_paired_differences(self) -> None:
        blocks = [
            PairedBlock("a", 10.0, 11.0),
            PairedBlock("b", 20.0, 22.0),
            PairedBlock("c", 30.0, 33.0),
        ]
        result = standardized_paired_effect(blocks)

        self.assertTrue(result.is_defined)
        self.assertEqual(result.status, "defined")
        self.assertAlmostEqual(result.mean_difference, 2.0)
        self.assertAlmostEqual(result.sample_sd_difference, 1.0)
        self.assertAlmostEqual(result.value or 0.0, 2.0)

    def test_zero_variance_zero_difference_is_explicitly_undefined(self) -> None:
        result = standardized_paired_effect(
            [PairedBlock("a", 1.0, 1.0), PairedBlock("b", -2.0, -2.0)]
        )
        self.assertIsNone(result.value)
        self.assertFalse(result.is_defined)
        self.assertEqual(result.status, "zero_variance_zero_difference")
        self.assertEqual(result.mean_difference, 0.0)

    def test_zero_variance_nonzero_difference_is_explicitly_undefined(self) -> None:
        result = standardized_paired_effect(
            [PairedBlock("a", 1.0, 0.5), PairedBlock("b", 7.0, 6.5)]
        )
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "zero_variance_constant_difference")
        self.assertEqual(result.mean_difference, -0.5)


class ExactPairedSignFlipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = [
            PairedBlock(f"seed-{index}", baseline=1.0, candidate=0.0)
            for index in range(3)
        ]

    def test_exact_one_and_two_sided_probabilities(self) -> None:
        justification = "paired method labels are exchangeable under the predeclared sharp null"
        one_sided = exact_paired_sign_flip_test(
            self.blocks,
            alternative="candidate_less",
            exchangeability_justification=justification,
        )
        two_sided = exact_paired_sign_flip_test(
            self.blocks,
            alternative="two-sided",
            exchangeability_justification=justification,
        )

        self.assertEqual(one_sided.estimate, -1.0)
        self.assertEqual(one_sided.permutation_count, 8)
        self.assertEqual(one_sided.p_value, 1.0 / 8.0)
        self.assertEqual(two_sided.p_value, 2.0 / 8.0)
        self.assertEqual(one_sided.exchangeability_justification, justification)

    def test_zero_differences_have_unit_p_value(self) -> None:
        result = exact_paired_sign_flip_test(
            [PairedBlock("a", 1.0, 1.0), PairedBlock("b", -2.0, -2.0)],
            exchangeability_justification="paired labels are exchangeable under the sharp null",
        )
        self.assertEqual(result.p_value, 1.0)

    def test_assumptions_and_computation_limit_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exchangeability_justification"):
            exact_paired_sign_flip_test(
                self.blocks,
                exchangeability_justification="",
            )
        with self.assertRaisesRegex(ValueError, "alternative"):
            exact_paired_sign_flip_test(
                self.blocks,
                alternative="less",  # type: ignore[arg-type]
                exchangeability_justification="predeclared sharp-null exchangeability",
            )
        with self.assertRaisesRegex(ValueError, "at most 20"):
            exact_paired_sign_flip_test(
                [PairedBlock(f"seed-{index}", 0.0, 1.0) for index in range(21)],
                exchangeability_justification="predeclared sharp-null exchangeability",
            )


class HolmCorrectionTests(unittest.TestCase):
    def test_known_family_adjustment_and_step_down_decisions(self) -> None:
        results = holm_adjust(
            [
                HypothesisPValue("h-medium", 0.04),
                HypothesisPValue("h-small", 0.01),
                HypothesisPValue("h-middle", 0.03),
            ],
            alpha=0.05,
        )
        by_id = {result.hypothesis_id: result for result in results}

        self.assertEqual(
            [result.hypothesis_id for result in results],
            ["h-medium", "h-small", "h-middle"],
        )
        self.assertAlmostEqual(by_id["h-small"].adjusted_p_value, 0.03)
        self.assertAlmostEqual(by_id["h-middle"].adjusted_p_value, 0.06)
        self.assertAlmostEqual(by_id["h-medium"].adjusted_p_value, 0.06)
        self.assertTrue(by_id["h-small"].rejected)
        self.assertFalse(by_id["h-middle"].rejected)
        self.assertFalse(by_id["h-medium"].rejected)

    def test_ties_are_deterministically_ranked_by_identifier(self) -> None:
        results = holm_adjust(
            [HypothesisPValue("z", 0.01), HypothesisPValue("a", 0.01)]
        )
        by_id = {result.hypothesis_id: result for result in results}
        self.assertEqual(by_id["a"].rank, 1)
        self.assertEqual(by_id["z"].rank, 2)
        self.assertTrue(all(result.rejected for result in results))

    def test_invalid_families_and_p_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            holm_adjust([])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("", 0.1)])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("h", math.inf)])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("h", -0.01)])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("h", 1.01)])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("h", 0.1), HypothesisPValue("h", 0.2)])
        with self.assertRaises(ValueError):
            holm_adjust([HypothesisPValue("h", 0.1)], alpha=0.0)


class ValidationParetoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = [
            ValidationScore("safe", safety_defect=0.10, utility_loss=0.50),
            ValidationScore("balanced", safety_defect=0.20, utility_loss=0.30),
            ValidationScore("dominated", safety_defect=0.30, utility_loss=0.60),
            ValidationScore("safe-tie", safety_defect=0.10, utility_loss=0.50),
            ValidationScore("utility", safety_defect=0.40, utility_loss=0.20),
        ]

    def test_frontier_minimizes_both_losses_and_keeps_exact_ties(self) -> None:
        frontier = validation_pareto_frontier(reversed(self.scores))

        self.assertEqual(
            [point.config_id for point in frontier.points],
            ["safe", "safe-tie", "balanced", "utility"],
        )
        self.assertEqual(frontier.dominated_config_ids, ("dominated",))
        self.assertEqual(frontier.candidate_count, 5)
        self.assertEqual(frontier.selection_split, "validation")

    def test_budget_selection_uses_predeclared_deterministic_tiebreakers(self) -> None:
        frontier = validation_pareto_frontier(self.scores)
        selected = select_under_safety_budget(frontier, max_safety_defect=0.20)
        self.assertEqual(selected.config_id, "balanced")

        tied = validation_pareto_frontier(
            [
                ValidationScore("z", 0.1, 0.2),
                ValidationScore("a", 0.1, 0.2),
            ]
        )
        self.assertEqual(
            select_under_safety_budget(tied, max_safety_defect=0.1).config_id,
            "a",
        )

    def test_budget_with_no_eligible_candidate_fails(self) -> None:
        frontier = validation_pareto_frontier(self.scores)
        with self.assertRaises(ValueError):
            select_under_safety_budget(frontier, max_safety_defect=0.01)
        with self.assertRaises(ValueError):
            select_under_safety_budget(frontier, max_safety_defect=-0.1)

    def test_budget_selection_revalidates_manually_constructed_frontier(self) -> None:
        malformed = ValidationParetoFrontier(
            points=(ValidationScore("candidate", 0.1, math.nan),),
            dominated_config_ids=(),
            candidate_count=1,
        )
        with self.assertRaises(ValueError):
            select_under_safety_budget(malformed, max_safety_defect=0.2)

    def test_invalid_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            validation_pareto_frontier([])
        with self.assertRaises(ValueError):
            validation_pareto_frontier(
                [ValidationScore("same", 0.1, 0.2), ValidationScore("same", 0.2, 0.1)]
            )
        with self.assertRaises(ValueError):
            validation_pareto_frontier([ValidationScore("negative", -0.1, 0.2)])
        with self.assertRaises(ValueError):
            validation_pareto_frontier([ValidationScore("nan", 0.1, math.nan)])
        with self.assertRaises(ValueError):
            validation_pareto_frontier([ValidationScore(" ", 0.1, 0.2)])

    def test_public_caveat_bans_final_test_selection(self) -> None:
        self.assertIn("Final-test safety labels must not select", VALIDATION_SELECTION_CAVEAT)


if __name__ == "__main__":
    unittest.main()
