from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
import tomllib
import unittest
from collections.abc import Callable
from itertools import product
from pathlib import Path

from latent_safety.analysis import (
    FROZEN_CONFIRMATORY_SPEC,
    ConfirmatoryObservation,
    run_confirmatory_inference,
)
from latent_safety.analysis.confirmatory import (
    COMPARATORS,
    DOMAINS,
    MODEL_FAMILIES,
    PAIRED_SEEDS,
    PROPOSED_ARM,
    _percentile_upper_bound,
)


ROOT = Path(__file__).resolve().parents[1]


def _complete_observations() -> list[ConfirmatoryObservation]:
    observations: list[ConfirmatoryObservation] = []
    for domain, family, arm, seed in product(
        DOMAINS,
        MODEL_FAMILIES,
        (PROPOSED_ARM, *COMPARATORS),
        PAIRED_SEEDS,
    ):
        proposed = arm == PROPOSED_ARM
        observations.append(
            ConfirmatoryObservation(
                domain=domain,
                model_family=family,
                arm=arm,
                seed=seed,
                safety=0.30 if proposed else 0.60,
                reconstruction=1.02 if proposed else 1.00,
                rollout=2.04 if proposed else 2.00,
                run_complete=True,
                coverage_complete=True,
            )
        )
    return observations


def _replace_matching(
    observations: list[ConfirmatoryObservation],
    predicate: Callable[[ConfirmatoryObservation], bool],
    **changes: object,
) -> list[ConfirmatoryObservation]:
    return [
        dataclasses.replace(observation, **changes)
        if predicate(observation)
        else observation
        for observation in observations
    ]


class FrozenConfirmatoryInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.observations = _complete_observations()
        cls.result = run_confirmatory_inference(cls.observations)

    def test_exact_factorial_and_all_passing_constant_differences(self) -> None:
        result = self.result

        self.assertEqual(result.input_observation_count, 240)
        self.assertEqual(result.expected_observation_count, 240)
        self.assertEqual(len(result.bounds), 36)
        self.assertEqual(result.safety_bound_count, 12)
        self.assertEqual(result.utility_bound_count, 24)
        self.assertEqual(result.passed_bound_count, 36)
        self.assertTrue(result.all_elementary_bounds_passed)
        self.assertTrue(result.confirmatory_success)
        self.assertTrue(all(gate.all_bounds_passed for gate in result.domain_gates))

        safety = next(bound for bound in result.bounds if bound.endpoint == "safety")
        reconstruction = next(
            bound for bound in result.bounds if bound.endpoint == "reconstruction"
        )
        self.assertAlmostEqual(safety.point_mean or 0.0, -0.30)
        self.assertAlmostEqual(safety.upper_confidence_bound or 0.0, -0.30)
        self.assertAlmostEqual(reconstruction.point_mean or 0.0, 0.02)
        self.assertAlmostEqual(reconstruction.upper_confidence_bound or 0.0, 0.02)
        self.assertEqual(len(safety.seed_differences), 8)

    def test_result_is_json_compatible_and_records_the_fixed_method(self) -> None:
        payload = self.result.to_dict()
        serialized = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertIn('"elementary_bound_count": 36', serialized)
        self.assertEqual(payload["spec"]["bootstrap_resamples"], 100_000)
        self.assertEqual(payload["spec"]["bootstrap_seed"], 20_260_822)
        self.assertAlmostEqual(
            payload["spec"]["ucb_probability"],
            1.0 - 0.05 / 36.0,
        )

    def test_equal_family_average_is_applied_inside_each_seed(self) -> None:
        observations = _replace_matching(
            self.observations,
            lambda row: (
                row.domain == DOMAINS[0]
                and row.model_family == "ae"
                and row.arm == PROPOSED_ARM
                and row.seed == PAIRED_SEEDS[0]
            ),
            safety=0.20,
        )
        observations = _replace_matching(
            observations,
            lambda row: (
                row.domain == DOMAINS[0]
                and row.model_family == "beta_vae"
                and row.arm == PROPOSED_ARM
                and row.seed == PAIRED_SEEDS[0]
            ),
            safety=0.40,
        )
        result = run_confirmatory_inference(observations)
        bound = next(
            item
            for item in result.bounds
            if item.domain == DOMAINS[0]
            and item.comparator == "none"
            and item.endpoint == "safety"
        )
        first_seed = bound.seed_differences[0]

        self.assertAlmostEqual(first_seed.family_differences[0].value or 0.0, -0.40)
        self.assertAlmostEqual(first_seed.family_differences[1].value or 0.0, -0.20)
        self.assertAlmostEqual(first_seed.equal_family_mean or 0.0, -0.30)

    def test_safety_requires_both_point_improvement_and_strict_negative_ucb(self) -> None:
        observations = _replace_matching(
            self.observations,
            lambda row: row.arm != PROPOSED_ARM,
            safety=0.35,
        )
        result = run_confirmatory_inference(observations)
        safety_bounds = [bound for bound in result.bounds if bound.endpoint == "safety"]

        self.assertTrue(all(bound.ucb_gate_passed for bound in safety_bounds))
        self.assertTrue(all(bound.point_gate_passed is False for bound in safety_bounds))
        self.assertTrue(all(not bound.passed for bound in safety_bounds))
        self.assertFalse(result.confirmatory_success)

    def test_zero_denominator_tie_is_zero_and_positive_over_zero_fails_closed(self) -> None:
        tied = _replace_matching(
            self.observations,
            lambda row: row.arm in {PROPOSED_ARM, "none"},
            reconstruction=0.0,
        )
        tied_result = run_confirmatory_inference(tied)
        tied_bound = next(
            bound
            for bound in tied_result.bounds
            if bound.comparator == "none" and bound.endpoint == "reconstruction"
        )
        self.assertEqual(tied_bound.point_mean, 0.0)
        self.assertEqual(tied_bound.upper_confidence_bound, 0.0)
        self.assertTrue(tied_bound.passed)
        self.assertTrue(
            all(
                family.status == "tied_zero_denominator"
                for seed in tied_bound.seed_differences
                for family in seed.family_differences
            )
        )

        inadmissible = _replace_matching(
            self.observations,
            lambda row: row.arm == "none",
            reconstruction=0.0,
        )
        inadmissible_result = run_confirmatory_inference(inadmissible)
        inadmissible_bound = next(
            bound
            for bound in inadmissible_result.bounds
            if bound.comparator == "none" and bound.endpoint == "reconstruction"
        )
        self.assertIsNone(inadmissible_bound.point_mean)
        self.assertIsNone(inadmissible_bound.upper_confidence_bound)
        self.assertFalse(inadmissible_bound.passed)
        self.assertTrue(
            all(
                "positive_over_zero_denominator" in reason
                for reason in inadmissible_bound.failure_reasons
            )
        )
        json.dumps(inadmissible_result.to_dict(), allow_nan=False)

    def test_partial_duplicate_failed_and_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete confirmatory factorial"):
            run_confirmatory_inference(self.observations[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate confirmatory observation"):
            run_confirmatory_inference([*self.observations, self.observations[0]])

        failed = [
            dataclasses.replace(self.observations[0], run_complete=False),
            *self.observations[1:],
        ]
        with self.assertRaisesRegex(ValueError, "run is not complete"):
            run_confirmatory_inference(failed)

        uncovered = [
            dataclasses.replace(self.observations[0], coverage_complete=False),
            *self.observations[1:],
        ]
        with self.assertRaisesRegex(ValueError, "metric coverage is incomplete"):
            run_confirmatory_inference(uncovered)

        invalid = [
            dataclasses.replace(self.observations[0], safety=math.nan),
            *self.observations[1:],
        ]
        with self.assertRaisesRegex(ValueError, "must be finite"):
            run_confirmatory_inference(invalid)

    def test_frozen_spec_matches_the_checked_in_config(self) -> None:
        with (ROOT / "configs/e2_frontier/confirmatory_core.toml").open("rb") as stream:
            config = tomllib.load(stream)
        inference = config["inference"]

        self.assertEqual(inference["proposed_arm"], FROZEN_CONFIRMATORY_SPEC.proposed_arm)
        self.assertEqual(tuple(inference["domains"]), FROZEN_CONFIRMATORY_SPEC.domains)
        self.assertEqual(tuple(inference["comparators"]), FROZEN_CONFIRMATORY_SPEC.comparators)
        self.assertEqual(tuple(inference["endpoints"]), FROZEN_CONFIRMATORY_SPEC.endpoints)
        self.assertEqual(tuple(inference["paired_seeds"]), FROZEN_CONFIRMATORY_SPEC.paired_seeds)
        self.assertEqual(
            inference["bootstrap_resamples"], FROZEN_CONFIRMATORY_SPEC.bootstrap_resamples
        )
        self.assertEqual(inference["bootstrap_seed"], FROZEN_CONFIRMATORY_SPEC.bootstrap_seed)
        self.assertEqual(
            inference["elementary_bound_count"],
            FROZEN_CONFIRMATORY_SPEC.elementary_bound_count,
        )
        self.assertEqual(inference["family_alpha"], FROZEN_CONFIRMATORY_SPEC.family_alpha)
        self.assertAlmostEqual(
            inference["primary_ucb_quantile"], FROZEN_CONFIRMATORY_SPEC.ucb_probability
        )
        self.assertFalse(inference["studentized"])
        self.assertEqual(
            config["inference"]["safety"]["point_mean_max"],
            FROZEN_CONFIRMATORY_SPEC.safety_point_mean_max,
        )
        self.assertEqual(
            config["inference"]["safety"]["ucb_max"],
            FROZEN_CONFIRMATORY_SPEC.safety_ucb_max,
        )
        self.assertEqual(
            config["inference"]["utility"]["ucb_max"],
            FROZEN_CONFIRMATORY_SPEC.utility_ucb_max,
        )


class CompressedBootstrapTests(unittest.TestCase):
    def test_compressed_counts_match_direct_fixed_seed_percentile_bootstrap(self) -> None:
        values = (-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20)
        probability = FROZEN_CONFIRMATORY_SPEC.ucb_probability
        resamples = FROZEN_CONFIRMATORY_SPEC.bootstrap_resamples
        seed = FROZEN_CONFIRMATORY_SPEC.bootstrap_seed
        generator = random.Random(seed)
        direct = sorted(
            statistics.fmean(values[generator.randrange(len(values))] for _ in values)
            for _ in range(resamples)
        )
        position = (resamples - 1) * probability
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        weight = position - lower_index
        expected = direct[lower_index] * (1.0 - weight) + direct[upper_index] * weight

        actual = _percentile_upper_bound(
            values,
            probability=probability,
            resamples=resamples,
            seed=seed,
        )

        self.assertAlmostEqual(actual, expected, places=15)


if __name__ == "__main__":
    unittest.main()
