from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.profile_protocol import (  # noqa: E402
    COVERAGE_BUNDLES_PER_DOMAIN_SEED,
    ProfileProtocolError,
    assign_coverage_teachers,
    assign_trajectory_folds,
    build_crossfit_manifest,
    evaluate_profile_gate,
    nearest_rank,
    teacher_seed,
)


class PredictedProfileProtocolTests(unittest.TestCase):
    def test_fold_assignment_is_order_invariant_and_balanced(self) -> None:
        ids = [f"trajectory-{index:03d}" for index in range(23)]
        first = assign_trajectory_folds(ids)
        second = assign_trajectory_folds(tuple(reversed(ids)))
        self.assertEqual(first, second)
        counts = [sum(fold == index for fold in first.values()) for index in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(
            [teacher_seed(107, fold) for fold in range(5)],
            list(range(11070, 11075)),
        )

    def test_coverage_assignment_uses_sorted_index_modulo_five(self) -> None:
        ids = [f"bundle-{index:03d}" for index in reversed(range(10))]
        assignment = assign_coverage_teachers(ids)
        self.assertEqual(assignment["bundle-000"], 0)
        self.assertEqual(assignment["bundle-004"], 4)
        self.assertEqual(assignment["bundle-005"], 0)

    def test_nearest_rank_uses_the_registered_index(self) -> None:
        values = list(range(1, 201))
        self.assertEqual(nearest_rank(values, 0.95), 190.0)

    def test_profile_gate_passes_at_boundary_and_reports_sign_errors(self) -> None:
        predictions = {}
        targets = {}
        for index in range(COVERAGE_BUNDLES_PER_DOMAIN_SEED):
            bundle = f"bundle-{index:03d}"
            targets[bundle] = (-0.02, 0.03, 0.08)
            predictions[bundle] = (-0.01, 0.03, 0.08)
        # Exactly 10% of the physical margin scale in every bundle.
        result = evaluate_profile_gate(
            predictions,
            targets,
            margin_scale=0.10,
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.normalized_p95_error, 0.10)
        self.assertEqual(result.false_safe_count, 0)
        self.assertEqual(result.unsafe_target_count, 200)

        predictions["bundle-000"] = (0.01, 0.03, 0.08)
        sign_result = evaluate_profile_gate(
            predictions,
            targets,
            margin_scale=0.10,
        )
        self.assertEqual(sign_result.false_safe_count, 1)
        self.assertAlmostEqual(sign_result.false_safe_rate, 1 / 200)

    def test_profile_gate_fails_at_registered_tail(self) -> None:
        predictions = {}
        targets = {}
        for index in range(200):
            bundle = f"bundle-{index:03d}"
            targets[bundle] = (0.0, 0.0)
            # Eleven large errors put the nearest-rank p95 above the threshold.
            predictions[bundle] = (0.02 if index < 11 else 0.0, 0.0)
        result = evaluate_profile_gate(predictions, targets, margin_scale=0.10)
        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.normalized_p95_error, 0.20)

    def test_manifest_is_deterministic_and_complete(self) -> None:
        kwargs = {
            "task": "controlled_cart_video",
            "model_family": "ae",
            "data_seed": 100,
            "training_trajectory_ids": [f"cart-{index:03d}" for index in range(20)],
            "action_grid": (-1.0, 0.0, 1.0),
            "horizon": 4,
            "margin_scale": 0.25,
            "inherited_config_sha256": "a" * 64,
        }
        first = build_crossfit_manifest(**kwargs)
        second = build_crossfit_manifest(**kwargs)
        self.assertEqual(first, second)
        payload, digest = first
        self.assertEqual(len(digest), 64)
        self.assertEqual(payload["teacher_seeds"]["0"], 11000)
        self.assertEqual(sum(len(ids) for ids in payload["folds"].values()), 20)

    def test_invalid_or_incomplete_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProfileProtocolError, "unique"):
            assign_trajectory_folds(["same", "same"])
        with self.assertRaisesRegex(ProfileProtocolError, "match exactly"):
            evaluate_profile_gate(
                {"a": (0.0,)},
                {"b": (0.0,)},
                margin_scale=1.0,
            )
        with self.assertRaisesRegex(ProfileProtocolError, "every cross-fit fold"):
            build_crossfit_manifest(
                task="controlled_cart_video",
                model_family="ae",
                data_seed=0,
                training_trajectory_ids=("a", "b", "c", "d"),
                action_grid=(-1.0, 1.0),
                horizon=4,
                margin_scale=0.25,
                inherited_config_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
