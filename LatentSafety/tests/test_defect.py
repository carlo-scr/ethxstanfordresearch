from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.metrics.defect import (  # noqa: E402
    empirical_defect_curve,
    empirical_robust_defect,
    empirical_robust_defect_details,
    nearest_unsafe_diagnostic,
)
from latent_safety.synthetic import make_aliasing_fixture  # noqa: E402


class DefectMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_aliasing_fixture(n_pairs=16, max_margin=0.8, seed=11)

    def test_injective_control_has_zero_exact_sample_witness(self) -> None:
        estimate = empirical_robust_defect(
            self.fixture.faithful_latents,
            self.fixture.margins,
            delta=0.0,
        )
        self.assertEqual(estimate.witness_margin, 0.0)
        self.assertEqual(estimate.confounded_safe_count, 0)

    def test_known_collisions_recover_known_margin(self) -> None:
        estimate = empirical_robust_defect(
            self.fixture.collapsed_latents,
            self.fixture.margins,
            delta=0.0,
        )
        self.assertAlmostEqual(estimate.witness_margin, 0.8)
        self.assertEqual(estimate.confounded_safe_count, 16)

    def test_curve_is_monotone(self) -> None:
        curve = empirical_defect_curve(
            self.fixture.faithful_latents,
            self.fixture.margins,
            deltas=(0.0, 0.05, 0.2, 1.0),
        )
        self.assertEqual(
            [item.witness_margin for item in curve],
            sorted(item.witness_margin for item in curve),
        )

    def test_radius_and_latent_rescaling_agree(self) -> None:
        original = empirical_robust_defect(
            self.fixture.faithful_latents,
            self.fixture.margins,
            delta=0.2,
        )
        scale = 7.0
        rescaled = tuple(
            tuple(scale * value for value in vector)
            for vector in self.fixture.faithful_latents
        )
        transformed = empirical_robust_defect(
            rescaled,
            self.fixture.margins,
            delta=scale * 0.2,
        )
        self.assertEqual(original.witness_margin, transformed.witness_margin)
        self.assertEqual(original.confounded_safe_count, transformed.confounded_safe_count)
        self.assertEqual(original.confounded_fraction, transformed.confounded_fraction)

    def test_nearest_neighbor_output_is_explicitly_diagnostic(self) -> None:
        latents = ((-0.1,), (0.1,), (10.0,))
        margins = (-0.1, 0.1, 1.0)
        diagnostic = nearest_unsafe_diagnostic(latents, margins)
        self.assertEqual(diagnostic.cross_boundary_nearest_count, 1)
        exact = empirical_robust_defect(latents, margins, delta=0.0)
        self.assertEqual(exact.witness_margin, 0.0)
        self.assertTrue(math.isclose(diagnostic.max_margin, 0.1))

    def test_detailed_witnesses_are_traceable_and_bounded(self) -> None:
        fixture = make_aliasing_fixture(n_pairs=4, max_margin=1.0, seed=5)
        details = empirical_robust_defect_details(
            fixture.collapsed_latents,
            fixture.margins,
            delta=0.0,
            max_witnesses=2,
        )

        self.assertEqual(details.estimate.confounded_safe_count, 4)
        self.assertEqual(len(details.witnesses), 2)
        self.assertEqual(details.witnesses[0].safe_margin, 1.0)
        for witness in details.witnesses:
            self.assertEqual(
                fixture.collapsed_latents[witness.safe_index],
                fixture.collapsed_latents[witness.unsafe_index],
            )

    def test_witness_limit_does_not_change_summary(self) -> None:
        fixture = make_aliasing_fixture(n_pairs=5, max_margin=0.8, seed=6)
        without = empirical_robust_defect_details(
            fixture.collapsed_latents,
            fixture.margins,
            delta=0.0,
            max_witnesses=0,
        )
        with_all = empirical_robust_defect_details(
            fixture.collapsed_latents,
            fixture.margins,
            delta=0.0,
            max_witnesses=100,
        )
        self.assertEqual(without.estimate, with_all.estimate)
        self.assertEqual(without.witnesses, ())


if __name__ == "__main__":
    unittest.main()
