from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.metrics.action import audit_exact_action_fibers  # noqa: E402
from latent_safety.metrics.defect import empirical_robust_defect  # noqa: E402
from latent_safety.synthetic import (  # noqa: E402
    localized_collision_encoder,
    localized_collision_mse_bound,
    signed_action_margins,
    smooth_unit_bump,
)


class SmoothAverageLossSeparationTests(unittest.TestCase):
    def test_bump_has_compact_support_and_unit_center(self) -> None:
        self.assertEqual(smooth_unit_bump(0.0), 1.0)
        self.assertEqual(smooth_unit_bump(-1.0), 0.0)
        self.assertEqual(smooth_unit_bump(1.0), 0.0)
        self.assertEqual(smooth_unit_bump(3.0), 0.0)

    def test_fixed_collision_survives_as_width_shrinks(self) -> None:
        margin = 0.4
        for width in (0.1, 0.01, 0.001):
            safe_code = localized_collision_encoder(
                margin,
                collision_margin=margin,
                width=width,
            )
            unsafe_code = localized_collision_encoder(
                -margin,
                collision_margin=margin,
                width=width,
            )
            self.assertAlmostEqual(safe_code, 0.0)
            self.assertAlmostEqual(unsafe_code, 0.0)
            defect = empirical_robust_defect(
                ((safe_code,), (unsafe_code,)),
                (margin, -margin),
                delta=0.0,
            )
            self.assertAlmostEqual(defect.witness_margin, margin)

    def test_uniform_reconstruction_mse_obeys_vanishing_analytic_bound(self) -> None:
        margin = 0.4
        width = 0.02
        # Midpoint quadrature is deterministic and sufficiently fine to check the analytic rate.
        count = 200_000
        squared_errors = []
        for index in range(count):
            state = -1.0 + 2.0 * (index + 0.5) / count
            code = localized_collision_encoder(
                state,
                collision_margin=margin,
                width=width,
            )
            squared_errors.append((state - code) ** 2)
        empirical_mse = sum(squared_errors) / count
        bound = localized_collision_mse_bound(
            collision_margin=margin,
            width=width,
        )
        self.assertLessEqual(empirical_mse, bound * (1.0 + 1e-6))
        self.assertLess(
            localized_collision_mse_bound(
                collision_margin=margin,
                width=width / 10.0,
            ),
            bound,
        )

    def test_same_collision_has_disjoint_safe_action_sets(self) -> None:
        margin = 0.4
        code = localized_collision_encoder(
            margin,
            collision_margin=margin,
            width=0.02,
        )
        audit = audit_exact_action_fibers(
            ((code,), (code,)),
            (signed_action_margins(margin), signed_action_margins(-margin)),
        )
        self.assertEqual(audit.conflicting_fiber_count, 1)
        self.assertAlmostEqual(audit.worst_required_violation, margin)

    def test_invalid_localization_parameters_fail_closed(self) -> None:
        for margin, width in ((0.0, 0.1), (1.0, 0.1), (0.4, 0.0), (0.4, 0.4)):
            with self.assertRaises(ValueError):
                localized_collision_encoder(
                    0.0,
                    collision_margin=margin,
                    width=width,
                )


if __name__ == "__main__":
    unittest.main()
