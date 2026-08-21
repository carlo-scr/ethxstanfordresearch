from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.metrics.action import (  # noqa: E402
    audit_exact_action_fibers,
    audit_radius_action_neighborhoods,
)


class ActionFiberTests(unittest.TestCase):
    def test_shared_code_with_disjoint_safe_actions_is_conflict(self) -> None:
        audit = audit_exact_action_fibers(
            latents=((0.0,), (0.0,)),
            action_safety_margins=((1.0, -0.25), (-0.5, 1.0)),
        )
        self.assertEqual(audit.conflicting_fiber_count, 1)
        self.assertAlmostEqual(audit.worst_required_violation, 0.25)

    def test_distinct_codes_allow_distinct_actions(self) -> None:
        audit = audit_exact_action_fibers(
            latents=((0.0,), (1.0,)),
            action_safety_margins=((1.0, -0.25), (-0.5, 1.0)),
        )
        self.assertEqual(audit.nontrivial_fiber_count, 0)
        self.assertEqual(audit.conflicting_fiber_count, 0)

    def test_shared_code_with_common_action_is_not_conflict(self) -> None:
        audit = audit_exact_action_fibers(
            latents=((0.0,), (0.0,)),
            action_safety_margins=((1.0, -0.25), (0.2, 1.0)),
        )
        self.assertEqual(audit.individually_viable_fiber_count, 1)
        self.assertEqual(audit.conflicting_fiber_count, 0)

    def test_radius_neighborhood_recovers_known_conflict(self) -> None:
        audit = audit_radius_action_neighborhoods(
            [(0.0,), (0.05,), (2.0,)],
            [(1.0, -0.25), (-0.5, 0.75), (1.0, 1.0)],
            delta=0.05,
        )
        self.assertEqual(audit.conflicting_neighborhood_count, 2)
        self.assertAlmostEqual(audit.worst_required_violation, 0.25)
        self.assertAlmostEqual(audit.mean_required_violation, 0.25)
        self.assertAlmostEqual(audit.tail_required_violation, 0.25)
        self.assertEqual(audit.center_required_violations, (0.25, 0.25, None))
        self.assertEqual({witness.center_index for witness in audit.witnesses}, {0, 1})

    def test_neighborhood_detects_higher_order_conflict(self) -> None:
        # Every pair has a common action; all three together do not.
        audit = audit_radius_action_neighborhoods(
            [(0.0,), (0.0,), (0.0,)],
            [(1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)],
            delta=0.0,
        )
        self.assertEqual(audit.conflicting_neighborhood_count, 3)
        self.assertEqual(audit.worst_required_violation, 1.0)
        self.assertEqual(audit.center_required_violations, (1.0, 1.0, 1.0))

    def test_nonviable_neighborhood_is_not_attributed_to_aliasing(self) -> None:
        audit = audit_radius_action_neighborhoods(
            [(0.0,), (0.0,)],
            [(-1.0, -0.5), (1.0, -1.0)],
            delta=0.0,
        )
        self.assertEqual(audit.individually_viable_neighborhood_count, 0)
        self.assertEqual(audit.conflicting_neighborhood_count, 0)
        self.assertEqual(audit.center_required_violations, (None, None))


if __name__ == "__main__":
    unittest.main()
