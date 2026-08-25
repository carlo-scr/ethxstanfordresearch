from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.foundation.audit import (  # noqa: E402
    audit_common_action_neighborhoods,
    audit_common_action_records,
)
from latent_safety.records import AuditRecord  # noqa: E402


class FoundationCommonActionAuditTests(unittest.TestCase):
    def test_nonviable_point_cannot_hide_viable_conflict(self) -> None:
        audit = audit_common_action_neighborhoods(
            ((0.0,), (0.0,), (0.0,)),
            ((-1.0, -1.0), (1.0, -0.5), (-0.25, 1.0)),
            delta=0.0,
            sample_ids=("nonviable", "left", "right"),
            trajectory_ids=("trajectory", "trajectory", "trajectory"),
        )
        self.assertEqual(audit.viable_sample_count, 2)
        self.assertEqual(audit.viable_fraction, 2 / 3)
        self.assertEqual(audit.eligible_center_count, 2)
        self.assertEqual(audit.eligible_given_viable, 1.0)
        self.assertEqual(audit.conflicting_center_count, 2)
        self.assertEqual(audit.center_required_violations, (None, 0.25, 0.25))
        self.assertEqual(audit.status, "ok")
        self.assertEqual(audit.trajectory_balanced_tail_required_violation, 0.25)

    def test_higher_order_conflict_keeps_one_action_minimum_per_set(self) -> None:
        audit = audit_common_action_neighborhoods(
            ((0.0,), (0.0,), (0.0,)),
            ((1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            delta=0.0,
        )
        self.assertEqual(audit.conflicting_center_count, 3)
        self.assertEqual(audit.worst_required_violation, 1.0)
        self.assertEqual(audit.center_best_actions, (0, 0, 0))

    def test_gamma_boundary_is_viable_and_nonviable_members_are_removed_first(self) -> None:
        audit = audit_common_action_neighborhoods(
            ((0.0,), (0.0,), (0.0,)),
            ((0.5, -1.0), (-1.0, 0.5), (0.49, 0.49)),
            delta=0.0,
            gamma=0.5,
        )
        self.assertEqual(audit.viable_sample_count, 2)
        self.assertEqual(audit.eligible_center_count, 2)
        self.assertAlmostEqual(audit.worst_required_violation or 0.0, 1.5)

    def test_missing_trajectory_coverage_is_none_not_zero(self) -> None:
        audit = audit_common_action_neighborhoods(
            ((0.0,), (0.0,), (10.0,)),
            ((1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
            delta=0.0,
            trajectory_ids=("paired", "paired", "isolated"),
        )
        self.assertEqual(audit.status, "trajectory_coverage_incomplete")
        self.assertEqual(audit.empty_trajectory_ids, ("isolated",))
        self.assertIsNone(audit.trajectory_balanced_tail_required_violation)

    def test_record_bridge_sorts_ids_and_preserves_traceable_witnesses(self) -> None:
        records = (
            AuditRecord(
                sample_id="b",
                trajectory_id="trajectory",
                split="validation",
                safety_margin=0.1,
                latent=(0.0,),
                action_safety_margins=(-0.25, 1.0),
            ),
            AuditRecord(
                sample_id="a",
                trajectory_id="trajectory",
                split="validation",
                safety_margin=0.1,
                latent=(0.0,),
                action_safety_margins=(1.0, -0.5),
            ),
        )
        audit = audit_common_action_records(records, delta=0.0)
        self.assertEqual(audit.witnesses[0].center_sample_id, "a")
        self.assertEqual(audit.witnesses[0].member_sample_ids, ("a", "b"))

    def test_group_mask_prevents_unintended_cross_group_edges(self) -> None:
        audit = audit_common_action_neighborhoods(
            ((0.0,), (0.0,)),
            ((1.0, -1.0), (-1.0, 1.0)),
            delta=0.0,
            comparison_group_ids=("scene-a", "scene-b"),
        )
        self.assertEqual(audit.status, "no_eligible_centers")
        self.assertIsNone(audit.worst_required_violation)
        self.assertEqual(audit.eligible_given_viable, 0.0)

    def test_gamma_is_a_nonnegative_buffer(self) -> None:
        with self.assertRaisesRegex(ValueError, "gamma must be nonnegative"):
            audit_common_action_neighborhoods(
                ((0.0,), (0.0,)),
                ((1.0, -1.0), (-1.0, 1.0)),
                delta=0.0,
                gamma=-0.01,
            )

    def test_tiny_radius_does_not_receive_a_fixed_absolute_tolerance(self) -> None:
        radius = 1e-15
        audit = audit_common_action_neighborhoods(
            ((0.0,), (radius * 10.0,)),
            ((1.0, -1.0), (-1.0, 1.0)),
            delta=radius,
        )
        self.assertEqual(audit.status, "no_eligible_centers")
        self.assertEqual(audit.center_member_counts, (1, 1))


if __name__ == "__main__":
    unittest.main()
