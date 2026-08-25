from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.matched_radius import (  # noqa: E402
    MATCHED_RADIUS_PROTOCOL,
    MATCHED_RADIUS_SCHEMA_VERSION,
    MATCHED_RADIUS_SEMANTICS_VERSION,
    RADIUS_AUDIT_PROTOCOL,
    RADIUS_AUDIT_SCHEMA_VERSION,
    RADIUS_AUDIT_SEMANTICS_VERSION,
    RadiusPoint,
    build_validation_radius_audit,
    match_validation_radius_audits,
    validate_paired_records,
    validation_radius_audit_from_dict,
)
from latent_safety.records import AuditRecord  # noqa: E402


PROFILES = (
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -0.5),
    (-0.5, 1.0),
)


def _records(
    latents: tuple[float, ...],
    *,
    split: str = "validation",
    trajectories: tuple[str, ...] = (
        "trajectory-a",
        "trajectory-a",
        "trajectory-b",
        "trajectory-b",
    ),
) -> tuple[AuditRecord, ...]:
    return tuple(
        AuditRecord(
            sample_id=f"sample-{index}",
            trajectory_id=trajectories[index],
            split=split,
            safety_margin=0.2 - 0.1 * index,
            latent=(latent,),
            action_safety_margins=PROFILES[index],
        )
        for index, latent in enumerate(latents)
    )


def _only_mass_candidate(
    audit,
    *,
    relative_radius: float,
    mass: float,
    eligible_center_count: int | None = None,
):
    curve = []
    for point in audit.curve:
        if point.relative_radius == relative_radius:
            eligible = (
                point.viable_sample_count
                if eligible_center_count is None
                else eligible_center_count
            )
            coverage = eligible / point.viable_sample_count
            curve.append(
                dataclasses.replace(
                    point,
                    eligible_center_count=eligible,
                    eligible_center_coverage=coverage,
                    eligible_given_viable=coverage,
                    median_nonself_neighborhood_mass=mass,
                    evaluable_trajectory_count=audit.trajectory_count,
                    empty_trajectory_count=0,
                    trajectory_balanced_p95_required_violation=0.1,
                )
            )
        else:
            curve.append(
                RadiusPoint(
                    relative_radius=point.relative_radius,
                    absolute_radius=point.absolute_radius,
                    viable_sample_count=point.viable_sample_count,
                    viable_fraction=point.viable_fraction,
                    eligible_center_count=0,
                    eligible_center_coverage=0.0,
                    eligible_given_viable=0.0,
                    median_nonself_neighborhood_mass=None,
                    trajectory_count=audit.trajectory_count,
                    evaluable_trajectory_count=0,
                    empty_trajectory_count=audit.trajectory_count,
                    trajectory_balanced_p95_required_violation=None,
                )
            )
    return dataclasses.replace(audit, curve=tuple(curve))


class MatchedRadiusTests(unittest.TestCase):
    def test_deterministic_mass_match_excludes_self_and_never_uses_safety_to_select(self) -> None:
        radii = (0.0, 0.4, 0.5, 1.0)
        control_records = _records((0.0, 1.0, 3.0, 4.0))
        # The learned geometry preserves the same mass while pairing compatible action profiles.
        learned_records = _records((0.0, 3.0, 1.0, 4.0))
        control = build_validation_radius_audit(control_records, relative_radii=radii)
        learned = build_validation_radius_audit(learned_records, relative_radii=radii)
        reversed_control = build_validation_radius_audit(
            reversed(control_records), relative_radii=radii
        )

        self.assertEqual(control.to_dict(), reversed_control.to_dict())
        result = match_validation_radius_audits(
            control,
            learned,
            control_reference_relative_radius=0.5,
        )
        self.assertEqual(result.control_point.median_nonself_neighborhood_mass, 1 / 3)
        self.assertEqual(result.learned_relative_radius, 0.4)
        self.assertEqual(result.relative_neighborhood_mass_mismatch, 0.0)
        self.assertFalse(result.radius_selection_used_safety)
        self.assertEqual(result.schema_version, MATCHED_RADIUS_SCHEMA_VERSION)
        self.assertEqual(result.protocol, MATCHED_RADIUS_PROTOCOL)
        self.assertEqual(
            result.semantics_version,
            MATCHED_RADIUS_SEMANTICS_VERSION,
        )
        self.assertEqual(
            result.control_point.trajectory_balanced_p95_required_violation, 0.75
        )
        self.assertEqual(
            result.learned_point.trajectory_balanced_p95_required_violation, 0.0
        )
        self.assertTrue(result.strict_safety_improvement)
        self.assertTrue(result.all_matched_radius_gates_passed)
        fields = result.to_dict()["selector_metric_fields"]
        self.assertEqual(fields["control"]["neighborhood_mass"], 1 / 3)
        self.assertEqual(fields["learned"]["empty_trajectory_count"], 0)
        provenance = result.to_dict()["selector_radius_provenance"]
        self.assertEqual(provenance["control_reference_relative_radius"], 0.5)
        self.assertEqual(provenance["learned_relative_radius"], 0.4)
        self.assertEqual(len(provenance["control_audit_sha256"]), 64)
        self.assertEqual(len(provenance["learned_audit_sha256"]), 64)
        self.assertEqual(provenance["pairing_sha256"], control.pairing_sha256)
        self.assertEqual(
            result.to_dict()["semantics"]["version"],
            MATCHED_RADIUS_SEMANTICS_VERSION,
        )

    def test_five_percent_mass_mismatch_is_inclusive_and_above_fails(self) -> None:
        radii = (0.0, 0.4, 0.5, 1.0)
        control = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)), relative_radii=radii
        )
        learned = build_validation_radius_audit(
            _records((0.0, 3.0, 1.0, 4.0)), relative_radii=radii
        )
        learned_at_boundary = _only_mass_candidate(
            learned, relative_radius=0.4, mass=0.35
        )
        boundary = match_validation_radius_audits(
            control,
            learned_at_boundary,
            control_reference_relative_radius=0.5,
        )
        self.assertAlmostEqual(boundary.relative_neighborhood_mass_mismatch, 0.05)
        self.assertTrue(boundary.mass_match_passed)

        learned_above = _only_mass_candidate(
            learned, relative_radius=0.4, mass=0.350001
        )
        above = match_validation_radius_audits(
            control,
            learned_above,
            control_reference_relative_radius=0.5,
        )
        self.assertFalse(above.mass_match_passed)
        self.assertIn(
            "relative_neighborhood_mass_mismatch_above_threshold",
            above.failure_reasons,
        )

    def test_matched_radius_cannot_pass_by_discarding_center_coverage(self) -> None:
        radii = (0.0, 0.4, 0.5, 1.0)
        control = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)), relative_radii=radii
        )
        learned = build_validation_radius_audit(
            _records((0.0, 3.0, 1.0, 4.0)), relative_radii=radii
        )
        sparse_candidate = _only_mass_candidate(
            learned,
            relative_radius=0.4,
            mass=1 / 3,
            eligible_center_count=2,
        )
        result = match_validation_radius_audits(
            control,
            sparse_candidate,
            control_reference_relative_radius=0.5,
        )
        self.assertTrue(result.mass_match_passed)
        self.assertTrue(result.strict_safety_improvement)
        self.assertEqual(result.learned_point.eligible_given_viable, 0.5)
        self.assertFalse(result.coverage_floor_passed)
        self.assertFalse(result.coverage_drop_passed)
        self.assertFalse(result.all_matched_radius_gates_passed)
        self.assertIn("eligible_center_coverage_below_minimum", result.failure_reasons)
        self.assertIn(
            "coverage_loss_vs_control_above_maximum", result.failure_reasons
        )

        inclusive = match_validation_radius_audits(
            control,
            sparse_candidate,
            control_reference_relative_radius=0.5,
            minimum_eligible_center_coverage=0.5,
            max_coverage_fraction_loss_vs_control=0.5,
        )
        self.assertTrue(inclusive.coverage_floor_passed)
        self.assertTrue(inclusive.coverage_drop_passed)
        self.assertTrue(inclusive.all_matched_radius_gates_passed)

    def test_sign_retention_and_empty_trajectory_are_explicit_failures(self) -> None:
        radii = (0.0, 0.4, 0.5, 1.0)
        control = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)), relative_radii=radii
        )
        no_effect = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)), relative_radii=radii
        )
        result = match_validation_radius_audits(
            control, no_effect, control_reference_relative_radius=0.5
        )
        self.assertFalse(result.strict_safety_improvement)
        self.assertIn(
            "matched_mass_safety_improvement_is_not_strict", result.failure_reasons
        )

        sparse = build_validation_radius_audit(
            _records(
                (0.0, 0.01, 100.0, 200.0),
                trajectories=("paired", "paired", "isolated-a", "isolated-b"),
            ),
            relative_radii=(0.0, 0.05),
        )
        sparse_result = match_validation_radius_audits(
            sparse, sparse, control_reference_relative_radius=0.05
        )
        self.assertEqual(sparse_result.control_point.empty_trajectory_count, 2)
        self.assertIn(
            "control_reference_has_empty_trajectory", sparse_result.failure_reasons
        )
        self.assertFalse(sparse_result.all_matched_radius_gates_passed)
        self.assertFalse(sparse_result.selector_metrics_ready)
        fields = sparse_result.to_dict()["selector_metric_fields"]
        self.assertFalse(fields["control"]["coverage_complete"])

    def test_empty_centers_and_zero_mass_fail_closed(self) -> None:
        empty = build_validation_radius_audit(
            _records((0.0, 1.0, 2.0, 3.0)), relative_radii=(0.0,)
        )
        result = match_validation_radius_audits(
            empty, empty, control_reference_relative_radius=0.0
        )
        self.assertIsNone(result.control_point.median_nonself_neighborhood_mass)
        self.assertIsNone(result.learned_point)
        self.assertFalse(result.selector_metrics_ready)
        self.assertIn(
            "control_reference_has_no_eligible_centers", result.failure_reasons
        )
        self.assertIn(
            "no_learned_radius_has_defined_nonzero_mass_match", result.failure_reasons
        )

        invalid_zero = dataclasses.replace(
            empty,
            curve=(
                dataclasses.replace(
                    empty.curve[0],
                    eligible_center_count=1,
                    eligible_center_coverage=0.25,
                    eligible_given_viable=0.25,
                    median_nonself_neighborhood_mass=0.0,
                    evaluable_trajectory_count=1,
                    empty_trajectory_count=1,
                    trajectory_balanced_p95_required_violation=0.0,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent with eligible-center"):
            match_validation_radius_audits(invalid_zero, invalid_zero)

    def test_nonviable_point_does_not_remove_viable_centers(self) -> None:
        records = (
            AuditRecord(
                sample_id="nonviable",
                trajectory_id="trajectory-a",
                split="validation",
                safety_margin=-0.1,
                latent=(0.0,),
                action_safety_margins=(-1.0, -1.0),
            ),
            AuditRecord(
                sample_id="left",
                trajectory_id="trajectory-a",
                split="validation",
                safety_margin=0.1,
                latent=(0.0,),
                action_safety_margins=(1.0, -0.5),
            ),
            AuditRecord(
                sample_id="right",
                trajectory_id="trajectory-a",
                split="validation",
                safety_margin=0.1,
                latent=(0.0,),
                action_safety_margins=(-0.25, 1.0),
            ),
        )
        audit = build_validation_radius_audit(records, relative_radii=(0.0,))
        point = audit.curve[0]
        self.assertEqual(point.viable_sample_count, 2)
        self.assertEqual(point.viable_fraction, 2 / 3)
        self.assertEqual(point.eligible_center_count, 2)
        self.assertEqual(point.eligible_given_viable, 1.0)
        self.assertEqual(point.eligible_center_coverage, 1.0)
        self.assertEqual(point.trajectory_balanced_p95_required_violation, 0.25)

    def test_tiny_radius_uses_only_one_ulp_of_boundary_slack(self) -> None:
        records = (
            AuditRecord(
                sample_id="left",
                trajectory_id="trajectory",
                split="validation",
                safety_margin=0.1,
                latent=(0.0,),
                action_safety_margins=(1.0, -1.0),
            ),
            AuditRecord(
                sample_id="right",
                trajectory_id="trajectory",
                split="validation",
                safety_margin=0.1,
                latent=(1e-14,),
                action_safety_margins=(-1.0, 1.0),
            ),
        )
        audit = build_validation_radius_audit(records, relative_radii=(0.1,))
        self.assertEqual(audit.curve[0].absolute_radius, 1e-15)
        self.assertEqual(audit.curve[0].eligible_center_count, 0)

    def test_absolute_radius_must_match_relative_radius_times_scale(self) -> None:
        audit = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)), relative_radii=(0.0, 0.5)
        )
        tampered = dataclasses.replace(
            audit,
            curve=(
                audit.curve[0],
                dataclasses.replace(
                    audit.curve[1],
                    absolute_radius=audit.curve[1].absolute_radius + 1e-6,
                ),
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "absolute_radius must equal relative_radius"
        ):
            tampered.validate()

    def test_split_and_physical_pairing_isolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation records only"):
            build_validation_radius_audit(
                _records((0.0, 1.0, 3.0, 4.0), split="test")
            )
        control = _records((0.0, 1.0, 3.0, 4.0))
        learned = list(_records((0.0, 3.0, 1.0, 4.0)))
        learned[0] = dataclasses.replace(learned[0], safety_margin=9.0)
        with self.assertRaisesRegex(ValueError, "identical sample IDs"):
            validate_paired_records(control, learned)

    def test_integrity_checked_audit_detects_valid_numeric_tampering(self) -> None:
        audit = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)),
            relative_radii=(0.0, 0.4, 0.5, 1.0),
        )
        payload = audit.to_dict()
        self.assertEqual(payload["schema_version"], RADIUS_AUDIT_SCHEMA_VERSION)
        self.assertEqual(payload["protocol"], RADIUS_AUDIT_PROTOCOL)
        self.assertEqual(
            payload["semantics_version"],
            RADIUS_AUDIT_SEMANTICS_VERSION,
        )
        self.assertEqual(
            payload["semantics"]["version"],
            RADIUS_AUDIT_SEMANTICS_VERSION,
        )
        round_trip = validation_radius_audit_from_dict(payload)
        self.assertEqual(round_trip, audit)
        defined = next(
            row
            for row in payload["curve"]
            if row["trajectory_balanced_p95_required_violation"] is not None
        )
        defined["trajectory_balanced_p95_required_violation"] += 0.01
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validation_radius_audit_from_dict(payload)
        json.dumps(audit.to_dict(), allow_nan=False, sort_keys=True)

    def test_radius_audit_requires_exact_v2_protocol_and_semantics(self) -> None:
        audit = build_validation_radius_audit(
            _records((0.0, 1.0, 3.0, 4.0)),
            relative_radii=(0.0, 0.4, 0.5, 1.0),
        )

        missing_semantics = audit.to_dict()
        missing_semantics.pop("semantics")
        with self.assertRaisesRegex(ValueError, "must include exact v2 semantics"):
            validation_radius_audit_from_dict(missing_semantics)

        changed_semantics = audit.to_dict()
        changed_semantics["semantics"]["coverage"] = "legacy all-sample coverage"
        with self.assertRaisesRegex(ValueError, "do not match the implementation"):
            validation_radius_audit_from_dict(changed_semantics)

        legacy = dataclasses.replace(
            audit,
            schema_version=1,
            protocol="e2_validation_radius_curve_v1",
        )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported validation radius audit protocol",
        ):
            legacy.validate()


if __name__ == "__main__":
    unittest.main()
