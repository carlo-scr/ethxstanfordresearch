from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import tomllib
import unittest
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.preconfirmation import (
    CONTROL_ARM,
    DOMAINS,
    EXPECTED_OBSERVATIONS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
    PreconfirmationObservation,
    run_preconfirmation_selection,
)


def _complete_observations() -> list[PreconfirmationObservation]:
    rows: list[PreconfirmationObservation] = []
    for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS):
        rows.append(
            PreconfirmationObservation(
                domain=domain,
                model_family=family,
                arm=CONTROL_ARM,
                seed=seed,
                weight=None,
                safety=0.50,
                reconstruction=1.00,
                rollout=2.00,
                eligible_center_coverage=0.90,
                neighborhood_mass=1.00,
                empty_trajectory_count=0,
                run_complete=True,
                coverage_complete=True,
            )
        )
    safety_by_weight = {
        0.001: 0.45,
        0.01: 0.40,
        0.1: 0.30,
        1.0: 0.30,
        10.0: 0.35,
    }
    utility_by_weight = {
        0.001: 1.00,
        0.01: 1.00,
        0.1: 1.02,
        1.0: 1.01,
        10.0: 1.00,
    }
    for domain, family, arm, seed, weight in product(
        DOMAINS,
        MODEL_FAMILIES,
        LEARNED_ARMS,
        PILOT_SEEDS,
        POSITIVE_WEIGHTS,
    ):
        profile = arm == PROFILE_ARM
        rows.append(
            PreconfirmationObservation(
                domain=domain,
                model_family=family,
                arm=arm,
                seed=seed,
                weight=weight,
                safety=safety_by_weight[weight],
                reconstruction=utility_by_weight[weight],
                rollout=2.0 * utility_by_weight[weight],
                eligible_center_coverage=0.88,
                neighborhood_mass=1.02,
                empty_trajectory_count=0,
                run_complete=True,
                coverage_complete=True,
                profile_normalized_p95_error=0.08 if profile else None,
                profile_teacher_count=5 if profile else None,
                profile_teacher_failure_count=0 if profile else None,
                profile_label_manifest_sha256=(
                    hashlib.sha256(
                        f"{domain}|{family}|{seed}".encode("utf-8")
                    ).hexdigest()
                    if profile
                    else None
                ),
            )
        )
    return rows


def _replace(
    rows: list[PreconfirmationObservation],
    predicate,
    **changes: object,
) -> list[PreconfirmationObservation]:
    return [
        dataclasses.replace(row, **changes) if predicate(row) else row for row in rows
    ]


class PreconfirmationSelectionTests(unittest.TestCase):
    def test_complete_factorial_selects_by_safety_then_utility_then_weight(self) -> None:
        result = run_preconfirmation_selection(_complete_observations())

        self.assertEqual(result.input_observation_count, EXPECTED_OBSERVATIONS)
        self.assertEqual(result.expected_observation_count, 288)
        self.assertEqual(result.required_selection_count, 18)
        self.assertEqual(result.passed_selection_count, 18)
        self.assertTrue(result.ready_for_confirmation)
        self.assertEqual(len(result.profile_label_manifests), 18)
        self.assertTrue(all(selection.selected_weight == 1.0 for selection in result.selections))
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False, sort_keys=True)

        tied = _replace(
            _complete_observations(),
            lambda row: row.arm != CONTROL_ARM and row.weight in {0.1, 1.0},
            reconstruction=1.01,
            rollout=2.02,
        )
        tied_result = run_preconfirmation_selection(tied)
        self.assertTrue(all(selection.selected_weight == 0.1 for selection in tied_result.selections))

    def test_every_gate_is_recorded_and_no_eligible_weight_fails_the_stratum(self) -> None:
        domain = DOMAINS[0]
        family = MODEL_FAMILIES[0]
        rows = _complete_observations()
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == PROFILE_ARM
            and row.weight == 0.001,
            reconstruction=1.06,
        )
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == PROFILE_ARM
            and row.weight == 0.01,
            eligible_center_coverage=0.79,
        )
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == PROFILE_ARM
            and row.weight == 0.1,
            neighborhood_mass=1.06,
        )
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == PROFILE_ARM
            and row.weight == 1.0,
            profile_normalized_p95_error=0.1000001,
        )
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == PROFILE_ARM
            and row.weight == 10.0,
            empty_trajectory_count=1,
        )
        result = run_preconfirmation_selection(rows)
        failed = next(
            selection
            for selection in result.selections
            if selection.domain == domain
            and selection.model_family == family
            and selection.arm == PROFILE_ARM
        )

        self.assertFalse(failed.passed)
        self.assertIsNone(failed.selected_weight)
        self.assertFalse(result.ready_for_confirmation)
        self.assertEqual(result.passed_selection_count, 17)
        reasons = {candidate.weight: candidate.failure_reasons for candidate in failed.candidates}
        self.assertIn("reconstruction_ratio_above_1.05", reasons[0.001])
        self.assertIn("eligible_center_coverage_below_0.80", reasons[0.01])
        self.assertIn("neighborhood_mass_mismatch_above_0.05", reasons[0.1])
        self.assertIn("profile_normalized_p95_error_above_0.10", reasons[1.0])
        self.assertIn("empty_trajectory", reasons[10.0])

    def test_zero_denominator_rule_is_explicit_and_json_safe(self) -> None:
        domain = DOMAINS[0]
        family = MODEL_FAMILIES[0]
        arm = LEARNED_ARMS[0]
        rows = _replace(
            _complete_observations(),
            lambda row: row.domain == domain
            and row.model_family == family
            and (row.arm == CONTROL_ARM or row.arm == arm),
            reconstruction=0.0,
        )
        rows = _replace(
            rows,
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == arm
            and row.weight == 0.001,
            reconstruction=0.01,
        )
        result = run_preconfirmation_selection(rows)
        selection = next(
            item
            for item in result.selections
            if item.domain == domain
            and item.model_family == family
            and item.arm == arm
        )
        inadmissible = next(
            candidate for candidate in selection.candidates if candidate.weight == 0.001
        )
        tied = next(candidate for candidate in selection.candidates if candidate.weight == 1.0)
        self.assertIsNone(inadmissible.reconstruction_ratio)
        self.assertIn(
            "inadmissible_reconstruction_zero_denominator",
            inadmissible.failure_reasons,
        )
        self.assertEqual(tied.reconstruction_ratio, 1.0)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_zero_neighborhood_mass_fails_closed_in_every_stratum(self) -> None:
        rows = _replace(
            _complete_observations(),
            lambda row: True,
            neighborhood_mass=0.0,
        )

        result = run_preconfirmation_selection(rows)

        self.assertEqual(result.passed_selection_count, 0)
        self.assertFalse(result.ready_for_confirmation)
        for selection in result.selections:
            self.assertFalse(selection.passed)
            self.assertTrue(
                all(
                    candidate.maximum_neighborhood_mass_mismatch is None
                    and "inadmissible_neighborhood_mass_zero_denominator"
                    in candidate.failure_reasons
                    for candidate in selection.candidates
                )
            )

    def test_empty_paired_none_trajectory_fails_all_three_cell_arms(self) -> None:
        domain = DOMAINS[0]
        family = MODEL_FAMILIES[0]
        rows = _replace(
            _complete_observations(),
            lambda row: row.domain == domain
            and row.model_family == family
            and row.arm == CONTROL_ARM
            and row.seed == PILOT_SEEDS[0],
            empty_trajectory_count=1,
        )
        result = run_preconfirmation_selection(rows)
        affected = [
            selection
            for selection in result.selections
            if selection.domain == domain and selection.model_family == family
        ]
        self.assertEqual(len(affected), 3)
        self.assertEqual(result.passed_selection_count, 15)
        self.assertFalse(result.ready_for_confirmation)
        for selection in affected:
            self.assertFalse(selection.passed)
            self.assertTrue(
                all(
                    "paired_none_empty_trajectory" in candidate.failure_reasons
                    for candidate in selection.candidates
                )
            )

    def test_missing_duplicate_failed_and_invalid_rows_are_rejected(self) -> None:
        rows = _complete_observations()
        with self.assertRaisesRegex(ValueError, "incomplete preconfirmation factorial"):
            run_preconfirmation_selection(rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate preconfirmation observation"):
            run_preconfirmation_selection([*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, "run is not complete"):
            run_preconfirmation_selection(
                [dataclasses.replace(rows[0], run_complete=False), *rows[1:]]
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            run_preconfirmation_selection(
                [dataclasses.replace(rows[0], safety=math.nan), *rows[1:]]
            )
        with self.assertRaisesRegex(ValueError, "profile-only fields must be null"):
            run_preconfirmation_selection(
                [
                    dataclasses.replace(rows[0], profile_teacher_count=5),
                    *rows[1:],
                ]
            )

        inconsistent_manifest = _replace(
            rows,
            lambda row: row.arm == PROFILE_ARM
            and row.domain == DOMAINS[0]
            and row.model_family == MODEL_FAMILIES[0]
            and row.seed == PILOT_SEEDS[0]
            and row.weight == POSITIVE_WEIGHTS[0],
            profile_label_manifest_sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "share one label manifest"):
            run_preconfirmation_selection(inconsistent_manifest)

    def test_constants_match_frozen_preconfirmation_config(self) -> None:
        with (ROOT / "configs" / "e2_frontier" / "intervention.toml").open(
            "rb"
        ) as stream:
            config = tomllib.load(stream)
        self.assertEqual(tuple(config["run"]["seeds"]), PILOT_SEEDS)
        self.assertEqual(tuple(config["regularization"]["positive_weights"]), POSITIVE_WEIGHTS)
        self.assertEqual(
            tuple(config["required_core_learned_arms"]["names"]), LEARNED_ARMS
        )
        self.assertEqual(
            config["eligibility"]["max_relative_reconstruction_degradation"],
            0.05,
        )
        self.assertEqual(
            config["eligibility"]["max_relative_rollout_degradation"], 0.05
        )
        self.assertEqual(
            config["eligibility"]["minimum_eligible_center_coverage"], 0.80
        )
        self.assertEqual(
            config["eligibility"]["max_coverage_fraction_loss_vs_none"], 0.05
        )
        self.assertEqual(
            config["eligibility"]["eligible_center_coverage_denominator"],
            "individually viable validation centers",
        )
        self.assertEqual(
            config["eligibility"]["max_relative_neighborhood_mass_mismatch"],
            0.05,
        )
        self.assertEqual(
            config["eligibility"]["predicted_profile_normalized_p95_error_max"],
            0.10,
        )
        artifact = config["selection_artifact"]
        self.assertEqual(
            artifact["entrypoint"], "scripts/select_preconfirmation_weights.py"
        )
        self.assertEqual(artifact["input_split"], "validation_only")
        self.assertEqual(artifact["expected_observations"], EXPECTED_OBSERVATIONS)
        self.assertEqual(artifact["required_weight_freezes"], 18)
        self.assertFalse(artifact["final_test_access"])
        self.assertIn("separately in every pilot seed", artifact["matched_mass_sign_scope"])
        self.assertIn("checksum-identical", artifact["profile_label_manifest_reuse"])


if __name__ == "__main__":
    unittest.main()
