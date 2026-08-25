from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plan_profile_teachers import build_plan  # noqa: E402
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.data import CartState, deterministic_step, safety_margin  # noqa: E402
from latent_safety.learning.profile_coverage import (  # noqa: E402
    COVERAGE_CENTER_TIMESTEP,
    COVERAGE_SEED_BASE,
    ProfileCoverageError,
    _observed_constant_action_profile,
    build_coverage_bundles,
    coverage_seed,
    load_profile_coverage_gate,
)
from latent_safety.learning.profile_artifacts import canonical_sha256, ids_sha256  # noqa: E402
from latent_safety.learning.profile_protocol import (  # noqa: E402
    PROFILE_PROTOCOL_VERSION,
    REQUIRED_NORMALIZED_P95,
    evaluate_profile_gate,
)
from latent_safety.learning.profile_teacher import resolve_teacher_config  # noqa: E402
from latent_safety.learning.runtime import require_torch  # noqa: E402


class ProfileCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan()
        cls.task = cls.plan["tasks"][0]
        cls.base = load_config(ROOT / cls.task["base_config"])
        cls.config = resolve_teacher_config(
            cls.base,
            cls.task,
            device="cpu",
            engineering_smoke=True,
        )

    def test_seed_rule_is_family_independent_and_domain_separated(self) -> None:
        self.assertEqual(coverage_seed("controlled_cart_video", 0), COVERAGE_SEED_BASE)
        self.assertEqual(
            coverage_seed("controlled_pendulum_video", 0), COVERAGE_SEED_BASE + 100_000
        )
        self.assertEqual(
            coverage_seed("controlled_dubins_navigation_pixels", 107),
            COVERAGE_SEED_BASE + 200_107,
        )
        with self.assertRaisesRegex(ProfileCoverageError, "unsupported"):
            coverage_seed("unknown", 0)
        with self.assertRaisesRegex(ProfileCoverageError, "non-negative"):
            coverage_seed("controlled_cart_video", -1)

    def test_observed_profile_is_explicit_t0_through_h_rollout(self) -> None:
        state = CartState(position=0.75, velocity=0.35, nuisance_phase=0.0)
        observed = _observed_constant_action_profile(state, self.config.data)
        manual = []
        for action in self.config.data.actions:
            branch = state
            margins = [safety_margin(branch, self.config.data)]
            for _ in range(self.config.data.action_profile_horizon):
                branch = deterministic_step(branch, action, self.config.data)
                margins.append(safety_margin(branch, self.config.data))
            manual.append(min(margins))
        self.assertEqual(observed, tuple(manual))

    def test_bundle_generator_is_disjoint_balanced_and_never_builds_oracle_profiles(self) -> None:
        try:
            modules = require_torch()
        except Exception as error:  # pragma: no cover - optional dependency environment
            self.skipTest(str(error))
        with patch(
            "latent_safety.learning.data.action_safety_profile",
            side_effect=AssertionError("oracle profile helper must not be called"),
        ):
            first = build_coverage_bundles(
                modules,
                self.config,
                data_seed=0,
                bundle_count=10,
            )
            second = build_coverage_bundles(
                modules,
                self.config,
                data_seed=0,
                bundle_count=10,
            )
        self.assertEqual(len(first), 10)
        self.assertEqual(
            [bundle.bundle_id for bundle in first],
            [bundle.bundle_id for bundle in second],
        )
        self.assertTrue(
            all(bundle.bundle_id.startswith("coverage-controlled_cart_video") for bundle in first)
        )
        self.assertFalse(any(bundle.bundle_id.startswith("cart-") for bundle in first))
        self.assertEqual(
            [bundle.teacher_fold for bundle in first], [0, 1, 2, 3, 4] * 2
        )
        self.assertTrue(
            all(
                tuple(bundle.observed_action_profile)
                == tuple(reference.observed_action_profile)
                for bundle, reference in zip(first, second, strict=True)
            )
        )
        self.assertTrue(
            all(bundle.history.shape[0] == self.config.data.history_length for bundle in first)
        )
        self.assertEqual(COVERAGE_CENTER_TIMESTEP, 3)

    def test_bundle_count_must_support_all_five_teachers(self) -> None:
        try:
            modules = require_torch()
        except Exception as error:  # pragma: no cover
            self.skipTest(str(error))
        with self.assertRaisesRegex(ProfileCoverageError, "at least five"):
            build_coverage_bundles(modules, self.config, data_seed=0, bundle_count=4)

    def _signed_gate(self) -> tuple[dict[str, object], dict[int, str]]:
        teacher_hashes = {fold: str(fold) * 64 for fold in range(5)}
        predictions = {
            f"coverage-controlled_cart_video-seed-000-bundle-{index:04d}": (
                0.0,
                0.0,
                0.0,
            )
            for index in range(10)
        }
        targets = {bundle_id: values for bundle_id, values in predictions.items()}
        gate = evaluate_profile_gate(
            predictions,
            targets,
            margin_scale=self.config.objective.margin_scale,
            required_max=REQUIRED_NORMALIZED_P95,
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "status": "gate_passed",
            "scientific_gate_passed": True,
            "engineering_smoke": True,
            "evidence_eligible": False,
            "plan_sha256": "a" * 64,
            "domain": self.config.data.task,
            "model_family": self.config.model.family,
            "data_seed": 0,
            "assembly": {
                "path": "crossfit_label_manifest.json",
                "manifest_sha256": "b" * 64,
                "file_sha256": "c" * 64,
            },
            "generator": {},
            "teacher_assignment": {},
            "action_grid": list(self.config.data.actions),
            "profile_rollout_horizon": self.config.data.action_profile_horizon,
            "margin_scale": self.config.objective.margin_scale,
            "required_normalized_p95": REQUIRED_NORMALIZED_P95,
            "bundle_count": 10,
            "bundle_ids_sha256": ids_sha256(sorted(predictions)),
            "scenario_counts": {"nominal": 10},
            "gate": gate.to_dict(),
            "bundles": [
                {
                    "bundle_id": bundle_id,
                    "scenario": "nominal",
                    "teacher_fold": index % 5,
                    "teacher_task_sha256": teacher_hashes[index % 5],
                    "prediction": list(predictions[bundle_id]),
                    "observed_target": list(targets[bundle_id]),
                    "normalized_max_error": 0.0,
                }
                for index, bundle_id in enumerate(sorted(predictions))
            ],
            "device": {},
            "execution_code": {},
            "producing_command": [],
            "remaining_scope": {},
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        return manifest, teacher_hashes

    def test_gate_loader_authenticates_teacher_inventory_and_self_digest(self) -> None:
        manifest, teacher_hashes = self._signed_gate()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_profile_coverage_gate(
                path,
                expected_domain=self.config.data.task,
                expected_model_family=self.config.model.family,
                expected_data_seed=0,
                expected_plan_sha256="a" * 64,
                expected_assembly_manifest_sha256="b" * 64,
                expected_action_grid=self.config.data.actions,
                expected_profile_horizon=self.config.data.action_profile_horizon,
                expected_margin_scale=self.config.objective.margin_scale,
                expected_teacher_task_sha256_by_fold=teacher_hashes,
                allow_engineering_failure=True,
            )
            self.assertEqual(loaded["verified_manifest_sha256"], manifest["manifest_sha256"])

            mismatched = json.loads(json.dumps(manifest))
            mismatched["bundles"][0]["teacher_task_sha256"] = "f" * 64
            mismatched.pop("manifest_sha256")
            mismatched["manifest_sha256"] = canonical_sha256(mismatched)
            path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ProfileCoverageError, "teacher identity"):
                load_profile_coverage_gate(
                    path,
                    expected_domain=self.config.data.task,
                    expected_model_family=self.config.model.family,
                    expected_data_seed=0,
                    expected_plan_sha256="a" * 64,
                    expected_assembly_manifest_sha256="b" * 64,
                    expected_action_grid=self.config.data.actions,
                    expected_profile_horizon=self.config.data.action_profile_horizon,
                    expected_margin_scale=self.config.objective.margin_scale,
                    expected_teacher_task_sha256_by_fold=teacher_hashes,
                    allow_engineering_failure=True,
                )


if __name__ == "__main__":
    unittest.main()
