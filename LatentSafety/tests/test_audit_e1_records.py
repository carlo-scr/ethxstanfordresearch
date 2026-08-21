from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.records import AuditRecord, write_jsonl  # noqa: E402


class AuditE1RecordsIntegrationTests(unittest.TestCase):
    def test_calibration_only_radius_and_traceable_witnesses(self) -> None:
        calibration = (
            AuditRecord(
                "cal:0",
                "cal-a",
                "calibration",
                0.1,
                (0.0,),
                observation_latent=(0.0,),
                state_latent=(0.0, 0.5),
                action_safety_margins=(1.0, -1.0),
            ),
            AuditRecord(
                "cal:1",
                "cal-b",
                "calibration",
                -0.1,
                (1.0,),
                observation_latent=(1.0,),
                state_latent=(1.0, -0.5),
                action_safety_margins=(-1.0, 1.0),
            ),
        )
        test = (
            AuditRecord(
                "test:0",
                "test-a",
                "test",
                0.5,
                (0.0,),
                observation_latent=(0.0,),
                state_latent=(0.0, 0.5),
                action_safety_margins=(1.0, -1.0),
            ),
            AuditRecord(
                "test:1",
                "test-b",
                "test",
                -0.2,
                (0.01,),
                observation_latent=(0.01,),
                state_latent=(0.01, -0.5),
                action_safety_margins=(-1.0, 1.0),
            ),
            AuditRecord(
                "test:2",
                "test-c",
                "test",
                0.8,
                (3.0,),
                observation_latent=(3.0,),
                state_latent=(3.0, 0.0),
                action_safety_margins=(1.0, 1.0),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            calibration_path = temporary / "calibration.jsonl"
            test_path = temporary / "test.jsonl"
            output_path = temporary / "audit.json"
            margin_oracle_path = temporary / "audit_margin_oracle.json"
            action_oracle_path = temporary / "audit_action_oracle.json"
            observation_oracle_path = temporary / "audit_observation_oracle.json"
            state_oracle_path = temporary / "audit_state_oracle.json"
            write_jsonl(calibration_path, calibration)
            write_jsonl(test_path, test)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_e1_records.py"),
                    "--config",
                    str(ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"),
                    "--calibration",
                    str(calibration_path),
                    "--test",
                    str(test_path),
                    "--output",
                    str(output_path),
                    "--relative-radii",
                    "0",
                    "0.02",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("2 static and 2 action radii", completed.stdout)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            for view, path in (
                ("latent_plus_margin", margin_oracle_path),
                ("latent_plus_action_profile", action_oracle_path),
                ("observation_oracle", observation_oracle_path),
                ("state_oracle", state_oracle_path),
            ):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "audit_e1_records.py"),
                        "--config",
                        str(ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"),
                        "--calibration",
                        str(calibration_path),
                        "--test",
                        str(test_path),
                        "--output",
                        str(path),
                        "--representation-view",
                        view,
                        "--relative-radii",
                        "0",
                        "0.02",
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            margin_oracle = json.loads(margin_oracle_path.read_text(encoding="utf-8"))
            action_oracle = json.loads(action_oracle_path.read_text(encoding="utf-8"))
            observation_oracle = json.loads(
                observation_oracle_path.read_text(encoding="utf-8")
            )
            state_oracle = json.loads(state_oracle_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["radius_calibration"]["calibration_scale"], 1.0)
        self.assertFalse(payload["radius_calibration"]["uses_safety_labels"])
        robust = payload["static_curve"][1]
        self.assertEqual(robust["estimate"]["witness_margin"], 0.5)
        self.assertEqual(robust["witnesses"][0]["safe_sample_id"], "test:0")
        self.assertEqual(robust["witnesses"][0]["unsafe_sample_id"], "test:1")
        self.assertEqual(payload["action_curve"][1]["conflicting_neighborhood_count"], 2)
        self.assertEqual(payload["action_curve"][1]["tail_required_violation"], 1.0)
        self.assertEqual(
            payload["action_curve"][1]["trajectory_tail_summary"],
            {
                "score": (
                    "95th percentile across eligible centered-neighborhood finite-action "
                    "common violations within each trajectory"
                ),
                "all_trajectory_count": 3,
                "evaluable_trajectory_count": 2,
                "mean_trajectory_tail_required_violation": 1.0,
                "median_trajectory_tail_required_violation": 1.0,
                "per_trajectory": [
                    {
                        "trajectory_id": "test-a",
                        "eligible_center_count": 1,
                        "tail_quantile": 0.95,
                        "tail_required_violation": 1.0,
                        "maximum_required_violation": 1.0,
                    },
                    {
                        "trajectory_id": "test-b",
                        "eligible_center_count": 1,
                        "tail_quantile": 0.95,
                        "tail_required_violation": 1.0,
                        "maximum_required_violation": 1.0,
                    },
                ],
            },
        )
        self.assertEqual(
            [
                row["sample_id"]
                for row in payload["action_curve"][1]["center_required_violations"]
            ],
            ["test:0", "test:1"],
        )
        self.assertEqual(payload["representation_view"]["name"], "latent")
        self.assertFalse(payload["representation_view"]["oracle_control"])
        self.assertEqual(
            margin_oracle["static_curve"][1]["estimate"]["witness_margin"], 0.0
        )
        self.assertTrue(margin_oracle["representation_view"]["oracle_control"])
        self.assertEqual(
            action_oracle["action_curve"][1]["conflicting_neighborhood_count"], 0
        )
        self.assertEqual(
            margin_oracle["radius_calibration"]["calibration_scale"],
            payload["radius_calibration"]["calibration_scale"],
        )
        self.assertEqual(
            observation_oracle["radius_calibration"]["scale_source"],
            "calibration_observation_oracle",
        )
        self.assertEqual(
            observation_oracle["static_curve"][1]["estimate"]["witness_margin"],
            payload["static_curve"][1]["estimate"]["witness_margin"],
        )
        self.assertEqual(
            state_oracle["radius_calibration"]["scale_source"],
            "calibration_state_oracle",
        )


if __name__ == "__main__":
    unittest.main()
