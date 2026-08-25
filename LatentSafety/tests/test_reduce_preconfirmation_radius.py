from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.matched_radius import (  # noqa: E402
    MATCHED_RADIUS_PROTOCOL,
    MATCHED_RADIUS_SCHEMA_VERSION,
    MATCHED_RADIUS_SEMANTICS_VERSION,
    RELATIVE_RADIUS_GRID,
)
from latent_safety.records import AuditRecord, write_jsonl  # noqa: E402


SCRIPT_PATH = ROOT / "scripts" / "reduce_preconfirmation_radius.py"
SPEC = importlib.util.spec_from_file_location("reduce_preconfirmation_radius", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/reduce_preconfirmation_radius.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PROFILES = (
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -0.5),
    (-0.5, 1.0),
)


def _records(latents: tuple[float, ...]) -> tuple[AuditRecord, ...]:
    return tuple(
        AuditRecord(
            sample_id=f"sample-{index}",
            trajectory_id="trajectory-a" if index < 2 else "trajectory-b",
            split="validation",
            safety_margin=0.1,
            latent=(latent,),
            action_safety_margins=PROFILES[index],
        )
        for index, latent in enumerate(latents)
    )


class ReducePreconfirmationRadiusScriptTests(unittest.TestCase):
    def test_record_mode_pins_inputs_and_emits_selector_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            control_path = directory / "none_validation.jsonl"
            learned_path = directory / "learned_validation.jsonl"
            write_jsonl(control_path, _records((0.0, 0.01, 1.0, 1.01)))
            write_jsonl(learned_path, _records((0.0, 1.0, 0.01, 1.01)))
            payload = MODULE.reduce_inputs(
                control_records_path=control_path,
                learned_records_path=learned_path,
            )

        self.assertEqual(payload["schema_version"], MATCHED_RADIUS_SCHEMA_VERSION)
        self.assertEqual(payload["analysis"], MATCHED_RADIUS_PROTOCOL)
        self.assertEqual(
            payload["semantics_version"],
            MATCHED_RADIUS_SEMANTICS_VERSION,
        )
        self.assertFalse(payload["evidence_eligible"])
        self.assertIn("canonical aggregator", payload["evidence_boundary"])
        self.assertEqual(payload["input"]["mode"], "paired_validation_record_files")
        self.assertEqual(payload["input"]["split"], "validation_only")
        self.assertFalse(payload["input"]["calibration_access"])
        self.assertFalse(payload["input"]["final_test_access"])
        self.assertEqual(len(payload["input"]["control"]["sha256"]), 64)
        self.assertEqual(len(payload["input"]["learned"]["sha256"]), 64)
        self.assertEqual(
            payload["protocol_config"]["relative_radius_grid"],
            list(RELATIVE_RADIUS_GRID),
        )
        self.assertEqual(
            [entry["path"] for entry in payload["implementation"]],
            [
                "scripts/reduce_preconfirmation_radius.py",
                "src/latent_safety/analysis/matched_radius.py",
                "src/latent_safety/records.py",
                "src/latent_safety/metrics/defect.py",
            ],
        )
        self.assertTrue(payload["result"]["selector_metrics_ready"])
        self.assertTrue(payload["result"]["mass_match_passed"])
        self.assertTrue(payload["result"]["strict_safety_improvement"])
        self.assertEqual(
            payload["result"]["semantics"]["version"],
            MATCHED_RADIUS_SEMANTICS_VERSION,
        )
        self.assertIsNotNone(
            payload["result"]["selector_metric_fields"]["learned"]
        )
        # Serialization remains strict and finite even though this artifact is diagnostic-only.
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_derived_audit_replay_mode_is_not_exposed_as_integrity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--control-audit",
                    "derived.json",
                    "--output",
                    str(Path(temporary) / "reduction.json"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized arguments: --control-audit", completed.stderr)

    def test_cli_refuses_incomplete_mode_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            control_path = directory / "control.jsonl"
            learned_path = directory / "learned.jsonl"
            output_path = directory / "matched.json"
            write_jsonl(control_path, _records((0.0, 0.01, 1.0, 1.01)))
            write_jsonl(learned_path, _records((0.0, 1.0, 0.01, 1.01)))

            incomplete = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--control-records",
                    str(control_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("requires both control and learned", incomplete.stderr)

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--control-records",
                str(control_path),
                "--learned-records",
                str(learned_path),
                "--output",
                str(output_path),
            ]
            first = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            second = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
