from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.records import AuditRecord, write_jsonl  # noqa: E402

SCRIPT_PATH = ROOT / "scripts" / "run_foundation_audit.py"
SPEC = importlib.util.spec_from_file_location("run_foundation_audit", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/run_foundation_audit.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records(*, split: str = "validation") -> tuple[AuditRecord, ...]:
    return (
        AuditRecord(
            sample_id="left",
            trajectory_id="trajectory",
            split=split,
            safety_margin=0.1,
            latent=(0.0,),
            action_safety_margins=(1.0, -0.5),
        ),
        AuditRecord(
            sample_id="right",
            trajectory_id="trajectory",
            split=split,
            safety_margin=0.1,
            latent=(0.0,),
            action_safety_margins=(-0.25, 1.0),
        ),
    )


def _write_test_freeze(
    path: Path,
    records_path: Path,
    *,
    revision: str,
    delta: float = 0.0,
    gamma: float = 0.0,
    tail_quantile: float = 0.95,
    max_witnesses: int = 10,
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": MODULE.TEST_FREEZE_PROTOCOL,
        "status": "frozen_before_test_access",
        "analysis": MODULE.FOUNDATION_AUDIT_PROTOCOL,
        "input": {
            "split": "test",
            "records_sha256": MODULE._sha256(records_path),
        },
        "audit_parameters": {
            "delta": delta,
            "gamma": gamma,
            "tail_quantile": tail_quantile,
            "max_witnesses": max_witnesses,
        },
        "code": {"git_revision": revision, "git_dirty": False},
    }
    payload["manifest_sha256"] = MODULE._canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return MODULE._sha256(path)


def _clean_base_manifest(revision: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at_utc": "2026-08-25T00:00:00+00:00",
        "config": {"path": "unused", "sha256": "0" * 64},
        "code": {"git_revision": revision, "git_dirty": False},
        "environment": {"python": "test", "installed_distributions": ()},
    }


class RunFoundationAuditTests(unittest.TestCase):
    def test_function_emits_traceable_nonpromoted_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_path = Path(temporary) / "records.jsonl"
            write_jsonl(records_path, _records())
            payload = MODULE.run_audit(
                records_path,
                delta=0.0,
                gamma=0.0,
                tail_quantile=0.95,
                max_witnesses=10,
            )
        self.assertEqual(payload["analysis"], "foundation_common_action_audit_v1")
        self.assertFalse(payload["evidence_eligible"])
        self.assertEqual(payload["input"]["split"], "validation")
        self.assertEqual(len(payload["input"]["sha256"]), 64)
        self.assertEqual(payload["audit"]["conflicting_center_count"], 2)
        self.assertTrue(payload["semantics"]["viability_filter_precedes_neighborhood"])
        self.assertIn("code", payload["run_provenance"])
        self.assertEqual(
            [entry["path"] for entry in payload["implementation"]],
            [
                "scripts/run_foundation_audit.py",
                "src/latent_safety/foundation/audit.py",
                "src/latent_safety/manifest.py",
                "src/latent_safety/records.py",
            ],
        )
        unsigned = dict(payload)
        declared = unsigned.pop("artifact_sha256")
        self.assertEqual(declared, MODULE._canonical_sha256(unsigned))
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_training_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_path = Path(temporary) / "records.jsonl"
            write_jsonl(records_path, _records(split="train"))
            with self.assertRaisesRegex(ValueError, "held-out split"):
                MODULE.run_audit(
                    records_path,
                    delta=0.0,
                    gamma=0.0,
                    tail_quantile=0.95,
                    max_witnesses=10,
                )

    def test_test_records_require_freeze_unblinding_and_access_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records_path = directory / "test_records.jsonl"
            freeze_path = directory / "test_freeze.json"
            write_jsonl(records_path, _records(split="test"))
            freeze_sha256 = _write_test_freeze(
                freeze_path,
                records_path,
                revision="a" * 40,
            )

            calls: list[Path] = []

            def forbidden_file_hash(path: Path) -> str:
                calls.append(path.resolve())
                raise AssertionError("authorization failure must precede file hashing")

            cases = (
                ({}, "checksum-pinned frozen manifest"),
                (
                    {"test_freeze_path": freeze_path},
                    "--test-freeze-sha256",
                ),
                (
                    {
                        "test_freeze_path": freeze_path,
                        "test_freeze_sha256": freeze_sha256,
                    },
                    "--unblind-test",
                ),
                (
                    {
                        "test_freeze_path": freeze_path,
                        "test_freeze_sha256": freeze_sha256,
                        "unblind_test": True,
                    },
                    "access provenance",
                ),
            )
            for options, message in cases:
                with self.subTest(message=message), patch.object(
                    MODULE,
                    "_sha256",
                    side_effect=forbidden_file_hash,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        MODULE.run_audit(
                            records_path,
                            delta=0.0,
                            gamma=0.0,
                            tail_quantile=0.95,
                            max_witnesses=10,
                            declared_split="test",
                            **options,
                        )
            self.assertEqual(calls, [])

    def test_valid_frozen_test_audit_records_explicit_access_boundary(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records_path = directory / "test_records.jsonl"
            freeze_path = directory / "test_freeze.json"
            write_jsonl(records_path, _records(split="test"))
            freeze_sha256 = _write_test_freeze(
                freeze_path,
                records_path,
                revision=revision,
            )
            with patch.object(
                MODULE,
                "base_manifest",
                return_value=_clean_base_manifest(revision),
            ):
                payload = MODULE.run_audit(
                    records_path,
                    delta=0.0,
                    gamma=0.0,
                    tail_quantile=0.95,
                    max_witnesses=10,
                    declared_split="test",
                    test_freeze_path=freeze_path,
                    test_freeze_sha256=freeze_sha256,
                    unblind_test=True,
                    access_provenance="unblinding-decision-2026-08-25",
                )
        self.assertTrue(payload["access"]["final_test_unblinded"])
        self.assertEqual(
            payload["access"]["unblinding_provenance"],
            "unblinding-decision-2026-08-25",
        )
        self.assertEqual(
            payload["access"]["test_freeze"]["protocol"],
            MODULE.TEST_FREEZE_PROTOCOL,
        )

    def test_test_freeze_pins_parameters_and_clean_revision(self) -> None:
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records_path = directory / "test_records.jsonl"
            freeze_path = directory / "test_freeze.json"
            write_jsonl(records_path, _records(split="test"))
            freeze_sha256 = _write_test_freeze(
                freeze_path,
                records_path,
                revision=revision,
            )
            with patch.object(
                MODULE,
                "base_manifest",
                return_value=_clean_base_manifest(revision),
            ):
                with self.assertRaisesRegex(ValueError, "parameters do not match"):
                    MODULE.run_audit(
                        records_path,
                        delta=0.1,
                        gamma=0.0,
                        tail_quantile=0.95,
                        max_witnesses=10,
                        declared_split="test",
                        test_freeze_path=freeze_path,
                        test_freeze_sha256=freeze_sha256,
                        unblind_test=True,
                        access_provenance="decision",
                    )
            dirty = _clean_base_manifest(revision)
            dirty["code"] = {"git_revision": revision, "git_dirty": True}
            with patch.object(MODULE, "base_manifest", return_value=dirty):
                with self.assertRaisesRegex(ValueError, "dirty worktree"):
                    MODULE.run_audit(
                        records_path,
                        delta=0.0,
                        gamma=0.0,
                        tail_quantile=0.95,
                        max_witnesses=10,
                        declared_split="test",
                        test_freeze_path=freeze_path,
                        test_freeze_sha256=freeze_sha256,
                        unblind_test=True,
                        access_provenance="decision",
                    )

    def test_cli_writes_once_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records_path = directory / "records.jsonl"
            output_path = directory / "audit.json"
            write_jsonl(records_path, _records())
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--records",
                str(records_path),
                "--delta",
                "0",
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
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(written["audit"]["status"], "ok")
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
