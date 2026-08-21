from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.records import (  # noqa: E402
    AuditRecord,
    read_jsonl,
    validate_records,
    write_jsonl,
)


def _record(sample_id: str, trajectory_id: str, split: str) -> AuditRecord:
    return AuditRecord(
        sample_id=sample_id,
        trajectory_id=trajectory_id,
        split=split,
        safety_margin=0.2,
        latent=(0.1, 0.3),
        observation_latent=(1.0, 2.0),
        state_latent=(0.25, -0.5),
        action_safety_margins=(-0.2, 0.4),
    )


class RecordTests(unittest.TestCase):
    def test_rejects_trajectory_leakage(self) -> None:
        records = (_record("a", "trajectory-1", "train"), _record("b", "trajectory-1", "test"))
        with self.assertRaisesRegex(ValueError, "occurs in both"):
            validate_records(records)

    def test_jsonl_round_trip(self) -> None:
        records = (
            _record("a", "trajectory-1", "train"),
            _record("b", "trajectory-2", "test"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            write_jsonl(path, records)
            self.assertEqual(read_jsonl(path), records)


if __name__ == "__main__":
    unittest.main()
