from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.metrics.tolerance import (  # noqa: E402
    required_maximum_samples,
    tolerance_confidence,
)


class ToleranceTests(unittest.TestCase):
    def test_maximum_confidence_matches_closed_form(self) -> None:
        confidence = tolerance_confidence(sample_count=100, tail_mass=0.05, order=1)
        self.assertAlmostEqual(confidence, 1.0 - 0.95**100)

    def test_required_count_reaches_requested_confidence(self) -> None:
        count = required_maximum_samples(tail_mass=0.01, failure_probability=0.05)
        confidence = tolerance_confidence(sample_count=count, tail_mass=0.01, order=1)
        self.assertGreaterEqual(confidence, 0.95)
        previous = tolerance_confidence(sample_count=count - 1, tail_mass=0.01, order=1)
        self.assertLess(previous, 0.95)


if __name__ == "__main__":
    unittest.main()

