from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.metrics.faithfulness import (  # noqa: E402
    cover_robust_defect_bound,
    pairwise_faithfulness_audit,
)


class FaithfulnessTests(unittest.TestCase):
    def test_identity_margin_is_one_faithful(self) -> None:
        audit = pairwise_faithfulness_audit(
            latents=((-1.0,), (0.0,), (2.0,)),
            margins=(-1.0, 0.0, 2.0),
            kappa=1.0,
        )
        self.assertEqual(audit.maximum_violation, 0.0)

    def test_collapsed_codes_violate_faithfulness(self) -> None:
        audit = pairwise_faithfulness_audit(
            latents=((0.0,), (0.0,)),
            margins=(-1.0, 1.0),
            kappa=100.0,
        )
        self.assertEqual(audit.maximum_violation, 2.0)

    def test_cover_bound_keeps_both_approximation_factors(self) -> None:
        bound = cover_robust_defect_bound(
            cover_radius=0.1,
            margin_lipschitz=2.0,
            encoder_lipschitz=3.0,
            kappa=4.0,
            latent_radius=0.5,
        )
        self.assertAlmostEqual(bound, 2 * 2 * 0.1 + 4 * (0.5 + 2 * 3 * 0.1))


if __name__ == "__main__":
    unittest.main()

