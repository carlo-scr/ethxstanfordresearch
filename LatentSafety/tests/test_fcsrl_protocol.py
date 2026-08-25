from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.fcsrl_protocol import (  # noqa: E402
    FCSRL_ATOM_COUNT,
    FCSRL_RETURN_LENGTH,
    FCSRLProtocolError,
    build_target_trace,
    categorical_mean,
    project_categorical_target,
    projected_cross_entropy,
    sequence_mask,
    symexp,
    symlog,
)


class FCSRLProtocolTests(unittest.TestCase):
    def test_symlog_and_symexp_are_inverse(self) -> None:
        for value in (-100.0, -1.0, -0.0, 0.0, 0.25, 5.0, 100.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(symexp(symlog(value)), value, places=12)

    def test_projection_has_two_adjacent_atoms_and_decodes_clipped_target(self) -> None:
        for target in (-100.0, -0.4, 0.0, 0.8, 10.0, 100.0):
            with self.subTest(target=target):
                projection = project_categorical_target(target)
                self.assertEqual(len(projection), FCSRL_ATOM_COUNT)
                self.assertAlmostEqual(sum(projection), 1.0)
                nonzero = [index for index, weight in enumerate(projection) if weight > 0.0]
                self.assertLessEqual(len(nonzero), 2)
                if len(nonzero) == 2:
                    self.assertEqual(nonzero[1] - nonzero[0], 1)
                expected = symexp(min(4.0, max(-2.0, symlog(target))))
                self.assertAlmostEqual(categorical_mean(projection), expected, places=11)

    def test_target_recursion_matches_closed_form_without_events(self) -> None:
        trace = build_target_trace(
            violations=(0,) * FCSRL_RETURN_LENGTH,
            terminations=(False,) * FCSRL_RETURN_LENGTH,
            truncations=(False,) * FCSRL_RETURN_LENGTH,
            bootstrap_values=(2.0,) * FCSRL_RETURN_LENGTH,
        )
        expected = tuple(2.0 * 0.9 ** (9 - index) for index in range(10))
        for observed, target in zip(trace.targets, expected, strict=True):
            self.assertAlmostEqual(observed, target)
        self.assertEqual(trace.mask, (1,) * 10)
        self.assertEqual(trace.train_targets, trace.targets[:4])
        self.assertEqual(trace.train_mask, (1,) * 4)

    def test_termination_and_truncation_semantics_are_distinct(self) -> None:
        terminations = [False] * 10
        terminations[2] = True
        terminated = build_target_trace(
            violations=(0,) * 10,
            terminations=terminations,
            truncations=(False,) * 10,
            bootstrap_values=(3.0,) * 10,
        )
        self.assertEqual(terminated.targets[2], 0.0)
        self.assertEqual(terminated.mask[:4], (1, 1, 1, 0))

        truncations = [False] * 10
        truncations[2] = True
        truncated = build_target_trace(
            violations=(0,) * 10,
            terminations=(False,) * 10,
            truncations=truncations,
            bootstrap_values=(3.0,) * 10,
        )
        self.assertAlmostEqual(truncated.targets[2], 2.7)
        self.assertEqual(truncated.mask[:4], (1, 1, 1, 0))

    def test_violation_is_a_lower_clamp(self) -> None:
        violations = [0] * 10
        violations[5] = 1
        trace = build_target_trace(
            violations=violations,
            terminations=(False,) * 10,
            truncations=(False,) * 10,
            bootstrap_values=(0.0,) * 10,
        )
        self.assertEqual(trace.targets[5], 1.0)
        self.assertAlmostEqual(trace.targets[4], 0.9)
        self.assertAlmostEqual(trace.targets[0], 0.9**5)

    def test_cross_entropy_is_finite_and_prefers_matching_logits(self) -> None:
        target = project_categorical_target(0.75)
        matching = [20.0 if weight > 0.0 else -20.0 for weight in target]
        flat = [0.0] * FCSRL_ATOM_COUNT
        self.assertLess(
            projected_cross_entropy(matching, target),
            projected_cross_entropy(flat, target),
        )
        self.assertTrue(math.isfinite(projected_cross_entropy(flat, target)))

    def test_invalid_sequences_fail_closed(self) -> None:
        with self.assertRaisesRegex(FCSRLProtocolError, "length 10"):
            sequence_mask((False,), (False,))
        with self.assertRaisesRegex(FCSRLProtocolError, "both"):
            sequence_mask(
                (True,) + (False,) * 9,
                (True,) + (False,) * 9,
            )
        with self.assertRaisesRegex(FCSRLProtocolError, "binary"):
            build_target_trace(
                violations=(2,) + (0,) * 9,
                terminations=(False,) * 10,
                truncations=(False,) * 10,
                bootstrap_values=(0.0,) * 10,
            )


if __name__ == "__main__":
    unittest.main()
