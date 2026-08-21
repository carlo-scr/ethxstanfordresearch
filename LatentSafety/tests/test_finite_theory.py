import unittest

from latent_safety.metrics.finite import (
    audit_finite_static_fibers,
    check_finite_data_processing,
    check_sound_completeness,
    enumerate_binary_certificates,
)


class FiniteStaticTheoryTests(unittest.TestCase):
    def test_exact_audit_and_maximal_sound_certificate(self) -> None:
        latents = [(0.0,), (0.0,), (1.0,), (2.0,)]
        margins = [0.4, -0.1, 0.8, -0.2]
        audit = audit_finite_static_fibers(latents, margins)

        self.assertEqual(audit.mixed_fiber_count, 1)
        self.assertEqual(audit.collided_safe_count, 1)
        self.assertAlmostEqual(audit.exact_defect, 0.4)
        self.assertEqual(audit.maximal_sound_codes, ((1.0,),))

    def test_finite_endpoint_is_explicit(self) -> None:
        latents = [(0.0,), (0.0,), (1.0,)]
        margins = [0.4, -0.1, 0.8]

        self.assertFalse(
            check_sound_completeness(
                latents, margins, completeness_margin=0.4
            ).exists
        )
        self.assertTrue(
            check_sound_completeness(
                latents, margins, completeness_margin=0.400001
            ).exists
        )

    def test_zero_defect_does_not_encode_endpoint_attainment(self) -> None:
        no_collision = check_sound_completeness(
            [(0.0,), (1.0,)], [0.0, -0.1], completeness_margin=0.0
        )
        boundary_collision = check_sound_completeness(
            [(0.0,), (0.0,)], [0.0, -0.1], completeness_margin=0.0
        )

        self.assertTrue(no_collision.exists)
        self.assertFalse(boundary_collision.exists)
        self.assertEqual(
            audit_finite_static_fibers([(0.0,), (1.0,)], [0.0, -0.1]).exact_defect,
            0.0,
        )
        self.assertEqual(
            audit_finite_static_fibers([(0.0,), (0.0,)], [0.0, -0.1]).exact_defect,
            0.0,
        )

    def test_data_processing_under_coarsening(self) -> None:
        margins = [0.8, -0.1, 0.3]
        fine = [(0.0,), (1.0,), (2.0,)]
        coarse = [(0.0,), (0.0,), (2.0,)]
        result = check_finite_data_processing(fine, coarse, margins)

        self.assertTrue(result.is_deterministic_postprocessing)
        self.assertTrue(result.monotone)
        self.assertEqual(result.fine_defect, 0.0)
        self.assertEqual(result.coarse_defect, 0.8)

    def test_detects_missing_postprocessing_premise(self) -> None:
        result = check_finite_data_processing(
            [(0.0,), (0.0,)], [(0.0,), (1.0,)], [0.2, -0.1]
        )
        self.assertFalse(result.is_deterministic_postprocessing)
        self.assertFalse(result.monotone)

    def test_certificate_enumeration_guard(self) -> None:
        certificates = enumerate_binary_certificates([(0.0,), (1.0,)])
        self.assertEqual(len(certificates), 4)
        with self.assertRaises(ValueError):
            enumerate_binary_certificates((float(index),) for index in range(21))


if __name__ == "__main__":
    unittest.main()
