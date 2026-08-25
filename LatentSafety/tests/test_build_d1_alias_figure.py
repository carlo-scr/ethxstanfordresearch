from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_d1_alias_figure_module",
    ROOT / "scripts/build_d1_alias_figure.py",
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load scripts/build_d1_alias_figure.py")
BUILD_FIGURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD_FIGURE
SPEC.loader.exec_module(BUILD_FIGURE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_render_dependencies() -> bool:
    return (
        importlib.util.find_spec("matplotlib") is not None
        and importlib.util.find_spec("numpy") is not None
    )


class D1AliasFigureDataTests(unittest.TestCase):
    def test_registered_roots_have_exact_alias_and_action_set_semantics(self) -> None:
        cart, pendulum = BUILD_FIGURE.collect_figure_data()

        self.assertEqual(
            (cart.domain, pendulum.domain),
            ("controlled_cart_video", "controlled_pendulum_video"),
        )
        for domain in (cart, pendulum):
            self.assertTrue(domain.current_frames_pixel_identical)
            self.assertTrue(domain.previous_frames_pixel_different)
            self.assertEqual(domain.roots[0].current_frame, domain.roots[1].current_frame)
            self.assertNotEqual(domain.roots[0].previous_frame, domain.roots[1].previous_frame)
            self.assertEqual(domain.viable_first_action_intersection, ())

        self.assertEqual(cart.roots[0].viable_first_actions, (1.0,))
        self.assertEqual(cart.roots[1].viable_first_actions, (-1.0,))
        self.assertEqual(pendulum.roots[0].viable_first_actions, (1.0,))
        self.assertEqual(pendulum.roots[1].viable_first_actions, (-1.0, 0.0))
        self.assertEqual(
            tuple(root.hidden_velocity for root in cart.roots),
            (-1.85, 3.85),
        )
        self.assertEqual(
            tuple(root.hidden_velocity for root in pendulum.roots),
            (-3.0, 3.25),
        )

    def test_tampered_evidence_fails_before_rendering(self) -> None:
        evidence = json.loads(BUILD_FIGURE.EVIDENCE_PATH.read_text(encoding="utf-8"))
        evidence["reports"][0]["fixture_roots"][0]["current_kinematics"][0] = 99.0
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "altered_evidence.json"
            altered.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current-root mismatch"):
                BUILD_FIGURE.collect_figure_data(evidence_path=altered)


@unittest.skipUnless(_has_render_dependencies(), "Matplotlib and NumPy are optional")
class D1AliasFigureArtifactTests(unittest.TestCase):
    def _build(self, directory: Path) -> tuple[Path, Path, Path, dict[str, object]]:
        pdf = directory / "d1_controlled_aliases.pdf"
        png = directory / "d1_controlled_aliases.png"
        sidecar = directory / "d1_controlled_aliases.json"
        payload = BUILD_FIGURE.build_artifacts(
            pdf_path=pdf,
            png_path=png,
            sidecar_path=sidecar,
        )
        return pdf, png, sidecar, payload

    def test_outputs_and_sidecar_are_complete_and_self_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pdf, png, sidecar, payload = self._build(Path(temporary))

            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(pdf.stat().st_size, 10_000)
            self.assertGreater(png.stat().st_size, 20_000)
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(stored, payload)
            self.assertEqual(stored["outputs"]["pdf"]["sha256"], _sha256(pdf))
            self.assertEqual(stored["outputs"]["png"]["sha256"], _sha256(png))
            self.assertEqual(
                stored["provenance"]["evidence"]["sha256"],
                _sha256(BUILD_FIGURE.EVIDENCE_PATH),
            )
            self.assertEqual(
                stored["assertions"],
                {
                    "all_current_frame_pairs_pixel_identical": True,
                    "all_previous_frame_pairs_pixel_different": True,
                    "all_viable_first_action_intersections_empty": True,
                },
            )
            self.assertEqual(len(stored["domains"]), 2)
            self.assertEqual(len(stored["provenance"]["config_files"]), 2)
            self.assertIn(
                "src/latent_safety/benchmarks/dynamic_oracle.py",
                stored["provenance"]["source_files"],
            )

    def test_pdf_and_png_are_byte_deterministic_across_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as first_temporary:
            first_pdf, first_png, _, _ = self._build(Path(first_temporary))
            first_hashes = _sha256(first_pdf), _sha256(first_png)
        with tempfile.TemporaryDirectory() as second_temporary:
            second_pdf, second_png, _, _ = self._build(Path(second_temporary))
            second_hashes = _sha256(second_pdf), _sha256(second_png)

        self.assertEqual(first_hashes, second_hashes)


if __name__ == "__main__":
    unittest.main()
