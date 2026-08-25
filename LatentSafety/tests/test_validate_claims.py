from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_claims_module", ROOT / "scripts" / "validate_claims.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load scripts/validate_claims.py")
VALIDATE_CLAIMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE_CLAIMS)


class ClaimValidatorTests(unittest.TestCase):
    def test_transitive_inputs_are_discovered_once_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "paper" / "sections"
            sections.mkdir(parents=True)
            (root / "paper" / "main.tex").write_text(
                r"\input{sections/a}\input{sections/c}", encoding="utf-8"
            )
            (sections / "a.tex").write_text(
                r"\input{sections/b}", encoding="utf-8"
            )
            (sections / "b.tex").write_text("proofs", encoding="utf-8")
            (sections / "c.tex").write_text(
                r"\input{sections/b}", encoding="utf-8"
            )

            paths = VALIDATE_CLAIMS._included_tex_paths(root)

            self.assertEqual(
                [path.relative_to(root / "paper").as_posix() for path in paths],
                ["sections/a.tex", "sections/b.tex", "sections/c.tex"],
            )

    def test_transitive_input_cycle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "paper" / "sections"
            sections.mkdir(parents=True)
            (root / "paper" / "main.tex").write_text(
                r"\input{sections/a}", encoding="utf-8"
            )
            (sections / "a.tex").write_text(
                r"\input{sections/b}", encoding="utf-8"
            )
            (sections / "b.tex").write_text(
                r"\input{sections/a}", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "cyclic manuscript input"):
                VALIDATE_CLAIMS._included_tex_paths(root)

    def test_formal_results_require_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.tex"
            path.write_text(
                r"\begin{theorem}[Named]\label{thm:one}x\end{theorem}",
                encoding="utf-8",
            )
            self.assertEqual(
                VALIDATE_CLAIMS._formal_result_labels(path), ("thm:one",)
            )

            path.write_text(
                r"\begin{proposition}x\end{proposition}", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "unlabeled formal-result proposition"
            ):
                VALIDATE_CLAIMS._formal_result_labels(path)

    def test_empirical_results_are_discovered_and_require_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.tex"
            path.write_text(
                r"\begin{table}\label{tab:one}x\end{table}"
                r"\begin{figure*}\label{fig:two}x\end{figure*}",
                encoding="utf-8",
            )
            self.assertEqual(
                VALIDATE_CLAIMS._empirical_result_labels(path),
                ("tab:one", "fig:two"),
            )

            path.write_text(r"\begin{table}x\end{table}", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unlabeled empirical-result table"
            ):
                VALIDATE_CLAIMS._empirical_result_labels(path)

    def test_reverse_empirical_registration_and_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "paper" / "sections"
            sections.mkdir(parents=True)
            (root / "paper" / "main.tex").write_text(
                r"\input{sections/results}", encoding="utf-8"
            )
            (sections / "results.tex").write_text(
                r"\begin{table}\label{tab:actual}x\end{table}",
                encoding="utf-8",
            )
            manifest = root / "evidence.json"
            manifest.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            registry = {
                "schema_version": 2,
                "claims": [
                    {
                        "id": "D-test",
                        "statement": "registered control",
                        "status": "verified_internal",
                        "scope": "included",
                        "evidence": "fixture",
                        "command": "python fixture.py",
                        "artifact_paths": ["evidence.json"],
                        "empirical_labels": ["tab:registered"],
                        "manifest": "evidence.json",
                        "manifest_sha256": digest,
                        "reviewer": "internal; independent review pending",
                        "promotion_state": "provisional_internal",
                    }
                ],
            }

            errors, _ = VALIDATE_CLAIMS.validate_registry(root, registry)

            self.assertIn(
                "empirical manuscript result is not claim-registered: 'tab:actual'",
                errors,
            )
            self.assertIn(
                "registered empirical label is not an included result: 'tab:registered'",
                errors,
            )

            registry["claims"][0]["empirical_labels"] = ["tab:actual"]
            registry["claims"][0]["manifest_sha256"] = "0" * 64
            errors, _ = VALIDATE_CLAIMS.validate_registry(root, registry)
            self.assertTrue(
                any("manifest_sha256 mismatch" in error for error in errors)
            )

    def test_table_value_check_detects_tex_cell_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "paper" / "sections"
            sections.mkdir(parents=True)
            (root / "paper" / "main.tex").write_text(
                r"\input{sections/results}", encoding="utf-8"
            )
            table_path = sections / "results.tex"

            def write_table(value: str) -> None:
                table_path.write_text(
                    r"\begin{table}\label{tab:control}"
                    r"\begin{tabular}{ll}\toprule A & B \\ \midrule "
                    f"Cart & ${value}$ "
                    r"\\ \bottomrule\end{tabular}\end{table}",
                    encoding="utf-8",
                )

            write_table("0.1")
            manifest = root / "evidence.json"
            manifest.write_text(
                '{"rows":[{"domain":"Cart","value":"0.1"}]}\n',
                encoding="utf-8",
            )
            registry = {
                "schema_version": 2,
                "claims": [
                    {
                        "id": "D-test",
                        "statement": "registered control",
                        "status": "verified_internal",
                        "scope": "included",
                        "evidence": "fixture",
                        "command": "python fixture.py",
                        "artifact_paths": ["evidence.json"],
                        "empirical_labels": ["tab:control"],
                        "manifest": "evidence.json",
                        "manifest_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        "reviewer": "internal; independent review pending",
                        "promotion_state": "provisional_internal",
                        "value_check": "latex_tabular_rows_v1",
                        "value_manifest_key": "rows",
                        "value_columns": ["domain", "value"],
                    }
                ],
            }

            errors, _ = VALIDATE_CLAIMS.validate_registry(root, registry)
            self.assertEqual(errors, [])

            write_table("0.2")
            errors, _ = VALIDATE_CLAIMS.validate_registry(root, registry)
            self.assertTrue(
                any("TeX table rows differ from manifest" in error for error in errors)
            )

    def test_table_value_check_selects_table_from_multiple_empirical_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "paper" / "sections"
            sections.mkdir(parents=True)
            (root / "paper" / "main.tex").write_text(
                r"\input{sections/results}", encoding="utf-8"
            )
            (sections / "results.tex").write_text(
                r"\begin{figure}\label{fig:control}x\end{figure}"
                r"\begin{table}\label{tab:control}"
                r"\begin{tabular}{ll}\toprule A & B \\ \midrule "
                r"Cart & $0.1$ \\ \bottomrule\end{tabular}\end{table}",
                encoding="utf-8",
            )
            manifest = root / "evidence.json"
            manifest.write_text(
                '{"rows":[{"domain":"Cart","value":"0.1"}]}\n',
                encoding="utf-8",
            )
            claim = {
                "id": "D-test",
                "statement": "registered control",
                "status": "verified_internal",
                "scope": "included",
                "evidence": "fixture",
                "command": "python fixture.py",
                "artifact_paths": ["evidence.json"],
                "empirical_labels": ["fig:control", "tab:control"],
                "manifest": "evidence.json",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "reviewer": "internal; independent review pending",
                "promotion_state": "provisional_internal",
                "value_check": "latex_tabular_rows_v1",
                "value_label": "tab:control",
                "value_manifest_key": "rows",
                "value_columns": ["domain", "value"],
            }
            registry = {"schema_version": 2, "claims": [claim]}

            errors, counts = VALIDATE_CLAIMS.validate_registry(root, registry)
            self.assertEqual(errors, [])
            self.assertEqual(counts.empirical_labels, 2)

            claim.pop("value_label")
            errors, _ = VALIDATE_CLAIMS.validate_registry(root, registry)
            self.assertTrue(
                any("requires value_label" in error for error in errors)
            )

    def test_repository_registry_validates_and_preserves_h1_ids(self) -> None:
        with (ROOT / "paper" / "claims.toml").open("rb") as stream:
            registry = tomllib.load(stream)

        errors, counts = VALIDATE_CLAIMS.validate_registry(ROOT, registry)

        self.assertEqual(errors, [])
        self.assertEqual(counts.formal_labels, 12)
        self.assertEqual(counts.empirical_labels, 2)
        claim_ids = {claim["id"] for claim in registry["claims"]}
        self.assertTrue(
            {
                "H1a-exact-sample-witness",
                "H1b-radius-rescaling",
                "H1c-nearest-neighbor-diagnostic",
                "D1-controlled-dynamic-oracle",
                "D2-complete-finite-q-bound",
            }.issubset(claim_ids)
        )


if __name__ == "__main__":
    unittest.main()
