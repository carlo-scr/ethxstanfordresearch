from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery guard
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_script("build_notebooks.py")
EXECUTE = _load_script("execute_notebooks.py")
VALIDATE = _load_script("validate_notebooks.py")


class NotebookPipelineTests(unittest.TestCase):
    def test_repository_inventory_and_executed_notebooks_validate(self) -> None:
        self.assertEqual(VALIDATE.validate_inventory(), [])
        for name in VALIDATE.EXPECTED_NOTEBOOKS:
            with self.subTest(name=name):
                self.assertEqual(
                    VALIDATE.validate(
                        ROOT / "notebooks" / name,
                        require_executed=True,
                    ),
                    [],
                )

    def test_contract_constants_match_across_pipeline(self) -> None:
        self.assertEqual(BUILD.HANDOFF_SCHEMA_VERSION, EXECUTE.HANDOFF_SCHEMA_VERSION)
        self.assertEqual(BUILD.HANDOFF_SCHEMA_VERSION, VALIDATE.HANDOFF_SCHEMA_VERSION)
        self.assertEqual(BUILD.OUTPUT_LIMITS, EXECUTE.OUTPUT_LIMITS)
        self.assertEqual(BUILD.OUTPUT_LIMITS, VALIDATE.OUTPUT_LIMITS)
        self.assertEqual(
            EXECUTE.DETERMINISTIC_ENVIRONMENT,
            VALIDATE.DETERMINISTIC_ENVIRONMENT,
        )
        self.assertEqual(tuple(BUILD.BUILDERS), VALIDATE.EXPECTED_NOTEBOOKS)

    def test_cell_ids_and_source_digest_are_deterministic(self) -> None:
        cell = {"cell_type": "code", "source": "value = 1"}
        first_id = BUILD._stable_cell_id(3, cell)
        self.assertEqual(first_id, BUILD._stable_cell_id(3, copy.deepcopy(cell)))
        self.assertNotEqual(first_id, BUILD._stable_cell_id(4, cell))
        self.assertNotEqual(
            first_id,
            BUILD._stable_cell_id(3, {**cell, "source": "value = 2"}),
        )
        cells = [cell, {"cell_type": "markdown", "source": "# Title"}]
        self.assertEqual(BUILD._source_sha256(cells), EXECUTE._source_sha256(cells))

    def test_output_budget_rejects_oversized_or_noisy_cells(self) -> None:
        oversized = {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": "x" * EXECUTE.OUTPUT_LIMITS["max_cell_output_bytes"],
                        }
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "serialized output bytes"):
            EXECUTE._enforce_output_limits(oversized)

        too_many = copy.deepcopy(oversized)
        too_many["cells"][0]["outputs"] = [
            {"output_type": "stream", "name": "stdout", "text": "x"}
            for _ in range(EXECUTE.OUTPUT_LIMITS["max_outputs_per_cell"] + 1)
        ]
        with self.assertRaisesRegex(RuntimeError, "produced 33 outputs"):
            EXECUTE._enforce_output_limits(too_many)

    def test_validator_rejects_persisted_absolute_workspace_path(self) -> None:
        source_path = ROOT / "notebooks" / VALIDATE.EXPECTED_NOTEBOOKS[0]
        notebook = json.loads(source_path.read_text(encoding="utf-8"))
        code_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        code_cell["outputs"].append(
            {
                "output_type": "stream",
                "name": "stdout",
                "text": f"artifact={ROOT}/runs/private.json\n",
            }
        )
        notebook["metadata"]["absolute_workspace_artifact"] = str(
            ROOT / "runs" / "private.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absolute.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            errors = VALIDATE.validate(path, require_executed=True)
        self.assertTrue(
            any("persists an absolute workspace path" in error for error in errors)
        )
        self.assertTrue(
            any("metadata persists an absolute workspace path" in error for error in errors)
        )

    def test_validator_rejects_source_or_timing_drift(self) -> None:
        source_path = ROOT / "notebooks" / VALIDATE.EXPECTED_NOTEBOOKS[0]
        notebook = json.loads(source_path.read_text(encoding="utf-8"))
        notebook["cells"][0]["source"][0] = "# Changed title\n"
        notebook["cells"][0]["metadata"]["execution"] = {"iopub.status.busy": "now"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            errors = VALIDATE.validate(path, require_executed=True)
        self.assertTrue(any("source_sha256" in error for error in errors))
        self.assertTrue(any("deterministic source id" in error for error in errors))
        self.assertTrue(any("timing metadata" in error for error in errors))

    def test_deterministic_environment_is_scoped(self) -> None:
        original = os.environ.get("LATENT_SAFETY_NOTEBOOK_E2E")
        os.environ["LATENT_SAFETY_NOTEBOOK_E2E"] = "1"
        try:
            with tempfile.TemporaryDirectory() as directory:
                with EXECUTE._deterministic_environment(Path(directory)):
                    self.assertEqual(os.environ["LATENT_SAFETY_NOTEBOOK_E2E"], "0")
                    self.assertEqual(os.environ["PYTHONHASHSEED"], "0")
                    self.assertEqual(os.environ["MPLCONFIGDIR"], directory)
            self.assertEqual(os.environ["LATENT_SAFETY_NOTEBOOK_E2E"], "1")
        finally:
            if original is None:
                os.environ.pop("LATENT_SAFETY_NOTEBOOK_E2E", None)
            else:
                os.environ["LATENT_SAFETY_NOTEBOOK_E2E"] = original

    def test_executor_rejects_notebooks_outside_owned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.ipynb"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be inside"):
                EXECUTE._resolve_notebook_path(path)


if __name__ == "__main__":
    unittest.main()
