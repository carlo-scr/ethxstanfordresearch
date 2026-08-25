#!/usr/bin/env python3
"""Validate generated notebook provenance, output bounds, and execution handoff state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = (
    "00_e0_estimator_validation.ipynb",
    "01_e1_torch_world_model_pilot.ipynb",
    "02_theory_counterexamples.ipynb",
    "03_pilot_analysis_template.ipynb",
    "04_two_domain_oracle_controls.ipynb",
    "05_dynamic_regret_oracle.ipynb",
    "06_confirmatory_inference_preflight.ipynb",
    "07_method_protocol_smoke.ipynb",
)
HANDOFF_SCHEMA_VERSION = 1
OUTPUT_LIMITS = {
    "max_outputs_per_cell": 32,
    "max_cell_output_bytes": 512_000,
    "max_notebook_output_bytes": 2_000_000,
}
DETERMINISTIC_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "LATENT_SAFETY_NOTEBOOK_E2E": "0",
    "LATENT_SAFETY_NOTEBOOK_OUTPUT": "runs/e1_world_models/notebook_cpu_smoke",
}
EXPECTED_HANDOFF = {
    "dependency_extra": "research",
    "working_directory": "notebooks",
    "build_command": "python scripts/build_notebooks.py",
    "execute_command": "python scripts/execute_notebooks.py",
    "validate_command": "python3 scripts/validate_notebooks.py --require-executed",
    "output_limits": OUTPUT_LIMITS,
}


def _source_sha256(cells: list[dict[str, Any]]) -> str:
    payload = [
        {"cell_type": str(cell.get("cell_type", "")), "source": _source(cell)}
        for cell in cells
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _stable_cell_id(index: int, cell: dict[str, Any]) -> str:
    payload = f"{index}\0{cell.get('cell_type', '')}\0{_source(cell)}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _serialized_output_bytes(output: dict[str, Any]) -> int:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return "".join(value)
            return [normalize(item) for item in value]
        return value

    return len(
        json.dumps(
            normalize(output),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _contains_absolute_workspace_path(serialized: str) -> bool:
    normalized = serialized.replace("\\\\", "/").replace("\\", "/")
    if str(ROOT.resolve()) in serialized:
        return True
    home_prefix = "/Users/" in normalized or "/home/" in normalized
    windows_home = ":/Users/" in normalized
    return (home_prefix or windows_home) and "/LatentSafety" in normalized


def _validate_output_contract(
    cells: list[dict[str, Any]], errors: list[str]
) -> tuple[int, int]:
    output_count = 0
    notebook_bytes = 0
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            errors.append(f"code cell {index} outputs must be a list")
            continue
        if len(outputs) > OUTPUT_LIMITS["max_outputs_per_cell"]:
            errors.append(
                f"code cell {index} has {len(outputs)} outputs; limit is "
                f"{OUTPUT_LIMITS['max_outputs_per_cell']}"
            )
        cell_bytes = 0
        for output in outputs:
            if not isinstance(output, dict):
                errors.append(f"code cell {index} contains a non-object output")
                continue
            serialized = json.dumps(output, ensure_ascii=False, sort_keys=True)
            if _contains_absolute_workspace_path(serialized):
                errors.append(f"code cell {index} persists an absolute workspace path")
            if output.get("output_type") == "error":
                errors.append(
                    f"code cell {index} contains {output.get('ename')}: "
                    f"{output.get('evalue')}"
                )
            cell_bytes += _serialized_output_bytes(output)
        if cell_bytes > OUTPUT_LIMITS["max_cell_output_bytes"]:
            errors.append(
                f"code cell {index} has {cell_bytes} serialized output bytes; limit is "
                f"{OUTPUT_LIMITS['max_cell_output_bytes']}"
            )
        output_count += len(outputs)
        notebook_bytes += cell_bytes
    if notebook_bytes > OUTPUT_LIMITS["max_notebook_output_bytes"]:
        errors.append(
            f"notebook has {notebook_bytes} serialized output bytes; limit is "
            f"{OUTPUT_LIMITS['max_notebook_output_bytes']}"
        )
    return output_count, notebook_bytes


def validate(path: Path, *, require_executed: bool) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read notebook JSON: {error}"]

    if not isinstance(notebook, dict):
        return ["notebook root must be an object"]

    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    cells = notebook.get("cells", [])
    if not isinstance(cells, list) or not cells:
        return ["notebook has no cells"]
    if not all(isinstance(cell, dict) for cell in cells):
        return ["every notebook cell must be an object"]
    first_source = _source(cells[0])
    if cells[0].get("cell_type") != "markdown" or not first_source.lstrip().startswith("# "):
        errors.append("first cell must be a level-one Markdown title")
    markdown = "\n".join(
        _source(cell)
        for cell in cells
        if cell.get("cell_type") == "markdown"
    ).lower()
    for required_section in ("tl;dr", "assumption", "takeaway"):
        if required_section not in markdown:
            errors.append(f"missing reader-facing section containing {required_section!r}")

    cell_ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        errors.append("every cell must have a nonempty deterministic id")
    elif len(set(cell_ids)) != len(cell_ids):
        errors.append("cell ids must be unique")
    for index, cell in enumerate(cells):
        if cell.get("id") != _stable_cell_id(index, cell):
            errors.append(f"cell {index} id does not match its deterministic source id")
        if "execution" in cell.get("metadata", {}):
            errors.append(f"cell {index} contains nondeterministic execution timing metadata")

    metadata = notebook.get("metadata", {})
    if _contains_absolute_workspace_path(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    ):
        errors.append("notebook metadata persists an absolute workspace path")
    latent_metadata = metadata.get("latent_safety") if isinstance(metadata, dict) else None
    source_sha256 = _source_sha256(cells)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    executable_cells = [
        cell
        for cell in code_cells
        if _source(cell).strip()
        and "skip-execution" not in cell.get("metadata", {}).get("tags", [])
    ]
    if not isinstance(latent_metadata, dict):
        errors.append("missing latent_safety handoff metadata")
        latent_metadata = {}
    if latent_metadata.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        errors.append(f"latent_safety.schema_version must be {HANDOFF_SCHEMA_VERSION}")
    if latent_metadata.get("generated_by") != "scripts/build_notebooks.py":
        errors.append("latent_safety.generated_by must identify scripts/build_notebooks.py")
    if latent_metadata.get("source_sha256") != source_sha256:
        errors.append("latent_safety.source_sha256 does not match notebook sources")
    if latent_metadata.get("cell_count") != len(cells):
        errors.append("latent_safety.cell_count does not match notebook cells")
    if latent_metadata.get("code_cell_count") != len(code_cells):
        errors.append("latent_safety.code_cell_count does not match notebook code cells")
    if latent_metadata.get("handoff") != EXPECTED_HANDOFF:
        errors.append("latent_safety.handoff does not match the execution contract")

    output_count, output_bytes = _validate_output_contract(cells, errors)
    execution = latent_metadata.get("execution")
    if not isinstance(execution, dict):
        errors.append("latent_safety.execution must be an object")
        execution = {}
    state = execution.get("state")
    if state not in {"not_executed", "complete"}:
        errors.append("latent_safety.execution.state must be not_executed or complete")
    if require_executed and state != "complete":
        errors.append("notebook handoff state is not complete")

    if state == "not_executed":
        if execution != {"state": "not_executed"}:
            errors.append("unexecuted notebook has unexpected execution metadata")
        for index, cell in enumerate(code_cells):
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(f"unexecuted code cell {index} retains stale execution state")
    elif state == "complete":
        expected_execution = {
            "state": "complete",
            "executed_by": "scripts/execute_notebooks.py",
            "source_sha256": source_sha256,
            "kernel_name": "python3",
            "working_directory": "notebooks",
            "timeout_seconds": execution.get("timeout_seconds"),
            "record_timing": False,
            "allow_errors": False,
            "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
            "code_cells_executed": len(executable_cells),
            "output_count": output_count,
            "output_bytes": output_bytes,
        }
        if execution != expected_execution:
            errors.append("latent_safety.execution does not match persisted execution state")
        timeout_seconds = execution.get("timeout_seconds")
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            errors.append("latent_safety.execution.timeout_seconds must be positive")
        execution_counts = [cell.get("execution_count") for cell in executable_cells]
        if execution_counts != list(range(1, len(executable_cells) + 1)):
            errors.append("executable code-cell counts must be sequential from one")
    return errors


def validate_inventory(notebook_dir: Path = NOTEBOOK_DIR) -> list[str]:
    observed = tuple(sorted(path.name for path in notebook_dir.glob("*.ipynb")))
    if observed == EXPECTED_NOTEBOOKS:
        return []
    missing = sorted(set(EXPECTED_NOTEBOOKS) - set(observed))
    unexpected = sorted(set(observed) - set(EXPECTED_NOTEBOOKS))
    return [f"notebook inventory mismatch: missing={missing!r}, unexpected={unexpected!r}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--require-executed", action="store_true")
    args = parser.parse_args()

    paths = args.paths or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not paths:
        raise SystemExit("no notebooks found")
    failures = 0
    if not args.paths:
        inventory_errors = validate_inventory()
        if inventory_errors:
            failures += 1
            print("FAIL notebooks inventory")
            for error in inventory_errors:
                print(f"  - {error}")
    for supplied_path in paths:
        path = supplied_path if supplied_path.is_absolute() else ROOT / supplied_path
        errors = validate(path, require_executed=args.require_executed)
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        if errors:
            failures += 1
            print(f"FAIL {display_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {display_path}")
    if failures:
        raise SystemExit(f"{failures} notebook(s) failed validation")


if __name__ == "__main__":
    main()
