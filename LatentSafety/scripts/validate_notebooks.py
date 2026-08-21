#!/usr/bin/env python3
"""Validate notebook structure and, optionally, persisted execution state using stdlib JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def validate(path: Path, *, require_executed: bool) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    cells = notebook.get("cells", [])
    if not cells:
        return ["notebook has no cells"]
    first_source = "".join(cells[0].get("source", []))
    if cells[0].get("cell_type") != "markdown" or not first_source.lstrip().startswith("# "):
        errors.append("first cell must be a level-one Markdown title")
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "markdown"
    ).lower()
    for required_section in ("tl;dr", "assumption", "takeaway"):
        if required_section not in markdown:
            errors.append(f"missing reader-facing section containing {required_section!r}")

    if require_executed:
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code" or not "".join(cell.get("source", [])).strip():
                continue
            tags = cell.get("metadata", {}).get("tags", [])
            if "skip-execution" not in tags and cell.get("execution_count") is None:
                errors.append(f"code cell {index} has no persisted execution count")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"code cell {index} contains {output.get('ename')}: {output.get('evalue')}"
                    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--require-executed", action="store_true")
    args = parser.parse_args()

    paths = args.paths or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not paths:
        raise SystemExit("no notebooks found")
    failures = 0
    for supplied_path in paths:
        path = supplied_path if supplied_path.is_absolute() else ROOT / supplied_path
        errors = validate(path, require_executed=args.require_executed)
        if errors:
            failures += 1
            print(f"FAIL {path.relative_to(ROOT)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit(f"{failures} notebook(s) failed validation")


if __name__ == "__main__":
    main()
