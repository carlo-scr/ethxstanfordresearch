#!/usr/bin/env python3
"""Execute generated research notebooks in deterministic fresh kernels.

Execution is fail-closed: source provenance is checked before a kernel starts, prior outputs are
cleared, errors and timeouts abort without replacing the notebook, and persisted output is subject
to fixed per-cell and per-notebook byte budgets.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
KERNEL_NAME = "python3"
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


def _execution_dependencies() -> tuple[Any, Any]:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "notebook execution requires the research dependencies; "
            "install with `python -m pip install -e '.[research]'`"
        ) from error
    return nbformat, NotebookClient


def _source_sha256(cells: list[Any]) -> str:
    payload = [
        {"cell_type": str(cell["cell_type"]), "source": str(cell["source"])}
        for cell in cells
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialized_output_bytes(output: Any) -> int:
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


def _output_stats(notebook: Any) -> tuple[int, int]:
    output_count = 0
    output_bytes = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        outputs = cell.get("outputs", [])
        output_count += len(outputs)
        output_bytes += sum(_serialized_output_bytes(output) for output in outputs)
    return output_count, output_bytes


def _enforce_output_limits(notebook: Any) -> tuple[int, int]:
    notebook_bytes = 0
    output_count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        outputs = cell.get("outputs", [])
        if len(outputs) > OUTPUT_LIMITS["max_outputs_per_cell"]:
            raise RuntimeError(
                f"code cell {index} produced {len(outputs)} outputs; limit is "
                f"{OUTPUT_LIMITS['max_outputs_per_cell']}"
            )
        cell_bytes = sum(_serialized_output_bytes(output) for output in outputs)
        if cell_bytes > OUTPUT_LIMITS["max_cell_output_bytes"]:
            raise RuntimeError(
                f"code cell {index} produced {cell_bytes} serialized output bytes; limit is "
                f"{OUTPUT_LIMITS['max_cell_output_bytes']}"
            )
        output_count += len(outputs)
        notebook_bytes += cell_bytes
    if notebook_bytes > OUTPUT_LIMITS["max_notebook_output_bytes"]:
        raise RuntimeError(
            f"notebook produced {notebook_bytes} serialized output bytes; limit is "
            f"{OUTPUT_LIMITS['max_notebook_output_bytes']}"
        )
    return output_count, notebook_bytes


def _resolve_notebook_path(supplied_path: Path) -> Path:
    candidate = supplied_path if supplied_path.is_absolute() else ROOT / supplied_path
    path = candidate.resolve()
    if path.suffix != ".ipynb" or not path.is_file():
        raise FileNotFoundError(f"not a notebook: {path}")
    try:
        path.relative_to(NOTEBOOK_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"notebook must be inside {NOTEBOOK_DIR}: {path}") from error
    return path


def _clear_execution_state(notebook: Any) -> None:
    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
        cell.get("metadata", {}).pop("execution", None)


def _strip_timing_metadata(notebook: Any) -> None:
    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook["cells"]:
        cell.get("metadata", {}).pop("execution", None)


@contextmanager
def _deterministic_environment(matplotlib_config_dir: Path) -> Iterator[None]:
    updates = dict(DETERMINISTIC_ENVIRONMENT)
    updates["MPLCONFIGDIR"] = str(matplotlib_config_dir)
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_atomic(nbformat: Any, notebook: Any, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        nbformat.write(notebook, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _warm_matplotlib_cache() -> None:
    """Populate the isolated font cache before notebook output capture begins."""

    subprocess.run(
        [sys.executable, "-c", "import matplotlib.font_manager"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def execute(path: Path, *, timeout: int) -> None:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    nbformat, notebook_client = _execution_dependencies()
    notebook = nbformat.read(path, as_version=4)
    latent_metadata = notebook.get("metadata", {}).get("latent_safety")
    if not isinstance(latent_metadata, dict):
        raise RuntimeError(f"{path.name} lacks latent_safety handoff metadata; rebuild it")
    if latent_metadata.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise RuntimeError(f"{path.name} has an unsupported handoff schema; rebuild it")
    source_sha256 = _source_sha256(notebook["cells"])
    if latent_metadata.get("source_sha256") != source_sha256:
        raise RuntimeError(f"{path.name} source digest is stale; rebuild it before execution")
    handoff = latent_metadata.get("handoff")
    if not isinstance(handoff, dict) or handoff.get("output_limits") != OUTPUT_LIMITS:
        raise RuntimeError(f"{path.name} has a stale output contract; rebuild it")

    _clear_execution_state(notebook)
    with tempfile.TemporaryDirectory(prefix="latent-safety-matplotlib-") as directory:
        with _deterministic_environment(Path(directory)):
            _warm_matplotlib_cache()
            client = notebook_client(
                notebook,
                timeout=timeout,
                iopub_timeout=timeout,
                kernel_name=KERNEL_NAME,
                resources={"metadata": {"path": str(path.parent)}},
                allow_errors=False,
                record_timing=False,
                interrupt_on_timeout=True,
                raise_on_iopub_timeout=True,
                coalesce_streams=True,
            )
            client.execute()

    _strip_timing_metadata(notebook)
    output_count, output_bytes = _enforce_output_limits(notebook)
    executable_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and str(cell.get("source", "")).strip()
        and "skip-execution" not in cell.get("metadata", {}).get("tags", [])
    ]
    executed_cells = sum(cell.get("execution_count") is not None for cell in executable_cells)
    if executed_cells != len(executable_cells):
        raise RuntimeError(
            f"{path.name} executed {executed_cells} of {len(executable_cells)} code cells"
        )
    latent_metadata["execution"] = {
        "state": "complete",
        "executed_by": "scripts/execute_notebooks.py",
        "source_sha256": source_sha256,
        "kernel_name": KERNEL_NAME,
        "working_directory": "notebooks",
        "timeout_seconds": timeout,
        "record_timing": False,
        "allow_errors": False,
        "deterministic_environment": dict(DETERMINISTIC_ENVIRONMENT),
        "code_cells_executed": executed_cells,
        "output_count": output_count,
        "output_bytes": output_bytes,
    }
    _write_atomic(nbformat, notebook, path)
    print(
        f"executed {path.relative_to(ROOT)} "
        f"({executed_cells} code cells, {output_bytes} output bytes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    supplied_paths = args.paths or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not supplied_paths:
        raise SystemExit("no notebooks found")
    paths = [_resolve_notebook_path(path) for path in supplied_paths]
    for path in paths:
        execute(path, timeout=args.timeout)


if __name__ == "__main__":
    main()
