#!/usr/bin/env python3
"""Execute research notebooks in fresh kernels and persist their bounded outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def execute(path: Path, *, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        record_timing=True,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"executed {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/latent-safety-matplotlib")
    paths = args.paths or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    for supplied_path in paths:
        path = supplied_path if supplied_path.is_absolute() else ROOT / supplied_path
        if path.suffix != ".ipynb" or not path.is_file():
            raise FileNotFoundError(f"not a notebook: {path}")
        execute(path, timeout=args.timeout)


if __name__ == "__main__":
    main()
