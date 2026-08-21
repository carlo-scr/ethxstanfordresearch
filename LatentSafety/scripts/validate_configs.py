#!/usr/bin/env python3
"""Parse every experiment TOML and enforce minimal provenance fields."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").rglob("*.toml"))
    errors: list[str] = []
    seen_experiments: set[str] = set()
    for path in paths:
        try:
            with path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
            continue
        for key in ("schema_version", "experiment"):
            if key not in config:
                errors.append(f"{path.relative_to(root)}: missing {key!r}")
        experiment = config.get("experiment")
        if experiment in seen_experiments:
            errors.append(f"{path.relative_to(root)}: duplicate experiment {experiment!r}")
        if isinstance(experiment, str):
            seen_experiments.add(experiment)

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(paths)} experiment configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

