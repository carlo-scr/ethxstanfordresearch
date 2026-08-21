#!/usr/bin/env python3
"""Expand a versioned E1 factorial grid into a deterministic, auditable task plan."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.config import load_config, validate_config  # noqa: E402
from latent_safety.manifest import config_sha256, write_json_atomic  # noqa: E402


_HISTORY_MODE_DEFINITIONS: dict[str, tuple[str, int]] = {
    "stack_h1": ("stack", 1),
    "stack_h4": ("stack", 4),
    "gru_h4": ("gru", 4),
}


def _repo_relative(path: Path, *, label: str) -> Path:
    """Return a portable repository-relative path, rejecting path escapes."""

    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside the repository: {resolved}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must name a file or subdirectory, not the repository root")
    return relative


def _output_root(value: object) -> Path:
    """Validate that task outputs are isolated below the managed run directory."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("execution.output_root must be a non-empty string")
    declared = Path(value)
    if declared.is_absolute():
        raise ValueError("execution.output_root must be repository-relative")
    relative = _repo_relative(ROOT / declared, label="execution.output_root")
    managed_root = Path("runs") / "e1_world_models"
    try:
        relative.relative_to(managed_root)
    except ValueError as error:
        raise ValueError(
            "execution.output_root must lie below runs/e1_world_models"
        ) from error
    return relative


def _git_code_state() -> dict[str, str | bool | None]:
    """Capture whether the plan was derived from an identifiable clean revision."""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _analysis_config(value: object) -> dict[str, Any]:
    """Validate audit inputs before any GPU time can be spent."""

    if not isinstance(value, dict):
        raise ValueError("grid [analysis] must be a TOML table")
    radii = value.get("relative_radii")
    if not isinstance(radii, list) or not radii:
        raise ValueError("analysis.relative_radii must be a non-empty list")
    normalized: list[float] = []
    for radius in radii:
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("analysis.relative_radii must contain only finite numbers")
        numeric = float(radius)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("analysis.relative_radii must be finite and non-negative")
        normalized.append(numeric)
    if normalized != sorted(set(normalized)):
        raise ValueError("analysis.relative_radii must be sorted and unique")
    primary = value.get("primary_relative_radius")
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not math.isfinite(float(primary))
        or float(primary) not in normalized
    ):
        raise ValueError(
            "analysis.primary_relative_radius must be one of analysis.relative_radii"
        )
    if value.get("selection_split") != "validation":
        raise ValueError("analysis.selection_split must be exactly 'validation'")
    return value


def _axis(table: dict[str, Any], name: str, expected: type) -> tuple[Any, ...]:
    values = table.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"axes.{name} must be a non-empty list")
    if any(not isinstance(value, expected) or isinstance(value, bool) for value in values):
        raise ValueError(f"axes.{name} contains an invalid value")
    if len(set(values)) != len(values):
        raise ValueError(f"axes.{name} must not contain duplicates")
    return tuple(values)


def build_plan(grid_path: Path) -> dict[str, Any]:
    grid_path = grid_path.resolve()
    grid_relative = _repo_relative(grid_path, label="grid path")
    with grid_path.open("rb") as stream:
        grid = tomllib.load(stream)
    if grid.get("schema_version") != 1:
        raise ValueError("only schema_version = 1 grids are supported")
    axes = grid.get("axes")
    execution = grid.get("execution")
    if not isinstance(axes, dict) or not isinstance(execution, dict):
        raise ValueError("grid requires [axes] and [execution] tables")

    families = _axis(axes, "model_families", str)
    history_modes = _axis(axes, "history_modes", str)
    unknown_history_modes = sorted(set(history_modes) - _HISTORY_MODE_DEFINITIONS.keys())
    if unknown_history_modes:
        raise ValueError(
            "axes.history_modes contains unsupported modes: "
            + ", ".join(unknown_history_modes)
        )
    dimensions = _axis(axes, "latent_dimensions", int)
    safety_arms = _axis(axes, "safety_arms", str)
    seeds = _axis(axes, "seeds", int)
    base_value = grid.get("base_config")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError("base_config must be a non-empty repository-relative path")
    base_declared = Path(base_value)
    if base_declared.is_absolute():
        raise ValueError("base_config must be repository-relative")
    base_relative = _repo_relative(ROOT / base_declared, label="base_config")
    base_path = ROOT / base_relative
    base = load_config(base_path)
    if not base.evaluation.emit_audit_records:
        raise ValueError(
            "the E1 sweep requires evaluation.emit_audit_records = true for automatic audit"
        )
    output_root = _output_root(execution.get("output_root"))
    device = str(execution.get("device", "cuda"))
    analysis = _analysis_config(grid.get("analysis"))

    tasks = []
    combinations = itertools.product(
        families,
        history_modes,
        dimensions,
        safety_arms,
        seeds,
    )
    for task_id, (family, history_mode, dimension, arm, seed) in enumerate(combinations):
        history_encoder, history_length = _HISTORY_MODE_DEFINITIONS[history_mode]
        kl_weight = 0.0 if family == "ae" else base.objective.kl_weight
        safety_weight = 0.0 if arm == "none" else base.objective.safety_weight
        resolved = dataclasses.replace(
            base,
            run=dataclasses.replace(base.run, seed=seed, device=device),
            data=dataclasses.replace(base.data, history_length=history_length),
            model=dataclasses.replace(
                base.model,
                family=family,
                history_encoder=history_encoder,
                latent_dim=dimension,
            ),
            objective=dataclasses.replace(
                base.objective,
                safety_arm=arm,
                safety_weight=safety_weight,
                kl_weight=kl_weight,
            ),
        )
        validate_config(resolved)
        output = (
            output_root
            / f"{family}_{history_mode}_d{dimension}"
            / arm
            / f"seed_{seed}"
        )
        tasks.append(
            {
                "task_id": task_id,
                "model_family": family,
                "history_mode": history_mode,
                "history_encoder": history_encoder,
                "history_length": history_length,
                "latent_dim": dimension,
                "safety_arm": arm,
                "seed": seed,
                "kl_weight": kl_weight,
                "safety_weight": safety_weight,
                "device": device,
                "output_dir": output.as_posix(),
            }
        )

    expected = int(execution.get("expected_tasks", len(tasks)))
    if len(tasks) != expected:
        raise ValueError(f"expanded {len(tasks)} tasks, expected {expected}")
    if len({task["output_dir"] for task in tasks}) != len(tasks):
        raise ValueError("grid produced duplicate output directories")
    plan = {
        "schema_version": 1,
        "experiment": str(grid["experiment"]),
        "status": str(grid.get("status", "planned")),
        "code": _git_code_state(),
        "grid": {"path": grid_relative.as_posix(), "sha256": config_sha256(grid_path)},
        "base_config": {
            "path": base_relative.as_posix(),
            "sha256": config_sha256(base_path),
        },
        "axes": {
            "model_families": list(families),
            "history_modes": list(history_modes),
            "latent_dimensions": list(dimensions),
            "safety_arms": list(safety_arms),
            "seeds": list(seeds),
        },
        "history_mode_definitions": {
            mode: {
                "history_encoder": _HISTORY_MODE_DEFINITIONS[mode][0],
                "history_length": _HISTORY_MODE_DEFINITIONS[mode][1],
            }
            for mode in history_modes
        },
        "task_count": len(tasks),
        "tasks": tasks,
        "analysis": analysis,
        "gate": grid.get("gate", {}),
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.grid.resolve())
    output = args.output if args.output.is_absolute() else ROOT / args.output
    write_json_atomic(output.resolve(), plan)
    print(f"wrote {plan['task_count']} tasks to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
