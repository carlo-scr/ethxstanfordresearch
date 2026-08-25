#!/usr/bin/env python3
"""Run one checksum-pinned cross-fitted predicted-profile teacher task."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plan_profile_teachers import (  # noqa: E402
    EXPECTED_TASKS,
    build_plan,
    task_sha256,
)
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.profile_teacher import (  # noqa: E402
    build_teacher_split_spec,
    resolve_teacher_config,
    run_profile_teacher_task,
    validate_task_split_hashes,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    declared = Path(value)
    if declared.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    resolved = (ROOT / declared).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository: {declared}") from error
    return resolved


def _task_output_path(value: object) -> Path:
    resolved = _repository_path(value, label="task.output_dir")
    managed = (ROOT / "runs" / "e2_frontier" / "profile_teachers").resolve()
    try:
        resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            "task.output_dir must resolve below runs/e2_frontier/profile_teachers"
        ) from error
    return resolved


def _smoke_output_path(value: Path) -> Path:
    if value.is_absolute():
        resolved = value.resolve()
        temporary_roots = {
            Path(tempfile.gettempdir()).resolve(),
            Path("/tmp").resolve(),
        }
        if not any(
            resolved == root or root in resolved.parents for root in temporary_roots
        ):
            raise ValueError(
                "absolute engineering-smoke outputs must resolve below the system temp directory"
            )
        return resolved
    resolved = (ROOT / value).resolve()
    managed = (ROOT / "runs" / "e2_frontier" / "profile_teacher_smoke").resolve()
    try:
        resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            "relative engineering-smoke outputs must resolve below "
            "runs/e2_frontier/profile_teacher_smoke"
        ) from error
    return resolved


def _verify_plan(plan: object) -> dict[str, Any]:
    """Verify self-digests, source hashes, task hashes, paths, and canonical expansion."""

    if not isinstance(plan, dict) or plan.get("schema_version") != 2:
        raise ValueError("plan must be a schema_version = 2 object")
    declared_plan_hash = plan.get("plan_sha256")
    if not isinstance(declared_plan_hash, str) or len(declared_plan_hash) != 64:
        raise ValueError("plan.plan_sha256 must be a SHA-256 string")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    actual_plan_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(declared_plan_hash, actual_plan_hash):
        raise ValueError(
            f"plan_sha256 mismatch: declared {declared_plan_hash}, "
            f"recomputed {actual_plan_hash}"
        )

    tasks = plan.get("tasks")
    if (
        not isinstance(tasks, list)
        or plan.get("task_count") != EXPECTED_TASKS
        or len(tasks) != EXPECTED_TASKS
    ):
        raise ValueError(f"plan must contain exactly {EXPECTED_TASKS} tasks")
    outputs: set[Path] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or task.get("task_id") != index:
            raise ValueError("plan task IDs must be exactly ordered as 0..329")
        declared_task_hash = task.get("task_sha256")
        if not isinstance(declared_task_hash, str) or len(declared_task_hash) != 64:
            raise ValueError(f"tasks[{index}].task_sha256 must be a SHA-256 string")
        actual_task_hash = task_sha256(task)
        if not hmac.compare_digest(declared_task_hash, actual_task_hash):
            raise ValueError(
                f"tasks[{index}].task_sha256 mismatch: declared {declared_task_hash}, "
                f"recomputed {actual_task_hash}"
            )
        output = _task_output_path(task.get("output_dir"))
        if output in outputs:
            raise ValueError(f"duplicate task output path: {output}")
        outputs.add(output)

    base_configs = plan.get("base_configs")
    if not isinstance(base_configs, dict) or not base_configs:
        raise ValueError("plan.base_configs must be a non-empty object")
    for domain, entry in base_configs.items():
        if not isinstance(domain, str) or not isinstance(entry, dict):
            raise ValueError("plan.base_configs entries must be objects")
        config_path = _repository_path(
            entry.get("path"), label=f"base_configs.{domain}.path"
        )
        declared_hash = entry.get("sha256")
        if not isinstance(declared_hash, str) or len(declared_hash) != 64:
            raise ValueError(f"base_configs.{domain}.sha256 must be a SHA-256 string")
        if not config_path.is_file():
            raise ValueError(f"base config is missing: {config_path}")
        actual_hash = _sha256(config_path)
        if not hmac.compare_digest(declared_hash, actual_hash):
            raise ValueError(
                f"base_configs.{domain}.sha256 mismatch: declared {declared_hash}, "
                f"recomputed {actual_hash}"
            )

    code = plan.get("code")
    if not isinstance(code, dict) or set(code) != {"git_revision", "git_dirty"}:
        raise ValueError("plan.code must contain exactly git_revision and git_dirty")
    if code["git_revision"] is not None and (
        not isinstance(code["git_revision"], str) or not code["git_revision"].strip()
    ):
        raise ValueError("plan.code.git_revision must be null or a non-empty string")
    if code["git_dirty"] is not None and not isinstance(code["git_dirty"], bool):
        raise ValueError("plan.code.git_dirty must be null or a boolean")

    # A self-digest can be recomputed after tampering.  Rebuild the exact 330-task factorial from
    # checked-in planner logic and verified config files.  Code state is prospective provenance,
    # so preserve the plan's frozen value during this structural comparison.
    expected = build_plan()
    expected["code"] = code
    expected_unsigned = dict(expected)
    expected_unsigned.pop("plan_sha256", None)
    if unsigned != expected_unsigned:
        raise ValueError("plan is not the canonical 330-task profile-teacher expansion")
    return plan


def _current_code_state() -> dict[str, str | bool | None]:
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


def _display_plan_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<external-plan>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help="allow an engineering run from dirty or revision-mismatched code",
    )
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help=(
            "run a one-epoch, 20-trajectory CPU model-path check; the result is never evidence"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "required only for --engineering-smoke; use a new path below the system temp "
            "directory or runs/e2_frontier/profile_teacher_smoke"
        ),
    )
    args = parser.parse_args(argv)

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    try:
        with plan_path.open(encoding="utf-8") as stream:
            plan = _verify_plan(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    tasks = plan["tasks"]
    if not 0 <= args.index < len(tasks):
        parser.error(f"--index must lie in [0, {len(tasks) - 1}]")
    task = tasks[args.index]
    domain = str(task["domain"])
    config_entry = plan["base_configs"].get(domain)
    if not isinstance(config_entry, dict):
        parser.error(f"plan has no base config for task domain {domain!r}")
    if (
        task.get("base_config") != config_entry.get("path")
        or task.get("base_config_sha256") != config_entry.get("sha256")
    ):
        parser.error("selected task disagrees with its domain base-config provenance")
    try:
        base_config_path = _repository_path(
            config_entry["path"], label=f"base_configs.{domain}.path"
        )
        base_config = load_config(base_config_path)
        resolved = resolve_teacher_config(
            base_config,
            task,
            device="cpu" if args.engineering_smoke else (args.device or base_config.run.device),
            engineering_smoke=args.engineering_smoke,
        )
        split_spec = build_teacher_split_spec(
            resolved,
            data_seed=int(task["data_seed"]),
            fold_index=int(task["fold_index"]),
        )
        if not args.engineering_smoke:
            validate_task_split_hashes(task, split_spec)
        if args.engineering_smoke:
            if args.output is None and not args.dry_run:
                raise ValueError("--engineering-smoke requires an explicit --output path")
            output_dir = (
                _smoke_output_path(args.output)
                if args.output is not None
                else Path(tempfile.gettempdir()) / "profile-teacher-dry-run-placeholder"
            )
        else:
            if args.output is not None:
                raise ValueError("--output is only valid with --engineering-smoke")
            output_dir = _task_output_path(task["output_dir"])
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    current_code = _current_code_state()
    planned_code = plan["code"]
    version_matches = (
        planned_code.get("git_revision") is not None
        and planned_code.get("git_dirty") is False
        and current_code["git_revision"] == planned_code.get("git_revision")
        and current_code["git_dirty"] is False
    )
    evidence_eligible = version_matches and not args.engineering_smoke
    if not args.dry_run and not evidence_eligible and not (
        args.allow_unversioned or args.engineering_smoke
    ):
        parser.error(
            "evidence-producing tasks require the same clean committed revision recorded in "
            "the plan; regenerate after committing, or use --allow-unversioned only for an "
            "engineering run"
        )

    preflight = {
        "status": "dry_run" if args.dry_run else "ready",
        "plan_sha256": plan["plan_sha256"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "base_config_sha256": task["base_config_sha256"],
        "domain": resolved.data.task,
        "model_family": resolved.model.family,
        "history_mode": f"{resolved.model.history_encoder}_h{resolved.data.history_length}",
        "data_seed": task["data_seed"],
        "teacher_seed": resolved.run.seed,
        "fold_index": task["fold_index"],
        "trajectory_counts": {
            "fitting": len(split_spec.fitting_ids),
            "held_out": len(split_spec.held_out_ids),
            "ordinary_validation": len(split_spec.validation_ids),
        },
        "materialized_splits": ["train", "validation"],
        "excluded_splits": ["calibration", "test"],
        "engineering_smoke": args.engineering_smoke,
        "evidence_eligible": evidence_eligible,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output path: {output_dir}")
    display_plan = _display_plan_path(plan_path)
    command = [
        "python",
        "scripts/run_profile_teacher.py",
        "--plan",
        display_plan,
        "--index",
        str(args.index),
    ]
    if args.engineering_smoke:
        command.extend(["--engineering-smoke", "--output", "<engineering-output>"])
    elif args.device is not None:
        command.extend(["--device", args.device])
    if args.allow_unversioned:
        command.append("--allow-unversioned")
    try:
        result = run_profile_teacher_task(
            base_config,
            task,
            base_config_path=str(task["base_config"]),
            plan_sha256=str(plan["plan_sha256"]),
            planned_code=planned_code,
            output_dir=output_dir,
            device=resolved.run.device,
            producing_command=command,
            evidence_eligible=evidence_eligible,
            engineering_smoke=args.engineering_smoke,
        )
    except Exception as error:
        print(f"profile teacher failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
