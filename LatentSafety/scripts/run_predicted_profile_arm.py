#!/usr/bin/env python3
"""Run one authenticated nonprivileged predicted-profile downstream arm."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_safety.analysis.preconfirmation import PROFILE_ARM  # noqa: E402
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.profile_artifacts import (  # noqa: E402
    ASSEMBLED_MANIFEST_FILENAME,
)
from latent_safety.learning.profile_coverage import (  # noqa: E402
    COVERAGE_MANIFEST_FILENAME,
)
from latent_safety.learning.profile_downstream import (  # noqa: E402
    run_predicted_profile_arm,
)
from run_profile_teacher import (  # noqa: E402
    _current_code_state,
    _repository_path,
    _smoke_output_path,
    _verify_plan,
)

REGISTERED_POSITIVE_WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _managed_file(value: Path, *, root_name: str, label: str) -> Path:
    if value.is_absolute():
        raise ValueError(f"production {label} must be repository-relative")
    resolved = (ROOT / value).resolve()
    managed = (ROOT / "runs" / "e2_frontier" / root_name).resolve()
    try:
        resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            f"production {label} must resolve below runs/e2_frontier/{root_name}"
        ) from error
    return resolved


def _engineering_file(value: Path, *, label: str) -> Path:
    resolved = value.resolve()
    # The common smoke-path verifier validates the parent without requiring it to exist.
    _smoke_output_path(resolved.parent)
    if not resolved.name:
        raise ValueError(f"engineering {label} must name a file")
    return resolved


def _registered_weight(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("safety weight must be numeric") from error
    if not math.isfinite(parsed) or parsed not in REGISTERED_POSITIVE_WEIGHTS:
        raise argparse.ArgumentTypeError(
            "safety weight must be one of "
            + ", ".join(format(weight, "g") for weight in REGISTERED_POSITIVE_WEIGHTS)
        )
    return parsed


def _sha256_token(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return normalized


def _resolve_e2_task_binding(
    *,
    e2_plan_value: Path,
    e2_task_id: int,
    e2_task_sha256: str,
    orchestration_plan_sha256: str | None,
    teacher_plan_path: Path,
    teacher_plan_sha256: str,
    domain: str,
    model_family: str,
    data_seed: int,
    safety_weight: float,
    device: str,
    assembly_path: Path,
    coverage_path: Path,
    requested_output: Path | None,
) -> tuple[Path, dict[str, object]]:
    """Authenticate the one E2 task allowed to use the preconfirmation namespace."""

    if e2_plan_value.is_absolute():
        raise ValueError("--e2-plan must be repository-relative")
    e2_plan_path = _repository_path(
        e2_plan_value.as_posix(), label="E2 preconfirmation plan"
    )
    try:
        with e2_plan_path.open(encoding="utf-8") as stream:
            raw_plan = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read E2 preconfirmation plan: {error}") from error

    # Importing here keeps the standalone profile runner independent unless an E2 binding is
    # explicitly requested.  The same verifier used by the indexed parent reopens every pinned
    # source and requires the exact canonical 288-task expansion.
    from plan_e2_preconfirmation import READY_PLAN_STATUS  # noqa: PLC0415
    from run_e2_preconfirmation_task import (  # noqa: PLC0415
        _task_output_path as _e2_output_path,
        _verify_pinned_profile_inputs,
        _verify_plan as _verify_e2_plan,
    )

    plan = _verify_e2_plan(raw_plan)
    profile_handoff = plan.get("profile_handoff")
    runner = plan.get("runner")
    if (
        plan.get("status") != READY_PLAN_STATUS
        or not isinstance(profile_handoff, dict)
        or profile_handoff.get("all_inputs_ready") is not True
        or not isinstance(runner, dict)
        or runner.get("execution_enabled") is not True
    ):
        raise ValueError("E2 preconfirmation plan is not ready for indexed execution")
    if orchestration_plan_sha256 is None:
        raise ValueError("E2-bound profile execution requires orchestration plan SHA-256")
    if plan.get("plan_sha256") != orchestration_plan_sha256:
        raise ValueError("E2 plan SHA-256 disagrees with --orchestration-plan-sha256")
    if teacher_plan_sha256 == orchestration_plan_sha256:
        raise ValueError("teacher and E2 orchestration plan SHA-256 values must be distinct")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 0 <= e2_task_id < len(tasks):
        raise ValueError("--e2-task-id is outside the verified E2 plan")
    task = tasks[e2_task_id]
    if not isinstance(task, dict) or task.get("task_id") != e2_task_id:
        raise ValueError("verified E2 task ID/index binding mismatch")
    if task.get("task_sha256") != e2_task_sha256:
        raise ValueError("--e2-task-sha256 disagrees with the verified E2 task")
    expected_identity = {
        "safety_arm": PROFILE_ARM,
        "domain": domain,
        "model_family": model_family,
        "seed": data_seed,
        "safety_weight": safety_weight,
        "device": device,
        "selection_split": "validation_only",
        "calibration_or_final_test_selection_access": False,
    }
    mismatches = {
        field: {"planned": task.get(field), "requested": expected}
        for field, expected in expected_identity.items()
        if task.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            "E2 task identity disagrees with profile invocation: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if _repository_path(
        task.get("profile_teacher_plan"), label="E2 task profile_teacher_plan"
    ) != teacher_plan_path:
        raise ValueError("E2 task binds a different profile teacher plan")
    if task.get("profile_teacher_plan_sha256") != teacher_plan_sha256:
        raise ValueError("E2 task teacher-plan SHA-256 mismatch")
    if _repository_path(
        task.get("profile_label_manifest"), label="E2 task profile_label_manifest"
    ) != assembly_path:
        raise ValueError("E2 task binds a different profile label assembly")
    if _repository_path(
        task.get("profile_coverage_manifest"),
        label="E2 task profile_coverage_manifest",
    ) != coverage_path:
        raise ValueError("E2 task binds a different profile coverage gate")
    _verify_pinned_profile_inputs(task)

    if requested_output is None:
        raise ValueError("E2-bound profile execution requires the exact planned --output")
    if requested_output.is_absolute():
        raise ValueError("E2-bound profile --output must be repository-relative")
    planned_output = _e2_output_path(task.get("output_dir"))
    requested_resolved = (ROOT / requested_output).resolve()
    if requested_resolved != planned_output:
        raise ValueError("E2-bound profile --output disagrees with the verified task path")
    return planned_output, plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-family", choices=("ae", "beta_vae"), required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--safety-weight", type=_registered_weight, required=True)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--orchestration-plan-sha256",
        type=_sha256_token,
        help="distinct SHA-256 of the E2 orchestration plan that scheduled this run",
    )
    parser.add_argument(
        "--e2-plan",
        type=Path,
        help="repository-relative canonical E2 plan authorizing a preconfirmation output",
    )
    parser.add_argument("--e2-task-id", type=int)
    parser.add_argument("--e2-task-sha256", type=_sha256_token)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-unversioned", action="store_true")
    args = parser.parse_args(argv)

    plan_path = (
        args.teacher_plan if args.teacher_plan.is_absolute() else ROOT / args.teacher_plan
    )
    try:
        with plan_path.open(encoding="utf-8") as stream:
            teacher_plan = _verify_plan(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    tasks = sorted(
        (
            task
            for task in teacher_plan["tasks"]
            if task["domain"] == args.domain
            and task["model_family"] == args.model_family
            and task["data_seed"] == args.data_seed
        ),
        key=lambda task: int(task["fold_index"]),
    )
    if len(tasks) != 5 or [task["fold_index"] for task in tasks] != list(range(5)):
        parser.error("selected teacher-plan cell must contain exactly folds 0 through 4")
    task = tasks[0]
    config_entry = teacher_plan["base_configs"].get(args.domain)
    if not isinstance(config_entry, dict):
        parser.error(f"teacher plan has no base config for domain {args.domain!r}")
    e2_binding_fields = (args.e2_plan, args.e2_task_id, args.e2_task_sha256)
    if any(value is not None for value in e2_binding_fields) and not all(
        value is not None for value in e2_binding_fields
    ):
        parser.error(
            "E2 task binding requires --e2-plan, --e2-task-id, and --e2-task-sha256"
        )
    if args.engineering_smoke and args.e2_plan is not None:
        parser.error("engineering smoke cannot claim an E2 preconfirmation task binding")
    if args.allow_unversioned and args.e2_plan is not None:
        parser.error(
            "--allow-unversioned cannot consume an E2-bound canonical output; use a "
            "separate engineering-smoke output"
        )
    e2_plan: dict[str, object] | None = None
    try:
        base_path = _repository_path(
            config_entry["path"], label=f"base_configs.{args.domain}.path"
        )
        base_config = load_config(base_path)
        if args.engineering_smoke:
            if args.assembly is None or args.coverage is None or args.output is None:
                raise ValueError(
                    "engineering smoke requires --assembly, --coverage, and --output"
                )
            assembly_path = _engineering_file(args.assembly, label="assembly")
            coverage_path = _engineering_file(args.coverage, label="coverage")
            output_dir = _smoke_output_path(args.output)
        else:
            default_assembly = Path(
                "runs/e2_frontier/profile_labels/"
                f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}/"
                f"{ASSEMBLED_MANIFEST_FILENAME}"
            )
            default_coverage = Path(
                "runs/e2_frontier/profile_coverage/"
                f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}/"
                f"{COVERAGE_MANIFEST_FILENAME}"
            )
            assembly_path = _managed_file(
                args.assembly or default_assembly,
                root_name="profile_labels",
                label="assembly",
            )
            coverage_path = _managed_file(
                args.coverage or default_coverage,
                root_name="profile_coverage",
                label="coverage",
            )
            if args.e2_plan is not None:
                output_dir, e2_plan = _resolve_e2_task_binding(
                    e2_plan_value=args.e2_plan,
                    e2_task_id=int(args.e2_task_id),
                    e2_task_sha256=str(args.e2_task_sha256),
                    orchestration_plan_sha256=args.orchestration_plan_sha256,
                    teacher_plan_path=plan_path.resolve(),
                    teacher_plan_sha256=str(teacher_plan["plan_sha256"]),
                    domain=args.domain,
                    model_family=args.model_family,
                    data_seed=args.data_seed,
                    safety_weight=args.safety_weight,
                    device=args.device,
                    assembly_path=assembly_path,
                    coverage_path=coverage_path,
                    requested_output=args.output,
                )
            else:
                weight_token = format(args.safety_weight, ".8g").replace(".", "p")
                default_output = Path(
                    "runs/e2_frontier/predicted_profile_arm/"
                    f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}/"
                    f"weight_{weight_token}"
                )
                output_dir = _managed_file(
                    args.output or default_output,
                    root_name="predicted_profile_arm",
                    label="output",
                )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    for label, path in (("assembly", assembly_path), ("coverage", coverage_path)):
        if not path.is_file():
            parser.error(f"{label} manifest is missing: {path}")
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing downstream output: {output_dir}")

    current_code = _current_code_state()
    planned_code = teacher_plan["code"]
    teacher_version_matches = (
        planned_code.get("git_revision") is not None
        and planned_code.get("git_dirty") is False
        and current_code["git_revision"] == planned_code.get("git_revision")
        and current_code["git_dirty"] is False
    )
    e2_code = e2_plan.get("code") if e2_plan is not None else None
    e2_version_matches = e2_plan is None or (
        isinstance(e2_code, dict)
        and e2_code.get("git_revision") is not None
        and e2_code.get("git_dirty") is False
        and current_code["git_revision"] == e2_code.get("git_revision")
        and current_code["git_dirty"] is False
    )
    version_matches = teacher_version_matches and e2_version_matches
    if not args.engineering_smoke:
        if args.orchestration_plan_sha256 is None:
            parser.error("production downstream runs require --orchestration-plan-sha256")
        if not version_matches and not args.allow_unversioned:
            parser.error(
                "production downstream execution requires the clean revision frozen in the "
                "teacher plan; use --allow-unversioned only for a non-evidence run"
            )
    command = [
        "python",
        "scripts/run_predicted_profile_arm.py",
        "--teacher-plan",
        "<verified-teacher-plan>",
        "--domain",
        args.domain,
        "--model-family",
        args.model_family,
        "--data-seed",
        str(args.data_seed),
        "--safety-weight",
        format(args.safety_weight, "g"),
    ]
    if args.orchestration_plan_sha256 is not None:
        command.extend(
            ["--orchestration-plan-sha256", args.orchestration_plan_sha256]
        )
    if args.e2_plan is not None:
        command.extend(
            [
                "--e2-plan",
                args.e2_plan.as_posix(),
                "--e2-task-id",
                str(args.e2_task_id),
                "--e2-task-sha256",
                str(args.e2_task_sha256),
                "--assembly",
                str(args.assembly),
                "--coverage",
                str(args.coverage),
                "--output",
                str(args.output),
            ]
        )
    if args.engineering_smoke:
        command.extend(
            [
                "--engineering-smoke",
                "--assembly",
                "<engineering-assembly>",
                "--coverage",
                "<engineering-coverage>",
                "--output",
                "<engineering-output>",
            ]
        )
    if args.allow_unversioned:
        command.append("--allow-unversioned")
    try:
        result = run_predicted_profile_arm(
            base_config,
            tasks,
            teacher_plan_sha256=str(teacher_plan["plan_sha256"]),
            orchestration_plan_sha256=args.orchestration_plan_sha256,
            assembled_manifest_path=assembly_path,
            coverage_manifest_path=coverage_path,
            output_dir=output_dir,
            device=args.device,
            safety_weight=args.safety_weight,
            producing_command=command,
            execution_code=current_code,
            execution_evidence_eligible=(
                version_matches and not args.allow_unversioned
            ),
            engineering_smoke=args.engineering_smoke,
        )
    except Exception as error:
        print(
            f"predicted-profile arm failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
