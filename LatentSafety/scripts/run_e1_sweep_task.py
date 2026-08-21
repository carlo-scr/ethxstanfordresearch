#!/usr/bin/env python3
"""Run one indexed task from a deterministic E1 sweep plan without shell evaluation."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"plan.{label}.path must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"plan.{label}.path must be repository-relative")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"plan.{label}.path escapes the repository: {path}") from error
    return resolved


def _task_output_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task.output_dir must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("task.output_dir must be repository-relative")
    resolved = (ROOT / relative).resolve()
    managed_root = (ROOT / "runs" / "e1_world_models").resolve()
    try:
        resolved.relative_to(managed_root)
    except ValueError as error:
        raise ValueError(
            "task.output_dir must resolve below runs/e1_world_models"
        ) from error
    return resolved


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_plan(plan: object) -> dict[str, object]:
    """Verify the immutable plan and source-config hashes before launching compute."""

    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("plan must be a schema_version = 1 object")
    declared_hash = plan.get("plan_sha256")
    if not isinstance(declared_hash, str) or not declared_hash:
        raise ValueError("plan.plan_sha256 must be a non-empty string")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(declared_hash, actual_hash):
        raise ValueError(
            f"plan_sha256 mismatch: declared {declared_hash}, recomputed {actual_hash}"
        )

    source_paths: dict[str, Path] = {}
    for label in ("grid", "base_config"):
        entry = plan.get(label)
        if not isinstance(entry, dict):
            raise ValueError(f"plan.{label} must be an object")
        source_path = _declared_path(entry.get("path"), label=label)
        declared_source_hash = entry.get("sha256")
        if not isinstance(declared_source_hash, str) or not declared_source_hash:
            raise ValueError(f"plan.{label}.sha256 must be a non-empty string")
        if not source_path.is_file():
            raise ValueError(f"plan.{label} source is missing: {source_path}")
        source_paths[label] = source_path
        actual_source_hash = _sha256(source_path)
        if not hmac.compare_digest(declared_source_hash, actual_source_hash):
            raise ValueError(
                f"plan.{label}.sha256 mismatch for {source_path}: "
                f"declared {declared_source_hash}, recomputed {actual_source_hash}"
            )

    code = plan.get("code")
    if not isinstance(code, dict):
        raise ValueError("plan.code must be an object")
    planned_revision = code.get("git_revision")
    planned_dirty = code.get("git_dirty")
    if planned_revision is not None and (
        not isinstance(planned_revision, str) or not planned_revision.strip()
    ):
        raise ValueError("plan.code.git_revision must be null or a non-empty string")
    if planned_dirty is not None and not isinstance(planned_dirty, bool):
        raise ValueError("plan.code.git_dirty must be null or a boolean")

    tasks = plan.get("tasks")
    task_count = plan.get("task_count")
    if (
        not isinstance(tasks, list)
        or isinstance(task_count, bool)
        or not isinstance(task_count, int)
    ):
        raise ValueError("plan.tasks must be a list and plan.task_count must be an integer")
    if not tasks or task_count != len(tasks):
        raise ValueError("plan.task_count must equal a non-empty tasks list")
    history_definitions = plan.get("history_mode_definitions")
    if not isinstance(history_definitions, dict):
        raise ValueError("plan.history_mode_definitions must be an object")
    output_dirs: set[Path] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"plan.tasks[{index}] must be an object")
        if task.get("task_id") != index:
            raise ValueError("plan task IDs must be exactly ordered as 0..task_count-1")
        output_dir = _task_output_path(task.get("output_dir"))
        if output_dir in output_dirs:
            raise ValueError(f"duplicate task.output_dir: {output_dir}")
        output_dirs.add(output_dir)
        history_mode = task.get("history_mode")
        definition = history_definitions.get(history_mode)
        if (
            not isinstance(definition, dict)
            or task.get("history_encoder") != definition.get("history_encoder")
            or task.get("history_length") != definition.get("history_length")
        ):
            raise ValueError(
                f"plan.tasks[{index}] disagrees with its history_mode definition"
            )
    analysis = plan.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("plan.analysis must be an object")
    relative_radii = analysis.get("relative_radii")
    if not isinstance(relative_radii, list) or not relative_radii:
        raise ValueError("plan.analysis.relative_radii must be a non-empty list")
    for radius in relative_radii:
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not float("-inf") < float(radius) < float("inf")
            or float(radius) < 0.0
        ):
            raise ValueError(
                "plan.analysis.relative_radii must contain finite non-negative numbers"
            )

    # A digest alone is self-attested: a manually modified plan can simply be re-hashed. Rebuild
    # the plan from the verified grid with the checked-in planner and require the unsigned payload
    # to match exactly. The original code-state field is retained here because a dry-run may occur
    # after creating an unignored plan file; evidence runs independently enforce the exact revision.
    from plan_e1_sweep import build_plan

    expected = build_plan(source_paths["grid"])
    expected["code"] = code
    expected.pop("plan_sha256", None)
    if expected != unsigned:
        raise ValueError(
            "plan is not the canonical expansion of its verified grid and base config"
        )
    return plan


def _versioned_code_state() -> tuple[str | None, str | None]:
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
    resolved_revision = revision.stdout.strip() if revision.returncode == 0 else None
    resolved_status = status.stdout.strip() if status.returncode == 0 else None
    return resolved_revision, resolved_status


def _record_task_status(
    *,
    output_dir: Path,
    plan_path: Path,
    plan: dict[str, object],
    task: dict[str, object],
    status: str,
    training_returncode: int,
    completed_audits: list[str],
    failed_audit: str | None = None,
) -> None:
    """Persist wrapper-level provenance once the trainer has created the run directory."""

    if not output_dir.is_dir():
        return
    payload: dict[str, object] = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "plan": {
            "path": str(plan_path),
            "sha256": plan["plan_sha256"],
            "code": plan["code"],
        },
        "task_id": task["task_id"],
        "task": task,
        "training_returncode": training_returncode,
        "completed_audits": completed_audits,
        "failed_audit": failed_audit,
    }
    _write_json_atomic(output_dir / "sweep_task_manifest.json", payload)


def _update_run_manifest_status(
    *,
    output_dir: Path,
    plan: dict[str, object],
    task: dict[str, object],
    status: str,
    completed_audits: list[str],
    failed_audit: str | None = None,
) -> None:
    """Make aggregate eligibility depend on completion of the automatic audit suite."""

    path = output_dir / "run_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot update trainer run manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"trainer run manifest must contain an object: {path}")
    payload["status"] = status
    payload["orchestration"] = {
        "plan_sha256": plan["plan_sha256"],
        "planned_code": plan["code"],
        "task_id": task["task_id"],
        "completed_audits": completed_audits,
        "failed_audit": failed_audit,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help=(
            "allow a write-producing engineering run without a clean Git revision; "
            "never use this for evidence-producing sweeps"
        ),
    )
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    try:
        with plan_path.open(encoding="utf-8") as stream:
            plan = _verify_plan(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 0 <= args.index < len(tasks):
        upper_bound = len(tasks) - 1 if isinstance(tasks, list) else -1
        parser.error(f"--index must lie in [0, {upper_bound}]")
    task = tasks[args.index]
    history_definitions = plan.get("history_mode_definitions")
    history_mode = task.get("history_mode")
    if not isinstance(history_definitions, dict) or history_mode not in history_definitions:
        parser.error("plan does not define the selected task's history_mode")
    history_definition = history_definitions[history_mode]
    if (
        not isinstance(history_definition, dict)
        or task.get("history_encoder") != history_definition.get("history_encoder")
        or task.get("history_length") != history_definition.get("history_length")
    ):
        parser.error("selected task disagrees with its history_mode definition")
    if not args.dry_run:
        revision, status = _versioned_code_state()
        planned_code = plan["code"]
        planned_revision = planned_code.get("git_revision")
        planned_dirty = planned_code.get("git_dirty")
        version_problem = (
            planned_revision is None
            or planned_dirty is not False
            or revision is None
            or status is None
            or bool(status)
            or revision != planned_revision
        )
        if version_problem and not args.allow_unversioned:
            parser.error(
                "evidence-producing tasks require a plan generated from the same clean "
                "committed Git revision as the execution checkout; regenerate the plan after "
                "committing, or use --allow-unversioned only for engineering checks"
            )
        if version_problem:
            print(
                "WARNING: plan/execution code is unversioned, dirty, or revision-mismatched; "
                "outputs are engineering-only",
                file=sys.stderr,
            )
    device = args.device or str(task["device"])
    try:
        base_config_path = _declared_path(plan["base_config"]["path"], label="base_config")
        output_dir = _task_output_path(task["output_dir"])
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if not args.dry_run and output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            parser.error(
                f"refusing to launch into non-empty task output: {output_dir}; "
                "preserve the failed/completed run and generate a new output root for a retry"
            )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_e1_torch.py"),
        "--config",
        str(base_config_path),
        "--output",
        str(output_dir),
        "--device",
        device,
        "--seed",
        str(task["seed"]),
        "--model-family",
        str(task["model_family"]),
        "--history-encoder",
        str(task["history_encoder"]),
        "--history-length",
        str(task["history_length"]),
        "--latent-dim",
        str(task["latent_dim"]),
        "--kl-weight",
        str(task["kl_weight"]),
        "--safety-arm",
        str(task["safety_arm"]),
        "--safety-weight",
        str(task["safety_weight"]),
    ]
    if args.dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if args.dry_run:
        return completed.returncode
    completed_audits: list[str] = []
    if completed.returncode != 0:
        _record_task_status(
            output_dir=output_dir,
            plan_path=plan_path.resolve(),
            plan=plan,
            task=task,
            status="training_failed",
            training_returncode=completed.returncode,
            completed_audits=completed_audits,
        )
        return completed.returncode
    if args.skip_audit:
        _update_run_manifest_status(
            output_dir=output_dir,
            plan=plan,
            task=task,
            status="audit_skipped",
            completed_audits=completed_audits,
        )
        _record_task_status(
            output_dir=output_dir,
            plan_path=plan_path.resolve(),
            plan=plan,
            task=task,
            status="training_success_audit_skipped",
            training_returncode=completed.returncode,
            completed_audits=completed_audits,
        )
        return 0

    _update_run_manifest_status(
        output_dir=output_dir,
        plan=plan,
        task=task,
        status="audit_pending",
        completed_audits=completed_audits,
    )

    relative_radii = plan.get("analysis", {}).get(
        "relative_radii", [0.0, 0.01, 0.02, 0.05, 0.10]
    )
    audit_command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_e1_records.py"),
        "--config",
        str(base_config_path),
        "--calibration",
        str(output_dir / "audit_calibration.jsonl"),
        "--validation",
        str(output_dir / "audit_validation.jsonl"),
        "--test",
        str(output_dir / "audit_test.jsonl"),
        "--output",
        str(output_dir / "latent_safety_audit.json"),
        "--relative-radii",
        *(str(value) for value in relative_radii),
    ]
    audit = subprocess.run(audit_command, cwd=ROOT, check=False)
    if audit.returncode != 0:
        _update_run_manifest_status(
            output_dir=output_dir,
            plan=plan,
            task=task,
            status="audit_failed",
            completed_audits=completed_audits,
            failed_audit="latent",
        )
        _record_task_status(
            output_dir=output_dir,
            plan_path=plan_path.resolve(),
            plan=plan,
            task=task,
            status="audit_failed",
            training_returncode=completed.returncode,
            completed_audits=completed_audits,
            failed_audit="latent",
        )
        return audit.returncode
    completed_audits.append("latent")

    # The unsupervised checkpoint also receives two privileged append controls.  These answer the
    # reviewer-facing questions "why not append h?" and "why not append the safe-action profile?"
    # without retraining or silently treating either oracle as deployable.
    if str(task["safety_arm"]) == "none":
        for view, filename in (
            ("observation_oracle", "latent_safety_audit_observation_oracle.json"),
            ("state_oracle", "latent_safety_audit_state_oracle.json"),
            ("latent_plus_margin", "latent_safety_audit_oracle_margin.json"),
            (
                "latent_plus_action_profile",
                "latent_safety_audit_oracle_action_profile.json",
            ),
        ):
            oracle_command = [
                *audit_command,
                "--representation-view",
                view,
            ]
            output_index = oracle_command.index("--output") + 1
            oracle_command[output_index] = str(output_dir / filename)
            oracle = subprocess.run(oracle_command, cwd=ROOT, check=False)
            if oracle.returncode != 0:
                _update_run_manifest_status(
                    output_dir=output_dir,
                    plan=plan,
                    task=task,
                    status="audit_failed",
                    completed_audits=completed_audits,
                    failed_audit=view,
                )
                _record_task_status(
                    output_dir=output_dir,
                    plan_path=plan_path.resolve(),
                    plan=plan,
                    task=task,
                    status="audit_failed",
                    training_returncode=completed.returncode,
                    completed_audits=completed_audits,
                    failed_audit=view,
                )
                return oracle.returncode
            completed_audits.append(view)
            _update_run_manifest_status(
                output_dir=output_dir,
                plan=plan,
                task=task,
                status="audit_pending",
                completed_audits=completed_audits,
            )
    _update_run_manifest_status(
        output_dir=output_dir,
        plan=plan,
        task=task,
        status="success",
        completed_audits=completed_audits,
    )
    _record_task_status(
        output_dir=output_dir,
        plan_path=plan_path.resolve(),
        plan=plan,
        task=task,
        status="success",
        training_returncode=completed.returncode,
        completed_audits=completed_audits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
