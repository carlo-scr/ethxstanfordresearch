#!/usr/bin/env python3
"""Verify and run one indexed task from the canonical E2 preconfirmation plan."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    EXPECTED_OBSERVATIONS,
    PROFILE_ARM,
)
from latent_safety.analysis.preconfirmation_artifacts import (  # noqa: E402
    authenticate_preconfirmation_task_manifest,
    write_preconfirmation_task_manifest,
)
from plan_e2_preconfirmation import (  # noqa: E402
    OUTPUT_ROOT,
    READY_PLAN_STATUS,
    build_plan,
    canonical_sha256,
    file_sha256,
    task_sha256,
)


def _declared_repo_path(value: object, *, label: str) -> Path:
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
    resolved = _declared_repo_path(value, label="task.output_dir")
    managed = (ROOT / OUTPUT_ROOT).resolve()
    try:
        relative = resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            f"task.output_dir must resolve below {OUTPUT_ROOT.as_posix()}"
        ) from error
    if relative == Path("."):
        raise ValueError("task.output_dir must not equal the managed output root")
    return resolved


def _lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_source_entry(entry: object, *, label: str) -> Path:
    if not isinstance(entry, dict):
        raise ValueError(f"plan.{label} must be an object")
    path = _declared_repo_path(entry.get("path"), label=f"plan.{label}.path")
    declared = _lower_sha256(entry.get("sha256"), label=f"plan.{label}.sha256")
    if not path.is_file():
        raise ValueError(f"plan.{label} source is missing: {path}")
    actual = file_sha256(path)
    if not hmac.compare_digest(declared, actual):
        raise ValueError(
            f"plan.{label}.sha256 mismatch for {path}: declared {declared}, "
            f"recomputed {actual}"
        )
    return path


def _verify_plan(plan: object) -> dict[str, Any]:
    """Authenticate source hashes and require the exact canonical expansion."""

    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("plan must be a schema_version = 1 object")
    declared_hash = _lower_sha256(
        plan.get("plan_sha256"), label="plan.plan_sha256"
    )
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    actual_hash = canonical_sha256(unsigned)
    if not hmac.compare_digest(declared_hash, actual_hash):
        raise ValueError(
            f"plan_sha256 mismatch: declared {declared_hash}, recomputed {actual_hash}"
        )

    intervention_path = _verify_source_entry(
        plan.get("intervention_config"), label="intervention_config"
    )
    base_entries = plan.get("base_configs")
    if not isinstance(base_entries, dict) or len(base_entries) != 3:
        raise ValueError("plan.base_configs must contain exactly three domains")
    for domain, entry in base_entries.items():
        if not isinstance(domain, str) or not domain:
            raise ValueError("plan.base_configs contains an invalid domain key")
        _verify_source_entry(entry, label=f"base_configs.{domain}")
    profile_handoff = plan.get("profile_handoff")
    if not isinstance(profile_handoff, dict):
        raise ValueError("plan.profile_handoff must be an object")
    teacher_plan_entry = profile_handoff.get("teacher_plan")
    if not isinstance(teacher_plan_entry, dict):
        raise ValueError("plan.profile_handoff.teacher_plan must be an object")
    teacher_plan_path = _declared_repo_path(
        teacher_plan_entry.get("path"),
        label="plan.profile_handoff.teacher_plan.path",
    )
    teacher_plan_file_sha = teacher_plan_entry.get("file_sha256")
    if teacher_plan_file_sha is not None:
        declared_teacher_file_sha = _lower_sha256(
            teacher_plan_file_sha,
            label="plan.profile_handoff.teacher_plan.file_sha256",
        )
        if not teacher_plan_path.is_file():
            raise ValueError(f"pinned profile teacher plan is missing: {teacher_plan_path}")
        if not hmac.compare_digest(
            declared_teacher_file_sha, file_sha256(teacher_plan_path)
        ):
            raise ValueError("pinned profile teacher plan file SHA-256 changed")

    code = plan.get("code")
    if not isinstance(code, dict) or set(code) != {"git_revision", "git_dirty"}:
        raise ValueError("plan.code must contain only git_revision and git_dirty")
    revision = code.get("git_revision")
    dirty = code.get("git_dirty")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("plan.code.git_revision must be null or a non-empty string")
    if dirty is not None and not isinstance(dirty, bool):
        raise ValueError("plan.code.git_dirty must be null or a boolean")

    tasks = plan.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != EXPECTED_OBSERVATIONS
        or plan.get("task_count") != EXPECTED_OBSERVATIONS
    ):
        raise ValueError("plan must contain exactly 288 indexed tasks")
    outputs: set[Path] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"plan.tasks[{index}] must be an object")
        if task.get("task_id") != index:
            raise ValueError("plan task IDs must be ordered exactly as 0..287")
        declared_task_hash = _lower_sha256(
            task.get("task_sha256"), label=f"plan.tasks[{index}].task_sha256"
        )
        actual_task_hash = task_sha256(task)
        if not hmac.compare_digest(declared_task_hash, actual_task_hash):
            raise ValueError(f"plan.tasks[{index}] task_sha256 mismatch")
        output = _task_output_path(task.get("output_dir"))
        if output in outputs:
            raise ValueError(f"duplicate task.output_dir: {output}")
        outputs.add(output)
        if (
            task.get("selection_split") != "validation_only"
            or task.get("calibration_or_final_test_selection_access") is not False
        ):
            raise ValueError(f"plan.tasks[{index}] is not validation-selection-only")

    expected = build_plan(intervention_path, teacher_plan_path)
    expected["code"] = code
    expected.pop("plan_sha256", None)
    if expected != unsigned:
        raise ValueError(
            "plan is not the canonical expansion of its verified intervention, base "
            "configs, and profile inputs"
        )
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


def _code_matches(plan: dict[str, Any]) -> tuple[bool, dict[str, str | bool | None]]:
    current = _current_code_state()
    planned = plan["code"]
    matches = (
        planned.get("git_revision") is not None
        and planned.get("git_dirty") is False
        and current["git_revision"] == planned.get("git_revision")
        and current["git_dirty"] is False
    )
    return matches, current


def _verify_pinned_profile_inputs(task: dict[str, Any]) -> dict[str, str]:
    """Recheck both raw files immediately before a profile task is launched."""

    if task.get("safety_arm") != PROFILE_ARM:
        return {}
    fields = {
        "teacher_plan_sha256": "profile_teacher_plan_sha256",
        "teacher_plan_file_sha256": "profile_teacher_plan_file_sha256",
        "label_manifest_sha256": "profile_label_manifest_sha256",
        "label_manifest_file_sha256": "profile_label_manifest_file_sha256",
        "coverage_manifest_sha256": "profile_coverage_manifest_sha256",
        "coverage_manifest_file_sha256": "profile_coverage_manifest_file_sha256",
    }
    pinned = {
        result_name: _lower_sha256(task.get(task_name), label=f"task.{task_name}")
        for result_name, task_name in fields.items()
    }
    assembly_path = _declared_repo_path(
        task.get("profile_label_manifest"), label="task.profile_label_manifest"
    )
    coverage_path = _declared_repo_path(
        task.get("profile_coverage_manifest"), label="task.profile_coverage_manifest"
    )
    teacher_plan_path = _declared_repo_path(
        task.get("profile_teacher_plan"), label="task.profile_teacher_plan"
    )
    for path, expected, label in (
        (
            teacher_plan_path,
            pinned["teacher_plan_file_sha256"],
            "profile teacher plan",
        ),
        (assembly_path, pinned["label_manifest_file_sha256"], "profile label manifest"),
        (
            coverage_path,
            pinned["coverage_manifest_file_sha256"],
            "profile coverage manifest",
        ),
    ):
        if not path.is_file():
            raise ValueError(f"pinned {label} is missing: {path}")
        actual = file_sha256(path)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"pinned {label} file SHA-256 changed: {path}")
    try:
        teacher_plan = json.loads(teacher_plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read pinned profile teacher plan: {error}") from error
    if not isinstance(teacher_plan, dict):
        raise ValueError("pinned profile teacher plan must contain an object")
    teacher_unsigned = dict(teacher_plan)
    teacher_declared = teacher_unsigned.pop("plan_sha256", None)
    if (
        teacher_declared != pinned["teacher_plan_sha256"]
        or canonical_sha256(teacher_unsigned) != pinned["teacher_plan_sha256"]
    ):
        raise ValueError("pinned profile teacher plan self-hash mismatch")
    for path, expected, label in (
        (assembly_path, pinned["label_manifest_sha256"], "profile label manifest"),
        (coverage_path, pinned["coverage_manifest_sha256"], "profile coverage manifest"),
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read pinned {label}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"pinned {label} must contain an object")
        declared = payload.get("manifest_sha256")
        unsigned = dict(payload)
        unsigned.pop("manifest_sha256", None)
        actual = canonical_sha256(unsigned)
        if (
            not isinstance(declared, str)
            or not hmac.compare_digest(declared, expected)
            or not hmac.compare_digest(actual, expected)
        ):
            raise ValueError(f"pinned {label} self-hash mismatch")
    return pinned


def _fit_invocation(
    plan: dict[str, Any],
    task: dict[str, Any],
    *,
    plan_path: Path,
    device: str,
) -> dict[str, Any]:
    """Build the exact subprocess invocation for one authenticated task."""

    if task["safety_arm"] == PROFILE_ARM:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_predicted_profile_arm.py"),
            "--teacher-plan",
            str(task["profile_teacher_plan"]),
            "--domain",
            str(task["domain"]),
            "--model-family",
            str(task["model_family"]),
            "--data-seed",
            str(task["seed"]),
            "--safety-weight",
            str(task["safety_weight"]),
            "--assembly",
            str(task["profile_label_manifest"]),
            "--coverage",
            str(task["profile_coverage_manifest"]),
            "--output",
            str(task["output_dir"]),
            "--device",
            device,
            "--orchestration-plan-sha256",
            str(plan["plan_sha256"]),
            "--e2-plan",
            str(plan_path),
            "--e2-task-id",
            str(task["task_id"]),
            "--e2-task-sha256",
            str(task["task_sha256"]),
        ]
        return {
            "kind": "subprocess",
            "command": command,
            "authenticated_binding": {
                "teacher_plan_sha256": task["profile_teacher_plan_sha256"],
                "orchestration_plan_sha256": plan["plan_sha256"],
                "task_sha256": task["task_sha256"],
            },
            "isolation": (
                "cross-fitted training labels plus ordinary validation only; teacher and "
                "orchestration plan hashes remain distinct"
            ),
        }
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_e1_torch.py"),
        "--config",
        str(ROOT / task["base_config"]),
        "--output",
        str(_task_output_path(task["output_dir"])),
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
        "--fcsrl-head-hidden-dim",
        str(task["fcsrl_head_hidden_dim"]),
        "--access-scope",
        "train_validation_only",
        "--orchestration-plan-sha256",
        str(plan["plan_sha256"]),
    ]
    return {
        "kind": "subprocess",
        "command": command,
        "isolation": "main trainer train_validation_only access scope",
    }


def _dry_run_payload(
    plan: dict[str, Any],
    task: dict[str, Any],
    *,
    plan_path: Path,
    device: str,
) -> dict[str, Any]:
    profile_ready = all(
        task.get(field) is not None
        for field in (
            "profile_teacher_plan_sha256",
            "profile_teacher_plan_file_sha256",
            "profile_label_manifest_sha256",
            "profile_label_manifest_file_sha256",
            "profile_coverage_manifest_sha256",
            "profile_coverage_manifest_file_sha256",
        )
    ) if task["safety_arm"] == PROFILE_ARM else True
    blockers: list[str] = []
    if (
        plan.get("status") != READY_PLAN_STATUS
        or not bool(plan["profile_handoff"]["all_inputs_ready"])
        or not bool(plan["runner"]["execution_enabled"])
    ):
        blockers.append(
            "the canonical plan is blocked until all 18 production profile handoffs are "
            "checksum-pinned"
        )
    if not profile_ready:
        blockers.append("selected predicted-profile task has no pinned teacher handoff")
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "evidence_written": False,
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": plan["plan_sha256"],
            "status": plan["status"],
        },
        "task": task,
        "resolved_device": device,
        "output_dir": str(_task_output_path(task["output_dir"])),
        "fit_invocation": _fit_invocation(
            plan,
            task,
            plan_path=plan_path,
            device=device,
        ),
        "postfit_validation_audit": {
            "implemented": True,
            "artifacts": [
                "validation_postfit_manifest.json",
                "audit_validation.jsonl",
            ],
            "available_outputs": [
                "validation reconstruction and maximum-horizon rollout utility",
                "explicit constant-action physical profiles after checkpoint selection",
            ],
        },
        "execution_scope": {
            "materialized_splits": ["train", "validation"],
            "selection_split": "validation_only",
            "calibration_materialized": False,
            "final_test_materialized": False,
            "fit_oracle_action_profiles_forbidden": True,
            "postfit_validation_profiles": (
                "materialized only by the post-selection validation audit; never a fitting "
                "or checkpoint-selection target"
            ),
        },
        "execution_ready": not blockers,
        "blocked_dependencies": blockers,
        "downstream_dependencies": [
            "paired frozen-radius and matched-sign reduction",
            "canonical 288-row validation-only aggregation and weight selection",
        ],
    }


def _execution_plan_path(plan_path: Path) -> Path:
    """Require evidence/engineering execution to bind a plan stored in the repository."""

    resolved = plan_path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "execution requires --plan to resolve inside the repository so the completed "
            "task handoff can authenticate it"
        ) from error
    if relative == Path(".") or not resolved.is_file():
        raise ValueError("execution --plan must name an existing repository file")
    return resolved


class _TaskClaim:
    """Owned exclusive claim retained for the lifetime of one indexed task launch."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor

    def release(self) -> None:
        """Remove only the still-identical claim inode, then close its descriptor."""

        if self._descriptor < 0:
            return
        descriptor = self._descriptor
        self._descriptor = -1
        try:
            owner = os.fstat(descriptor)
            try:
                observed = os.stat(self.path, follow_symlinks=False)
            except FileNotFoundError:
                return
            if (observed.st_dev, observed.st_ino) == (owner.st_dev, owner.st_ino):
                self.path.unlink()
        finally:
            os.close(descriptor)


def _task_claim_path(output_dir: Path) -> Path:
    """Return the adjacent claim without changing the frozen task output path."""

    return output_dir.parent / f".{output_dir.name}.preconfirmation_task.claim"


def _acquire_task_claim(
    *,
    output_dir: Path,
    plan: dict[str, Any],
    task: dict[str, Any],
) -> _TaskClaim:
    """Atomically claim an unused task path; stale claims deliberately fail closed."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    claim_path = _task_claim_path(output_dir)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "task output is already claimed by another or interrupted indexed launch: "
            f"{claim_path}; stale claims require explicit operator review"
        ) from error
    except OSError as error:
        raise ValueError(f"cannot atomically claim task output {claim_path}: {error}") from error

    claim = _TaskClaim(claim_path, descriptor)
    payload = {
        "schema_version": 1,
        "protocol": "e2_preconfirmation_exclusive_task_claim_v1",
        "pid": os.getpid(),
        "plan_sha256": plan["plan_sha256"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "output_dir": task["output_dir"],
    }
    try:
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short write while recording exclusive task claim")
        os.fsync(descriptor)
    except Exception as error:
        try:
            claim.release()
        except OSError as cleanup_error:
            raise ValueError(
                "exclusive task claim initialization and cleanup both failed: "
                f"{error}; cleanup: {cleanup_error}"
            ) from error
        raise ValueError(f"cannot initialize exclusive task claim: {error}") from error

    if output_dir.exists():
        try:
            claim.release()
        except OSError as error:
            raise ValueError(
                f"task output already exists and claim cleanup failed: {error}"
            ) from error
        raise ValueError(
            f"refusing to overwrite existing task output: {output_dir}; "
            "the canonical indexed task path is single-use"
        )
    return claim


def _run_task_subprocess(command: list[str]) -> int:
    """Launch without swallowing trainer output and preserve an ordinary failure code."""

    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    except OSError as error:
        print(f"failed to launch E2 task subprocess: {error}", file=sys.stderr)
        return 1
    if completed.returncode != 0:
        print(
            f"E2 task subprocess failed with exit code {completed.returncode}; "
            "no preconfirmation task handoff was written",
            file=sys.stderr,
        )
        return completed.returncode if 0 < completed.returncode < 256 else 1
    return 0


def _execute_claimed_task(
    *,
    plan: dict[str, Any],
    task: dict[str, Any],
    plan_path: Path,
    device: str,
) -> int:
    """Run and authenticate one task while its caller retains the exclusive claim."""

    invocation = _fit_invocation(
        plan,
        task,
        plan_path=plan_path.relative_to(ROOT),
        device=device,
    )
    command = invocation.get("command")
    if invocation.get("kind") != "subprocess" or not isinstance(command, list):
        print(
            "verified task did not resolve to a subprocess invocation",
            file=sys.stderr,
        )
        return 1
    returncode = _run_task_subprocess(command)
    if returncode != 0:
        return returncode

    try:
        manifest_path = write_preconfirmation_task_manifest(
            repo_root=ROOT,
            plan_path=plan_path,
            plan_sha256=plan["plan_sha256"],
            task=task,
        )
        manifest, authenticated_path = authenticate_preconfirmation_task_manifest(
            repo_root=ROOT,
            plan_path=plan_path,
            plan_sha256=plan["plan_sha256"],
            task=task,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(
            "task subprocess completed, but authenticated handoff creation failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    if authenticated_path != manifest_path:
        print("authenticated task handoff path changed unexpectedly", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "success",
                "task_id": task["task_id"],
                "task_sha256": task["task_sha256"],
                "output_dir": task["output_dir"],
                "evidence_eligible": True,
                "evidence_written": True,
                "preconfirmation_task_manifest": {
                    "path": manifest_path.relative_to(ROOT).as_posix(),
                    "manifest_sha256": manifest["manifest_sha256"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help=(
            "retained for fail-closed compatibility: canonical execution is refused; use "
            "the separate trainer/profile engineering-smoke entrypoints"
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
    device = args.device or str(task["device"])
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(
                    plan,
                    task,
                    plan_path=plan_path,
                    device=device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        plan_path = _execution_plan_path(plan_path)
    except ValueError as error:
        parser.error(str(error))
    if args.allow_unversioned:
        parser.error(
            "--allow-unversioned never launches into the canonical single-use E2 output "
            "namespace; use the existing trainer/profile engineering-smoke entrypoints with "
            "a separate temporary output"
        )
    if (
        plan.get("status") != READY_PLAN_STATUS
        or not bool(plan["profile_handoff"]["all_inputs_ready"])
        or not bool(plan["runner"]["execution_enabled"])
    ):
        parser.error(
            "indexed execution requires a fully ready canonical plan with all 18 production "
            "profile label/coverage handoffs pinned; regenerate the plan after those inputs "
            "exist"
        )
    if device != task["device"]:
        parser.error(
            f"production execution must use the plan-declared device {task['device']!r}; "
            f"received {device!r}"
        )
    matches, _ = _code_matches(plan)
    if not matches:
        parser.error(
            "evidence tasks require the same clean committed revision frozen in the plan; "
            "regenerate after committing"
        )
    try:
        _verify_pinned_profile_inputs(task)
        output_dir = _task_output_path(task["output_dir"])
    except ValueError as error:
        parser.error(str(error))
    try:
        claim = _acquire_task_claim(
            output_dir=output_dir,
            plan=plan,
            task=task,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        return _execute_claimed_task(
            plan=plan,
            task=task,
            plan_path=plan_path,
            device=device,
        )
    finally:
        try:
            claim.release()
        except OSError as error:
            print(
                f"warning: exclusive task claim cleanup failed closed: {error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
