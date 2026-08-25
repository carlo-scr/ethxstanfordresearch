#!/usr/bin/env python3
"""Run a diagnostic safe-action-sufficiency audit on frozen representation records."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.foundation.audit import audit_common_action_records  # noqa: E402
from latent_safety.manifest import base_manifest, write_json_atomic  # noqa: E402
from latent_safety.records import read_jsonl  # noqa: E402


FOUNDATION_AUDIT_PROTOCOL = "foundation_common_action_audit_v1"
TEST_FREEZE_PROTOCOL = "foundation_test_audit_freeze_v1"
HELD_OUT_SPLITS = ("validation", "calibration", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _audit_parameters(
    *,
    delta: float,
    gamma: float,
    tail_quantile: float,
    max_witnesses: int,
) -> dict[str, float | int]:
    return {
        "delta": delta,
        "gamma": gamma,
        "tail_quantile": tail_quantile,
        "max_witnesses": max_witnesses,
    }


def _load_test_freeze(
    path: Path,
    *,
    expected_file_sha256: str,
    parameters: dict[str, float | int],
    current_code: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Integrity-check a freeze produced before final-test labels were unblinded."""

    resolved = path.resolve()
    actual_file_sha256 = _sha256(resolved)
    if not hmac.compare_digest(expected_file_sha256, actual_file_sha256):
        raise ValueError("test freeze file does not match --test-freeze-sha256")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read test freeze manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("test freeze manifest must contain a JSON object")
    expected_fields = {
        "schema_version",
        "protocol",
        "status",
        "analysis",
        "input",
        "audit_parameters",
        "code",
        "manifest_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError(
            "test freeze manifest fields must be exactly "
            f"{sorted(expected_fields)!r}"
        )
    if payload.get("schema_version") != 1 or payload.get("protocol") != TEST_FREEZE_PROTOCOL:
        raise ValueError("unsupported test freeze manifest protocol")
    if payload.get("status") != "frozen_before_test_access":
        raise ValueError("test freeze manifest was not frozen before test access")
    if payload.get("analysis") != FOUNDATION_AUDIT_PROTOCOL:
        raise ValueError("test freeze manifest names a different analysis")

    declared_manifest_sha256 = _lower_sha256(
        payload.get("manifest_sha256"),
        label="test freeze manifest_sha256",
    )
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    actual_manifest_sha256 = _canonical_sha256(unsigned)
    if not hmac.compare_digest(declared_manifest_sha256, actual_manifest_sha256):
        raise ValueError("test freeze manifest SHA-256 mismatch")

    input_entry = payload.get("input")
    if not isinstance(input_entry, dict) or set(input_entry) != {
        "split",
        "records_sha256",
    }:
        raise ValueError("test freeze input must pin exactly split and records_sha256")
    if input_entry.get("split") != "test":
        raise ValueError("test freeze input split must be exactly 'test'")
    declared_records_sha256 = _lower_sha256(
        input_entry.get("records_sha256"),
        label="test freeze records_sha256",
    )
    if payload.get("audit_parameters") != parameters:
        raise ValueError("test audit parameters do not match the frozen manifest")

    code = payload.get("code")
    if not isinstance(code, dict) or set(code) != {"git_revision", "git_dirty"}:
        raise ValueError("test freeze code must pin git_revision and git_dirty")
    frozen_revision = code.get("git_revision")
    if (
        not isinstance(frozen_revision, str)
        or len(frozen_revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in frozen_revision)
    ):
        raise ValueError("test freeze git_revision must be a lowercase Git object ID")
    if code.get("git_dirty") is not False:
        raise ValueError("test freeze must originate from a clean worktree")
    if current_code.get("git_dirty") is not False:
        raise ValueError("final-test audit refuses to run from a dirty worktree")
    if current_code.get("git_revision") != frozen_revision:
        raise ValueError("current Git revision does not match the frozen test manifest")
    return (
        {
            "path": resolved.name,
            "file_sha256": actual_file_sha256,
            "manifest_sha256": declared_manifest_sha256,
            "protocol": TEST_FREEZE_PROTOCOL,
        },
        declared_records_sha256,
    )


def run_audit(
    records_path: Path,
    *,
    delta: float,
    gamma: float,
    tail_quantile: float,
    max_witnesses: int,
    declared_split: str = "validation",
    test_freeze_path: Path | None = None,
    test_freeze_sha256: str | None = None,
    unblind_test: bool = False,
    access_provenance: str | None = None,
    producing_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit one held-out split and return a checksum-pinned diagnostic payload."""

    resolved = records_path.resolve()
    if declared_split not in HELD_OUT_SPLITS:
        raise ValueError(f"declared_split must be one of {HELD_OUT_SPLITS!r}")
    if declared_split == "test":
        # These authorization checks intentionally precede every read of ``records_path``.  The
        # freeze checksum is verified below before the final-test file is hashed or parsed.
        if test_freeze_path is None:
            raise ValueError("test audit requires a checksum-pinned frozen manifest")
        normalized_freeze_sha256 = _lower_sha256(
            test_freeze_sha256,
            label="--test-freeze-sha256",
        )
        if unblind_test is not True:
            raise ValueError("test audit requires explicit --unblind-test authorization")
        if not isinstance(access_provenance, str) or not access_provenance.strip():
            raise ValueError("test audit requires non-empty unblinding access provenance")
    elif (
        test_freeze_path is not None
        or test_freeze_sha256 is not None
        or unblind_test
        or access_provenance is not None
    ):
        raise ValueError(
            "test freeze and unblinding options are valid only for declared test input"
        )

    parameters = _audit_parameters(
        delta=delta,
        gamma=gamma,
        tail_quantile=tail_quantile,
        max_witnesses=max_witnesses,
    )
    provenance_input = test_freeze_path if declared_split == "test" else resolved
    assert provenance_input is not None
    run_provenance = base_manifest(repo_root=ROOT, config_path=provenance_input)
    # ``base_manifest`` calls this input a config because most repository runners use a TOML file.
    # The audit has no free-standing config yet, so retain only its common run provenance here.
    run_provenance.pop("config", None)
    code = run_provenance.get("code")
    if not isinstance(code, dict):
        raise ValueError("could not determine repository code provenance")

    freeze: dict[str, Any] | None = None
    frozen_records_sha256: str | None = None
    if declared_split == "test":
        assert test_freeze_path is not None
        freeze, frozen_records_sha256 = _load_test_freeze(
            test_freeze_path,
            expected_file_sha256=normalized_freeze_sha256,
            parameters=parameters,
            current_code=code,
        )

    # A final-test record file is touched only after authorization, access provenance, the
    # integrity-checked freeze, and its clean code revision have all passed validation.
    records_sha256 = _sha256(resolved)
    if frozen_records_sha256 is not None and not hmac.compare_digest(
        frozen_records_sha256,
        records_sha256,
    ):
        raise ValueError("test freeze manifest does not match the supplied record file")

    records = read_jsonl(resolved)
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ValueError("foundation audit input must contain exactly one split")
    split = next(iter(splits))
    if split not in HELD_OUT_SPLITS:
        raise ValueError("foundation audit input must be a held-out split, not training data")
    if split != declared_split:
        raise ValueError(
            f"record split {split!r} does not match declared input split {declared_split!r}"
        )
    audit = audit_common_action_records(
        records,
        delta=delta,
        gamma=gamma,
        tail_quantile=tail_quantile,
        max_witnesses=max_witnesses,
    )
    implementation_paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "latent_safety" / "foundation" / "audit.py",
        ROOT / "src" / "latent_safety" / "manifest.py",
        ROOT / "src" / "latent_safety" / "records.py",
    )
    payload = {
        "schema_version": 1,
        "analysis": FOUNDATION_AUDIT_PROTOCOL,
        "evidence_eligible": False,
        "evidence_boundary": (
            "diagnostic core only; promotion additionally requires pinned checkpoint, "
            "preprocessing, normalization, action-library, simulator, and split manifests"
        ),
        "input": {
            "path": resolved.name,
            "sha256": records_sha256,
            "split": split,
            "record_count": len(records),
        },
        "access": {
            "declared_split": declared_split,
            "final_test_unblinded": split == "test",
            "unblinding_provenance": (
                access_provenance.strip() if split == "test" and access_provenance else None
            ),
            "test_freeze": freeze,
        },
        "run_provenance": {
            **run_provenance,
            "producing_command": list(producing_command) or ["python_api:run_audit"],
        },
        "implementation": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in implementation_paths
        ],
        "semantics": {
            "viability_filter_precedes_neighborhood": True,
            "audit_ball": "closed_euclidean_ball",
            "physical_infeasibility_excluded_from_members": True,
            "empty_trajectory_endpoint": None,
            "finite_reference_only": True,
        },
        "audit": dataclasses.asdict(audit),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--input-split",
        choices=HELD_OUT_SPLITS,
        default="validation",
        help="declare the record split before access; validation is the safe default",
    )
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--max-witnesses", type=int, default=100)
    parser.add_argument(
        "--test-freeze",
        type=Path,
        help="integrity-checked manifest that froze the exact final-test audit",
    )
    parser.add_argument(
        "--test-freeze-sha256",
        help="pre-access SHA-256 pin for the exact test-freeze file",
    )
    parser.add_argument(
        "--unblind-test",
        action="store_true",
        help="explicitly authorize the one frozen final-test audit",
    )
    parser.add_argument(
        "--access-provenance",
        help="immutable decision, approval, or handoff identifier for final-test unblinding",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    output_path = arguments.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    try:
        payload = run_audit(
            arguments.records,
            delta=arguments.delta,
            gamma=arguments.gamma,
            tail_quantile=arguments.tail_quantile,
            max_witnesses=arguments.max_witnesses,
            declared_split=arguments.input_split,
            test_freeze_path=arguments.test_freeze,
            test_freeze_sha256=arguments.test_freeze_sha256,
            unblind_test=arguments.unblind_test,
            access_provenance=arguments.access_provenance,
            producing_command=(
                tuple(sys.argv)
                if argv is None
                else ("scripts/run_foundation_audit.py", *argv)
            ),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_json_atomic(output_path, payload)
    audit = payload["audit"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": audit["status"],
                "viable_sample_count": audit["viable_sample_count"],
                "eligible_center_count": audit["eligible_center_count"],
                "conflicting_center_count": audit["conflicting_center_count"],
                "trajectory_balanced_tail_required_violation": audit[
                    "trajectory_balanced_tail_required_violation"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
