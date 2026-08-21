#!/usr/bin/env python3
"""Audit frozen E1 latent records using calibration-only radii and traceable witnesses."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.manifest import base_manifest, write_json_atomic  # noqa: E402
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.metrics.action import audit_radius_action_neighborhoods  # noqa: E402
from latent_safety.metrics.defect import (  # noqa: E402
    empirical_robust_defect_details,
    median_pairwise_distance,
)
from latent_safety.records import read_jsonl, validate_records  # noqa: E402


def _radius_grid(values: list[float]) -> tuple[float, ...]:
    radii = tuple(float(value) for value in values)
    if not radii or any(not math.isfinite(value) or value < 0.0 for value in radii):
        raise ValueError("relative radii must be a non-empty list of finite non-negative values")
    if tuple(sorted(set(radii))) != radii:
        raise ValueError("relative radii must be unique and sorted")
    return radii


def _require_split(records: tuple[object, ...], expected: str) -> None:
    observed = {getattr(record, "split") for record in records}
    if observed != {expected}:
        raise ValueError(f"expected only split {expected!r}, found {sorted(observed)!r}")


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _audit_records(
    records: tuple[Any, ...],
    *,
    representation_view: str,
    oracle_margin_scale: float,
    relative_radii: tuple[float, ...],
    absolute_radii: tuple[float, ...],
    quantile: float,
    max_witnesses: int,
) -> dict[str, Any]:
    latents = [
        _representation_vector(
            record,
            view=representation_view,
            margin_scale=oracle_margin_scale,
        )
        for record in records
    ]
    margins = [record.safety_margin for record in records]
    static_curve = []
    for relative, delta in zip(relative_radii, absolute_radii, strict=True):
        details = empirical_robust_defect_details(
            latents,
            margins,
            delta=delta,
            quantile=quantile,
            max_witnesses=max_witnesses,
        )
        static_curve.append(
            {
                "relative_radius": relative,
                "absolute_radius": delta,
                "estimate": dataclasses.asdict(details.estimate),
                "witnesses": [
                    {
                        **dataclasses.asdict(witness),
                        "safe_sample_id": records[witness.safe_index].sample_id,
                        "unsafe_sample_id": records[witness.unsafe_index].sample_id,
                    }
                    for witness in details.witnesses
                ],
            }
        )

    action_presence = [record.action_safety_margins is not None for record in records]
    if any(action_presence) and not all(action_presence):
        raise ValueError("records mix present and absent action_safety_margins")
    action_curve = []
    if all(action_presence):
        action_rows = [record.action_safety_margins for record in records]
        for relative, delta in zip(relative_radii, absolute_radii, strict=True):
            audit = audit_radius_action_neighborhoods(
                latents,
                action_rows,
                delta=delta,
                max_witnesses=max_witnesses,
            )
            payload = dataclasses.asdict(audit)
            payload["relative_radius"] = relative
            center_scores = [
                {
                    "center_index": index,
                    "sample_id": records[index].sample_id,
                    "trajectory_id": records[index].trajectory_id,
                    "required_violation": value,
                }
                for index, value in enumerate(audit.center_required_violations)
                if value is not None
            ]
            payload["center_required_violations"] = center_scores
            by_trajectory: dict[str, list[float]] = {}
            for score in center_scores:
                by_trajectory.setdefault(str(score["trajectory_id"]), []).append(
                    float(score["required_violation"])
                )
            tail_quantile = float(audit.tail_quantile)
            trajectory_rows = [
                {
                    "trajectory_id": trajectory_id,
                    "eligible_center_count": len(values),
                    "tail_quantile": tail_quantile,
                    "tail_required_violation": _linear_quantile(values, tail_quantile),
                    "maximum_required_violation": max(values),
                }
                for trajectory_id, values in sorted(by_trajectory.items())
            ]
            trajectory_tails = [
                float(row["tail_required_violation"]) for row in trajectory_rows
            ]
            payload["trajectory_tail_summary"] = {
                "score": (
                    "95th percentile across eligible centered-neighborhood finite-action "
                    "common violations within each trajectory"
                ),
                "all_trajectory_count": len({record.trajectory_id for record in records}),
                "evaluable_trajectory_count": len(trajectory_rows),
                "mean_trajectory_tail_required_violation": (
                    statistics.fmean(trajectory_tails) if trajectory_tails else 0.0
                ),
                "median_trajectory_tail_required_violation": (
                    float(statistics.median(trajectory_tails))
                    if trajectory_tails
                    else 0.0
                ),
                "per_trajectory": trajectory_rows,
            }
            payload["witnesses"] = [
                {
                    **dataclasses.asdict(witness),
                    "center_sample_id": records[witness.center_index].sample_id,
                    "member_sample_ids": [
                        records[index].sample_id for index in witness.member_indices
                    ],
                }
                for witness in audit.witnesses
            ]
            action_curve.append(payload)
    return {
        "record_count": len(records),
        "static_curve": static_curve,
        "action_curve": action_curve,
    }


def _representation_vector(
    record: Any,
    *,
    view: str,
    margin_scale: float,
) -> tuple[float, ...]:
    """Return one preregistered representation view for an oracle-control audit.

    The appended coordinates use the physical margin scale frozen in the training config.  They
    are privileged controls, not deployable representations when the corresponding oracle is
    unavailable at runtime.
    """

    latent = tuple(float(value) for value in record.latent)
    if view == "latent":
        return latent
    if view == "latent_plus_margin":
        return (*latent, float(record.safety_margin) / margin_scale)
    if view == "latent_plus_action_profile":
        profile = record.action_safety_margins
        if profile is None:
            raise ValueError(
                "latent_plus_action_profile requires action_safety_margins on every record"
            )
        return (*latent, *(float(value) / margin_scale for value in profile))
    if view == "observation_oracle":
        observation = record.observation_latent
        if observation is None:
            raise ValueError(
                "observation_oracle requires observation_latent on every record"
            )
        return tuple(float(value) for value in observation)
    if view == "state_oracle":
        state = record.state_latent
        if state is None:
            raise ValueError("state_oracle requires state_latent on every record")
        return tuple(float(value) for value in state)
    raise ValueError(f"unknown representation view: {view}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--relative-radii",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.10],
        help="multipliers of the calibration median pairwise latent distance",
    )
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--max-witnesses", type=int, default=100)
    parser.add_argument(
        "--representation-view",
        choices=(
            "latent",
            "latent_plus_margin",
            "latent_plus_action_profile",
            "observation_oracle",
            "state_oracle",
        ),
        default="latent",
        help=(
            "audit the learned latent or a privileged oracle append control; oracle dimensions "
            "are normalized by objective.margin_scale"
        ),
    )
    parser.add_argument(
        "--max-scale-points",
        type=int,
        default=1024,
        help="deterministic label-free cap for the O(n^2) calibration scale calculation",
    )
    args = parser.parse_args()

    relative_radii = _radius_grid(args.relative_radii)
    if not 0.0 <= args.quantile <= 1.0:
        parser.error("--quantile must lie in [0, 1]")
    if args.max_witnesses < 0:
        parser.error("--max-witnesses must be non-negative")
    if args.max_scale_points < 2:
        parser.error("--max-scale-points must be at least two")

    learning_config = load_config(args.config.resolve())
    oracle_margin_scale = learning_config.objective.margin_scale

    calibration = read_jsonl(args.calibration.resolve())
    validation = read_jsonl(args.validation.resolve()) if args.validation else None
    test = read_jsonl(args.test.resolve())
    _require_split(calibration, "calibration")
    if validation is not None:
        _require_split(validation, "validation")
    _require_split(test, "test")
    validate_records((*calibration, *(validation or ()), *test))

    scale_records = tuple(
        sorted(
            calibration,
            key=lambda record: hashlib.sha256(record.sample_id.encode("utf-8")).digest(),
        )[: args.max_scale_points]
    )
    if args.representation_view in {"observation_oracle", "state_oracle"}:
        scale_vectors = [
            _representation_vector(
                record,
                view=args.representation_view,
                margin_scale=oracle_margin_scale,
            )
            for record in scale_records
        ]
        scale_source = f"calibration_{args.representation_view}"
    else:
        # Oracle append controls deliberately retain the learned representation's radius anchor;
        # otherwise adding a coordinate would silently redefine the comparison radius.
        scale_vectors = [record.latent for record in scale_records]
        scale_source = "calibration_raw_latent"
    calibration_scale = median_pairwise_distance(scale_vectors)
    absolute_radii = tuple(relative * calibration_scale for relative in relative_radii)
    test_audit = _audit_records(
        test,
        representation_view=args.representation_view,
        oracle_margin_scale=oracle_margin_scale,
        relative_radii=relative_radii,
        absolute_radii=absolute_radii,
        quantile=args.quantile,
        max_witnesses=args.max_witnesses,
    )
    validation_audit = (
        _audit_records(
            validation,
            representation_view=args.representation_view,
            oracle_margin_scale=oracle_margin_scale,
            relative_radii=relative_radii,
            absolute_radii=absolute_radii,
            quantile=args.quantile,
            max_witnesses=args.max_witnesses,
        )
        if validation is not None
        else None
    )

    manifest = base_manifest(repo_root=ROOT, config_path=args.config.resolve())
    manifest.update(
        {
            "experiment": "e1_frozen_representation_audit",
            "status": "success",
            "inputs": {
                "calibration_records": str(args.calibration.resolve()),
                "validation_records": str(args.validation.resolve()) if args.validation else None,
                "test_records": str(args.test.resolve()),
                "calibration_count": len(calibration),
                "validation_count": len(validation) if validation is not None else None,
                "test_count": len(test),
            },
            "radius_calibration": {
                "method": "relative_to_calibration_median_pairwise_distance",
                "scale_source": scale_source,
                "uses_safety_labels": False,
                "selection": "smallest_sha256_sample_ids",
                "available_points": len(calibration),
                "used_points": len(scale_records),
                "max_scale_points": args.max_scale_points,
                "calibration_scale": calibration_scale,
                "relative_radii": relative_radii,
                "absolute_radii": absolute_radii,
                "deployment_certificate": False,
            },
            "representation_view": {
                "name": args.representation_view,
                "oracle_control": args.representation_view != "latent",
                "appended_coordinate_scale": (
                    oracle_margin_scale
                    if args.representation_view
                    in {"latent_plus_margin", "latent_plus_action_profile"}
                    else None
                ),
                "runtime_availability_assumed": args.representation_view == "latent",
                "note": (
                    "visible-factor observation oracle; evaluation control, not raw pixels"
                    if args.representation_view == "observation_oracle"
                    else (
                        "privileged ground-truth state; evaluation control"
                        if args.representation_view == "state_oracle"
                        else (
                            "privileged specification oracle; use only as a comparison control"
                            if args.representation_view != "latent"
                            else "frozen learned representation"
                        )
                    )
                ),
            },
            "semantics": {
                "static": "finite-test-set lower witnesses for radius-indexed supremum",
                "action": "finite-test-set centered-neighborhood common-action conflicts",
                "action_tail": (
                    "per-trajectory 95th percentile of centered-neighborhood required "
                    "violations for the configured finite constant-action profile; not a "
                    "full robust Bellman-Q regret"
                ),
                "warning": "neither curve certifies unsampled states or continuous actions",
            },
            "split_audits": {
                **({"validation": validation_audit} if validation_audit is not None else {}),
                "test": test_audit,
            },
            "static_curve": test_audit["static_curve"],
            "action_curve": test_audit["action_curve"],
        }
    )
    write_json_atomic(args.output.resolve(), manifest)
    print(
        f"wrote {args.output.resolve()} with {len(test_audit['static_curve'])} static and "
        f"{len(test_audit['action_curve'])} action radii on test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
