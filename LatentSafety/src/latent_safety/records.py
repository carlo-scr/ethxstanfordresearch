"""Portable audit records and trajectory-split validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRecord:
    """One privileged-evaluation record emitted by a frozen representation."""

    sample_id: str
    trajectory_id: str
    split: str
    safety_margin: float
    latent: tuple[float, ...]
    observation_latent: tuple[float, ...] | None = None
    state_latent: tuple[float, ...] | None = None
    action_safety_margins: tuple[float, ...] | None = None

    def validate(self) -> None:
        if not self.sample_id or not self.trajectory_id:
            raise ValueError("sample_id and trajectory_id must be non-empty")
        if self.split not in {"train", "validation", "calibration", "test"}:
            raise ValueError(f"invalid split: {self.split!r}")
        if not math.isfinite(self.safety_margin):
            raise ValueError("safety_margin must be finite")
        if not self.latent or any(not math.isfinite(value) for value in self.latent):
            raise ValueError("latent must be non-empty and finite")
        if self.observation_latent is not None and (
            not self.observation_latent
            or any(not math.isfinite(value) for value in self.observation_latent)
        ):
            raise ValueError("observation_latent must be non-empty and finite when present")
        if self.state_latent is not None and (
            not self.state_latent
            or any(not math.isfinite(value) for value in self.state_latent)
        ):
            raise ValueError("state_latent must be non-empty and finite when present")
        if self.action_safety_margins is not None and (
            not self.action_safety_margins
            or any(not math.isfinite(value) for value in self.action_safety_margins)
        ):
            raise ValueError("action_safety_margins must be non-empty and finite when present")


def validate_records(records: Iterable[AuditRecord]) -> tuple[AuditRecord, ...]:
    """Validate IDs and ensure no trajectory leaks across data splits."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("at least one audit record is required")
    sample_ids: set[str] = set()
    trajectory_splits: dict[str, str] = {}
    for record in materialized:
        record.validate()
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id!r}")
        sample_ids.add(record.sample_id)
        previous_split = trajectory_splits.setdefault(record.trajectory_id, record.split)
        if previous_split != record.split:
            raise ValueError(
                f"trajectory {record.trajectory_id!r} occurs in both "
                f"{previous_split!r} and {record.split!r}"
            )
    return materialized


def write_jsonl(path: Path, records: Iterable[AuditRecord]) -> None:
    """Write validated records atomically as stable JSON Lines."""

    materialized = validate_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in materialized:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> tuple[AuditRecord, ...]:
    records: list[AuditRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                payload = json.loads(line)
                records.append(
                    AuditRecord(
                        sample_id=str(payload["sample_id"]),
                        trajectory_id=str(payload["trajectory_id"]),
                        split=str(payload["split"]),
                        safety_margin=float(payload["safety_margin"]),
                        latent=tuple(float(value) for value in payload["latent"]),
                        observation_latent=(
                            tuple(float(value) for value in payload["observation_latent"])
                            if payload.get("observation_latent") is not None
                            else None
                        ),
                        state_latent=(
                            tuple(float(value) for value in payload["state_latent"])
                            if payload.get("state_latent") is not None
                            else None
                        ),
                        action_safety_margins=(
                            tuple(float(value) for value in payload["action_safety_margins"])
                            if payload.get("action_safety_margins") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    return validate_records(records)
