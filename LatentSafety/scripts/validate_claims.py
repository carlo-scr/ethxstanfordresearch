#!/usr/bin/env python3
"""Validate that every manuscript claim has an explicit evidence state."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ALLOWED_STATUSES = {
    "hypothesis",
    "theorem_needs_proof",
    "pilot_unreproduced",
    "verified_internal",
    "verified_independent",
    "rejected",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "paper" / "claims.toml"
    with path.open("rb") as stream:
        registry = tomllib.load(stream)
    errors: list[str] = []
    seen: set[str] = set()
    for claim in registry.get("claims", []):
        claim_id = claim.get("id")
        if not claim_id or claim_id in seen:
            errors.append(f"invalid or duplicate claim id: {claim_id!r}")
        seen.add(claim_id)
        if claim.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{claim_id}: invalid status {claim.get('status')!r}")
        for field in ("statement", "status", "evidence"):
            if not str(claim.get(field, "")).strip():
                errors.append(f"{claim_id}: missing {field}")
        if claim.get("status", "").startswith("verified") and not claim.get("command"):
            errors.append(f"{claim_id}: verified claims require a producing command")
    if not seen:
        errors.append("claim registry is empty")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(seen)} paper claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

