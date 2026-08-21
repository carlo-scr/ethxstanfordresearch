"""Small, dependency-free run manifest helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def _git(command: list[str], cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", *command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_distributions() -> tuple[str, ...]:
    """Return a path-free snapshot of installed Python package names and versions."""

    resolved: set[str] = set()
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            resolved.add(f"{name}=={distribution.version}")
    return tuple(sorted(resolved, key=str.casefold))


def base_manifest(*, repo_root: Path, config_path: Path) -> dict[str, Any]:
    status = _git(["status", "--porcelain"], repo_root)
    installed = _installed_distributions()
    installed_payload = "\n".join(installed).encode("utf-8")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": config_sha256(config_path),
        },
        "code": {
            "git_revision": _git(["rev-parse", "HEAD"], repo_root),
            "git_dirty": bool(status) if status is not None else None,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "installed_distributions": installed,
            "installed_distributions_sha256": hashlib.sha256(
                installed_payload
            ).hexdigest(),
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
