#!/usr/bin/env python3
"""Repository-local launcher for the optional E1 PyTorch pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from latent_safety.learning.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

