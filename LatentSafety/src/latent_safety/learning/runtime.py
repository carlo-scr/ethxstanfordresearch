"""Lazy PyTorch loading, device selection, and reproducibility controls."""

from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass
from typing import Any


class TorchUnavailableError(RuntimeError):
    """Raised when a training run is requested without the optional dependency."""


@dataclass(frozen=True)
class TorchModules:
    torch: Any
    nn: Any
    functional: Any


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    accelerator_name: str


def require_torch() -> TorchModules:
    """Import PyTorch only at the point where numerical execution is requested."""

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise TorchUnavailableError(
            "PyTorch is required for training but is not installed. Install the project's "
            "research extras (for example: pip install -e '.[research]') or use --dry-run."
        ) from error
    return TorchModules(torch=torch, nn=nn, functional=functional)


def select_device(torch: Any, requested: str) -> tuple[Any, DeviceInfo]:
    """Resolve an explicit or automatic accelerator request without silent fallback."""

    if requested == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device='cuda' was requested, but CUDA is unavailable")
    if selected == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("run.device='mps' was requested, but MPS is unavailable")
    device = torch.device(selected)
    if selected == "cuda":
        name = str(torch.cuda.get_device_name(device))
    elif selected == "mps":
        name = "Apple Metal Performance Shaders"
    else:
        name = "CPU"
    return device, DeviceInfo(
        requested=requested,
        selected=str(device),
        accelerator_name=name,
    )


def seed_everything(
    torch: Any,
    *,
    seed: int,
    deterministic: bool,
    warn_only: bool,
) -> dict[str, Any]:
    """Seed all randomness used by the pipeline and configure deterministic kernels."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": deterministic,
        "deterministic_warn_only": warn_only,
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends, "cudnn")
            else None
        ),
    }


def seed_data_loader_worker(worker_id: int) -> None:
    """Derive standard-library worker randomness from PyTorch's worker seed."""

    del worker_id
    modules = require_torch()
    worker_seed = int(modules.torch.initial_seed() % 2**32)
    random.seed(worker_seed)
