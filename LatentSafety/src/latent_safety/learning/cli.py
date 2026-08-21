"""Command-line interface for dependency-free planning and optional GPU execution."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from latent_safety.learning.config import (
    ConfigError,
    cpu_smoke_config,
    dry_run_plan,
    load_config,
    validate_config,
)
from latent_safety.learning.runtime import TorchUnavailableError


_SAFETY_ARMS = ("none", "h_prediction", "boundary_contrastive", "safe_action_profile")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_e1_torch",
        description="Train and audit a synthetic visual latent world model.",
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML experiment config")
    parser.add_argument("--output", type=Path, help="override run.output_dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, help="override the paired experiment seed")
    parser.add_argument("--model-family", choices=("ae", "beta_vae"))
    parser.add_argument("--history-encoder", choices=("stack", "gru"))
    parser.add_argument(
        "--history-length",
        type=int,
        help="override data.history_length; sweep tasks pair this with --history-encoder",
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--safety-arm", choices=_SAFETY_ARMS, help="override one ablation arm")
    parser.add_argument("--safety-weight", type=float, help="override its supervision weight")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the execution plan without importing PyTorch",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a one-epoch, 16-trajectory CPU integration check (not paper evidence)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config_path = arguments.config.resolve()
        config = load_config(config_path)
        run = config.run
        data = config.data
        model = config.model
        objective = config.objective
        if arguments.device is not None:
            run = dataclasses.replace(run, device=arguments.device)
        if arguments.seed is not None:
            run = dataclasses.replace(run, seed=arguments.seed)
        if arguments.model_family is not None:
            model = dataclasses.replace(model, family=arguments.model_family)
            if arguments.model_family == "ae" and arguments.kl_weight is None:
                objective = dataclasses.replace(objective, kl_weight=0.0)
        if arguments.history_encoder is not None:
            model = dataclasses.replace(model, history_encoder=arguments.history_encoder)
        if arguments.history_length is not None:
            data = dataclasses.replace(data, history_length=arguments.history_length)
        if arguments.latent_dim is not None:
            model = dataclasses.replace(model, latent_dim=arguments.latent_dim)
        if arguments.kl_weight is not None:
            objective = dataclasses.replace(objective, kl_weight=float(arguments.kl_weight))
        if arguments.safety_arm is not None:
            objective = dataclasses.replace(objective, safety_arm=arguments.safety_arm)
            if arguments.safety_arm == "none" and arguments.safety_weight is None:
                objective = dataclasses.replace(objective, safety_weight=0.0)
        if arguments.safety_weight is not None:
            objective = dataclasses.replace(
                objective,
                safety_weight=float(arguments.safety_weight),
            )
        config = dataclasses.replace(
            config,
            run=run,
            data=data,
            model=model,
            objective=objective,
        )
        validate_config(config)
        if arguments.smoke:
            config = cpu_smoke_config(config)
        repo_root = Path(__file__).resolve().parents[3]
        configured_output = Path(config.run.output_dir)
        output_dir = arguments.output or configured_output
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        plan = dry_run_plan(config)
        plan["config"] = str(config_path)
        plan["output_dir"] = str(output_dir.resolve())
        if arguments.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        # Importing the trainer is intentionally below the dry-run return. The trainer itself also
        # loads torch lazily, which keeps error messages controlled and the minimal install useful.
        from latent_safety.learning.trainer import run_experiment

        result = run_experiment(
            config,
            config_path=config_path,
            output_dir=output_dir.resolve(),
            repo_root=repo_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ConfigError, TorchUnavailableError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
