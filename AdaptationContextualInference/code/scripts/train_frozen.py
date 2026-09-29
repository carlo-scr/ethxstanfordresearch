"""Collect training-condition transitions and fit the frozen dynamics + noise model.

    python scripts/train_frozen.py --out runs/frozen.pt
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cira.config import EnvConfig, FrozenConfig
from cira.frozen import FrozenModel, collect_training_data, residual_table, rollout_errors


def report(model: FrozenModel, env_cfg: EnvConfig) -> None:
    table = residual_table(model, env_cfg)
    print("mean |velocity residual| / sigma per context (rows: surfaces, cols: loads; ~0.8 = training):")
    for (name, _), row in zip(env_cfg.surfaces, table):
        print(f"  {name:8s}", " ".join(f"{v:6.2f}" for v in row))
    errs = rollout_errors(model, env_cfg)
    print(f"training-condition rollout: final-position error median {100 * np.median(errs):.1f} cm, "
          f"90th pct {100 * np.quantile(errs, 0.9):.1f} cm over {len(errs)} strikes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/frozen.pt"))
    ap.add_argument("--episodes", type=int, default=FrozenConfig.n_episodes)
    ap.add_argument("--epochs", type=int, default=FrozenConfig.epochs)
    ap.add_argument("--hidden", type=int, default=FrozenConfig.hidden)
    ap.add_argument("--eval-only", action="store_true", help="only report on an existing --out model")
    args = ap.parse_args()
    env_cfg = EnvConfig()
    if args.eval_only:
        model = FrozenModel.load(args.out)
        report(model, env_cfg)
        return
    cfg = replace(FrozenConfig(), n_episodes=args.episodes, epochs=args.epochs, hidden=args.hidden)
    t0 = time.time()
    Z, A, Zn = collect_training_data(env_cfg, cfg.n_episodes, cfg.seed)
    print(f"collected {len(Z)} transitions from {cfg.n_episodes} episodes in {time.time() - t0:.1f}s")
    model = FrozenModel(env_cfg, cfg.hidden, cfg.sigma_floor).fit(Z, A, Zn, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out} ({time.time() - t0:.1f}s total)")
    report(model, env_cfg)


if __name__ == "__main__":
    main()
