#!/usr/bin/env bash
#SBATCH --job-name=latent-safety-e1-pendulum
#SBATCH --array=0-143%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/e1-pendulum-%A_%a.out

set -euo pipefail

E1_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${E1_REPO_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"

python scripts/run_e1_sweep_task.py \
  --plan runs/plans/e1_pendulum_pilot.json \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda
