#!/bin/bash
# GPU job array: one array index per config entry (sweep_list[index]).
#
# Usage (from the repo root on Adroit; `make submit SWEEP=<name>` does this for you):
#   sbatch --array=0-9 --job-name=<sweep> slurm/array_template.slurm scripts/run_<sweep>.py [hydra overrides...]
#
# The log directory must exist before submitting:
#   mkdir -p /scratch/network/$USER/flyvis-closedloop/logs
#
#SBATCH --job-name=flyloop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00          # GPU jobs are capped at 2 days (48:00:00)
#SBATCH --array=0-0              # overridden by sbatch --array on the command line
#SBATCH --output=/scratch/network/%u/flyvis-closedloop/logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ag0226@princeton.edu

set -euo pipefail

SCRIPT=${1:?usage: sbatch slurm/array_template.slurm <script.py> [overrides...]}
shift

# Conda, the way Princeton Research Computing recommends: purge modules, load
# their anaconda3 module, activate the env created on the login node.
# Use the same version that created the env (checked with `module avail anaconda3`, Oct 2026).
module purge
module load anaconda3/2026.7
conda activate flyloop

# Outputs go to scratch, never to /home.
export FLYLOOP_RUNS=/scratch/network/$USER/flyvis-closedloop/runs
mkdir -p "$FLYLOOP_RUNS"

# Headless MuJoCo rendering on the cluster (do not set this on the Mac).
export MUJOCO_GL=egl
# Compute nodes may have no internet: log W&B offline, `wandb sync` from the login node.
export WANDB_MODE=offline
export WANDB_DIR=/scratch/network/$USER/flyvis-closedloop/wandb
export PYTHONUNBUFFERED=1

echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "commit $(git rev-parse --short HEAD)  script $SCRIPT  args $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$SCRIPT" --index "$SLURM_ARRAY_TASK_ID" "$@"
