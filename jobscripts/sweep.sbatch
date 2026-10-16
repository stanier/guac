#!/bin/bash
#SBATCH --job-name=guac_sweep
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm/sweep_%A_%a.out
#SBATCH --error=logs/slurm/sweep_%A_%a.err

# Runs a sweep spec as a job array: array task I of N runs every Nth run of the
# sweep (sweep.py --shard I/N), all writing to one output directory, and each
# shard's results.md includes every shard's finished runs. Resubmitting skips
# finished runs. See sweeps/README.md.
#
#   sbatch jobscripts/sweep.sbatch                                         # sweeps/recall_scaled.yaml
#   sbatch --export=ALL,SPEC=sweeps/lm_scaled.yaml jobscripts/sweep.sbatch
#   sbatch --export=ALL,BUDGET="--flop-match full_attention" jobscripts/sweep.sbatch
#   sbatch --array=0-15 jobscripts/sweep.sbatch                            # more GPUs, shorter jobs
#
# Afterwards, python guac/pipeline/sweep.py <spec> [BUDGET] --summarize rebuilds the table.

set -euo pipefail

export SPEC=${SPEC:-sweeps/recall_scaled.yaml}
export BUDGET=${BUDGET:-}                 # empty: the spec's budget
export OUTPUT_DIR=${OUTPUT_DIR:-}         # empty: scratch/sweeps/<name>[-param|-flop]
export TRAINING_VENV=${TRAINING_VENV:-$HOME/venvs/guac}

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$TRAINING_VENV/bin/activate"

SHARD_COUNT=${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))

ARGS=("$SPEC" --shard "$SHARD_INDEX/$SHARD_COUNT")
if [ -n "$BUDGET" ]; then read -r -a BUDGET_ARGS <<< "$BUDGET"; ARGS+=("${BUDGET_ARGS[@]}"); fi
if [ -n "$OUTPUT_DIR" ]; then ARGS+=(--output-dir "$OUTPUT_DIR"); fi

srun python guac/pipeline/sweep.py "${ARGS[@]}"
deactivate
