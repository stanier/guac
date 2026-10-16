#!/bin/bash
#SBATCH --job-name=guac_full_pipeline
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/full_pipeline_%j.out
#SBATCH --error=logs/slurm/full_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# Runs tokenizer training (if needed), pre-tokenization, pre-training,
# mid-training and post-training in order. Override any setting below with
# sbatch --export=ALL,PRETRAIN_CONFIG=guac-430m,... (see docs/SLURM.md).
# Outputs go where the pipeline puts them by default (scratch/ via site_config.yml).

set -euo pipefail

# Tokenizer: the preset's tokenizer_model_path (trained or built if missing), unless
# TOKENIZER_DIR names one, or VOCAB_SIZE / TOKENIZER_TYPE ask for a different one
export TOKENIZER_DIR=${TOKENIZER_DIR:-}
export VOCAB_SIZE=${VOCAB_SIZE:-}
export TOKENIZER_TYPE=${TOKENIZER_TYPE:-}
export TOKENIZER_DATASETS=${TOKENIZER_DATASETS:-fineweb}
export MAX_LENGTH=${MAX_LENGTH:-}   # empty: the preset's max_sequence_length

# Pre-training (streaming datasets such as fineweb need a step budget)
export PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-guac-3b}
export PRETRAIN_DATASETS=${PRETRAIN_DATASETS:-fineweb}
export PRETRAIN_STEPS=${PRETRAIN_STEPS:-10000}

# Mid-training
export MIDTRAIN_TASK=${MIDTRAIN_TASK:-arithmetic}
export MIDTRAIN_LEARNING_RATE=${MIDTRAIN_LEARNING_RATE:-1e-5}
export MIDTRAIN_SAMPLES=${MIDTRAIN_SAMPLES:-50000}
export MIDTRAIN_DATASETS=${MIDTRAIN_DATASETS:-}   # required for MIDTRAIN_TASK=custom

# Post-training (empty: the mode's default dataset, alpaca or ultrachat)
export POSTTRAIN_MODE=${POSTTRAIN_MODE:-instruction}
export POSTTRAIN_DATASETS=${POSTTRAIN_DATASETS:-}

# General
export ENABLE_PROFILING=${ENABLE_PROFILING:-false}
export FORCE_RETOKENIZE=${FORCE_RETOKENIZE:-false}
export DEBUG_MODE=${DEBUG_MODE:-false}
export TRAINING_VENV=${TRAINING_VENV:-$HOME/venvs/guac}   # created with: pip install -e ".[test]"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

# The venv supplies the pinned Python packages (transformers 4.46.3); load only
# what it can't, such as CUDA. A Transformers module on PYTHONPATH would shadow
# the venv's version.
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$TRAINING_VENV/bin/activate"

DEBUG_FLAGS=()
if [ "$DEBUG_MODE" = "true" ]; then DEBUG_FLAGS+=(--debug); fi
COMMON_FLAGS=("${DEBUG_FLAGS[@]}")
if [ "$ENABLE_PROFILING" = "true" ]; then COMMON_FLAGS+=(--enable-profiling); fi
RETOKENIZE_FLAGS=()
if [ "$FORCE_RETOKENIZE" = "true" ]; then RETOKENIZE_FLAGS+=(--force-retokenize); fi

# Every stage gets the same tokenizer and length, so pretrain.py finds the pre-tokenized cache
DATA_FLAGS=()
if [ -n "$MAX_LENGTH" ]; then DATA_FLAGS+=(--max-length "$MAX_LENGTH"); fi

# Paths resolve through site_config.yml's scratch_dir, as the Python scripts' defaults do
SCRATCH_DIR=$(python -c "from guac.core.path_manager import get_scratch_dir; print(get_scratch_dir())")
if [ -z "$TOKENIZER_DIR" ] && [ -n "$VOCAB_SIZE$TOKENIZER_TYPE" ]; then
    TOKENIZER_DIR="$SCRATCH_DIR/tokenizers/${TOKENIZER_TYPE:-sentencepiece}/shared_$(( ${VOCAB_SIZE:-60000} / 1000 ))k"
elif [ -z "$TOKENIZER_DIR" ]; then
    TOKENIZER_DIR=$(python -c "
import os
from guac.core.pipeline_utils import load_model_config, resolve_scratch_path
path = resolve_scratch_path(load_model_config('$PRETRAIN_CONFIG').get('tokenizer_model_path', 'scratch/tokenizers/sentencepiece/shared_60k'))
print(os.path.dirname(path) if path.endswith('.model') else path)" | tail -n 1)
fi
PRETRAIN_DIR="$SCRATCH_DIR/models/pretraining/${PRETRAIN_CONFIG}"
MIDTRAIN_DIR="$SCRATCH_DIR/models/midtraining/${PRETRAIN_CONFIG}_${MIDTRAIN_TASK}"
POSTTRAIN_DIR="$SCRATCH_DIR/models/posttraining/${PRETRAIN_CONFIG}_${MIDTRAIN_TASK}_${POSTTRAIN_MODE}"
echo "Tokenizer: $TOKENIZER_DIR"

train_tokenizer() {
    if [ -f "$TOKENIZER_DIR/tokenizer_config.json" ]; then
        echo "Tokenizer exists at $TOKENIZER_DIR; skipping tokenizer training"
        return
    fi
    if [[ "$TOKENIZER_DIR" == */character/* ]]; then
        srun -J tokenizer python -m guac.tokenizer.build_char_tokenizer --output "$TOKENIZER_DIR"
        return
    fi
    srun -J tokenizer python guac/pipeline/train_tokenizer.py --output-dir "$TOKENIZER_DIR" \
        --vocab-size "${VOCAB_SIZE:-60000}" --tokenizer-type "${TOKENIZER_TYPE:-sentencepiece}" \
        --datasets "$TOKENIZER_DATASETS" --max-samples 50000
}

tokenize_datasets() {
    # Fills the cache pretrain.py reads; streaming datasets are tokenized during training instead
    srun -J tokenize python guac/pipeline/tokenize_datasets.py \
        --model-config "$PRETRAIN_CONFIG" --tokenizer "$TOKENIZER_DIR" --datasets "$PRETRAIN_DATASETS" \
        "${DATA_FLAGS[@]}" "${DEBUG_FLAGS[@]}" "${RETOKENIZE_FLAGS[@]}"
}

pretrain() {
    srun -J pretrain python guac/pipeline/pretrain.py \
        --model-config "$PRETRAIN_CONFIG" --tokenizer "$TOKENIZER_DIR" --datasets "$PRETRAIN_DATASETS" \
        --max-steps "$PRETRAIN_STEPS" --output-dir "$PRETRAIN_DIR" \
        "${DATA_FLAGS[@]}" "${COMMON_FLAGS[@]}" "${RETOKENIZE_FLAGS[@]}"
}

midtrain() {
    local args=(--base-model "$PRETRAIN_DIR" --task "$MIDTRAIN_TASK" --learning-rate "$MIDTRAIN_LEARNING_RATE"
                --output-dir "$MIDTRAIN_DIR")
    if [ "$MIDTRAIN_TASK" = "arithmetic" ]; then args+=(--num-arithmetic-samples "$MIDTRAIN_SAMPLES"); fi
    if [ -n "$MIDTRAIN_DATASETS" ]; then args+=(--datasets $MIDTRAIN_DATASETS); fi
    srun -J midtrain python guac/pipeline/midtrain.py "${args[@]}" "${COMMON_FLAGS[@]}" "${RETOKENIZE_FLAGS[@]}"
}

posttrain() {
    local args=(--base-model "$MIDTRAIN_DIR" --mode "$POSTTRAIN_MODE" --output-dir "$POSTTRAIN_DIR")
    if [ -n "$POSTTRAIN_DATASETS" ]; then args+=(--datasets $POSTTRAIN_DATASETS); fi
    srun -J posttrain python guac/pipeline/posttrain.py "${args[@]}" "${COMMON_FLAGS[@]}"
}

train_tokenizer
tokenize_datasets
pretrain
midtrain
posttrain

echo "Pipeline complete: $POSTTRAIN_DIR"
deactivate
