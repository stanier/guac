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

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail

# Tokenizer configuration
export VOCAB_SIZE=${VOCAB_SIZE:-60000}
export TOKENIZER_TYPE=${TOKENIZER_TYPE:-sentencepiece}
export TOKENIZER_DATASETS=${TOKENIZER_DATASETS:-fineweb}
export MAX_LENGTH=${MAX_LENGTH:-1024}

# Pre-training configuration
export PRETRAIN_CONFIG=${PRETRAIN_CONFIG:-guac-3b}
export PRETRAIN_DATASETS=${PRETRAIN_DATASETS:-fineweb}
export PRETRAIN_STEPS=${PRETRAIN_STEPS:-10000}

# Mid-training configuration
export MIDTRAIN_TASK=${MIDTRAIN_TASK:-arithmetic}
export MIDTRAIN_LEARNING_RATE=${MIDTRAIN_LEARNING_RATE:-1e-5}
export MIDTRAIN_SAMPLES=${MIDTRAIN_SAMPLES:-50000}

# Post-training configuration
export POSTTRAIN_MODE=${POSTTRAIN_MODE:-instruction}
export POSTTRAIN_DATASETS=${POSTTRAIN_DATASETS:-}  # Empty = use synthetic data

# General configuration
export ENABLE_PROFILING=${ENABLE_PROFILING:-false}
export FORCE_RETOKENIZE=${FORCE_RETOKENIZE:-false}
export DEBUG_MODE=${DEBUG_MODE:-false}

export JOB_CUSTODIAN="$USER"
export JOB_GROUP="your_lab_group"

export PROJECT_DIR="/work/$JOB_GROUP/$JOB_CUSTODIAN/"
export SCRATCH_DIR="/scratch/$JOB_CUSTODIAN/"
export LSCRATCH_DIR="/lscratch/$JOB_CUSTODIAN/"

export VANILLA_LAUNCHER="python "
export DISTRIBUTED_LAUNCHER="python -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $SLURM_NNODES \
        --rdzv_id $RDZV_ID \
        --rdzv_backend $RDZV_BACKEND \
        --rdzv_endpoint $head_node_ip:$RDZV_PORT \
"

export LAUNCHER="$VANILLA_LAUNCHER"

cd $SLURM_SUBMIT_DIR

module load \
    Transformers/4.39.3-gfbf-2023a \
    datasets/3.1.0-foss-2023a \
    huggingface_hub/0.26.3-GCCcore-12.3.0 \
    PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 \
    SentencePiece/0.2.0-GCC-12.3.0

source $TRAINING_VENV/bin/activate

train_tokenizer() {
    srun -J "tokenizer_training" --jobid $SLURM_JOB_ID bash -c """
$LAUNCHER scripts/training/pipeline/train_tokenizer.py \
    --vocab-size $VOCAB_SIZE \
    --tokenizer-type $TOKENIZER_TYPE \
    --datasets $TOKENIZER_DATASETS \
    --max-samples 50000
rsync -r ${LSCRATCH_DIR}/${TOKENIZER_PATH} ${SCRATCH_DIR}/${TOKENIZER_PATH}
"""
}

RETOKENIZE_FLAG=""
if [ "$FORCE_RETOKENIZE" = "true" ]; then RETOKENIZE_FLAG="--force-retokenize"; fi

load_datasets() {

}

tokenize_dataset() {
    srun -J "dataset_tokenization" --jobid $SLURM_JOB_ID bash -c """
rsync -r $SCRATCH_TODO $LSCRATCH_TODO
$LAUNCHER scripts/training/pipeline/tokenize_datasets.py \
    --tokenizer $LSCRATCH_DIR/$TOKENIZER_PATH \
    --datasets $PRETRAIN_DATASETS \
    --max-length $MAX_LENGTH \
    --stage pretraining \
    $RETOKENIZE_FLAG
rsync -r $LSCRATCH_TODO $SCRATCH_TODO
"""
}

pretrain() {
    PRETRAIN_ARGS=""
    if [ "$ENABLE_PROFILING" = "true" ]; then PRETRAIN_ARGS="$PRETRAIN_ARGS --enable-profiling"; fi
    if [ "$DEBUG_MODE" = "true" ]; then PRETRAIN_ARGS="$PRETRAIN_ARGS --debug"; fi

    srun -J "pretraining" --jobid $SLURM_JOB_ID bash -c """
rsync -r $SCRATCH_DIR/$TOKENIZER_NAME $LSCRATCH_DIR/$TOKENIZER_NAME
$LAUNCHER scripts/training/pipeline/pretrain.py \
    --config $PRETRAIN_CONFIG \
    --max-steps $PRETRAIN_STEPS \
    $PRETRAIN_ARGS
rsync -r $LSCRATCH_DIR/$CHECKPOINT_NAME $SCRATCH_DIR/$CHECKPOINT_NAME
"""
}

midtrain() {
    # Find the pretrained model directory
    PRETRAIN_OUTPUT=$(ls -td scratch/models/pretrain_${PRETRAIN_CONFIG}_* | head -1)
    if [ -z "$PRETRAIN_OUTPUT" ]; then echo "Could not find pretrained model output!"; exit 1; fi

    MIDTRAIN_ARGS="--base-model $PRETRAIN_OUTPUT --task $MIDTRAIN_TASK --learning-rate $MIDTRAIN_LEARNING_RATE"
    if [ "$MIDTRAIN_TASK" = "arithmetic" ]; then MIDTRAIN_ARGS="$MIDTRAIN_ARGS --num-arithmetic-samples $MIDTRAIN_SAMPLES"; fi
    if [ "$FORCE_RETOKENIZE" = "true" ]; then MIDTRAIN_ARGS="$MIDTRAIN_ARGS --force-retokenize"; fi
    if [ "$ENABLE_PROFILING" = "true" ]; then MIDTRAIN_ARGS="$MIDTRAIN_ARGS --enable-profiling"; fi
    if [ "$DEBUG_MODE" = "true" ]; then MIDTRAIN_ARGS="$MIDTRAIN_ARGS --debug"; fi

    srun -J "midtraining" --jobid $SLURM_JOB_ID bash -c """
rsync -r $SCRATCH_DIR/$CHECKPOINT_NAME $LSCRATCH_DIR/$CHECKPOINT_NAME
$LAUNCHER scripts/training/pipeline/midtrain.py $MIDTRAIN_ARGS
rsync -r $LSCRATCH_DIR/$CHECKPOINT_NAME $SCRATCH_DIR/$CHECKPOINT_NAME
"""
}

posttrain() {
    # Find the midtrained model directory
    BASE_MODEL_NAME=$(basename $PRETRAIN_OUTPUT)
    MIDTRAIN_OUTPUT=$(ls -td scratch/models/midtrain_${BASE_MODEL_NAME}_${MIDTRAIN_TASK}_* | head -1)
    if [ -z "$MIDTRAIN_OUTPUT" ]; then echo "Could not find midtrained model output"; exit 1; fi

    POSTTRAIN_ARGS="--base-model $MIDTRAIN_OUTPUT --mode $POSTTRAIN_MODE"
    if [ -n "$POSTTRAIN_DATASETS" ]; then POSTTRAIN_ARGS="$POSTTRAIN_ARGS --datasets $POSTTRAIN_DATASETS"; fi
    if [ "$FORCE_RETOKENIZE" = "true" ]; then POSTTRAIN_ARGS="$POSTTRAIN_ARGS --force-retokenize"; fi
    if [ "$ENABLE_PROFILING" = "true" ]; then POSTTRAIN_ARGS="$POSTTRAIN_ARGS --enable-profiling"; fi
    if [ "$DEBUG_MODE" = "true" ]; then POSTTRAIN_ARGS="$POSTTRAIN_ARGS --debug"; fi

    srun -J "posttraining" --jobid $SLURM_JOB_ID bash -c """
rsync -r $SCRATCH_DIR/$CHECKPOINT_NAME $LSCRATCH_DIR/$CHECKPOINT_NAME
$LAUNCHER scripts/training/pipeline/posttrain.py $POSTTRAIN_ARGS
rsync -r $LSCRATCH_DIR/$CHECKPOINT_NAME $SCRATCH_DIR/$CHECKPOINT_NAME
"""
}

main() {
    ## Environment variables
    #export TOKENIZERS_PARALLELISM=false
    #export PYTHONUNBUFFERED=1
    #export CUDA_LAUNCH_BLOCKING=0
    #
    #TOKENIZER_PATH="tokenizers/${TOKENIZER_TYPE}/shared_${VOCAB_SIZE}"
    #
    #if [ -f "${SCRATCH_DIR}/${TOKENIZER_PATH}/tokenizer_config.json" ]; then
    #    echo "Tokenizer already exists at ${SCRATCH_DIR}/${TOKENIZER_PATH}"
    #    echo "Skipping tokenizer training..."
    #else
    #    train_tokenizer()
    #fi
    #
    ## Find the final model directory
    #MIDTRAIN_BASE=$(basename $MIDTRAIN_OUTPUT)
    #FINAL_OUTPUT=$(ls -td scratch/models/posttrain_${MIDTRAIN_BASE}_${POSTTRAIN_MODE}_* | head -1)
    #if [ -z "$FINAL_OUTPUT" ]; then echo "Could not find final model output"; exit 1; fi


}

main

deactivate
