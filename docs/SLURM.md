# Guac Slurm Training Scripts

This directory contains Slurm jobscripts for running Guac training pipelines on HPC clusters.

## Quick Start

### Basic Usage

Submit the full pipeline with default settings:
```bash
sbatch jobscripts/full_pipeline.sbatch
```

### Customizing Parameters

You can customize the pipeline by setting environment variables when submitting:

```bash
# Train a smaller model with custom vocab
sbatch --export=ALL,PRETRAIN_CONFIG=guac-910m,VOCAB_SIZE=32000 \
    jobscripts/full_pipeline.sbatch

# Use different tasks and modes
sbatch --export=ALL,MIDTRAIN_TASK=code,POSTTRAIN_MODE=sft \
    jobscripts/full_pipeline.sbatch

# Debug mode with profiling
sbatch --export=ALL,DEBUG_MODE=true,ENABLE_PROFILING=true \
    jobscripts/full_pipeline.sbatch
```

### Configuration Variables

#### Tokenizer Settings
- `VOCAB_SIZE`: Vocabulary size (default: 60000)
- `TOKENIZER_TYPE`: sentencepiece or bpe (default: sentencepiece)
- `TOKENIZER_DATASETS`: Datasets for tokenizer training (default: fineweb)
- `MAX_LENGTH`: Maximum sequence length (default: 1024)

#### Pre-training Settings
- `PRETRAIN_CONFIG`: Model configuration - guac-3b, guac-4.5b, guac-910m, guac-430m (default: guac-3b)
- `PRETRAIN_DATASETS`: Training datasets (default: fineweb)
- `PRETRAIN_STEPS`: Maximum training steps (default: 10000)

#### Mid-training Settings
- `MIDTRAIN_TASK`: Specialization task - arithmetic, code, reasoning, custom (default: arithmetic)
- `MIDTRAIN_LEARNING_RATE`: Learning rate (default: 1e-5)
- `MIDTRAIN_SAMPLES`: Number of arithmetic samples (default: 50000)
- `MIDTRAIN_DATASETS`: Space-separated dataset paths or names; required for `custom`, optional for code and reasoning

#### Post-training Settings
- `POSTTRAIN_MODE`: instruction or sft (default: instruction)
- `POSTTRAIN_DATASETS`: Space-separated dataset names (default: alpaca for instruction, ultrachat for sft)

#### General Settings
- `ENABLE_PROFILING`: Enable performance profiling (default: false)
- `FORCE_RETOKENIZE`: Force dataset re-tokenization (default: false)
- `DEBUG_MODE`: Use smaller datasets for testing (default: false)
- `TRAINING_VENV`: Virtual environment with `pip install -e ".[test]"` (default: `$HOME/venvs/guac`)

Pass variables with `--export=ALL,...`: a bare `--export=VAR=value` exports only the listed
variables, dropping `PATH` and the rest of the submitting environment.

## SLURM Resource Configuration

### Modifying Resources

Edit the `#SBATCH` directives at the top of the script:

```bash
#SBATCH --partition=gpu          # Queue/partition name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs (e.g., gpu:2, gpu:a100:1)
#SBATCH --cpus-per-task=8        # CPU cores
#SBATCH --mem=64G                # Memory
#SBATCH --time=48:00:00          # Maximum runtime (HH:MM:SS)
```

### Multi-GPU Training

For multi-GPU training, modify the GPU request:
```bash
#SBATCH --gres=gpu:4             # Request 4 GPUs
```

And update the Python launch command to use `torchrun` or similar.

## Pipeline Stages

The full pipeline runs these stages in sequence:

1. **Tokenizer Training** (Stage 1)
   - Trains a SentencePiece or BPE tokenizer
   - Skips if tokenizer already exists
   - Output: `scratch/tokenizers/{type}/shared_{vocab_size / 1000}k/`

2. **Dataset Pre-tokenization** (Stage 2)
   - Pre-tokenizes and caches datasets
   - Significantly speeds up training
   - Uses caching - only runs once per dataset/vocab combination

3. **Pre-training** (Stage 3)
   - Trains base language model, with validation perplexity
   - Uses the preset `PRETRAIN_CONFIG` from `model_configs.yaml`
   - Output: `scratch/models/pretraining/{config}/`

4. **Mid-training** (Stage 4)
   - Specializes model on specific tasks
   - Tasks: arithmetic, code, reasoning, custom
   - Output: `scratch/models/midtraining/{config}_{task}/`

5. **Post-training** (Stage 5)
   - Instruction tuning or supervised fine-tuning
   - Modes: instruction, sft
   - Output: `scratch/models/posttraining/{config}_{task}_{mode}/`

## Environment Setup

### Before Running

1. **Update the environment setup section** in `full_pipeline.sbatch`:
   ```bash
   # Option 1: Using conda
   source ~/.bashrc
   conda activate guac

   # Option 2: Using venv
   source venv/bin/activate

   # Option 3: Using module system
   module load python/3.10
   module load cuda/11.8
   ```

2. **Update email notifications**:
   ```bash
   #SBATCH --mail-user=your.email@example.com
   ```

3. **Verify partition names** match your cluster:
   ```bash
   #SBATCH --partition=gpu  # Change to your cluster's GPU partition name
   ```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View output logs
```bash
# While running
tail -f logs/slurm/full_pipeline_<jobid>.out

# View errors
tail -f logs/slurm/full_pipeline_<jobid>.err
```

### Cancel a job
```bash
scancel <jobid>
```

## Examples

### Quick Test Run (Debug Mode)
```bash
sbatch --export=ALL,DEBUG_MODE=true,PRETRAIN_CONFIG=guac-430m,PRETRAIN_STEPS=100 \
    jobscripts/full_pipeline.sbatch
```

### Code Specialization Pipeline
```bash
sbatch --export=ALL,PRETRAIN_CONFIG=guac-4.5b,MIDTRAIN_TASK=code,POSTTRAIN_MODE=sft \
    jobscripts/full_pipeline.sbatch
```

### Reasoning Specialization with Large Vocab
```bash
sbatch --export=ALL,VOCAB_SIZE=100000,PRETRAIN_CONFIG=guac-3b,MIDTRAIN_TASK=reasoning \
    jobscripts/full_pipeline.sbatch
```

### Production Training (Longer Runtime)
```bash
sbatch --time=72:00:00 \
    --export=PRETRAIN_CONFIG=guac-3b,PRETRAIN_STEPS=50000 \
    jobscripts/full_pipeline.sbatch
```

## Troubleshooting

### Job Fails Immediately
- Check logs: `logs/slurm/full_pipeline_<jobid>.err`
- Verify Python environment is activated
- Check GPU availability: `nvidia-smi`

### Out of Memory
- Reduce batch size in training configs
- Use gradient accumulation
- Request more memory: `#SBATCH --mem=128G`
- Use smaller model config (guac-430m < guac-910m < guac-4.5b < guac-3b)

### Timeout
- Increase time limit: `#SBATCH --time=96:00:00`
- Reduce training steps: `PRETRAIN_STEPS=5000`
- Use checkpointing to resume jobs

### Cannot Find Tokenizer
- Verify tokenizer training completed successfully
- Check path: `ls scratch/tokenizers/sentencepiece/shared_60k/`
- Run tokenizer training manually first

## Advanced Usage

### Running Individual Stages

You can run stages independently by commenting out sections in the script or creating separate jobscripts for each stage.

### Checkpointing and Resuming

The scripts support checkpointing. To resume from a checkpoint, modify the training calls to include:
```bash
--resume-from-checkpoint path/to/checkpoint
```

### Custom Datasets

For custom datasets in mid-training:
```bash
sbatch --export=ALL,MIDTRAIN_TASK=custom,MIDTRAIN_DATASETS="/path/to/dataset1 /path/to/dataset2" \
    jobscripts/full_pipeline.sbatch
```

## Additional Resources

- Main documentation: `docs/`
- Model configs: `configs/models/`
- Training scripts: `guac/pipeline/`
- Makefile targets: `Makefile`
