# Guac Training Makefile
# Provides convenient commands for common training workflows

.PHONY: help debug train arithmetic eval-arithmetic clean cache test install pipeline pretrain midtrain posttrain \
	status clean-models clean-cache progressive multitask benchmark expert examples \
	pretrain-debug midtrain-debug posttrain-debug pipeline-debug pipeline-profiling \
	pipeline-code pipeline-reasoning pipeline-guac-3b pipeline-guac-1.9b pipeline-guac-1.1b pipeline-guac-590m \
	configs arch download-datasets download-datasets-from download-pretraining-datasets \
	download-instruction-datasets download-from-site-config download-help train-tokenizer \
	tokenize-datasets

# Default target
help:
	@echo "Guac Training Commands:"
	@echo ""
	@echo "=== Tokenizer & Dataset Preparation ==="
	@echo "  make train-tokenizer   - Train a new tokenizer on specified datasets"
	@echo "  make tokenize-datasets - Pre-tokenize and cache datasets for training"
	@echo ""
	@echo "=== Three-Stage Pipeline ==="
	@echo "  make pipeline   - Run complete pre→mid→post training pipeline"
	@echo "  make pretrain   - Run pre-training stage (foundation language modeling)"
	@echo "  make midtrain   - Run mid-training stage (task specialization)"
	@echo "  make posttrain  - Run post-training stage (instruction tuning/alignment)"
	@echo "  make pipeline-fresh - Force re-tokenization and run full pipeline"
	@echo ""
	@echo "=== Legacy Commands ==="
	@echo "  make debug      - Quick debug training with guac-3b model (fastest)"
	@echo "  make stable     - Stable training with guac-1.1b model (recommended)"
	@echo "  make train      - Standard training with mixed architecture"
	@echo "  make arithmetic - Train arithmetic-specific model (now mid-training task)"
	@echo "  make eval-arithmetic - Exact-match arithmetic accuracy (MODEL=dir, default: mid-trained model)"
	@echo ""
	@echo "=== Utilities ==="
	@echo "  make test       - Run all tests to verify setup"
	@echo "  make test-path  - Run path manager tests"
	@echo "  make download-datasets          - Download datasets from CSV configuration"
	@echo "  make download-pretraining-datasets - Download datasets from site config"
	@echo "  make clean      - Clean cached datasets and outputs"
	@echo "  make cache      - Show cached dataset information"
	@echo "  make install    - Install requirements and setup"
	@echo ""
	@echo "=== Pipeline Variables ==="
	@echo "  VOCAB_SIZE=60000 - Vocabulary size for tokenizer training"
	@echo "  TOKENIZER_PATH=  - Path to tokenizer for dataset tokenization"
	@echo "  MAX_LENGTH=1024  - Maximum sequence length for tokenization"
	@echo "  CONFIG=guac-3b      - Model config (guac-3b, guac-1.9b, guac-1.1b, guac-590m, guac-280m, etc.)"
	@echo "  TASK=arithmetic  - Mid-training task (arithmetic, code, reasoning, custom)"
	@echo "  MODE=instruction - Post-training mode (instruction, sft)"
	@echo "  DATASETS=fineweb - Datasets for training"
	@echo "  DEBUG=false      - Enable debug mode with smaller datasets"
	@echo "  FORCE_RETOKENIZE=false - Force re-tokenization even if cached"

# Configuration variables
CONFIG ?= guac-3b
DATASETS ?= fineweb
TASK ?= arithmetic
MODE ?= instruction
DEBUG ?= false
TEMPLATES ?= --no-templates
VOCAB_SIZE ?= 60000
MAX_LENGTH ?= 1024
TOKENIZER_PATH ?=
FORCE_RETOKENIZE ?= false

# Dataset download configuration
DOWNLOAD_CSV ?= datasets_download.csv
DOWNLOAD_OUTPUT ?= $(SCRATCH_DIR)/datasets/downloaded

# Pipeline directories - using path manager structure
SCRATCH_DIR ?= ./scratch
MODEL_DIR = $(SCRATCH_DIR)/models
PRETRAIN_DIR = $(MODEL_DIR)/pretraining/$(CONFIG)
MIDTRAIN_DIR = $(MODEL_DIR)/midtraining/$(CONFIG)_$(TASK)
POSTTRAIN_DIR = $(MODEL_DIR)/posttraining/$(CONFIG)_$(TASK)_$(MODE)

# Debug flags
DEBUG_FLAG = $(if $(filter true,$(DEBUG)),--debug,)
PROFILING_FLAG = $(if $(filter true,$(ENABLE_PROFILING)),--enable-profiling,)
RETOKENIZE_FLAG = $(if $(filter true,$(FORCE_RETOKENIZE)),--force-retokenize,)

# Installation and setup
install:
	pip install -r requirements.txt
	@if [ ! -f site_config.yml ]; then \
		cp site_config.example.yml site_config.yml; \
		echo "Created site_config.yml - please configure for your environment"; \
	fi

# ============================================================================
# TOKENIZER TRAINING
# ============================================================================

# Train a new tokenizer on specified datasets
train-tokenizer:
	@echo "🔤 Training tokenizer..."
	@echo "📊 Vocab size: $(VOCAB_SIZE) | Datasets: $(DATASETS)"
	TOKENIZERS_PARALLELISM=false python scripts/training/pipeline/train_tokenizer.py \
		--vocab-size $(VOCAB_SIZE) \
		--datasets $(DATASETS)
	@echo "✅ Tokenizer training complete"

# Pre-tokenize and cache datasets for training
tokenize-datasets:
	@echo "📦 Pre-tokenizing datasets..."
	@echo "📊 Tokenizer: $(TOKENIZER_PATH) | Datasets: $(DATASETS) | Max length: $(MAX_LENGTH)"
	@if [ -z "$(TOKENIZER_PATH)" ]; then \
		echo "❌ TOKENIZER_PATH not set!"; \
		echo "Please specify tokenizer path:"; \
		echo "  make tokenize-datasets TOKENIZER_PATH=scratch/tokenizers/sentencepiece/shared_60k"; \
		exit 1; \
	fi
	TOKENIZERS_PARALLELISM=false python scripts/training/pipeline/tokenize_datasets.py \
		--tokenizer $(TOKENIZER_PATH) \
		--datasets $(DATASETS) \
		--max-length $(MAX_LENGTH) \
		$(RETOKENIZE_FLAG)
	@echo "✅ Dataset tokenization complete"

# ============================================================================
# THREE-STAGE TRAINING PIPELINE
# ============================================================================

# Complete pipeline: pre-training → mid-training → post-training
pipeline: pretrain midtrain posttrain
	@echo ""
	@echo "🎉 Complete training pipeline finished!"
	@echo "📊 Pre-trained model:  $(PRETRAIN_DIR)"
	@echo "🎯 Mid-trained model:  $(MIDTRAIN_DIR)"
	@echo "🤖 Final model:        $(POSTTRAIN_DIR)"

# Stage 1: Pre-training (foundation language modeling)
pretrain:
	@echo "🚀 Starting pre-training stage..."
	@echo "📊 Config: $(CONFIG) | Datasets: $(DATASETS)"
	@mkdir -p $(MODEL_DIR)
	TOKENIZERS_PARALLELISM=false python scripts/training/pipeline/pretrain.py \
		--model-config $(CONFIG) \
		--datasets $(DATASETS) \
		--output-dir $(PRETRAIN_DIR) \
		$(DEBUG_FLAG) $(PROFILING_FLAG) $(RETOKENIZE_FLAG)
	@echo "✅ Pre-training complete: $(PRETRAIN_DIR)"

# Stage 2: Mid-training (task specialization)
midtrain:
	@echo "🎯 Starting mid-training stage..."
	@echo "📊 Task: $(TASK) | Base: $(PRETRAIN_DIR)"
	@if [ ! -d "$(PRETRAIN_DIR)" ]; then \
		echo "❌ Pre-trained model not found: $(PRETRAIN_DIR)"; \
		echo "Run 'make pretrain' first or check CONFIG=$(CONFIG)"; \
		exit 1; \
	fi
	TOKENIZERS_PARALLELISM=false python scripts/training/pipeline/midtrain.py \
		--base-model $(PRETRAIN_DIR) \
		--task $(TASK) \
		--output-dir $(MIDTRAIN_DIR) \
		$(DEBUG_FLAG) \
		$(PROFILING_FLAG) \
		$(RETOKENIZE_FLAG)
	@echo "✅ Mid-training complete: $(MIDTRAIN_DIR)"

# Stage 3: Post-training (instruction tuning/alignment)
posttrain:
	@echo "🤖 Starting post-training stage..."
	@echo "📊 Mode: $(MODE) | Base: $(MIDTRAIN_DIR)"
	@if [ ! -d "$(MIDTRAIN_DIR)" ]; then \
		echo "❌ Mid-trained model not found: $(MIDTRAIN_DIR)"; \
		echo "Run 'make midtrain' first or check TASK=$(TASK) CONFIG=$(CONFIG)"; \
		exit 1; \
	fi
	TOKENIZERS_PARALLELISM=false python scripts/training/pipeline/posttrain.py \
		--base-model $(MIDTRAIN_DIR) \
		--mode $(MODE) \
		--output-dir $(POSTTRAIN_DIR) \
		$(DEBUG_FLAG) \
		$(PROFILING_FLAG) \
		$(RETOKENIZE_FLAG)
	@echo "✅ Post-training complete: $(POSTTRAIN_DIR)"

# Individual stage variants with custom settings
pretrain-debug:
	$(MAKE) pretrain DEBUG=true

midtrain-debug:
	$(MAKE) midtrain DEBUG=true

posttrain-debug:
	$(MAKE) posttrain DEBUG=true

# Pipeline variants
pipeline-debug:
	$(MAKE) pipeline DEBUG=true

pipeline-profiling:
	$(MAKE) pipeline ENABLE_PROFILING=true

pipeline-fresh:
	$(MAKE) pipeline FORCE_RETOKENIZE=true

# Pipeline with different tasks
pipeline-code:
	$(MAKE) pipeline TASK=code

pipeline-reasoning:
	$(MAKE) pipeline TASK=reasoning

# Different model sizes
pipeline-guac-3b:
	$(MAKE) pipeline CONFIG=guac-3b

pipeline-guac-1.9b:
	$(MAKE) pipeline CONFIG=guac-1.9b

pipeline-guac-1.1b:
	$(MAKE) pipeline CONFIG=guac-1.1b

pipeline-guac-590m:
	$(MAKE) pipeline CONFIG=guac-590m

# ============================================================================
# LEGACY COMMANDS (Updated to use new scripts)
# ============================================================================

# Quick debug training (minimal model, fast iteration)
debug:
	@echo "🐛 Quick debug mode - using guac-3b model with small dataset"
	$(MAKE) pretrain CONFIG=guac-3b DEBUG=true

# Standard training with mixed architecture (now uses new pipeline)
train:
	@echo "🔧 Standard training - using $(CONFIG) model"
	$(MAKE) pretrain CONFIG=$(CONFIG) DATASETS=$(DATASETS)

# Stable training with conservative settings (now uses new pipeline)
stable:
	@echo "📊 Stable training - using guac-590m model"
	$(MAKE) pretrain CONFIG=guac-590m DATASETS=$(DATASETS)

# Arithmetic-specific training (now mid-training task)
arithmetic:
	@echo "🧮 Arithmetic training - now part of mid-training pipeline"
	@echo "💡 Recommendation: Run 'make pipeline TASK=arithmetic' for full pipeline"
	@if [ -d "$(PRETRAIN_DIR)" ]; then \
		$(MAKE) midtrain TASK=arithmetic; \
	else \
		echo "❌ No pre-trained model found. Running full pipeline..."; \
		$(MAKE) pipeline TASK=arithmetic; \
	fi

# Evaluate arithmetic accuracy (defaults to the mid-trained model for CONFIG/TASK)
EVAL_N ?= 50
eval-arithmetic:
	python scripts/training/pipeline/eval_arithmetic.py \
		--model $(or $(MODEL),$(MIDTRAIN_DIR)) \
		--num-per-category $(EVAL_N)

# Test installation and basic functionality
test:
	@echo "🧪 Running Guac Test Suite..."
	python tests/run_tests.py

# Test path manager specifically
test-path:
	@echo "🧪 Testing Path Manager..."
	python tests/run_tests.py --test path_manager

# Legacy installation test
test-install:
	@echo "🧪 Testing Guac installation..."
	@python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch not installed"
	@python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || echo "❌ Transformers not installed"
	@python -c "from scripts.training.core.custom_architectures import MixedLayerForCausalLM; print('✅ Guac architectures')" || echo "❌ Guac imports failed"
	@echo "🏃 Running minimal pipeline test..."
	$(MAKE) pretrain-debug CONFIG=guac-3b
	@echo "✅ Installation test complete!"

# Show pipeline status and available models
status:
	@echo "📊 Guac Training Pipeline Status"
	@echo "================================="
	@echo "Current configuration: CONFIG=$(CONFIG) TASK=$(TASK) MODE=$(MODE)"
	@echo ""
	@echo "📁 Model directories:"
	@echo "  Pre-train:  $(PRETRAIN_DIR)"
	@echo "  Mid-train:  $(MIDTRAIN_DIR)"
	@echo "  Post-train: $(POSTTRAIN_DIR)"
	@echo ""
	@echo "🗂️  Available models:"
	@ls -la $(MODEL_DIR)/ 2>/dev/null || echo "  No models found in $(MODEL_DIR)/"
	@echo ""
	@echo "💡 Quick commands:"
	@echo "  make pipeline         - Full 3-stage pipeline"
	@echo "  make pipeline-debug   - Debug mode pipeline"
	@echo "  make clean            - Clean all outputs"

# Clean cached datasets and outputs
clean:
	@echo "🧹 Cleaning Guac training outputs..."
	@echo "  Removing cached datasets..."
	@rm -rf ./tokenized_* ./tokenized_cache_* $(SCRATCH_DIR)/tokenized_datasets 2>/dev/null || true
	@echo "  Removing model outputs..."
	@rm -rf $(MODEL_DIR) 2>/dev/null || true
	@echo "  Removing legacy outputs..."
	@rm -rf ./output ./logs ./experiments 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Clean only models but keep datasets
clean-models:
	@echo "🧹 Cleaning model outputs only..."
	@rm -rf $(MODEL_DIR) 2>/dev/null || true
	@echo "✅ Model cleanup complete!"

# Clean only cached datasets
clean-cache:
	@echo "🧹 Cleaning cached datasets only..."
	@rm -rf ./tokenized_* ./tokenized_cache_* $(SCRATCH_DIR)/tokenized_datasets 2>/dev/null || true
	@echo "✅ Cache cleanup complete!"

# Show cached dataset information
cache:
	@echo "📦 Cached datasets:"
	@ls -la ./tokenized_* ./tokenized_cache_* 2>/dev/null || echo "  No cached datasets found"
	@echo ""
	@echo "💡 To force re-tokenization, use FORCE_RETOKENIZE=true"
	@echo "💡 Example: make pipeline FORCE_RETOKENIZE=true"

# ============================================================================
# ADVANCED PIPELINE COMMANDS
# ============================================================================

# Progressive training: guac-3b → guac-1.9b → guac-1.1b
progressive:
	@echo "📈 Progressive training pipeline..."
	$(MAKE) pipeline CONFIG=guac-3b
	$(MAKE) pipeline CONFIG=guac-1.9b
	$(MAKE) pipeline CONFIG=guac-1.1b
	@echo "✅ Progressive training complete!"

# Multi-task training: Train all tasks
multitask:
	@echo "🎯 Multi-task training pipeline..."
	$(MAKE) pretrain CONFIG=$(CONFIG)
	$(MAKE) midtrain TASK=arithmetic
	$(MAKE) midtrain TASK=code
	$(MAKE) midtrain TASK=reasoning
	@echo "✅ Multi-task training complete!"

# Benchmark training: Train popular configs
benchmark:
	@echo "🏆 Benchmark training..."
	$(MAKE) pipeline CONFIG=guac-3b TASK=arithmetic
	$(MAKE) pipeline CONFIG=guac-1.1b TASK=code
	$(MAKE) pipeline CONFIG=guac-590m TASK=reasoning
	@echo "✅ Benchmark training complete!"

# Expert training: Large model with all stages
expert:
	@echo "🎓 Expert model training..."
	$(MAKE) pipeline CONFIG=guac-9b TASK=reasoning MODE=instruction
	@echo "✅ Expert training complete!"

# ============================================================================
# QUICK REFERENCE AND EXAMPLES
# ============================================================================

# Show example commands
examples:
	@echo "🚀 Guac Training Examples"
	@echo "========================="
	@echo ""
	@echo "🔥 Quick Start:"
	@echo "  make train-tokenizer         # Train tokenizer first"
	@echo "  make pipeline-debug          # Debug mode pipeline"
	@echo "  make pipeline-fresh          # Force re-tokenization pipeline"
	@echo "  make status                  # Check current status"
	@echo ""
	@echo "🔤 Tokenizer Training:"
	@echo "  make train-tokenizer                           # Default (60K vocab, fineweb)"
	@echo "  make train-tokenizer VOCAB_SIZE=32000         # Custom vocab size"
	@echo "  make train-tokenizer DATASETS=wikitext-103    # Different dataset"
	@echo ""
	@echo "📦 Dataset Tokenization:"
	@echo "  make tokenize-datasets TOKENIZER_PATH=scratch/tokenizers/sentencepiece/shared_60k"
	@echo "  make tokenize-datasets TOKENIZER_PATH=./my_tokenizer DATASETS=wikitext-103"
	@echo "  make tokenize-datasets TOKENIZER_PATH=./my_tokenizer MAX_LENGTH=2048"
	@echo ""
	@echo "📊 Standard Training:"
	@echo "  make pipeline                # Default: guac-3b + arithmetic + instruction"
	@echo "  make pipeline CONFIG=guac-1.1b   # Femto model pipeline"
	@echo "  make pipeline TASK=code      # Code specialization"
	@echo "  make pipeline FORCE_RETOKENIZE=true  # Force re-tokenization"
	@echo ""
	@echo "🎯 Advanced Pipelines:"
	@echo "  make pipeline CONFIG=guac-590m TASK=reasoning MODE=sft"
	@echo "  make multitask CONFIG=guac-1.1b  # Train all tasks"
	@echo "  make progressive             # Train guac-3b→guac-1.9b→guac-1.1b"
	@echo ""
	@echo "🛠️  Utilities:"
	@echo "  make test                    # Test installation"
	@echo "  make clean                   # Clean all outputs"
	@echo "  make configs                 # Show available configs"

# Show model configurations
configs:
	@echo "📋 Available model configurations:"
	@echo "=================================="
	@grep -A 1 "^  [a-z]" model_configs.yaml | grep -v "^--" | head -20 || echo "Could not read model_configs.yaml"
	@echo ""
	@echo "💡 Use any config name with: make pipeline CONFIG=<name>"

# Show architecture information
arch:
	@echo "🏗️  Guac Architecture Information:"
	@echo "=================================="
	@echo "🧠 Available layer types:"
	@echo "  - mamba2: Mamba-2 state space model"
	@echo "  - conv: 1D convolution layer"
	@echo "  - based: BASED linear attention"
	@echo "  - attention: Standard multi-head attention"
	@echo "  - moh_attention: Mixture-of-Head attention"
	@echo "  - shared_expert_moe: Mixture-of-Experts with shared experts"
	@echo ""
	@echo "🔧 Modern features enabled by default:"
	@echo "  - RMSNorm (faster than LayerNorm)"
	@echo "  - Flash Attention 2 (memory efficient)"
	@echo "  - Mixed precision training (FP16)"
	@echo "  - Gradient checkpointing (for large models)"

# ============================================================================
# DATASET DOWNLOAD TARGETS
# ============================================================================

# Download datasets from CSV configuration
download-datasets:
	@echo "📥 Downloading datasets from $(DOWNLOAD_CSV)..."
	@if [ ! -f "$(DOWNLOAD_CSV)" ]; then \
		echo "❌ CSV file '$(DOWNLOAD_CSV)' not found!"; \
		echo "Please create it with format: repo,size,split,config,format"; \
		echo "Example:"; \
		echo "  HuggingFaceFW/fineweb,100MB,train,,jsonl"; \
		echo "  HuggingFaceFW/fineweb-edu,100MB,train,,jsonl"; \
		exit 1; \
	fi
	@mkdir -p $(DOWNLOAD_OUTPUT)
	@echo "Output directory: $(DOWNLOAD_OUTPUT)"
	@echo ""
	@failed=0; \
	line_num=0; \
	while IFS=, read -r repo size split config format; do \
		line_num=$$((line_num + 1)); \
		echo "Processing line $$line_num: $$repo,$$size,$$split,$$config,$$format"; \
		if [ -z "$$repo" ] || [ "$${repo#\#}" != "$$repo" ]; then \
			echo "⏭️  Skipping comment/empty line"; \
			continue; \
		fi; \
		if [ -z "$$size" ]; then \
			echo "⚠️  Warning: Empty size for $$repo, skipping"; \
			continue; \
		fi; \
		echo "📡 Downloading $$repo ($$size)..."; \
		cmd="python hf_partial_download.py \"$$repo\" --size \"$$size\" --output \"$(DOWNLOAD_OUTPUT)\""; \
		if [ -n "$$split" ]; then \
			cmd="$$cmd --split \"$$split\""; \
		fi; \
		if [ -n "$$config" ]; then \
			cmd="$$cmd --config \"$$config\""; \
		fi; \
		if [ -n "$$format" ]; then \
			cmd="$$cmd --format \"$$format\""; \
		fi; \
		echo "Running: $$cmd"; \
		eval $$cmd; \
		exit_code=$$?; \
		if [ $$exit_code -eq 0 ]; then \
			echo "✅ Successfully downloaded $$repo"; \
		else \
			echo "❌ Failed to download $$repo (exit code: $$exit_code)"; \
			failed=$$((failed + 1)); \
		fi; \
		echo ""; \
	done < $(DOWNLOAD_CSV); \
	if [ $$failed -eq 0 ]; then \
		echo "🎉 All datasets downloaded successfully!"; \
		echo "📁 Downloaded files are in: $(DOWNLOAD_OUTPUT)"; \
	else \
		echo "⚠️  $$failed dataset(s) failed to download"; \
		echo "📁 Any successful downloads are in: $(DOWNLOAD_OUTPUT)"; \
		echo "💡 Check logs above for specific error details"; \
		exit 1; \
	fi

# Download datasets with custom CSV file
download-datasets-from:
	@if [ -z "$(CSV)" ]; then \
		echo "❌ Please specify CSV file: make download-datasets-from CSV=your_file.csv"; \
		exit 1; \
	fi
	$(MAKE) download-datasets DOWNLOAD_CSV=$(CSV)

# Site config dataset download targets
SITE_CONFIG_PATH ?= site_config.yml
DATASET_TYPE ?= pretraining_datasets

# Download all default pretraining datasets from site config
download-pretraining-datasets:
	@echo "📥 Downloading default pretraining datasets from site config..."
	@if [ ! -f "$(SITE_CONFIG_PATH)" ]; then \
		echo "❌ Site config file not found: $(SITE_CONFIG_PATH)"; \
		exit 1; \
	fi
	@failed=0; \
	for dataset in fineweb fineweb-edu wikitext-103; do \
		echo "⬇️  Downloading $$dataset..."; \
		if python hf_partial_download.py --from-site-config $$dataset --output "$(DOWNLOAD_OUTPUT)" --site-config-path "$(SITE_CONFIG_PATH)"; then \
			echo "✅ Successfully downloaded $$dataset"; \
		else \
			echo "❌ Failed to download $$dataset"; \
			failed=$$((failed + 1)); \
		fi; \
		echo ""; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "🎉 All pretraining datasets downloaded successfully!"; \
		echo "📁 Downloaded to: $(DOWNLOAD_OUTPUT)"; \
	else \
		echo "⚠️  $$failed dataset(s) failed to download"; \
	fi

# Download specific dataset from site config
download-from-site-config:
	@if [ -z "$(DATASET)" ]; then \
		echo "❌ Please specify dataset: make download-from-site-config DATASET=fineweb"; \
		echo "Available pretraining datasets: fineweb, fineweb-edu, finewiki, wikitext-103, c4, openwebtext"; \
		echo "Available instruction datasets: openorca, alpaca, ultrachat"; \
		exit 1; \
	fi
	@echo "📥 Downloading $(DATASET) from site config..."
	python hf_partial_download.py --from-site-config $(DATASET) \
		--output "$(DOWNLOAD_OUTPUT)" \
		--site-config-path "$(SITE_CONFIG_PATH)" \
		--dataset-type $(DATASET_TYPE)

# Download instruction datasets from site config
download-instruction-datasets:
	@echo "📥 Downloading default instruction datasets from site config..."
	@if [ ! -f "$(SITE_CONFIG_PATH)" ]; then \
		echo "❌ Site config file not found: $(SITE_CONFIG_PATH)"; \
		exit 1; \
	fi
	@failed=0; \
	for dataset in openorca; do \
		echo "⬇️  Downloading $$dataset..."; \
		if python hf_partial_download.py --from-site-config $$dataset --dataset-type instruction_datasets --output "$(DOWNLOAD_OUTPUT)" --site-config-path "$(SITE_CONFIG_PATH)"; then \
			echo "✅ Successfully downloaded $$dataset"; \
		else \
			echo "❌ Failed to download $$dataset"; \
			failed=$$((failed + 1)); \
		fi; \
		echo ""; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "🎉 All instruction datasets downloaded successfully!"; \
		echo "📁 Downloaded to: $(DOWNLOAD_OUTPUT)"; \
	else \
		echo "⚠️  $$failed dataset(s) failed to download"; \
	fi

# Show available dataset download commands
download-help:
	@echo "📥 Dataset Download Commands"
	@echo "============================"
	@echo ""
	@echo "🎯 Main commands:"
	@echo "  make download-datasets              - Download from datasets_download.csv"
	@echo "  make download-datasets-from CSV=... - Download from custom CSV file"
	@echo ""
	@echo "🏗️  Site config commands:"
	@echo "  make download-pretraining-datasets  - Download default pretraining datasets"
	@echo "  make download-instruction-datasets  - Download default instruction datasets"
	@echo "  make download-from-site-config      - Download specific dataset from site config"
	@echo ""
	@echo "📝 Site config examples:"
	@echo "  make download-from-site-config DATASET=fineweb"
	@echo "  make download-from-site-config DATASET=openorca DATASET_TYPE=instruction_datasets"
	@echo ""
	@echo "📋 CSV Format:"
	@echo "  repo,size,split,config,format"
	@echo ""
	@echo "📝 CSV Examples:"
	@echo "  # Download 100MB of FineWeb"
	@echo "  HuggingFaceFW/fineweb,100MB,train,,jsonl"
	@echo ""
	@echo "  # Download 50K samples from specific config"
	@echo "  mozilla-foundation/common_voice_16_0,50K,train,en,parquet"
	@echo ""
	@echo "  # Download 10% of dataset"
	@echo "  username/dataset-name,10%,validation,,jsonl"
	@echo ""
	@echo "🔧 Variables:"
	@echo "  DOWNLOAD_CSV=file.csv     - CSV configuration file (default: datasets_download.csv)"
	@echo "  DOWNLOAD_OUTPUT=path      - Output directory (default: $(DOWNLOAD_OUTPUT))"
	@echo "  SITE_CONFIG_PATH=file.yml - Site config file (default: site_config.yml)"
	@echo "  DATASET=name              - Dataset name for download-from-site-config"
	@echo "  DATASET_TYPE=type         - Dataset type (pretraining_datasets|instruction_datasets)"
	@echo ""
	@echo "💡 Size formats supported:"
	@echo "  - Sample counts: 1000, 50K, 1M, 1B"
	@echo "  - Byte sizes: 100MB, 1GB, 500KB"
	@echo "  - Percentages: 10%, 25%, 50%"