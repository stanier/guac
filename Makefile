# Guac Training Makefile
# Provides convenient commands for common training workflows

.PHONY: help debug train arithmetic eval-arithmetic clean cache test install pipeline pretrain midtrain posttrain \
	status clean-models clean-cache multitask examples \
	pretrain-debug midtrain-debug posttrain-debug pipeline-debug pipeline-profiling \
	pipeline-code pipeline-reasoning pipeline-guac-3b pipeline-guac-1.7b pipeline-guac-910m pipeline-guac-430m \
	configs arch download-datasets download-datasets-from download-pretraining-datasets \
	download-instruction-datasets download-from-site-config download-help train-tokenizer \
	tokenize-datasets diagnostics sweep

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
	@echo "  make stable     - Stable training with guac-430m model (recommended)"
	@echo "  make train      - Standard training with mixed architecture"
	@echo "  make arithmetic - Train arithmetic-specific model (now mid-training task)"
	@echo "  make eval-arithmetic - Exact-match arithmetic accuracy (MODEL=dir, default: mid-trained model)"
	@echo "  make diagnostics DIAG_TASK=mqar LAYERS=conv,based - Train a small model on a synthetic diagnostic"
	@echo "  make sweep SPEC=sweeps/recall_small.yaml - Layer-type sweep at a matched budget, one results table"
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
	@echo "  TOKENIZER_TYPE=sentencepiece - Tokenizer to train (sentencepiece, bpe)"
	@echo "  TOKENIZER_PATH=  - Tokenizer for tokenize-datasets and pretrain (default: CONFIG's)"
	@echo "  MAX_LENGTH=      - Sequence length for tokenize-datasets and pretrain (default: CONFIG's)"
	@echo "  CONFIG=guac-3b      - Model config (guac-3b, guac-1.7b, guac-910m, guac-430m, guac-160m, etc.)"
	@echo "  TASK=arithmetic  - Mid-training task (arithmetic, code, reasoning, custom)"
	@echo "  MODE=instruction - Post-training mode (instruction, sft)"
	@echo "  DATASETS=fineweb - Datasets for training"
	@echo "  DEBUG=false      - Enable debug mode with smaller datasets"
	@echo "  FORCE_RETOKENIZE=false - Force re-tokenization even if cached"
	@echo "  PRETRAIN_ARGS=   - Extra pretrain.py flags, e.g. \"--batch-size 8 --grad-accum 4 --max-length 512\""
	@echo "  MIDTRAIN_ARGS=   - Extra midtrain.py flags, e.g. \"--learning-rate 1e-3 --reverse-answers\""
	@echo "  MAX_STEPS=       - Pretrain step budget (default: config's max_steps, else its epochs; 20 with DEBUG=true)"

# Configuration variables
CONFIG ?= guac-3b
DATASETS ?= fineweb
TASK ?= arithmetic
MODE ?= instruction
DEBUG ?= false
TEMPLATES ?= --no-templates
VOCAB_SIZE ?= 60000
TOKENIZER_TYPE ?= sentencepiece
MAX_LENGTH ?=
TOKENIZER_PATH ?=
FORCE_RETOKENIZE ?= false
PRETRAIN_ARGS ?=
MIDTRAIN_ARGS ?=
MAX_STEPS ?=

# Dataset download configuration
DOWNLOAD_CSV ?= datasets_download.csv
DOWNLOAD_OUTPUT ?= $(SCRATCH_DIR)/datasets/downloaded

# Pipeline directories - site_config.yml's scratch_dir, where the scripts write by default
SCRATCH_DIR ?= $(shell python -c "from guac.core.path_manager import get_scratch_dir; print(get_scratch_dir())" 2>/dev/null || echo ./scratch)
MODEL_DIR = $(SCRATCH_DIR)/models
PRETRAIN_DIR = $(MODEL_DIR)/pretraining/$(CONFIG)
MIDTRAIN_DIR = $(MODEL_DIR)/midtraining/$(CONFIG)_$(TASK)
POSTTRAIN_DIR = $(MODEL_DIR)/posttraining/$(CONFIG)_$(TASK)_$(MODE)

# Debug flags
DEBUG_FLAG = $(if $(filter true,$(DEBUG)),--debug,)
PROFILING_FLAG = $(if $(filter true,$(ENABLE_PROFILING)),--enable-profiling,)
RETOKENIZE_FLAG = $(if $(filter true,$(FORCE_RETOKENIZE)),--force-retokenize,)
MAX_STEPS_FLAG = $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),)
# The same tokenizer and length for tokenize-datasets and pretrain, so pretrain finds the cache
DATA_FLAGS = $(if $(TOKENIZER_PATH),--tokenizer $(TOKENIZER_PATH),) $(if $(MAX_LENGTH),--max-length $(MAX_LENGTH),)

# Installation and setup
install:
	pip install -e ".[test]"
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
	@echo "📊 Vocab size: $(VOCAB_SIZE) | Type: $(TOKENIZER_TYPE) | Datasets: $(DATASETS)"
	TOKENIZERS_PARALLELISM=false python guac/pipeline/train_tokenizer.py \
		--vocab-size $(VOCAB_SIZE) \
		--tokenizer-type $(TOKENIZER_TYPE) \
		--datasets $(DATASETS)
	@echo "✅ Tokenizer training complete"

# Pre-tokenize and cache datasets for training
tokenize-datasets:
	@echo "📦 Pre-tokenizing datasets..."
	@echo "📊 Config: $(CONFIG) | Datasets: $(DATASETS)"
	TOKENIZERS_PARALLELISM=false python guac/pipeline/tokenize_datasets.py \
		--model-config $(CONFIG) \
		--datasets $(DATASETS) \
		$(DATA_FLAGS) $(DEBUG_FLAG) $(RETOKENIZE_FLAG)
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
	TOKENIZERS_PARALLELISM=false python guac/pipeline/pretrain.py \
		--model-config $(CONFIG) \
		--datasets $(DATASETS) \
		--output-dir $(PRETRAIN_DIR) \
		$(DATA_FLAGS) $(DEBUG_FLAG) $(PROFILING_FLAG) $(RETOKENIZE_FLAG) $(MAX_STEPS_FLAG) $(PRETRAIN_ARGS)
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
	TOKENIZERS_PARALLELISM=false python guac/pipeline/midtrain.py \
		--base-model $(PRETRAIN_DIR) \
		--task $(TASK) \
		--output-dir $(MIDTRAIN_DIR) \
		$(DEBUG_FLAG) \
		$(PROFILING_FLAG) \
		$(RETOKENIZE_FLAG) \
		$(MIDTRAIN_ARGS)
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
	TOKENIZERS_PARALLELISM=false python guac/pipeline/posttrain.py \
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

pipeline-guac-1.7b:
	$(MAKE) pipeline CONFIG=guac-1.7b

pipeline-guac-910m:
	$(MAKE) pipeline CONFIG=guac-910m

pipeline-guac-430m:
	$(MAKE) pipeline CONFIG=guac-430m

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
	@echo "📊 Stable training - using guac-430m model"
	$(MAKE) pretrain CONFIG=guac-430m DATASETS=$(DATASETS)

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
	python guac/pipeline/eval_arithmetic.py \
		--model $(or $(MODEL),$(MIDTRAIN_DIR)) \
		--num-per-category $(EVAL_N)

# Synthetic diagnostics (MQAR, selective copying, induction, passkey) on a small model trained from scratch
DIAG_TASK ?= mqar
LAYERS ?= full_attention,full_attention
DIAG_ARGS ?=
diagnostics:
	python guac/pipeline/run_diagnostics.py --task $(DIAG_TASK) --layer-types $(LAYERS) $(DIAG_ARGS)

# Layer-type sweep from a spec (see sweeps/README.md)
SPEC ?= sweeps/recall_small.yaml
sweep:
	python guac/pipeline/sweep.py $(SPEC) $(SWEEP_ARGS)

# Test installation and basic functionality
test:
	@echo "🧪 Running Guac Test Suite..."
	python -m pytest

# Test path manager specifically
test-path:
	@echo "🧪 Testing Path Manager..."
	python -m pytest tests/test_path_manager.py

# Legacy installation test
test-install:
	@echo "🧪 Testing Guac installation..."
	@python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch not installed"
	@python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || echo "❌ Transformers not installed"
	@python -c "from guac.core.custom_architectures import MixedLayerForCausalLM; print('✅ Guac architectures')" || echo "❌ Guac imports failed"
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
	@ls -d $(SCRATCH_DIR)/tokenized_datasets/*/tokenized_cache_* 2>/dev/null | while read dir; do \
		echo "  $$(du -sh "$$dir" | cut -f1)	$$dir"; done | grep . || echo "  No cached datasets found"
	@echo ""
	@echo "💡 To force re-tokenization, use FORCE_RETOKENIZE=true"
	@echo "💡 Example: make pipeline FORCE_RETOKENIZE=true"

# ============================================================================
# ADVANCED PIPELINE COMMANDS
# ============================================================================

# Multi-task training: Train all tasks
multitask:
	@echo "🎯 Multi-task training pipeline..."
	$(MAKE) pretrain CONFIG=$(CONFIG)
	$(MAKE) midtrain TASK=arithmetic
	$(MAKE) midtrain TASK=code
	$(MAKE) midtrain TASK=reasoning
	@echo "✅ Multi-task training complete!"

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
	@echo "  make tokenize-datasets CONFIG=guac-430m DATASETS=wikitext-103   # CONFIG's tokenizer and length"
	@echo "  make tokenize-datasets TOKENIZER_PATH=./my_tokenizer MAX_LENGTH=2048"
	@echo "  (pass the same TOKENIZER_PATH / MAX_LENGTH to make pretrain so it finds the cache)"
	@echo ""
	@echo "📊 Standard Training:"
	@echo "  make pipeline                # Default: guac-3b + arithmetic + instruction"
	@echo "  make pipeline CONFIG=guac-910m   # Femto model pipeline"
	@echo "  make pipeline TASK=code      # Code specialization"
	@echo "  make pipeline FORCE_RETOKENIZE=true  # Force re-tokenization"
	@echo ""
	@echo "🎯 Advanced Pipelines:"
	@echo "  make pipeline CONFIG=guac-430m TASK=reasoning MODE=sft"
	@echo "  make multitask CONFIG=guac-910m  # Train all tasks"
	@echo ""
	@echo "🛠️  Utilities:"
	@echo "  make test                    # Test installation"
	@echo "  make clean                   # Clean all outputs"
	@echo "  make configs                 # Show available configs"

# Show model configurations
configs:
	@echo "📋 Available model configurations:"
	@echo "=================================="
	@python -c "import yaml; c = yaml.safe_load(open('model_configs.yaml'))['model_training_configs']; \
		[print(f\"  {n:16} {m.get('hidden_size', '?'):>5} hidden, {m.get('num_hidden_layers', '?'):>3} layers\") for n, m in c.items()]" \
		|| echo "Could not read model_configs.yaml"
	@echo ""
	@echo "💡 Use any config name with: make pipeline CONFIG=<name>"

# Show architecture information
arch:
	@echo "🏗️  Guac Architecture Information:"
	@echo "=================================="
	@echo "🧠 Available layer types (mixers; each block also has an FFN, see docs/ARCHITECTURE.md):"
	@echo "  - full_attention / flash_attention: grouped-query softmax attention with RoPE"
	@echo "  - sliding_window_attention: softmax attention over the last sliding_window_size positions"
	@echo "  - moh_attention: Mixture-of-Head attention"
	@echo "  - conv: causal 1D convolution"
	@echo "  - mamba2: Mamba-2 selective state space (chunked SSD scan)"
	@echo "  - based: BASED linear attention (Taylor feature map)"
	@echo "  - mlp_only, shared_expert_moe: no mixer (FFN or MoE only)"
	@echo ""
	@echo "🔧 Defaults: pre-norm RMSNorm blocks, SwiGLU FFNs, tied embeddings, FP16 on CUDA,"
	@echo "   gradient checkpointing, cached generation"

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
		cmd="python scripts/hf_partial_download.py \"$$repo\" --size \"$$size\" --output \"$(DOWNLOAD_OUTPUT)\""; \
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
	for dataset in $$(python -c "from guac.core.path_manager import PathManager; print(' '.join(PathManager('$(SITE_CONFIG_PATH)').get_dataset_configs().get('pretraining_datasets', {})))"); do \
		echo "⬇️  Downloading $$dataset..."; \
		if python scripts/hf_partial_download.py --from-site-config $$dataset --output "$(DOWNLOAD_OUTPUT)" --site-config-path "$(SITE_CONFIG_PATH)"; then \
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
		exit 1; \
	fi

# Download specific dataset from site config
download-from-site-config:
	@if [ -z "$(DATASET)" ]; then \
		echo "❌ Please specify dataset: make download-from-site-config DATASET=fineweb"; \
		python -c "from guac.core.path_manager import PathManager; \
			[print(f'Available {kind}: ' + ', '.join(names)) for kind, names in PathManager('$(SITE_CONFIG_PATH)').get_dataset_configs().items()]"; \
		exit 1; \
	fi
	@echo "📥 Downloading $(DATASET) from site config..."
	python scripts/hf_partial_download.py --from-site-config $(DATASET) \
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
	for dataset in $$(python -c "from guac.core.path_manager import PathManager; print(' '.join(PathManager('$(SITE_CONFIG_PATH)').get_dataset_configs().get('instruction_datasets', {})))"); do \
		echo "⬇️  Downloading $$dataset..."; \
		if python scripts/hf_partial_download.py --from-site-config $$dataset --dataset-type instruction_datasets --output "$(DOWNLOAD_OUTPUT)" --site-config-path "$(SITE_CONFIG_PATH)"; then \
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
		exit 1; \
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