import os, sys, yaml, re, torch
from typing import List, Tuple
from torch import nn
from tqdm.auto import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from transformers import (AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                          PreTrainedTokenizerFast, PrinterCallback, EarlyStoppingCallback)
from tokenizers import Tokenizer, trainers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from datasets import load_from_disk, concatenate_datasets, DatasetDict
try:
    from ..callbacks import CustomProgressCallback
except ImportError:
    from scripts.training.callbacks import CustomProgressCallback


def create_character_math_vocab():
    """Creates a character-level math tokenizer vocabulary."""
    # Special tokens
    special_tokens = ["<**inseq**>", "</**inseq**>", "<unk>", "<pad>"]
    
    # Individual digits (0-9)
    digits = [str(i) for i in range(10)]
    
    # Mathematical operators
    operators = ["+", "-", "*", "/", "//", "%", "="]
    
    # Grouping and formatting
    formatting = ["(", ")", " ", "."]
    
    # Common letters (in case there are any in the dataset)
    letters = []
    # letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    tokens = special_tokens + digits + operators + formatting + letters
    return {token: idx for idx, token in enumerate(tokens)}

def train_character_tokenizer(dataset):
    """Creates a character-level tokenizer for math expressions."""
    vocab = create_character_math_vocab()
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer.normalizer = None
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<**inseq**>",
                                   eos_token="</**inseq**>", unk_token="<unk>", pad_token="<pad>")


def tokenize_dataset(tokenizer, dataset):
    def tokenize_function(examples):
        texts = [f"<**inseq**>{text}</**inseq**>" for text in examples["text"]]

        return tokenizer(
            texts,
            padding='max_length',
            max_length=128,  # Adjust based on your model's context window
            return_tensors=None,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[], #'x', 'y', 'z', 'op', 'text'],
        desc="Tokenizing dataset"
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset

def init_model(tokenizer, model_kwargs, architecture_name=None):
    """Initialize model with specified architecture from config."""
    # Load architecture config
    arch_config = load_architecture_config()

    # Select architecture
    if architecture_name is None: architecture_name = select_runtime_architecture(arch_config)

    # Get the classes for this architecture
    config_class, model_class = get_architecture_classes(arch_config, architecture_name)

    # Get architecture-specific defaults
    arch_defaults = arch_config['architecture_defaults'].get(architecture_name, {})

    # Get runtime overrides
    runtime_overrides = arch_config['runtime_config'].get('override_params', {})
    if runtime_overrides == 'pass': runtime_overrides = {}

    # Build config parameters with proper precedence:
    # 1. Base required parameters
    # 2. Architecture defaults
    # 3. Runtime overrides
    # 4. Model-specific kwargs (highest priority)
    config_params = {
        'vocab_size': tokenizer.vocab_size,
        **arch_defaults,
        **runtime_overrides,
        **model_kwargs,
    }

    print(f"Initializing {architecture_name} model with config: {config_class.__name__}")
    config = config_class(**config_params)
    model = model_class(config)

    return model

def train_model(model, tokenizer, training_dataset, validation_dataset, training_kwargs, log_steps):
    training_args = TrainingArguments(
        report_to="none",
        eval_strategy="steps",
        metric_for_best_model='eval_input_ids_loss',
        disable_tqdm=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        #optim="adamw_torch_fused",
        #optim="schedule_free_radamw",
        #torch_compile=True,
        **training_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        ),
        callbacks=[CustomProgressCallback(tokenizer, log_steps=log_steps)]
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    return trainer

def load_model_configs():
    """Load model configurations from the YAML config file."""
    config_path = os.path.join(project_root, "model_configs.yaml")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)

    def convert_numeric_values(obj):
        if isinstance(obj, dict): return {k: convert_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, str) and ('e-' in obj.lower() or 'e+' in obj.lower()):
            try: return float(obj)
            except ValueError: return obj
        else: return obj

    return convert_numeric_values(config)

def load_architecture_config():
    """Load architecture configurations from the YAML config file."""
    config_path = os.path.join(project_root, "architecture_configs.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_site_config():
    """Load site-specific configurations from YAML config file."""
    config_path = os.path.join(project_root, "site_config.yml")
    example_path = os.path.join(project_root, "site_config.example.yml")

    if not os.path.exists(config_path):
        if os.path.exists(example_path):
            print(f"Warning: {config_path} not found. Please copy {example_path} to {config_path} and customize it.")
            print("Using example config for now...")
            config_path = example_path
        else:
            raise FileNotFoundError(f"Neither {config_path} nor {example_path} found. Please create site configuration.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand path variables in the config
    def expand_vars(text, var_dict):
        """Expand ${var} references in text using var_dict."""
        if not isinstance(text, str) or "${" not in text: return text

        result = text
        for var_name, var_value in var_dict.items():
            if isinstance(var_value, str):
                result = result.replace(f"${{{var_name}}}", var_value)
        return result

    # Build a flat dictionary of all variables for expansion
    all_vars = {}

    # First, collect base directory values
    dirs = config.get('directories', {})
    if 'scratch_dir' in dirs:
        all_vars['scratch_dir'] = dirs['scratch_dir']

    # Then expand base_dir references within each section
    for section_name, section_data in dirs.items():
        if isinstance(section_data, dict):
            if 'base_dir' in section_data:
                base_dir = expand_vars(section_data['base_dir'], all_vars)
                all_vars['base_dir'] = base_dir
                section_data['base_dir'] = base_dir

            # Expand all other paths in this section
            for key, value in section_data.items():
                if key != 'base_dir' and isinstance(value, str):
                    section_data[key] = expand_vars(value, all_vars)

    return config

def get_architecture_classes(arch_config, architecture_name):
    """Dynamically import and return the config and model classes for the specified architecture."""
    arch_info = arch_config['architectures'][architecture_name]

    if arch_info['import_path'] == 'transformers':
        from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                                  Lfm2MoeConfig, Lfm2MoeForCausalLM, Lfm2Config, Lfm2ForCausalLM,
                                  GraniteConfig, GraniteForCausalLM, GraniteMoeConfig, GraniteMoeForCausalLM,
                                  GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM,
                                  GraniteMoeSharedConfig, GraniteMoeSharedForCausalLM)

        class_map = {
            'LlamaConfig': LlamaConfig, 'LlamaForCausalLM': LlamaForCausalLM,
            'Qwen2Config': Qwen2Config, 'Qwen2ForCausalLM': Qwen2ForCausalLM,
            'Lfm2MoeConfig': Lfm2MoeConfig, 'Lfm2MoeForCausalLM': Lfm2MoeForCausalLM,
            'Lfm2Config': Lfm2Config, 'Lfm2ForCausalLM': Lfm2ForCausalLM,
            'GraniteConfig': GraniteConfig, 'GraniteForCausalLM': GraniteForCausalLM,
            'GraniteMoeConfig': GraniteMoeConfig, 'GraniteMoeForCausalLM': GraniteMoeForCausalLM,
            'GraniteMoeHybridConfig': GraniteMoeHybridConfig, 'GraniteMoeHybridForCausalLM': GraniteMoeHybridForCausalLM,
            'GraniteMoeSharedConfig': GraniteMoeSharedConfig, 'GraniteMoeSharedForCausalLM': GraniteMoeSharedForCausalLM,
        }

        return class_map[arch_info['config_class']], class_map[arch_info['model_class']]
    else:
        raise ValueError(f"Unsupported import path: {arch_info['import_path']}")

def select_runtime_architecture(arch_config, override_arch=None):
    """Select the architecture to use at runtime."""
    # Priority: override_arch > environment variable > config file > default
    selected = override_arch
    if selected is None: selected = os.environ.get('GUAC_ARCHITECTURE')
    if selected is None: selected = arch_config['runtime_config']['selected_architecture']
    if selected is None: selected = arch_config['default_architecture']
    if selected not in arch_config['architectures']:
        available = list(arch_config['architectures'].keys())
        raise ValueError(f"Unknown architecture '{selected}'. Available: {available}")

    print(f"Selected architecture: {selected}")
    return selected

def do_train(model, mc, train, val, step_val, output_dir):
    trainer = train_model(
            model = model,
            tokenizer = tokenizer,
            training_dataset = tok_smol_simple_dataset["train"],
            validation_dataset = tok_smol_simple_dataset["validation"][:100],
            log_steps = step_val,
            training_kwargs = {
                "output_dir": output_dir,
                "eval_steps": step_val,
                "logging_steps": step_val,
                "save_steps": 5*step_val,
                "learning_rate": mc['learning_rate'],
                "lr_scheduler_type": "reduce_lr_on_plateau",
                "warmup_ratio": mc["warmup_ratio"],
                "weight_decay": mc["weight_decay"],
                "per_device_train_batch_size": mc["pdbs_train"],
                "per_device_eval_batch_size": mc["pdbs_eval"],
                "gradient_accumulation_steps": mc["grad_accum_steps"],
                "num_train_epochs": mc["num_train_epochs"],
            },
        )

    trainer.save_model(f"{output_dir}_complete")

if __name__ == "__main__":
    # Load site-specific configuration
    site_config = load_site_config()
    dirs = site_config['directories']

    smol_simple_dataset = load_from_disk(dirs['datasets']['arithmetic_simple'])
    smol_complex_dataset = load_from_disk(dirs['datasets']['arithmetic_complex'])
    smol_dataset = DatasetDict({
        'train': concatenate_datasets([smol_simple_dataset['train'], smol_complex_dataset['train']]),
        'validation': concatenate_datasets([smol_simple_dataset['validation'], smol_complex_dataset['validation']])
    })

    tokenizer = train_character_tokenizer(smol_dataset)
    tokenizer.save_pretrained(dirs['tokenizers']['char_arithmetic'])

    tok_smol_simple_dataset = tokenize_dataset(tokenizer, smol_simple_dataset)
    tok_smol_complex_dataset = tokenize_dataset(tokenizer, smol_complex_dataset)
    tok_smol_simple_dataset.save_to_disk(dirs['tokenized_datasets']['simple'])
    tok_smol_complex_dataset.save_to_disk(dirs['tokenized_datasets']['complex'])

    # Load model configurations from YAML file
    config = load_model_configs()
    model_training_configs = config['model_training_configs']
    model_training_configs['defaults'] = config['defaults']
    common_training_kwargs = config['common_training_kwargs']

    targets = [ "iota" ]

    for target in targets:
        print(f"> TRAINING: {target}")

        selected_config = model_training_configs[target]
        d = common_training_kwargs | model_training_configs["defaults"]
        mc = d | selected_config

        model = init_model(tokenizer, {
            "hidden_size": mc['hidden_size'], "intermediate_size": mc['hidden_size'] * mc['intermediate_layers_fac'],
            "num_hidden_layers": mc['num_hidden_layers'], "num_attention_heads": mc['num_attention_heads'],
            "max_position_embeddings": mc['max_position_embeddings'], "num_key_value_heads": mc['num_kv_heads'],
            "torch_dtype": torch.float16, "num_experts": mc['num_experts'],
            "num_experts_per_tok": mc['num_experts_per_token'], "layer_types": mc['layer_types']
        })

        print(f"Model has {model.num_parameters()} parameters")

        if model.config.vocab_size != tokenizer.vocab_size:
            print(f"MODEL ERROR: Mismatch between model vocab_size ({model.config.vocab_size}) and tokenizer vocab_size ({tokenizer.vocab_size})")

        step_val = mc['step_val']

        do_train(model, mc, tok_smol_simple_dataset["train"], tok_smol_simple_dataset["validation"][:100], step_val, dirs['checkpoints']['pretrain_simple'])
        print("\n")

