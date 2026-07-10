import argparse

import jinja2
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from datasets import load_from_disk
from trl import (
    SFTTrainer,
    ORPOTrainer,
    AutoModelForCausalLMWithValueHead,
    ModelConfig,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
)
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model,
)

defaults = {
    "model_library"         : "./scratch/models/",
    "dataset_library"       : "./scratch/datasets/",
    "output_path"           : "./scratch/output/guac0/",
    "dataset_name"          : "alpaca",
    "dataset_split_train"   : "train",
    "model_name"            : "Meta-Llama-3-8B",
    "precision"             : "4bit",
    "stack_name"            : "sft_alpaca_basic",
    "max_seq_length"        : 16384,
    "task_type"             : "CAUSAL_LM",
    "lora"                  : {
        "r"         : 64,
        "alpha"     : 16,
        "dropout"   : 0.05,
        "bias"      : "none",
    },
    "max_steps"             : 100000,
    "attn_implementation"   : "flash_attention_2", #"sdpa",
    "epochs"                : 3,
    "per-device-batch-size" : 1,
    "log_level"             : "passive",
    "logging_first_step"    : True,
    "logging_steps"         : 500,
    "logging_strategy"      : "steps",
    "learning_rate"         : "2e-4",
}

class TrainingStack:
    def __init__(self, **kwargs):
        self.max_seq_length         = int(kwargs.get("max_seq_length", defaults["max_seq_length"]))
        self.model_library          = kwargs.get("model_library", defaults["model_library"])
        self.dataset_library        = kwargs.get("dataset_library", defaults["dataset_library"])
        self.output_path            = kwargs.get("output_path", defaults["output_path"])
        self.model_name             = kwargs.get("model_name", defaults["model_name"])
        self.precision              = kwargs.get("precision", defaults["precision"])
        self.stack_name             = kwargs.get("stack_name", defaults["stack_name"])
        self.dataset_name           = kwargs.get("dataset_name", defaults["dataset_name"])
        self.dataset_split_train    = kwargs.get("dataset_split_train", defaults["dataset_split_train"])
        self.max_steps              = int(kwargs.get("max_steps", defaults["max_steps"]))
        self.epochs                 = float(kwargs.get("epochs", defaults["epochs"]))
        self.pdbs                   = int(kwargs.get("per_device_batch_size", defaults["per-device-batch-size"]))
        self.logging_first_step     = bool(kwargs.get("logging_first_step", defaults["logging_first_step"]))
        self.log_level              = kwargs.get("log_level", defaults["log_level"])
        self.logging_steps          = int(kwargs.get("logging_steps", defaults["logging_steps"]))
        self.logging_strategy       = kwargs.get("logging_strategy", defaults["logging_strategy"])
        self.learning_rate          = float(kwargs.get("learning_rate", defaults["learning_rate"]))

    def prompt_formatting_func(self, article):
        output_texts = []

        environment = jinja2.Environment()

        for i in range(len(article['hash'])):
            text = environment.from_string("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n{% for message in messages %}\n{% if message['role'] == 'system' %}### Instruction:\n{% elif message['role'] == 'user' %}### Input:\n{% elif message['role'] == 'assistant' %}### Response:\n{% endif %}{{message['content']}}\n{% endfor %}").render(messages = article['messages'][i])
            output_texts.append(text)
        return output_texts

    def loadModel(self, **kwargs):
        model_path = f"{self.model_library}/{self.model_name}"

        self.model_config = ModelConfig(
            model_name_or_path      = model_path,
            attn_implementation     = kwargs.get("attn_implementation", defaults["attn_implementation"]),
        )

        response_template = "### Response:\n"
        self.quant_config = get_quantization_config(self.model_config)

        model_kwargs = dict(
            torch_dtype         = "auto",
            load_in_8bit        = (self.precision == "8bit"),
            load_in_4bit        = (self.precision == "4bit"),
            trust_remote_code   = False, # Don't
            attn_implementation = kwargs.get("attn_implementation", defaults["attn_implementation"]),
            use_cache           = False, # false if grad chkpnting
            quantization_config = self.quant_config,
            device_map          = get_kbit_device_map(),
        )

        self.tokenizer  = AutoTokenizer.from_pretrained(model_path)
        self.model      = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def loadDataset(self, **kwargs):
        self.train_dataset = load_from_disk(f"{self.dataset_library}/{self.dataset_name}") #, split = self.dataset_split_train)
    
    def setupLora(self, **kwargs):
        self.lora_config = LoraConfig(
            r               = kwargs.get("r", defaults["lora"]["r"]),
            lora_alpha      = kwargs.get("lora_alpha", defaults["lora"]["alpha"]),
            lora_dropout    = kwargs.get("lora_dropout", defaults["lora"]["dropout"]),
            bias            = kwargs.get("bias", defaults["lora"]["bias"]),
            task_type       = kwargs.get("task_type", defaults["task_type"]),
        )

        self.model = get_peft_model(self.model, self.lora_config)

    def setupTrainer(self, report_to = "tensorboard", **kwargs):
        self.training_args = TrainingArguments(
            logging_strategy            = kwargs.get("logging_strategy", self.logging_strategy),
            logging_steps               = kwargs.get("logging_steps", self.logging_steps),
            logging_first_step          = kwargs.get("logging_first_step", self.logging_first_step),
            report_to                   = report_to,
            num_train_epochs            = int(kwargs.get("epochs", self.epochs)),
            output_dir                  = kwargs.get("output_dir", self.output_path),
            per_device_train_batch_size = kwargs.get("pdbs", self.pdbs),
            learning_rate               = float(kwargs.get("learning_rate", self.learning_rate)),
            #packing                     = False,
        )

        self.response_template = "### Response:\n"
        self.collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer = self.tokenizer)

    def trainModel(self):
        self.trainer.train()

    def saveModel(self, **kwargs):
        self.trainer.save_model(kwargs.get("output_dir", self.training_args.output_dir))

class SFTTrainingStack(TrainingStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setupTrainer(self, report_to = "tensorboard", **kwargs):
        super().setupTrainer(report_to = report_to, **kwargs)

        self.trainer = SFTTrainer(
            self.model,
            args                = self.training_args,
            train_dataset       = self.train_dataset,
            #dataset_text_field  = "text",
            max_seq_length      = self.max_seq_length,
            peft_config         = self.lora_config,
            formatting_func     = self.prompt_formatting_func,
            data_collator       = self.collator,
        )
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--hf-model-library', action = 'store', default = defaults["model_library"])
    parser.add_argument('-o', '--outdir', action = 'store', default = defaults["output_path"], dest = 'outdir')
    parser.add_argument('-b', '--per-device-batch-size', default = 1, dest = 'per_device_batch_size')
    parser.add_argument('-e', '--epochs', default = 1, dest = 'epochs')
    parser.add_argument('-m', '--base-model', default = defaults["model_name"], dest = 'model_name')
    parser.add_argument('-s', '--max-seq-length', default = defaults["max_seq_length"], dest = 'max_seq_length')
    parser.add_argument('-d', '--dataset', default = defaults["dataset_name"], dest = 'training_dataset')

    args = parser.parse_args()

    stack = SFTTrainingStack(
        epochs                  = args.epochs,
        model_name              = args.model_name,
        max_seq_length          = args.max_seq_length,
        per_device_batch_size   = args.per_device_batch_size,
        dataset_name            = args.training_dataset,
    )

    stack.loadDataset()
    stack.loadModel()
    stack.setupLora()
    stack.setupTrainer(report_to = "wandb")
    stack.trainModel()
    stack.saveModel(output_dir = args.outdir)
