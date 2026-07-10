import uuid
import hashlib
import json
import re

from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
)

INSTRUCT_INSTRUCT = "instruct"
INSTRUCT_QUESTION = "question"
INSTRUCT_RESPONSE = "response"

DATASET_DIR = "./scratch/datasets/"

defaults = {
    "system": "Respond accurately to user's query",
}

def strip(message):
    return re.sub("^\s+|\s+$", "", message, flags=re.UNICODE)

def instruct_to_dialogue(article):
    article["messages"] = []

    if "instruct" in article: article["messages"].append({"role": "system", "content": strip(article["instruct"])})
    if "question" in article: article["messages"].append({"role": "user", "content": strip(article["question"])})
    article["messages"].append({"role": "assistant", "content": strip(article["response"])})

    return article

def inoutpair_to_dialogue(article):
    article["messages"] = []

    article["messages"].append({"role": "system", "content": defaults["system"]})

    for pair in article["conversation"]:
        article["messages"].append({"role": "user", "content": strip(pair["input"])})
        article["messages"].append({"role": "assistant", "content": strip(pair["output"])})

    return article

def gen_hash(x): hashlib.sha256(json.dumps(x).encode()).hexdigest()

if __name__ == "__main__":
    # SFT Instruct
    alpaca_dataset = (
        load_dataset(f"{DATASET_DIR}/alpaca/data", split = "train")
        .rename_column("instruction", INSTRUCT_INSTRUCT)
        .rename_column("input", INSTRUCT_QUESTION)
        .rename_column("output", INSTRUCT_RESPONSE)
        .map(instruct_to_dialogue)
        .remove_columns([INSTRUCT_INSTRUCT, INSTRUCT_QUESTION, INSTRUCT_RESPONSE])
        .map(lambda x: {**x, "source": "alpaca", "hash": gen_hash(x["messages"])})
    )
    #openorca_dataset = (
    #    load_dataset(f"{DATASET_DIR}/OpenOrca", split = "train")
    #    .rename_column("id", "openorca_id")
    #    .rename_column("system_prompt", INSTRUCT_INSTRUCT)
    #    .map(instruct_to_dialogue)
    #    .remove_columns([INSTRUCT_INSTRUCT, INSTRUCT_QUESTION, INSTRUCT_RESPONSE])
    #    .map(lambda x: {**x, "source": "openorca", "hash": gen_hash(x["messages"])})
    #)

    # SFT Conversational
    capybara_dataset = (
        load_dataset(f"{DATASET_DIR}/Capybara", split = "train")
        .rename_column("source", "capybara_source")
        .map(inoutpair_to_dialogue)
        .remove_columns("conversation")
        .map(lambda x: {**x, "source": "capybara", "hash": gen_hash(x["messages"])})
    )
    norobots_dataset = (
        load_dataset(f"{DATASET_DIR}/no_robots/data", split = "train")
        .remove_columns(["prompt", "category"])
        .rename_column("prompt_id", "norobots_prompt_id")
        .map(lambda x: {**x, "source": "no_robots", "hash": gen_hash(x["messages"])})
    )

    merged_dataset = concatenate_datasets([
        alpaca_dataset,
        #openorca_dataset,
        capybara_dataset,
        norobots_dataset,
    ])

    merged_dataset.save_to_disk(f"{DATASET_DIR}/guac-merge0")
