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
        article["messages"].append({"role": "user", "content": pair["input"].strip()})
        article["messages"].append({"role": "assistant", "content": pair["output"].strip()})

    return article

def gen_hash(x): hashlib.sha256(json.dumps(x).encode()).hexdigest()

if __name__ == "__main__":
    alpaca_dataset = (
        load_dataset(f"{DATASET_DIR}/alpaca/data", split = "train")
        .rename_column("instruction", INSTRUCT_INSTRUCT)
        .rename_column("input", INSTRUCT_QUESTION)
        .rename_column("output", INSTRUCT_RESPONSE)
        .map(instruct_to_dialogue)
        .remove_columns([INSTRUCT_INSTRUCT, INSTRUCT_QUESTION, INSTRUCT_RESPONSE])
        .map(lambda x: {**x, "source": "alpaca", "hash": gen_hash(x["messages"])})
    )

    alpaca_dataset.save_to_disk(f"{DATASET_DIR}/guac0-alpaca")
