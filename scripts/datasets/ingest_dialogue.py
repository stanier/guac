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
    return re.sub(r"^\s+|\s+$", "", message, flags=re.UNICODE)

def instruct_to_dialogue(article):
    """A system turn (the instruct column, else the default), the user's question and the response."""
    system = strip(article.get(INSTRUCT_INSTRUCT) or "") or defaults["system"]
    article["messages"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": strip(article[INSTRUCT_QUESTION])},
        {"role": "assistant", "content": strip(article[INSTRUCT_RESPONSE])},
    ]
    return article

def alpaca_to_question(article):
    """Alpaca's instruction is the user's request, and its optional input is context for it."""
    context = strip(article["input"])
    return {INSTRUCT_QUESTION: strip(article["instruction"]) + (f"\n\n{context}" if context else "")}

def inoutpair_to_dialogue(article):
    article["messages"] = []

    article["messages"].append({"role": "system", "content": defaults["system"]})

    for pair in article["conversation"]:
        article["messages"].append({"role": "user", "content": strip(pair["input"])})
        article["messages"].append({"role": "assistant", "content": strip(pair["output"])})

    return article

def gen_hash(x): return hashlib.sha256(json.dumps(x).encode()).hexdigest()

if __name__ == "__main__":
    # SFT Instruct
    alpaca_dataset = (
        load_dataset(f"{DATASET_DIR}/alpaca/data", split = "train")
        .map(alpaca_to_question, remove_columns = ["instruction", "input"])
        .rename_column("output", INSTRUCT_RESPONSE)
        .map(instruct_to_dialogue)
        .remove_columns([INSTRUCT_QUESTION, INSTRUCT_RESPONSE])
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
