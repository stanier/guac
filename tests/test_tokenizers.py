"""Tokenizer building: SentencePiece -> HF conversion, the vocab pipeline, BPE, characters."""

import os
import random
import subprocess
import sys

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

import sentencepiece as spm

from guac.core.prompt_templates import create_prompt_tokenizer
from guac.tokenizer.utils import load_model_proto, train_spm, wrap_as_hf_tokenizer

CORE = create_prompt_tokenizer().get_core_tokens()
SPECIALS = create_prompt_tokenizer().get_all_special_tokens()


def corpus(words, n, seed=0):
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        line = " ".join(rng.choice(words) for _ in range(rng.randint(3, 25)))
        if rng.random() < 0.3:
            line += f"\n\n{rng.randint(0, 99999)}.  {rng.choice(words).upper()}"
        lines.append(line)
    return lines


def make_words(alphabet, n, seed):
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(2, 9))) for _ in range(n)]


ENGLISH = make_words("etaoinshrdlucmfwyp", 300, 0)
CODE = [w + s for w in make_words("abcdefxyz_", 60, 1) for s in ("()", "[i]", " = 0;", "->next")]


@pytest.fixture(scope="module")
def sp_tokenizer(tmp_path_factory):
    out = tmp_path_factory.mktemp("sp")
    text = out / "train.txt"
    text.write_text("\n".join(corpus(ENGLISH, 2000)) + "\n")
    model = train_spm(str(text), str(out / "tokenizer"), 600, user_defined_symbols=SPECIALS,
                      character_coverage=0.9995)
    wrap_as_hf_tokenizer(model, CORE, str(out / "hf"))
    return model, str(out / "hf")


def test_sentencepiece_conversion_matches_sentencepiece(sp_tokenizer):
    model, hf_dir = sp_tokenizer
    sp = spm.SentencePieceProcessor(model_file=model)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    assert tok.is_fast
    texts = corpus(ENGLISH, 200, seed=5) + [
        "the   model\n\nlearns 漢字", "trailing  ", "ﬁ ① ｆｕｌｌ", "x=12345.678\tq", "emoji 🥑 ok", "",
        "\n".join(corpus(ENGLISH, 5, seed=7)),
    ]
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids == sp.encode(text), text
        assert tok.decode(ids) == sp.decode(ids), text


def test_sentencepiece_conversion_tokens(sp_tokenizer):
    model, hf_dir = sp_tokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    proto_ids = {p.piece: i for i, p in enumerate(load_model_proto(model).pieces)}
    # Core tokens are the model's pad/unk/bos/eos at their SentencePiece ids
    for name, token in CORE.items():
        assert getattr(tok, name) == token
        assert tok.convert_tokens_to_ids(token) == proto_ids[token]
    # EOS ends every encoded text, and nothing becomes UNK (byte fallback)
    ids = tok("new line\nand 漢")["input_ids"]
    assert ids[-1] == tok.eos_token_id
    assert tok.unk_token_id not in ids and proto_ids["<unk>"] not in ids
    assert tok.decode(ids, skip_special_tokens=True) == "new line\nand 漢"
    # Prompt template tokens (which contain spaces) come out whole
    think = "<& THINK &>"
    ids = tok(f"a{think}b", add_special_tokens=False)["input_ids"]
    assert tok.convert_ids_to_tokens(ids)[1] == think
    assert tok.decode(ids) == f"a{think}b"


def test_special_token_order_is_deterministic():
    code = "from guac.core.prompt_templates import create_prompt_tokenizer as c; print(c().get_all_special_tokens())"
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout
               for seed in ("1", "2", "3")}
    assert len(outputs) == 1
    assert SPECIALS[:3] == [CORE['pad_token'], CORE['unk_token'], CORE['bos_token']]


def test_char_tokenizer_keeps_newlines(tmp_path):
    from guac.tokenizer.build_char_tokenizer import build_char_tokenizer
    build_char_tokenizer(str(tmp_path))
    tok = AutoTokenizer.from_pretrained(str(tmp_path))
    ids = tok("a\n\nb")["input_ids"]
    assert tok.unk_token_id not in ids
    assert tok.decode(ids, skip_special_tokens=True) == "a\n\nb"


def test_bpe_tokenizer_round_trips(tmp_path):
    from guac.pipeline.train_tokenizer import train_bpe_tokenizer
    text = tmp_path / "train.txt"
    text.write_text("\n".join(corpus(ENGLISH, 500)) + "\n")
    train_bpe_tokenizer(str(text), str(tmp_path / "bpe"), 500, SPECIALS, CORE)
    tok = AutoTokenizer.from_pretrained(str(tmp_path / "bpe"))
    sample = "hello world, it's 2026.\n\n  漢 🥑"
    ids = tok(sample)["input_ids"]
    assert ids[-1] == tok.eos_token_id and tok.unk_token_id not in ids
    assert tok.decode(ids, skip_special_tokens=True) == sample
    assert [tok.pad_token_id, tok.unk_token_id, tok.bos_token_id, tok.eos_token_id] == [0, 1, 2, 3]
    assert len(tok(f"a{SPECIALS[5]}b", add_special_tokens=False)["input_ids"]) == 3


def test_vocab_pipeline_end_to_end(tmp_path):
    from guac.tokenizer import VocabPipeline, VocabSpec, PartitionSpec
    from guac.tokenizer.utils import extract_tokens_from_model

    paths = {}
    for name, words in (("general", ENGLISH), ("code", CODE)):
        paths[name] = str(tmp_path / name)
        Dataset.from_dict({"text": corpus(words, 1500)}).save_to_disk(paths[name])

    spec = VocabSpec(total_size=900, core_size=600, output_dir=str(tmp_path / "out"),
                     final_data=[paths["general"]],
                     partitions=[PartitionSpec("code", [paths["code"]], budget=120)])
    hf_dir = VocabPipeline(spec).run()

    core = load_model_proto(spec.core_model_path).pieces
    final = load_model_proto(spec.final_model_path).pieces
    assert len(final) == spec.total_size
    # Core pieces keep their ids; partition tokens follow them
    assert [p.piece for p in final[:len(core)]] == [p.piece for p in core]
    code_tokens = [p.piece for p in final[len(core):len(core) + 120]]
    assert any("(" in t or "_" in t for t in code_tokens)

    tok = AutoTokenizer.from_pretrained(hf_dir)
    sp = spm.SentencePieceProcessor(model_file=str(spec.final_model_path))
    for text in corpus(CODE, 20, seed=3) + corpus(ENGLISH, 20, seed=4):
        assert tok(text, add_special_tokens=False)["input_ids"] == sp.encode(text)
    # The code partition's tokens are actually used on code
    code_ids = {i for t in corpus(CODE, 50, seed=9) for i in sp.encode(t)}
    assert any(len(core) <= i < len(core) + 120 for i in code_ids)
    assert tok.pad_token == CORE['pad_token'] and tok.eos_token_id < len(core)
    assert set(SPECIALS) <= set(extract_tokens_from_model(str(spec.final_model_path)))
