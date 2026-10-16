"""
Tests for reset_at_document_boundaries: with packed documents, every layer
type gives each document the outputs it would get on its own, and gradients
don't cross documents. Generation (with a cache) ignores the setting.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from guac.core.custom_architectures import (
    SUPPORTED_LAYER_TYPES, MixedLayerForCausalLM, documents_from_eos, positions_within_documents,
)
from tests.test_generation import make_model

EOS = 2


def packed_documents(lengths, seed=0):
    """Documents of the given lengths (each ending in EOS), and the row that packs them."""
    generator = torch.Generator().manual_seed(seed)
    documents = [torch.cat([torch.randint(3, 40, (n - 1,), generator=generator), torch.tensor([EOS])])
                 for n in lengths]
    return documents, torch.cat(documents)[None]


# Long enough to cross the test models' chunks (4) and windows (4), with a one-token document
LENGTHS = [7, 1, 12, 5, 9]


def test_document_ids_count_earlier_eos_tokens():
    input_ids = torch.tensor([[5, 2, 6, 7, 2, 2, 8]])
    assert documents_from_eos(input_ids, 2).tolist() == [[0, 0, 1, 1, 1, 2, 3]]
    assert documents_from_eos(input_ids, [2, 8]).tolist() == [[0, 0, 1, 1, 1, 2, 3]]


def test_positions_restart_at_each_document_and_skip_padding():
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    assert positions_within_documents(document_ids).tolist() == [[0, 1, 2, 0, 1, 0]]
    mask = torch.tensor([[0, 1, 1, 1, 1, 1]])
    assert positions_within_documents(document_ids, mask).tolist() == [[0, 0, 1, 0, 1, 0]]


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_packed_documents_match_documents_run_alone(layer_type):
    model = make_model([layer_type, layer_type], reset_at_document_boundaries=True, eos_token_id=EOS,
                       use_absolute_position_embeddings=True)
    documents, packed = packed_documents(LENGTHS)
    with torch.no_grad():
        logits = model(input_ids=packed, use_cache=False).logits
        separate = torch.cat([model(input_ids=doc[None], use_cache=False).logits for doc in documents], dim=1)
    torch.testing.assert_close(logits, separate, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("layer_type", ["full_attention", "conv", "mamba2", "based", "sliding_window_attention"])
def test_without_the_setting_documents_do_mix(layer_type):
    model = make_model([layer_type, layer_type], eos_token_id=EOS)
    documents, packed = packed_documents(LENGTHS)
    with torch.no_grad():
        logits = model(input_ids=packed, use_cache=False).logits
        alone = model(input_ids=documents[2][None], use_cache=False).logits
    start = sum(LENGTHS[:2])
    assert not torch.allclose(logits[:, start:start + LENGTHS[2]], alone, atol=1e-4)


def test_gradients_do_not_cross_documents():
    model = make_model(list(SUPPORTED_LAYER_TYPES), reset_at_document_boundaries=True, eos_token_id=EOS)
    _, packed = packed_documents(LENGTHS)
    embeddings = model.get_input_embeddings()(packed).detach().requires_grad_()
    logits = model(inputs_embeds=embeddings, document_ids=documents_from_eos(packed, EOS), use_cache=False).logits
    # The loss on the last document reaches no earlier token
    logits[:, -LENGTHS[-1]:].sum().backward()
    assert embeddings.grad[:, :-LENGTHS[-1]].abs().max() == 0
    assert embeddings.grad[:, -LENGTHS[-1]:].abs().max() > 0


def test_batched_rows_with_different_documents_and_left_padding():
    model = make_model(["full_attention", "mamba2", "based", "conv"], reset_at_document_boundaries=True,
                       eos_token_id=EOS)
    documents, packed = packed_documents([6, 9])
    # Row 1: one document, left-padded with pad token 0
    single = torch.cat([torch.zeros(1, 6, dtype=torch.long), documents[1][None]], dim=1)
    mask = torch.cat([torch.zeros(1, 6), torch.ones(1, 9)], dim=1).long()
    with torch.no_grad():
        batch = model(input_ids=torch.cat([packed, single]), use_cache=False,
                      attention_mask=torch.cat([torch.ones(1, 15, dtype=torch.long), mask])).logits
        alone = model(input_ids=documents[1][None], use_cache=False).logits
    torch.testing.assert_close(batch[0, 6:], alone[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batch[1, 6:], alone[0], atol=1e-5, rtol=1e-4)


def test_generation_ignores_the_setting():
    model = make_model(["full_attention", "mamba2"], reset_at_document_boundaries=True, eos_token_id=EOS)
    _, packed = packed_documents([4, 5])
    with torch.no_grad():
        out = model(input_ids=packed, use_cache=True)
    assert out.past_key_values is not None
    with pytest.raises(ValueError, match="generation cache"):
        model(input_ids=packed, use_cache=True, document_ids=documents_from_eos(packed, EOS))


def test_setting_survives_save_and_load(tmp_path):
    model = make_model(["conv"], reset_at_document_boundaries=True, eos_token_id=EOS)
    model.save_pretrained(tmp_path)
    assert MixedLayerForCausalLM.from_pretrained(tmp_path).config.reset_at_document_boundaries


def test_preset_setting_and_tokenizer_eos_reach_the_model_config():
    """Documents split at the tokenizer's EOS, which needn't be the config default (2)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from guac.pipeline.pretrain import build_model_config

    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                        bos_token="[BOS]", eos_token="[EOS]")
    settings = {'hidden_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'layer_types': ['conv'],
                'reset_at_document_boundaries': True}
    config = build_model_config(settings, len(tokenizer), tokenizer)
    assert config.reset_at_document_boundaries
    assert (config.pad_token_id, config.bos_token_id, config.eos_token_id) == (0, 2, 3)
    assert build_model_config(settings, len(tokenizer)).eos_token_id == 2  # no tokenizer: the default
