#!/usr/bin/env python3
"""
Custom architecture implementations for Guac training.

This module provides custom model architectures that can be used alongside
standard transformers architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from typing import Optional, Tuple, Union, List, Dict, Any

from .linear_attention import causal_linear_attention, taylor_feature_map
from .mamba2 import Mamba2Mixer


SUPPORTED_LAYER_TYPES = (
    "full_attention", "conv", "mlp_only", "mamba2", "based",
    "flash_attention", "moh_attention", "shared_expert_moe", "sliding_window_attention",
)

# Layer types whose mixer is softmax attention, and so gets rotary position embeddings when use_rope is set
ROTARY_LAYER_TYPES = ("full_attention", "flash_attention", "sliding_window_attention", "moh_attention")

# Layer types whose block has only the FFN (shared_expert_moe defaults its FFN to MoE)
MIXERLESS_LAYER_TYPES = ("mlp_only", "shared_expert_moe")

# Per-layer FFN choices for MixedLayerConfig.ffn_types
FFN_TYPES = ("swiglu", "moe", "none")


def default_ffn_type(layer_type: str) -> str:
    return "moe" if layer_type == "shared_expert_moe" else "swiglu"


def causal_attention(query, key, value, attention_mask=None, dropout_p=0.0, window=None):
    """
    Causal scaled dot-product attention shared by every softmax-attention layer.

    query is (batch, heads, seq, head_dim); key and value are
    (batch, kv_heads, seq, head_dim) with heads a multiple of kv_heads
    (grouped-query attention). attention_mask is an optional (batch, seq)
    key padding mask with 1 for real tokens and 0 for padding. With `window`,
    each position attends only to the last `window` positions, itself included.

    F.scaled_dot_product_attention picks a FlashAttention or memory-efficient
    kernel when the device and dtype support it, and the math kernel otherwise.
    """
    num_kv_groups = query.size(1) // key.size(1)
    if num_kv_groups > 1:
        key = key.repeat_interleave(num_kv_groups, dim=1)
        value = value.repeat_interleave(num_kv_groups, dim=1)

    if window is not None and window < query.size(2):
        return sliding_window_attention(query, key, value, window, attention_mask, dropout_p)

    # Without padding the causal mask alone is exact, and is_causal=True lets
    # PyTorch use its fastest kernels
    if attention_mask is None or bool(attention_mask.all()):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=True)

    seq_len = query.size(2)
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
    allowed = allowed & attention_mask[:, None, None, :].bool()
    # Always let a position attend to itself so no row is fully masked: a fully
    # masked row makes softmax return NaN at that padding position, and later
    # layers spread it to real tokens through 0 * NaN
    allowed = allowed | torch.eye(seq_len, dtype=torch.bool, device=query.device)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=dropout_p)


def sliding_window_attention(query, key, value, window, attention_mask=None, dropout_p=0.0):
    """
    Causal attention where position t sees positions t - window + 1 through t.

    query, key and value are (batch, heads, seq, head_dim) with the same head
    count. The sequence is split into blocks of `window` positions, and each
    block attends to itself and the block before it, so time and memory grow
    as seq * window rather than seq^2.
    """
    batch, heads, length, _ = query.shape
    key_valid = (torch.ones(batch, length, dtype=torch.bool, device=query.device)
                 if attention_mask is None else attention_mask.bool())
    pad = (-length) % window
    if pad:
        query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
        key_valid = F.pad(key_valid, (0, pad))
    num_blocks = query.size(2) // window

    def blocks(t):  # (batch, heads, blocks * window, dim) -> (batch, blocks, heads, window, dim)
        return t.reshape(batch, heads, num_blocks, window, -1).transpose(1, 2)

    def with_previous_block(t):  # concatenate each block after the one before it (zeros for the first)
        return torch.cat([torch.cat([torch.zeros_like(t[:, :1]), t[:, :-1]], dim=1), t], dim=-2)

    query = blocks(query)
    key, value = (with_previous_block(blocks(t)) for t in (key, value))
    key_valid = key_valid.reshape(batch, num_blocks, window)
    key_valid = torch.cat([torch.cat([torch.zeros_like(key_valid[:, :1]), key_valid[:, :-1]], dim=1),
                           key_valid], dim=-1)  # (batch, blocks, 2 * window)

    # Query i of a block is at offset window + i among its 2 * window keys, so it
    # may see keys i + 1 through i + window (the last one being itself)
    offsets = torch.arange(2 * window, device=query.device)
    rows = torch.arange(window, device=query.device)[:, None]
    in_window = (offsets > rows) & (offsets <= rows + window)
    allowed = in_window & key_valid[:, :, None, :]
    # As in causal_attention, always allow attending to itself so no row is fully masked
    allowed = allowed | (offsets == rows + window)

    out = F.scaled_dot_product_attention(
        query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1),
        attn_mask=allowed.flatten(0, 1)[:, None], dropout_p=dropout_p,
    )
    out = out.reshape(batch, num_blocks, heads, window, -1).transpose(1, 2)
    return out.reshape(batch, heads, num_blocks * window, -1)[:, :, :length]


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings (RoPE; Su et al. 2021, as in LLaMA) for queries and keys.

    Rotates each pair of channels (i, i + head_dim / 2) by position * theta^(-2i / head_dim),
    so query-key scores depend only on the distance between positions. The
    frequencies are computed on the fly in float32: there are no parameters or
    buffers, so checkpoints don't change and casting the model can't round them.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"Rotary embeddings need an even head size, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, query, key, offset: int = 0):
        """query and key are (batch, heads, seq, head_dim); their positions are offset, offset + 1, ..."""
        seq_len = query.size(2)
        positions = torch.arange(offset, offset + seq_len, device=query.device, dtype=torch.float32)
        inv_freq = self.theta ** (-torch.arange(0, self.head_dim, 2, device=query.device, dtype=torch.float32)
                                  / self.head_dim)
        angles = torch.outer(positions, inv_freq).repeat(1, 2)  # (seq, head_dim)
        cos, sin = angles.cos(), angles.sin()
        return apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)


def apply_rotary(x, cos, sin):
    half = x.size(-1) // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient and stable than LayerNorm. Used in LLaMA, Mistral, and other modern models.
    Formula: x * (1 / sqrt(mean(x^2) + eps)) * weight
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_norm_layer(config, hidden_size=None):
    """Get the appropriate normalization layer based on config."""
    if hidden_size is None:
        hidden_size = config.hidden_size

    if getattr(config, 'use_rms_norm', False):
        return RMSNorm(hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)


class BasicTransformerConfig(PretrainedConfig):
    """
    Configuration class for BasicTransformer model.

    A simple Transformer architecture for educational/example purposes.
    """
    model_type = "basic_transformer"

    def __init__(
        self,
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        layer_norm_eps=1e-12,
        use_rms_norm=False,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = float(layer_norm_eps)
        self.use_rms_norm = use_rms_norm
        self.initializer_range = initializer_range
        self.use_cache = use_cache


class BasicMultiHeadAttention(nn.Module):
    """Basic multi-head attention implementation."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention_dropout = config.attention_probs_dropout_prob
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        def transpose_for_scores(x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        query_layer = transpose_for_scores(query_layer)
        key_layer = transpose_for_scores(key_layer)
        value_layer = transpose_for_scores(value_layer)

        context_layer = causal_attention(
            query_layer, key_layer, value_layer, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final projection
        attention_output = self.output_projection(context_layer)

        return attention_output


class BasicTransformerLayer(nn.Module):
    """A single transformer layer with attention and feed-forward."""

    def __init__(self, config):
        super().__init__()
        self.attention = BasicMultiHeadAttention(config)
        self.attention_layer_norm = get_norm_layer(config)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.output_layer_norm = get_norm_layer(config)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)

        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(attention_output)
        layer_output = self.output_layer_norm(attention_output + feed_forward_output)

        return layer_output


class BasicTransformerForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Basic Transformer model for causal language modeling.

    This is a simple, educational implementation of a transformer decoder
    suitable for text generation tasks.
    """
    config_class = BasicTransformerConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([
            BasicTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm and output projection
        self.final_layer_norm = get_norm_layer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and tie lm_head to the token embeddings when
        # config.tie_word_embeddings is set, matching what from_pretrained does
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation."""
        return {"input_ids": input_ids}

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search (not implemented for simplicity)."""
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not implemented for simplicity
            hidden_states=None,
            attentions=None,
        )


class MixedLayerConfig(PretrainedConfig):
    """
    Configuration class for MixedLayer model.

    Supports heterogeneous layer types like Granite and LFM2, allowing different
    layer architectures (attention, conv, MLP, etc.) within the same model.
    """
    model_type = "mixed_layer"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=None,
        intermediate_size=11008,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=4096,
        layer_norm_eps=1e-6,
        use_rms_norm=True,  # Default to True for modern architecture
        initializer_range=0.02,
        use_cache=False,
        # Rotary position embeddings in the softmax-attention mixers (ROTARY_LAYER_TYPES)
        use_rope=True,
        rope_theta=10000.0,
        # A learned table of max_position_embeddings positions added to the token
        # embeddings. Off by default: attention mixers get RoPE, and conv, Mamba-2
        # and BASED mixers see order through their causal structure
        use_absolute_position_embeddings=False,
        layer_types=None,
        # Per-layer FFN: "swiglu", "moe" or "none". None -> MoE for shared_expert_moe, SwiGLU otherwise
        ffn_types=None,
        # Per-layer SwiGLU width. None -> 2/3 of intermediate_size everywhere (see swiglu_hidden_size)
        ffn_hidden_sizes=None,
        conv_kernel_size=3,
        conv_groups=1,
        # Mamba-2 specific parameters (defaults follow the Mamba-2 paper)
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        mamba_ngroups=1,
        mamba_chunk_size=64,  # Scan chunk length; affects speed and memory, not results
        # BASED linear attention parameters (Arora et al. 2024)
        based_feature_dim=16,            # Query/key size per head before the Taylor feature map
        based_num_key_value_heads=None,  # Defaults to num_kv_heads
        based_num_heads=None,            # Defaults to num_attention_heads
        based_chunk_size=64,             # Chunk length of the linear-attention scan; affects speed, not results
        # Sliding-window attention: each position sees the last sliding_window_size positions
        sliding_window_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = float(layer_norm_eps)
        self.use_rms_norm = use_rms_norm
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.conv_kernel_size = conv_kernel_size
        self.conv_groups = conv_groups

        # Mamba-2 parameters
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_headdim = mamba_headdim
        self.mamba_ngroups = mamba_ngroups
        self.mamba_chunk_size = mamba_chunk_size

        # BASED parameters - default to main attention config if None
        self.based_feature_dim = based_feature_dim
        self.based_num_key_value_heads = based_num_key_value_heads if based_num_key_value_heads is not None else self.num_kv_heads
        self.based_num_heads = based_num_heads if based_num_heads is not None else num_attention_heads
        self.based_chunk_size = based_chunk_size
        self.sliding_window_size = sliding_window_size

        # Position encoding
        self.use_rope = use_rope
        self.rope_theta = float(rope_theta)
        self.use_absolute_position_embeddings = use_absolute_position_embeddings

        # MoH (Mixture-of-Head) parameters
        self.moh_top_k = kwargs.get('moh_top_k', max(1, num_attention_heads // 2))
        self.moh_temperature = kwargs.get('moh_temperature', 1.0)
        self.moh_noise_std = kwargs.get('moh_noise_std', 0.1)

        # MoE (Mixture-of-Experts) parameters - Granite 4 style with shared experts
        self.num_experts = kwargs.get('num_experts', 1)
        self.num_experts_per_token = kwargs.get('num_experts_per_token', 1)
        self.num_shared_experts = kwargs.get('num_shared_experts', 0)
        self.moe_top_k = kwargs.get('moe_top_k', self.num_experts_per_token)
        self.moe_temperature = kwargs.get('moe_temperature', 1.0)
        self.moe_noise_std = kwargs.get('moe_noise_std', 0.1)
        self.moe_load_balancing_weight = kwargs.get('moe_load_balancing_weight', 0.01)

        # Layer types configuration - defaults to all attention if not specified
        if layer_types is None:
            self.layer_types = ["full_attention"] * num_hidden_layers
        else:
            self.layer_types = layer_types
            # Ensure layer_types matches num_hidden_layers
            if len(self.layer_types) != num_hidden_layers:
                raise ValueError(f"layer_types length ({len(self.layer_types)}) must match num_hidden_layers ({num_hidden_layers})")
            unsupported = sorted(set(self.layer_types) - set(SUPPORTED_LAYER_TYPES))
            if unsupported:
                raise ValueError(f"Unsupported layer types {unsupported}. Supported types: {list(SUPPORTED_LAYER_TYPES)}")

        num_layers = len(self.layer_types)
        self.ffn_types = list(ffn_types) if ffn_types is not None else [default_ffn_type(t) for t in self.layer_types]
        if len(self.ffn_types) != num_layers:
            raise ValueError(f"ffn_types length ({len(self.ffn_types)}) must match num_hidden_layers ({num_layers})")
        unsupported = sorted(set(self.ffn_types) - set(FFN_TYPES))
        if unsupported:
            raise ValueError(f"Unsupported FFN types {unsupported}. Supported types: {list(FFN_TYPES)}")
        empty = [i for i, (layer_type, ffn_type) in enumerate(zip(self.layer_types, self.ffn_types))
                 if layer_type in MIXERLESS_LAYER_TYPES and ffn_type == "none"]
        if empty:
            raise ValueError(f"Layers {empty} have neither a mixer nor an FFN")

        if ffn_hidden_sizes is None:
            self.ffn_hidden_sizes = [swiglu_hidden_size(intermediate_size)] * num_layers
        else:
            self.ffn_hidden_sizes = [int(size) for size in ffn_hidden_sizes]
        if len(self.ffn_hidden_sizes) != num_layers:
            raise ValueError(f"ffn_hidden_sizes length ({len(self.ffn_hidden_sizes)}) must match "
                             f"num_hidden_layers ({num_layers})")
        if min(self.ffn_hidden_sizes, default=1) < 1:
            raise ValueError(f"ffn_hidden_sizes must all be positive, got {self.ffn_hidden_sizes}")

        if use_rope and set(self.layer_types) & set(ROTARY_LAYER_TYPES) and (hidden_size // num_attention_heads) % 2:
            raise ValueError(f"Rotary embeddings need an even head size; hidden_size {hidden_size} / "
                             f"num_attention_heads {num_attention_heads} is odd. Set use_rope=False or change the sizes")

        if "based" in self.layer_types:
            if hidden_size % self.based_num_heads:
                raise ValueError(f"hidden_size {hidden_size} must be divisible by based_num_heads {self.based_num_heads}")
            if self.based_num_heads % self.based_num_key_value_heads:
                raise ValueError(f"based_num_heads {self.based_num_heads} must be divisible by "
                                 f"based_num_key_value_heads {self.based_num_key_value_heads}")
        if "sliding_window_attention" in self.layer_types and sliding_window_size < 1:
            raise ValueError(f"sliding_window_size must be positive, got {sliding_window_size}")

        if "mamba2" in self.layer_types:
            mamba_inner = int(mamba_expand * hidden_size)
            if mamba_inner % mamba_headdim:
                raise ValueError(f"Mamba-2 inner size {mamba_inner} (mamba_expand * hidden_size) must be "
                                 f"divisible by mamba_headdim {mamba_headdim}")
            if (mamba_inner // mamba_headdim) % mamba_ngroups:
                raise ValueError(f"Mamba-2 head count {mamba_inner // mamba_headdim} must be divisible by "
                                 f"mamba_ngroups {mamba_ngroups}")


# ---------------------------------------------------------------------------
# Mixers: the part of a block that moves information between positions.
# Each takes (hidden_states, attention_mask=None) and returns a tensor of the
# same shape. Norms, residuals and the feed-forward network live in
# MixedTransformerLayer, so every layer type gets the same block around its mixer.
# ---------------------------------------------------------------------------


class ConvMixer(nn.Module):
    """Causal 1D convolution over the sequence (the short convolution used in LFM2)."""

    def __init__(self, config):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            # Causal: pad k-1 on both sides, then keep only the first seq_len outputs
            padding=config.conv_kernel_size - 1,
            groups=config.conv_groups
        )

    def forward(self, hidden_states, attention_mask=None):
        # Conv1d expects (batch, channels, sequence)
        conv_output = self.conv1d(hidden_states.transpose(1, 2))[:, :, :hidden_states.shape[1]]
        return conv_output.transpose(1, 2)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention implementation similar to Granite/LFM2.
    Supports both multi-head attention and grouped query attention patterns.
    """

    # Attention span in positions; None attends to the whole prefix
    window = None

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_kv_heads == 0

        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads

        # Query, Key, Value projections - all use same head_dim for compatibility
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.attention_dropout = config.attention_probs_dropout_prob
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta) if config.use_rope else None

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.rotary is not None:
            query_states, key_states = self.rotary(query_states, key_states)

        attn_output = causal_attention(
            query_states, key_states, value_states, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0, window=self.window,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        return self.o_proj(attn_output)


class SlidingWindowAttention(GroupedQueryAttention):
    """
    Grouped-query attention over only the last config.sliding_window_size
    positions. Small windows (64 in BASED) give exact local recall at linear cost.
    """

    def __init__(self, config):
        super().__init__(config)
        self.window = config.sliding_window_size


class ConfigMamba2Mixer(Mamba2Mixer):
    """The Mamba-2 mixer from mamba2.py, built from a MixedLayerConfig."""

    def __init__(self, config):
        super().__init__(
            config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
        )

    def forward(self, hidden_states, attention_mask=None):
        # Right padding can't affect earlier positions in a causal scan, so the mask isn't needed
        return super().forward(hidden_states)


class BasedMixer(nn.Module):
    """
    BASED linear attention (Arora et al. 2024, arXiv:2402.18668).

    Queries and keys are projected to based_feature_dim per head and expanded
    by the 2nd-order Taylor feature map, so phi(q) . phi(k) approximates
    exp(q . k / sqrt(d)). Attention is the normalized causal linear form

        y_i = sum_{j<=i} phi(q_i) . phi(k_j) v_j / sum_{j<=i} phi(q_i) . phi(k_j)

    computed in chunks (see linear_attention.py), so cost is linear in the
    sequence length and generation keeps a fixed-size state. The BASED paper
    pairs these layers with small sliding_window_attention layers (and short
    convolutions) for exact local recall; build that as a layer_types pattern.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_dim = config.based_feature_dim
        self.num_heads = config.based_num_heads
        self.num_key_value_heads = config.based_num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = config.based_chunk_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.feature_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.feature_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # The feature map squares its inputs and the scan sums over the sequence, so run both in float32
        compute_dtype = torch.promote_types(v.dtype, torch.float32)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            q = taylor_feature_map(q.to(compute_dtype))
            k = taylor_feature_map(k.to(compute_dtype))
            v = v.to(compute_dtype)
            if attention_mask is not None:
                # The feature map has a constant term, so padded keys must be zeroed explicitly
                keep = attention_mask[:, None, :, None].to(compute_dtype)
                k, v = k * keep, v * keep

            groups = self.num_heads // self.num_key_value_heads
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)

            y, _ = causal_linear_attention(q, k, v, self.chunk_size)

        y = y.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(y.to(hidden_states.dtype))


class MixtureOfHeadAttention(nn.Module):
    """
    Mixture-of-Head (MoH) Attention implementation.

    Treats attention heads as experts in a Mixture-of-Experts framework,
    allowing dynamic head selection for improved efficiency while maintaining
    or improving performance.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        # MoH specific parameters
        self.moh_top_k = getattr(config, 'moh_top_k', max(1, self.num_attention_heads // 2))
        self.moh_temperature = getattr(config, 'moh_temperature', 1.0)
        self.moh_noise_std = getattr(config, 'moh_noise_std', 0.1)

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_kv_heads == 0

        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads

        # Standard attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # MoH router for head selection
        self.head_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_attention_heads),
        )

        self.attention_dropout = config.attention_probs_dropout_prob
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta) if config.use_rope else None

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Compute head selection scores per token so routing at position t
        # only sees position t (pooling over the sequence would leak the future)
        head_scores = self.head_router(hidden_states)  # [batch_size, seq_len, num_attention_heads]

        # Add noise for exploration during training
        if self.training and self.moh_noise_std > 0:
            noise = torch.randn_like(head_scores) * self.moh_noise_std
            head_scores = head_scores + noise

        # Select top-k heads per token and scatter their mixture weights into a
        # sparse [batch_size, seq_len, num_attention_heads] tensor
        head_scores = head_scores / self.moh_temperature
        top_k_scores, top_k_indices = torch.topk(head_scores, self.moh_top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)  # [batch_size, seq_len, top_k]
        head_weights = torch.zeros_like(head_scores).scatter(-1, top_k_indices, top_k_weights)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.rotary is not None:
            q, k = self.rotary(q, k)

        # Since heads are selected per token, compute attention for all heads
        # (Q head i uses KV head i // num_kv_groups)
        attn_output = causal_attention(
            q, k, v, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Scale each head's output by its gate (0 for heads not selected at that
        # token) and concatenate the heads, so O = sum_h g_h * head_h @ W_O^h as in
        # the MoH paper. Every selected head keeps its own subspace.
        attn_output = attn_output * head_weights.transpose(1, 2).unsqueeze(-1)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)

        # Apply output projection
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Feed-forward networks: the per-position part of a block.
# ---------------------------------------------------------------------------


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA/PaLM.
    More efficient than GELU-based FFNs.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: gate * SiLU(up) where gate and up are different projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


def swiglu_hidden_size(intermediate_size: int) -> int:
    """
    SwiGLU width with the parameters of a two-matrix FFN of width intermediate_size.
    SwiGLU has three matrices, so it gets 2/3 of the width (as in LLaMA).
    """
    return int(intermediate_size * 2 / 3)


class SharedExpertMoE(nn.Module):
    """
    Shared Expert Mixture-of-Experts implementation inspired by Granite 4.

    Combines shared experts (always active) with routing experts (conditionally active)
    for improved parameter efficiency and stable training.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.aux_loss = None

        # MoE configuration
        self.num_experts = getattr(config, 'num_experts', 8)
        self.num_experts_per_token = getattr(config, 'num_experts_per_token', 2)
        self.num_shared_experts = getattr(config, 'num_shared_experts', 2)
        self.moe_top_k = getattr(config, 'moe_top_k', self.num_experts_per_token)
        self.moe_temperature = getattr(config, 'moe_temperature', 1.0)
        self.moe_noise_std = getattr(config, 'moe_noise_std', 0.1)
        self.load_balancing_weight = getattr(config, 'moe_load_balancing_weight', 0.01)

        # Shared experts - always active
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                self._create_expert() for _ in range(self.num_shared_experts)
            ])

        # Routing experts - conditionally active based on router
        if self.num_experts > 0:
            self.routing_experts = nn.ModuleList([
                self._create_expert() for _ in range(self.num_experts)
            ])

            # Router for expert selection
            self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def _create_expert(self):
        """Create a single expert (FFN) with SwiGLU activation."""
        return SwiGLUFeedForward(self.hidden_size, swiglu_hidden_size(self.intermediate_size))

    def route(self, tokens):
        """
        Router probabilities and each token's top-k experts.

        tokens is (num_tokens, hidden). Returns the probabilities over all experts
        (num_tokens, num_experts), and the top-k expert indices with their weights
        renormalized to sum to 1 (both (num_tokens, top_k)).
        """
        router_logits = self.router(tokens)

        # Add noise for exploration during training (like Switch Transformer)
        if self.training and self.moe_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.moe_noise_std

        router_probs = torch.softmax(router_logits / self.moe_temperature, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.moe_top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return router_probs, top_k_weights, top_k_indices

    def load_balancing_loss(self, router_probs, top_k_indices, token_weights=None):
        """
        Switch Transformer auxiliary loss, moe_load_balancing_weight * N * sum_i f_i * P_i,
        where f_i is the fraction of top-k assignments that go to expert i and P_i
        is the mean router probability of expert i. It equals the weight when
        routing is uniform and grows as routing concentrates; the gradient reaches
        the router through P_i. token_weights (num_tokens,) excludes padding.
        """
        if token_weights is None:
            token_weights = torch.ones(router_probs.size(0), device=router_probs.device)
        token_weights = token_weights.to(router_probs.dtype)
        num_tokens = token_weights.sum().clamp(min=1)
        assignments = F.one_hot(top_k_indices, self.num_experts).sum(dim=1).to(router_probs.dtype)
        fraction_routed = (assignments * token_weights[:, None]).sum(0) / (num_tokens * self.moe_top_k)
        mean_probability = (router_probs * token_weights[:, None]).sum(0) / num_tokens
        return self.load_balancing_weight * self.num_experts * (fraction_routed * mean_probability).sum()

    def dispatch(self, tokens, top_k_weights, top_k_indices):
        """
        Sum of each token's top-k expert outputs, weighted by the router.

        Assignments are sorted by expert, so each expert runs once on a contiguous
        batch of its tokens, and the results are scattered back with index_add_.
        """
        num_tokens, top_k = top_k_indices.shape
        flat_indices = top_k_indices.reshape(-1)
        order = flat_indices.argsort(stable=True)
        token_of_assignment = torch.arange(num_tokens, device=tokens.device).repeat_interleave(top_k)[order]
        weight_of_assignment = top_k_weights.reshape(-1)[order]
        counts = torch.bincount(flat_indices, minlength=self.num_experts).tolist()

        output = torch.zeros_like(tokens)
        for expert, token_idx, weight in zip(self.routing_experts, token_of_assignment.split(counts),
                                             weight_of_assignment.split(counts)):
            if token_idx.numel():
                expert_output = expert(tokens[token_idx]) * weight[:, None]
                output.index_add_(0, token_idx, expert_output.to(output.dtype))
        return output

    def forward(self, hidden_states, attention_mask=None):
        """
        Shared experts plus routed experts. Sets self.aux_loss to the load-balancing
        loss (None without routed experts); the model adds it to the training loss.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        expert_output = torch.zeros_like(hidden_states)

        # Shared experts (always active)
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                expert_output = expert_output + shared_expert(hidden_states) / self.num_shared_experts

        self.aux_loss = None
        if self.num_experts > 0:
            tokens = hidden_states.reshape(-1, hidden_size)
            router_probs, top_k_weights, top_k_indices = self.route(tokens)
            routed = self.dispatch(tokens, top_k_weights, top_k_indices)
            expert_output = expert_output + routed.view(batch_size, seq_len, hidden_size)

            token_weights = None if attention_mask is None else attention_mask[:, -seq_len:].reshape(-1)
            self.aux_loss = self.load_balancing_loss(router_probs, top_k_indices, token_weights)

        return expert_output


# ---------------------------------------------------------------------------
# The block shared by every layer type
# ---------------------------------------------------------------------------

# Mixer class per layer type. full_attention and flash_attention are the same
# mixer: F.scaled_dot_product_attention already picks FlashAttention kernels
# where it can. Types mapped to None have no mixer (see MIXERLESS_LAYER_TYPES).
MIXERS = {
    "full_attention": GroupedQueryAttention,
    "conv": ConvMixer,
    "mlp_only": None,
    "mamba2": ConfigMamba2Mixer,
    "based": BasedMixer,
    "flash_attention": GroupedQueryAttention,
    "moh_attention": MixtureOfHeadAttention,
    "shared_expert_moe": None,
    "sliding_window_attention": SlidingWindowAttention,
}
assert tuple(MIXERS) == SUPPORTED_LAYER_TYPES
assert {t for t, mixer in MIXERS.items() if mixer is None} == set(MIXERLESS_LAYER_TYPES)


class MixedTransformerLayer(nn.Module):
    """
    One pre-norm residual block. Every layer type has the same shape:

        h = x + mixer(norm(x))     skipped for layer types without a mixer
        y = h + ffn(norm(h))       skipped when the layer's FFN type is "none"

    Only the mixer differs between layer types, so comparing them isn't
    confounded by norm placement or a missing FFN. The FFN is chosen per layer
    by config.ffn_types (SwiGLU of width config.ffn_hidden_sizes[i], MoE, or none).
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.ffn_type = config.ffn_types[layer_idx]
        self.gradient_checkpointing = False

        mixer_class = MIXERS[self.layer_type]
        self.mixer_norm = get_norm_layer(config) if mixer_class is not None else None
        self.mixer = mixer_class(config) if mixer_class is not None else None

        if self.ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(config.hidden_size, config.ffn_hidden_sizes[layer_idx])
        elif self.ffn_type == "moe":
            self.ffn = SharedExpertMoE(config)
        else:
            self.ffn = None
        self.ffn_norm = get_norm_layer(config) if self.ffn is not None else None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        if self.mixer is not None:
            hidden_states = hidden_states + self.dropout(self.mixer(self.mixer_norm(hidden_states), attention_mask))
        if self.ffn_type == "moe":
            # The mask keeps padding out of the load-balancing statistics
            hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states), attention_mask))
        elif self.ffn is not None:
            hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states

    @property
    def aux_loss(self):
        """The MoE load-balancing loss from the last forward pass, or None for other FFNs."""
        return self.ffn.aux_loss if self.ffn_type == "moe" else None


def _run_layer(layer, hidden_states, attention_mask):
    hidden_states = layer(hidden_states, attention_mask)
    return hidden_states, layer.aux_loss


def param_matched_ffn_hidden_sizes(config, reference_layer_type: str = "full_attention") -> List[int]:
    """
    Per-layer SwiGLU widths that give every block about the same parameter count.

    The target is a `reference_layer_type` block with the config's default
    SwiGLU width. Each SwiGLU layer's width is then set so that its mixer, norms
    and FFN add up to the target, to within one unit of width (3 * hidden_size
    parameters). Layers whose FFN is MoE or none keep their current width.
    Pass the result as `ffn_hidden_sizes`.

    Raises ValueError if a layer's mixer alone is already larger than the target.
    """
    if reference_layer_type not in MIXERS:
        raise ValueError(f"Unknown reference layer type {reference_layer_type!r}. "
                         f"Supported types: {list(SUPPORTED_LAYER_TYPES)}")

    norm_params = sum(p.numel() for p in get_norm_layer(config).parameters())
    params_per_unit = 3 * config.hidden_size  # one unit of SwiGLU width: gate, up and down rows

    def non_ffn_params(layer_type):
        """Parameters of a block of this type apart from the FFN (its mixer and both norms)."""
        mixer_class = MIXERS[layer_type]
        if mixer_class is None:
            return norm_params
        with torch.device("meta"):
            mixer = mixer_class(config)
        return sum(p.numel() for p in mixer.parameters()) + 2 * norm_params

    target = non_ffn_params(reference_layer_type) + params_per_unit * swiglu_hidden_size(config.intermediate_size)

    sizes = []
    for i, (layer_type, ffn_type) in enumerate(zip(config.layer_types, config.ffn_types)):
        if ffn_type != "swiglu":
            sizes.append(config.ffn_hidden_sizes[i])
            continue
        width = round((target - non_ffn_params(layer_type)) / params_per_unit)
        if width < 1:
            raise ValueError(f"Layer {i} ({layer_type}) has more parameters without its FFN than a whole "
                             f"{reference_layer_type} block, so no FFN width can match it")
        sizes.append(width)
    return sizes


def block_parameter_counts(config) -> List[int]:
    """Parameter count of each block, built on the meta device so it costs no memory."""
    with torch.device("meta"):
        return [sum(p.numel() for p in MixedTransformerLayer(config, i).parameters())
                for i in range(config.num_hidden_layers)]


class MixedLayerForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Mixed Layer Transformer model for causal language modeling.

    This model supports heterogeneous layer types within the same architecture,
    similar to Granite and LFM2 models that can mix attention, convolutional,
    and other layer types.
    """
    config_class = MixedLayerConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (nn.Embedding(config.max_position_embeddings, config.hidden_size)
                                    if config.use_absolute_position_embeddings else None)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Mixed transformer layers
        self.layers = nn.ModuleList([
            MixedTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer norm and output projection
        self.final_layer_norm = get_norm_layer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and tie lm_head to the token embeddings when
        # config.tie_word_embeddings is set, matching what from_pretrained does
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing for the model."""
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation."""
        return {"input_ids": input_ids}

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search (not implemented for simplicity)."""
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        hidden_states = inputs_embeds
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        # Pass through mixed transformer layers, collecting MoE load-balancing losses
        aux_loss = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # The aux loss is returned from inside the checkpoint, so its graph is recomputed with the layer's
                hidden_states, layer_aux_loss = torch.utils.checkpoint.checkpoint(
                    _run_layer, layer, hidden_states, attention_mask, use_reentrant=False
                )
            else:
                hidden_states, layer_aux_loss = _run_layer(layer, hidden_states, attention_mask)
            if layer_aux_loss is not None:
                aux_loss = layer_aux_loss if aux_loss is None else aux_loss + layer_aux_loss

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # The load-balancing loss trains the routers. Eval loss leaves it out so it stays perplexity
            if aux_loss is not None and self.training:
                loss = loss + aux_loss

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=None,  # Not implemented for simplicity
            hidden_states=None,
            attentions=None,
        )