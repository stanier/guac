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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from typing import Optional, Tuple, Union, List, Dict, Any

from .mamba2 import Mamba2Mixer


SUPPORTED_LAYER_TYPES = (
    "full_attention", "conv", "mlp_only", "mamba2", "based",
    "flash_attention", "moh_attention", "shared_expert_moe",
)


def causal_attention(query, key, value, attention_mask=None, dropout_p=0.0):
    """
    Causal scaled dot-product attention shared by every softmax-attention layer.

    query is (batch, heads, seq, head_dim); key and value are
    (batch, kv_heads, seq, head_dim) with heads a multiple of kv_heads
    (grouped-query attention). attention_mask is an optional (batch, seq)
    key padding mask with 1 for real tokens and 0 for padding.

    F.scaled_dot_product_attention picks a FlashAttention or memory-efficient
    kernel when the device and dtype support it, and the math kernel otherwise.
    """
    num_kv_groups = query.size(1) // key.size(1)
    if num_kv_groups > 1:
        key = key.repeat_interleave(num_kv_groups, dim=1)
        value = value.repeat_interleave(num_kv_groups, dim=1)

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
        layer_types=None,
        conv_kernel_size=3,
        conv_groups=1,
        # Mamba-2 specific parameters (defaults follow the Mamba-2 paper)
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        mamba_ngroups=1,
        mamba_chunk_size=64,  # Scan chunk length; affects speed and memory, not results
        # BASED specific parameters
        based_feature_dim=16,
        based_num_key_value_heads=None,  # Will default to num_attention_heads
        based_num_heads=None,            # Will default to num_attention_heads
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

        # Rotary Position Embedding parameters
        self.use_rope = kwargs.get('use_rope', True)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        self.rope_scaling = kwargs.get('rope_scaling', None)

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

        if "mamba2" in self.layer_types:
            mamba_inner = int(mamba_expand * hidden_size)
            if mamba_inner % mamba_headdim:
                raise ValueError(f"Mamba-2 inner size {mamba_inner} (mamba_expand * hidden_size) must be "
                                 f"divisible by mamba_headdim {mamba_headdim}")
            if (mamba_inner // mamba_headdim) % mamba_ngroups:
                raise ValueError(f"Mamba-2 head count {mamba_inner // mamba_headdim} must be divisible by "
                                 f"mamba_ngroups {mamba_ngroups}")


class ConvolutionalLayer(nn.Module):
    """
    Convolutional layer implementation similar to those used in LFM2.
    Uses 1D convolution for sequence modeling.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Convolutional transformation
        self.conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            # Causal: pad k-1 on both sides, then keep only the first seq_len outputs
            padding=config.conv_kernel_size - 1,
            groups=config.conv_groups
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # Layer normalization
        self.input_layer_norm = get_norm_layer(config)
        self.output_layer_norm = get_norm_layer(config)

    def forward(self, hidden_states, attention_mask=None):
        # Apply input layer norm
        normed_states = self.input_layer_norm(hidden_states)

        # Convolutional processing
        # Conv1d expects (batch, channels, sequence) format
        conv_input = normed_states.transpose(1, 2)  # (batch, hidden_size, seq_len)
        conv_output = self.conv1d(conv_input)[:, :, :hidden_states.shape[1]]
        conv_output = conv_output.transpose(1, 2)  # back to (batch, seq_len, hidden_size)

        # Apply residual connection
        hidden_states = hidden_states + conv_output

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.output_layer_norm(hidden_states))
        hidden_states = hidden_states + ff_output

        return hidden_states


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention implementation similar to Granite/LFM2.
    Supports both multi-head attention and grouped query attention patterns.
    """

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

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_output = causal_attention(
            query_states, key_states, value_states, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        return self.o_proj(attn_output)


class Mamba2Layer(nn.Module):
    """
    Pre-norm residual block around a Mamba-2 mixer (see mamba2.py). Like the
    original Mamba-2 blocks, it has no separate feed-forward network.
    """

    def __init__(self, config):
        super().__init__()
        self.norm = get_norm_layer(config)
        self.mixer = Mamba2Mixer(
            config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
        )

    def forward(self, hidden_states, attention_mask=None):
        return hidden_states + self.mixer(self.norm(hidden_states))


class BasedLayer(nn.Module):
    """
    BASED (Bidirectional Attention with Sparse Efficient Decoding) layer implementation.
    Uses bidirectional attention with feature maps for efficient computation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.feature_dim = config.based_feature_dim
        self.num_heads = config.based_num_heads
        self.num_key_value_heads = config.based_num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert self.hidden_size % self.num_heads == 0

        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # Feature map projections for efficient attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.feature_dim),
            nn.ReLU()
        )

        # Output projection
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Layer norms
        self.input_norm = get_norm_layer(config, self.hidden_size)
        self.output_norm = get_norm_layer(config, self.hidden_size)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Input normalization
        hidden_states = self.input_norm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Expand k, v for grouped attention if needed
        if self.num_heads != self.num_key_value_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # Apply feature maps to queries and keys
        q_features = self.feature_map(q)  # (B, H, L, F)
        k_features = self.feature_map(k)  # (B, H, L, F)

        # BASED attention using feature maps
        # Compute attention via feature map dot products
        attn_output = self._based_attention(q_features, k_features, v, attention_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # First residual connection
        hidden_states = residual + attn_output

        # Feed-forward with second residual connection
        residual = hidden_states
        hidden_states = self.output_norm(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states

    def _based_attention(self, q_features, k_features, v, attention_mask=None):
        """BASED attention computation using feature maps."""
        batch_size, num_heads, seq_len, feature_dim = q_features.shape

        # Compute cumulative sums for efficient attention
        # This is a simplified version - real BASED uses more sophisticated feature maps
        k_cumsum = torch.cumsum(k_features, dim=2)  # (B, H, L, F)

        # Compute attention scores via feature map
        scores = torch.einsum('bhif,bhjf->bhij', q_features, k_cumsum)

        # Apply causal mask with safe large negative value instead of -inf
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q_features.device))
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, -1e4)  # Safe large negative

        # Apply attention mask if provided
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -1e4  # Safe large negative
            scores = scores + extended_mask

        # Clamp scores to prevent extreme values
        scores = torch.clamp(scores, min=-1e4, max=1e4)

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


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

        # Layer norms
        self.input_norm = get_norm_layer(config, self.hidden_size)
        self.output_norm = get_norm_layer(config, self.hidden_size)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.attention_dropout = config.attention_probs_dropout_prob

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Input normalization
        hidden_states = self.input_norm(hidden_states)

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

        # Since heads are selected per token, compute attention for all heads
        # (Q head i uses KV head i // num_kv_groups)
        attn_output = causal_attention(
            q, k, v, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Combine heads with per-token sparse mixture weights (unselected heads get 0)
        attn_output = torch.einsum('bhld,blh->bld', attn_output, head_weights)  # [batch_size, seq_len, head_dim]

        # The result should be [batch_size, seq_len, head_dim * top_k] but we want [batch_size, seq_len, hidden_size]
        # We need to carefully handle the output projection to map from reduced dimensions to full hidden_size
        current_dim = attn_output.size(-1)

        if current_dim != self.hidden_size:
            # Create a temporary projection that maps from current_dim to hidden_size
            # In practice, this would be learned, but for now we'll use a subset of the output projection
            scale_factor = self.hidden_size / current_dim
            attn_output = attn_output * scale_factor  # Scale to compensate for reduced heads

            # Repeat across the head dimension to restore full size
            repeat_count = self.hidden_size // current_dim
            remainder = self.hidden_size % current_dim

            if repeat_count > 0:
                repeated = attn_output.repeat(1, 1, repeat_count)
                if remainder > 0:
                    attn_output = torch.cat([repeated, attn_output[:, :, :remainder]], dim=-1)
                else:
                    attn_output = repeated

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        # First residual connection
        hidden_states = residual + attn_output

        # Feed-forward with second residual connection
        residual = hidden_states
        hidden_states = self.output_norm(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states


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

        # MoE configuration
        self.num_experts = getattr(config, 'num_experts', 8)
        self.num_experts_per_token = getattr(config, 'num_experts_per_token', 2)
        self.num_shared_experts = getattr(config, 'num_shared_experts', 2)
        self.moe_top_k = getattr(config, 'moe_top_k', self.num_experts_per_token)
        self.moe_temperature = getattr(config, 'moe_temperature', 1.0)
        self.moe_noise_std = getattr(config, 'moe_noise_std', 0.1)
        self.load_balancing_weight = getattr(config, 'moe_load_balancing_weight', 0.01)

        # Input and output layer norms
        self.input_norm = get_norm_layer(config, self.hidden_size)
        self.output_norm = get_norm_layer(config, self.hidden_size)

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

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _create_expert(self):
        """Create a single expert (FFN) with SwiGLU activation."""
        # SwiGLU requires 2/3 intermediate size for same params due to gate
        gate_size = int(self.intermediate_size * 2 / 3)
        return SwiGLUFeedForward(self.hidden_size, gate_size)

    def _compute_routing_weights(self, hidden_states):
        """Compute routing weights with load balancing."""
        batch_size, seq_len, _ = hidden_states.shape

        # Router logits
        router_logits = self.router(hidden_states)  # [batch, seq, num_experts]

        # Add noise for training stability (like Switch Transformer)
        if self.training and self.moe_noise_std > 0:
            noise = torch.randn_like(router_logits) * self.moe_noise_std
            router_logits = router_logits + noise

        # Apply temperature
        router_logits = router_logits / self.moe_temperature

        # Compute routing probabilities
        routing_weights = torch.softmax(router_logits, dim=-1)

        # Top-k expert selection
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.moe_top_k, dim=-1
        )

        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Load balancing loss (encourage balanced expert usage)
        if self.training:
            # Auxiliary loss to encourage equal expert usage
            router_probs = torch.softmax(router_logits, dim=-1)
            expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]
            balance_loss = self.load_balancing_weight * (
                expert_usage * expert_usage.numel()
            ).sum()
            self.balance_loss = balance_loss
        else:
            self.balance_loss = 0.0

        return top_k_weights, top_k_indices, routing_weights

    def forward(self, hidden_states):
        """Forward pass through shared + routed experts."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Store residual
        residual = hidden_states

        # Input normalization
        hidden_states = self.input_norm(hidden_states)

        # Initialize output
        expert_output = torch.zeros_like(hidden_states)

        # Shared experts (always active)
        if self.num_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(hidden_states)
                expert_output = expert_output + shared_output / self.num_shared_experts

        # Routing experts (conditionally active)
        if self.num_experts > 0:
            # Compute routing
            top_k_weights, top_k_indices, _ = self._compute_routing_weights(hidden_states)

            # Flatten for expert processing
            flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
            flat_top_k_weights = top_k_weights.view(-1, self.moe_top_k)  # [batch*seq, top_k]
            flat_top_k_indices = top_k_indices.view(-1, self.moe_top_k)  # [batch*seq, top_k]

            # Process through selected experts
            flat_expert_output = torch.zeros_like(flat_hidden)

            for i, expert in enumerate(self.routing_experts):
                # Find tokens that use this expert
                expert_mask = (flat_top_k_indices == i).any(dim=-1)

                if expert_mask.any():
                    # Get tokens for this expert
                    expert_tokens = flat_hidden[expert_mask]

                    # Process through expert
                    expert_result = expert(expert_tokens)

                    # Get weights for this expert
                    token_indices = torch.where(expert_mask)[0]
                    for j, token_idx in enumerate(token_indices):
                        # Find weight for this expert for this token
                        expert_positions = (flat_top_k_indices[token_idx] == i).nonzero().squeeze(-1)
                        if len(expert_positions) > 0:
                            weight = flat_top_k_weights[token_idx, expert_positions[0]]
                            flat_expert_output[token_idx] += weight * expert_result[j]

            # Reshape back and add to output
            routing_output = flat_expert_output.view(batch_size, seq_len, hidden_size)
            expert_output = expert_output + routing_output

        # Apply dropout
        expert_output = self.dropout(expert_output)

        # Residual connection and output normalization
        output = self.output_norm(expert_output + residual)

        return output


class FlashAttentionLayer(nn.Module):
    """
    Pre-norm attention + feed-forward block. Attention runs through
    F.scaled_dot_product_attention, which uses FlashAttention kernels on
    supported GPUs and dtypes and falls back to the math kernel elsewhere.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0

        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Layer norms
        self.input_norm = get_norm_layer(config, self.hidden_size)
        self.output_norm = get_norm_layer(config, self.hidden_size)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.attention_dropout = config.attention_probs_dropout_prob

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Input normalization
        hidden_states = self.input_norm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_output = causal_attention(
            q, k, v, attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # First residual connection
        hidden_states = residual + attn_output

        # Feed-forward with second residual connection
        residual = hidden_states
        hidden_states = self.output_norm(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states


class MixedTransformerLayer(nn.Module):
    """
    A transformer layer that can use different layer types based on configuration.
    Supports 'full_attention', 'conv', 'mlp_only', etc.
    """

    def __init__(self, config, layer_type):
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.gradient_checkpointing = False

        if layer_type == "full_attention":
            self.attention = GroupedQueryAttention(config)
            self.attention_layer_norm = get_norm_layer(config)

            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
            self.output_layer_norm = get_norm_layer(config)

        elif layer_type == "conv":
            self.conv_layer = ConvolutionalLayer(config)

        elif layer_type == "mlp_only":
            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
            self.layer_norm = get_norm_layer(config)

        elif layer_type == "mamba2":
            self.mamba_layer = Mamba2Layer(config)

        elif layer_type == "based":
            self.based_layer = BasedLayer(config)

        elif layer_type == "flash_attention":
            self.flash_attn_layer = FlashAttentionLayer(config)

        elif layer_type == "moh_attention":
            self.moh_layer = MixtureOfHeadAttention(config)

        elif layer_type == "shared_expert_moe":
            self.moe_layer = SharedExpertMoE(config)

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {list(SUPPORTED_LAYER_TYPES)}")

    def forward(self, hidden_states, attention_mask=None):
        if self.layer_type == "full_attention":
            # Self-attention with residual connection
            attention_output = self.attention(hidden_states, attention_mask)
            attention_output = self.attention_layer_norm(hidden_states + attention_output)

            # Feed-forward with residual connection
            feed_forward_output = self.feed_forward(attention_output)
            layer_output = self.output_layer_norm(attention_output + feed_forward_output)

            return layer_output

        elif self.layer_type == "conv":
            return self.conv_layer(hidden_states, attention_mask)

        elif self.layer_type == "mlp_only":
            normed_states = self.layer_norm(hidden_states)
            ff_output = self.feed_forward(normed_states)
            return hidden_states + ff_output

        elif self.layer_type == "mamba2":
            return self.mamba_layer(hidden_states, attention_mask)

        elif self.layer_type == "based":
            return self.based_layer(hidden_states, attention_mask)

        elif self.layer_type == "flash_attention":
            return self.flash_attn_layer(hidden_states, attention_mask)

        elif self.layer_type == "moh_attention":
            return self.moh_layer(hidden_states, attention_mask)

        elif self.layer_type == "shared_expert_moe":
            return self.moe_layer(hidden_states)


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
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Mixed transformer layers
        self.layers = nn.ModuleList([
            MixedTransformerLayer(config, layer_type)
            for layer_type in config.layer_types
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

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        # Pass through mixed transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
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