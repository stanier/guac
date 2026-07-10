#!/usr/bin/env python3
"""
Matryoshka Representation Learning for Guac.

Implementation based on "Matryoshka Representation Learning" (Kusupati et al., 2022)
and "MRL: Multi-Resolution Learning" papers.

This allows models to work at multiple representation dimensions during inference,
enabling flexible efficiency trade-offs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import math


class MatryoshkaEmbedding(nn.Module):
    """
    Matryoshka-capable embedding layer that supports multiple representation dimensions.

    The key insight is that we can slice the embedding to different dimensions
    and still get meaningful representations for downstream tasks.
    """

    def __init__(self, vocab_size: int, max_hidden_size: int, nesting_list: List[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_hidden_size = max_hidden_size

        # Default nesting list follows powers of 2, similar to the paper
        if nesting_list is None:
            self.nesting_list = [64, 128, 256, 512, max_hidden_size]
        else:
            self.nesting_list = sorted(nesting_list)

        # Ensure max dimension is included
        if max_hidden_size not in self.nesting_list:
            self.nesting_list.append(max_hidden_size)
            self.nesting_list = sorted(self.nesting_list)

        # Main embedding layer
        self.embedding = nn.Embedding(vocab_size, max_hidden_size)

        # Optional: Add dimension-specific normalization layers
        self.dimension_norms = nn.ModuleDict({
            str(dim): nn.LayerNorm(dim) for dim in self.nesting_list
        })

    def forward(self, input_ids: torch.Tensor, representation_dim: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with optional dimension specification.

        Args:
            input_ids: Token IDs
            representation_dim: Desired representation dimension. If None, uses max_hidden_size

        Returns:
            Embeddings of shape (batch_size, seq_len, representation_dim)
        """
        if representation_dim is None:
            representation_dim = self.max_hidden_size

        # Get full embeddings
        embeddings = self.embedding(input_ids)

        # Slice to desired dimension
        if representation_dim < self.max_hidden_size:
            embeddings = embeddings[..., :representation_dim]

        # Apply dimension-specific normalization if available
        if str(representation_dim) in self.dimension_norms:
            embeddings = self.dimension_norms[str(representation_dim)](embeddings)

        return embeddings


class MatryoshkaLinear(nn.Module):
    """
    Linear layer that supports Matryoshka representations.

    This layer can efficiently compute outputs for multiple representation dimensions
    by organizing the weight matrix appropriately.
    """

    def __init__(self, max_in_features: int, out_features: int, nesting_list: List[int] = None):
        super().__init__()
        self.max_in_features = max_in_features
        self.out_features = out_features

        if nesting_list is None:
            self.nesting_list = [64, 128, 256, 512, max_in_features]
        else:
            self.nesting_list = sorted(nesting_list)

        if max_in_features not in self.nesting_list:
            self.nesting_list.append(max_in_features)
            self.nesting_list = sorted(self.nesting_list)

        # Main linear layer
        self.linear = nn.Linear(max_in_features, out_features)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, representation_dim: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with optional dimension specification.

        Args:
            x: Input tensor of shape (..., representation_dim)
            representation_dim: Input dimension. If None, uses max_in_features

        Returns:
            Output tensor of shape (..., out_features)
        """
        if representation_dim is None:
            representation_dim = self.max_in_features

        # Use only the relevant portion of the weight matrix
        weight_slice = self.linear.weight[:, :representation_dim]

        # Compute output
        output = F.linear(x, weight_slice, self.bias)

        return output


class MatryoshkaAttention(nn.Module):
    """
    Multi-head attention that supports Matryoshka representations.

    This allows attention computation at different representation dimensions
    while maintaining the attention mechanism's effectiveness.
    """

    def __init__(self, config, nesting_list: List[int] = None):
        super().__init__()
        self.config = config
        self.max_hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_kv_heads', self.num_attention_heads)

        if nesting_list is None:
            self.nesting_list = [64, 128, 256, 512, self.max_hidden_size]
        else:
            self.nesting_list = sorted(nesting_list)

        if self.max_hidden_size not in self.nesting_list:
            self.nesting_list.append(self.max_hidden_size)
            self.nesting_list = sorted(self.nesting_list)

        # Ensure all nesting dimensions are divisible by number of heads
        self.valid_nesting_list = []
        for dim in self.nesting_list:
            if dim % self.num_attention_heads == 0:
                self.valid_nesting_list.append(dim)

        if not self.valid_nesting_list:
            # Fallback: use dimensions that work with the head count
            self.valid_nesting_list = [
                h * self.num_attention_heads
                for h in [4, 8, 16, 32, 64]
                if h * self.num_attention_heads <= self.max_hidden_size
            ]
            self.valid_nesting_list.append(self.max_hidden_size)

        # Projection layers
        self.q_proj = MatryoshkaLinear(self.max_hidden_size, self.max_hidden_size, self.valid_nesting_list)
        self.k_proj = MatryoshkaLinear(self.max_hidden_size, self.max_hidden_size, self.valid_nesting_list)
        self.v_proj = MatryoshkaLinear(self.max_hidden_size, self.max_hidden_size, self.valid_nesting_list)
        self.o_proj = MatryoshkaLinear(self.max_hidden_size, self.max_hidden_size, self.valid_nesting_list)

        self.dropout = nn.Dropout(getattr(config, 'attention_probs_dropout_prob', 0.1))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                representation_dim: Optional[int] = None) -> torch.Tensor:
        """
        Matryoshka attention forward pass.

        Args:
            hidden_states: Input of shape (batch_size, seq_len, representation_dim)
            attention_mask: Attention mask
            representation_dim: Desired representation dimension

        Returns:
            Attention output of shape (batch_size, seq_len, representation_dim)
        """
        if representation_dim is None:
            representation_dim = self.max_hidden_size

        # Ensure representation_dim is valid
        if representation_dim not in self.valid_nesting_list:
            # Find the largest valid dimension <= representation_dim
            valid_dims = [d for d in self.valid_nesting_list if d <= representation_dim]
            if valid_dims:
                representation_dim = max(valid_dims)
            else:
                representation_dim = min(self.valid_nesting_list)

        batch_size, seq_len, _ = hidden_states.shape
        head_dim = representation_dim // self.num_attention_heads

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states, representation_dim)
        key_states = self.k_proj(hidden_states, representation_dim)
        value_states = self.v_proj(hidden_states, representation_dim)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        attention_output = torch.matmul(attention_probs, value_states)

        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, representation_dim)

        # Final projection
        attention_output = self.o_proj(attention_output, representation_dim)

        return attention_output


class MatryoshkaLoss(nn.Module):
    """
    Loss function for Matryoshka Representation Learning.

    This computes losses at multiple representation dimensions and combines them
    with appropriate weights to encourage good representations at all scales.
    """

    def __init__(self, nesting_list: List[int], loss_weights: Optional[List[float]] = None,
                 base_loss_fn: nn.Module = None):
        super().__init__()
        self.nesting_list = sorted(nesting_list)

        # Default to equal weights if not specified
        if loss_weights is None:
            self.loss_weights = [1.0] * len(nesting_list)
        else:
            assert len(loss_weights) == len(nesting_list), "Loss weights must match nesting list length"
            self.loss_weights = loss_weights

        # Default to CrossEntropyLoss if not specified
        if base_loss_fn is None:
            self.base_loss_fn = nn.CrossEntropyLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def forward(self, logits_dict: Dict[int, torch.Tensor], labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute Matryoshka loss across multiple representation dimensions.

        Args:
            logits_dict: Dictionary mapping dimension -> logits tensor
            labels: Ground truth labels

        Returns:
            total_loss: Combined loss across all dimensions
            individual_losses: Dictionary of losses per dimension
        """
        individual_losses = {}
        total_loss = 0.0

        for i, dim in enumerate(self.nesting_list):
            if dim in logits_dict:
                loss = self.base_loss_fn(logits_dict[dim], labels)
                individual_losses[dim] = loss
                total_loss += self.loss_weights[i] * loss

        return total_loss, individual_losses


class MatryoshkaLMHead(nn.Module):
    """
    Language modeling head that supports Matryoshka representations.

    This can compute logits for multiple representation dimensions simultaneously
    during training, or efficiently for a single dimension during inference.
    """

    def __init__(self, config, nesting_list: List[int] = None):
        super().__init__()
        self.config = config
        self.max_hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        if nesting_list is None:
            self.nesting_list = [64, 128, 256, 512, self.max_hidden_size]
        else:
            self.nesting_list = sorted(nesting_list)

        if self.max_hidden_size not in self.nesting_list:
            self.nesting_list.append(self.max_hidden_size)
            self.nesting_list = sorted(self.nesting_list)

        # Create separate heads for each dimension for training
        # During inference, we'll use the appropriate slice
        self.lm_heads = nn.ModuleDict({
            str(dim): nn.Linear(dim, self.vocab_size, bias=False)
            for dim in self.nesting_list
        })

    def forward(self, hidden_states: torch.Tensor,
                compute_all_dimensions: bool = False,
                target_dimension: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Compute language modeling logits.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            compute_all_dimensions: If True, compute logits for all nesting dimensions
            target_dimension: If specified, compute logits only for this dimension

        Returns:
            Dictionary mapping dimension -> logits tensor
        """
        logits_dict = {}

        if compute_all_dimensions:
            # Training mode: compute for all dimensions
            dimensions_to_compute = self.nesting_list
        elif target_dimension is not None:
            # Inference mode: compute for specific dimension
            dimensions_to_compute = [target_dimension]
        else:
            # Default: compute for maximum dimension
            dimensions_to_compute = [self.max_hidden_size]

        for dim in dimensions_to_compute:
            if dim <= hidden_states.size(-1):
                # Slice hidden states to target dimension
                sliced_hidden_states = hidden_states[..., :dim]

                # Compute logits using appropriate head
                if str(dim) in self.lm_heads:
                    logits = self.lm_heads[str(dim)](sliced_hidden_states)
                else:
                    # Fallback: use the largest available head with slicing
                    available_dims = [int(k) for k in self.lm_heads.keys() if int(k) >= dim]
                    if available_dims:
                        fallback_dim = min(available_dims)
                        fallback_head = self.lm_heads[str(fallback_dim)]
                        # Use only the relevant portion of the weight matrix
                        weight_slice = fallback_head.weight[:, :dim]
                        logits = F.linear(sliced_hidden_states, weight_slice)
                    else:
                        continue

                logits_dict[dim] = logits

        return logits_dict


def create_matryoshka_config_update(base_config: Dict[str, Any],
                                  nesting_list: List[int] = None,
                                  loss_weights: List[float] = None) -> Dict[str, Any]:
    """
    Update a model configuration to support Matryoshka representations.

    Args:
        base_config: Base model configuration
        nesting_list: List of representation dimensions to support
        loss_weights: Weights for combining losses at different dimensions

    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()

    # Add Matryoshka-specific configurations
    config['use_matryoshka'] = True
    config['matryoshka_nesting_list'] = nesting_list or [64, 128, 256, 512, config['hidden_size']]
    config['matryoshka_loss_weights'] = loss_weights or [1.0] * len(config['matryoshka_nesting_list'])

    return config