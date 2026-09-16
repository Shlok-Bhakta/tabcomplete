"""Tiny random Qwen3.5 models for CPU-only cache experiments.

Mirrors the real Qwen3.5-0.8B hybrid pattern (3 linear-attention layers per
1 full-attention layer) at toy dimensions. Never downloads weights.
Config choices were read off the installed Transformers 5.17 Qwen3.5
source (models/qwen3_5/modular_qwen3_5.py + qwen3_next modeling), not guessed.
"""

from __future__ import annotations

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

__all__ = ["TINY_LAYER_PATTERN", "build_tiny_config", "build_tiny_model"]

# 3 linear (Gated DeltaNet) + 1 full softmax attention, repeated.
TINY_LAYER_PATTERN = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


def build_tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        layer_types=list(TINY_LAYER_PATTERN),
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=True,
    )


def build_tiny_model(seed: int = 0) -> Qwen3_5ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(build_tiny_config())
    model.eval()
    return model
