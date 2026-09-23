"""Bounded inference diagnostic: explicit KV repetition and forced efficient SDPA.

Use only in a dedicated evaluation process, never around concurrent model calls.
No positional encoding, attention mask, cache, or target-scoring rule is changed.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def explicit_kv_efficient_sdpa():
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = functional.scaled_dot_product_attention
    observed = {"calls": 0, "shapes": [], "backend_requested": "EFFICIENT_ATTENTION"}

    def checked(query, key, value, *args, **kwargs):
        if kwargs.get("enable_gqa", False):
            raise RuntimeError("native GQA unexpectedly enabled")
        if query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]:
            raise RuntimeError("KV heads were not explicitly repeated")
        observed["calls"] += 1
        shape = {
            "q": list(query.shape),
            "k": list(key.shape),
            "v": list(value.shape),
            "dtype": str(query.dtype),
            "causal": kwargs.get("is_causal"),
            "has_mask": kwargs.get("attn_mask") is not None,
        }
        if shape not in observed["shapes"]:
            observed["shapes"].append(shape)
        return original(query, key, value, *args, **kwargs)

    with (
        patch("transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False),
        patch.object(functional, "scaled_dot_product_attention", checked),
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
    ):
        yield observed
