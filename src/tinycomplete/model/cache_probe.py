"""Introspection helpers for Qwen3.5 hybrid caches (duck-typed, no guessing)."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["describe_cache", "cache_bytes", "cache_seq_length"]


def _tensor_info(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "bytes": t.numel() * t.element_size(),
    }


def describe_cache(cache: Any) -> dict[str, Any]:
    """Describe a HF cache object without assuming its concrete class."""
    info: dict[str, Any] = {"class": type(cache).__name__}
    layers = getattr(cache, "layers", None)
    if layers is None:
        # Legacy tuple-style cache.
        info["style"] = "legacy-tuple"
        info["num_layers"] = len(cache)
        return info
    info["style"] = "layered"
    info["num_layers"] = len(layers)
    info["layers"] = []
    for i, layer in enumerate(layers):
        entry: dict[str, Any] = {"index": i, "class": type(layer).__name__}
        attrs = (
            "conv_states",
            "recurrent_states",
            "keys",
            "values",
            "key_cache",
            "value_cache",
        )
        for attr in attrs:
            val = getattr(layer, attr, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                entry[attr] = _tensor_info(val)
            elif isinstance(val, dict):
                entry[attr] = {
                    k: _tensor_info(v) if isinstance(v, torch.Tensor) else repr(v)
                    for k, v in val.items()
                }
            elif isinstance(val, (list, tuple)):
                entry[attr] = [
                    _tensor_info(v) if isinstance(v, torch.Tensor) else repr(v) for v in val
                ]
            else:
                entry[attr] = repr(val)[:200]
        # Any other tensor-ish attributes worth reporting.
        for attr in dir(layer):
            if attr.startswith("_") or attr in entry:
                continue
            try:
                val = getattr(layer, attr)
            except Exception:
                continue
            if isinstance(val, torch.Tensor):
                entry[attr] = _tensor_info(val)
        info["layers"].append(entry)
    try:
        info["seq_length"] = cache.get_seq_length()
    except Exception:
        pass
    info["total_bytes"] = cache_bytes(cache)
    return info


def cache_bytes(cache: Any) -> int:
    total = 0
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, torch.Tensor):
            if id(obj) not in seen:
                seen.add(id(obj))
                total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                visit(v)

    visit(cache)
    return total


def cache_seq_length(cache: Any) -> int | None:
    try:
        return int(cache.get_seq_length())
    except Exception:
        return None
