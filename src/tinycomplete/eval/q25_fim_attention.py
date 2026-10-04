"""Fail-closed efficient SDPA override used only by the Q25 FIM campaign."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

ATTENTION_BACKEND = "torch-efficient-sdpa-explicit-kv-repeat-v1"

_ORIGINAL_SDPA: Callable[..., Any] | None = None
_REGISTERED_FORWARD: Callable[..., Any] | None = None


def _efficient_sdpa_context(_torch: Any) -> AbstractContextManager[Any]:
    return _make_efficient_sdpa_context_factory()(_torch)


def _make_efficient_sdpa_context_factory() -> Callable[[Any], AbstractContextManager[Any]]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("required efficient SDPA backend is unavailable") from exc
    backend = getattr(SDPBackend, "EFFICIENT_ATTENTION", None)
    if backend is None:
        raise RuntimeError("required efficient SDPA backend is unavailable")

    def context(_torch: Any) -> AbstractContextManager[Any]:
        return sdpa_kernel([backend])

    return context


def _make_sdpa_forward(
    original_forward: Callable[..., Any],
    *,
    context_factory: Callable[[Any], AbstractContextManager[Any]],
    require_cuda: bool,
) -> Callable[..., Any]:
    """Build a wrapper that explicitly expands GQA before calling HF SDPA."""

    def q25_fim_sdpa_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        position_bias: Any = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        if require_cuda and query.device.type != "cuda":
            raise RuntimeError("FIM attention requires the frozen CUDA efficient SDPA backend")
        if query.device != key.device or query.device != value.device:
            raise ValueError("attention tensors must share a device")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention tensors must have [batch, heads, tokens, width] shape")
        if key.shape != value.shape:
            raise ValueError("key and value tensor shapes must match")
        if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key tensor dimensions are incompatible")

        groups = getattr(module, "num_key_value_groups", 1)
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
            raise ValueError("module has invalid grouped-query attention metadata")
        if query.shape[1] != key.shape[1] * groups:
            raise ValueError("query and key head counts do not match grouped-query metadata")
        if groups > 1:
            key = _repeat_key_value_heads(key, groups)
            value = _repeat_key_value_heads(value, groups)

        effective_causal = (
            is_causal if is_causal is not None else getattr(module, "is_causal", True)
        )
        sdpa_module = SimpleNamespace(
            num_key_value_groups=1,
            is_causal=effective_causal,
        )
        with context_factory(torch):
            return original_forward(
                sdpa_module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=effective_causal,
                position_bias=position_bias,
                **kwargs,
            )

    q25_fim_sdpa_forward.__name__ = "q25_fim_efficient_sdpa_forward"
    return q25_fim_sdpa_forward


def _repeat_key_value_heads(states: Any, groups: int) -> Any:
    if groups == 1:
        return states
    batch, kv_heads, tokens, head_dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, tokens, head_dim)
        .reshape(batch, kv_heads * groups, tokens, head_dim)
    )


def install_q25_fim_attention(attention_backend: str) -> None:
    """Install the frozen backend over HF's existing ``sdpa`` registry entry."""
    global _ORIGINAL_SDPA, _REGISTERED_FORWARD

    if attention_backend != ATTENTION_BACKEND:
        raise ValueError("frozen FIM attention backend is missing or unsupported")

    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    registered = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if _REGISTERED_FORWARD is not None:
        if registered is not _REGISTERED_FORWARD:
            raise RuntimeError("the process SDPA registry changed after FIM backend setup")
        return
    if registered is not sdpa_attention_forward:
        raise RuntimeError("the pinned Transformers SDPA implementation changed")

    _ORIGINAL_SDPA = sdpa_attention_forward
    _REGISTERED_FORWARD = _make_sdpa_forward(
        _ORIGINAL_SDPA,
        context_factory=_make_efficient_sdpa_context_factory(),
        require_cuda=True,
    )
    AttentionInterface.register("sdpa", _REGISTERED_FORWARD)


def registered_sdpa_forward() -> Callable[..., Any]:
    """Return the installed function for the FIM-only CUDA smoke probe."""
    if _REGISTERED_FORWARD is None:
        raise RuntimeError("FIM efficient SDPA backend has not been installed")
    return _REGISTERED_FORWARD
