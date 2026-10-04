"""CPU parity checks for the campaign-local grouped-query SDPA override."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinycomplete.eval import q25_fim_attention as fim_attention  # noqa: E402


def _model() -> Any:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(91)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def transformers_sdpa_registry(monkeypatch: pytest.MonkeyPatch):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa", sdpa_attention_forward)
    monkeypatch.setattr(fim_attention, "_ORIGINAL_SDPA", None)
    monkeypatch.setattr(fim_attention, "_REGISTERED_FORWARD", None)
    return ALL_ATTENTION_FUNCTIONS


def _call(model: Any, wrapper: Any, **kwargs: Any):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_attention_forward)
    with sdpa_kernel([SDPBackend.MATH]):
        expected = model(**kwargs).logits
    ALL_ATTENTION_FUNCTIONS.register("sdpa", wrapper)
    with sdpa_kernel([SDPBackend.MATH]):
        actual = model(**kwargs).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return expected, actual


def test_small_qwen_forward_matches_math_gqa_for_causal_and_padding_masks(
    transformers_sdpa_registry: Any,
) -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    model = _model()
    wrapper = fim_attention._make_sdpa_forward(
        sdpa_attention_forward,
        context_factory=lambda torch_module: sdpa_kernel([SDPBackend.MATH]),
        require_cuda=False,
    )
    unpadded = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "use_cache": False}
    causal_expected, _ = _call(model, wrapper, **unpadded)

    padded = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "use_cache": False,
    }
    padding_expected, _ = _call(model, wrapper, **padded)
    assert torch.isfinite(causal_expected).all()
    assert torch.isfinite(padding_expected).all()
    assert transformers_sdpa_registry["sdpa"] is wrapper


def test_cached_prefill_and_single_token_decode_match_math_gqa(
    transformers_sdpa_registry: Any,
) -> None:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = _model()
    wrapper = fim_attention._make_sdpa_forward(
        sdpa_attention_forward,
        context_factory=lambda torch_module: sdpa_kernel([SDPBackend.MATH]),
        require_cuda=False,
    )
    tokens = torch.tensor([[9, 8, 7, 6]])
    mask = torch.ones_like(tokens)
    with sdpa_kernel([SDPBackend.MATH]):
        expected_prefill = model(
            input_ids=tokens[:, :3], attention_mask=mask[:, :3], use_cache=True
        )
    ALL_ATTENTION_FUNCTIONS.register("sdpa", wrapper)
    with sdpa_kernel([SDPBackend.MATH]):
        actual_prefill = model(input_ids=tokens[:, :3], attention_mask=mask[:, :3], use_cache=True)
    torch.testing.assert_close(actual_prefill.logits, expected_prefill.logits, rtol=1e-5, atol=1e-6)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_attention_forward)
    with sdpa_kernel([SDPBackend.MATH]):
        expected_decode = model(
            input_ids=tokens[:, 3:],
            attention_mask=mask,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
    ALL_ATTENTION_FUNCTIONS.register("sdpa", wrapper)
    with sdpa_kernel([SDPBackend.MATH]):
        actual_decode = model(
            input_ids=tokens[:, 3:],
            attention_mask=mask,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(actual_decode.logits, expected_decode.logits, rtol=1e-5, atol=1e-6)


def test_cuda_wrapper_refuses_cpu_and_registry_restore_is_test_scoped(
    transformers_sdpa_registry: Any,
) -> None:
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = transformers_sdpa_registry["sdpa"]
    fim_attention.install_q25_fim_attention(fim_attention.ATTENTION_BACKEND)
    installed = transformers_sdpa_registry["sdpa"]
    assert installed is fim_attention.registered_sdpa_forward()
    fim_attention.install_q25_fim_attention(fim_attention.ATTENTION_BACKEND)
    assert transformers_sdpa_registry["sdpa"] is installed
    query = torch.zeros((1, 4, 2, 8))
    key = torch.zeros((1, 2, 2, 8))
    with pytest.raises(RuntimeError, match="requires the frozen CUDA efficient SDPA backend"):
        installed(
            type("Attention", (), {"num_key_value_groups": 2, "is_causal": True})(),
            query,
            key,
            key,
            None,
        )
    assert original is not installed
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is installed


def test_efficient_context_disables_automatic_backend_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calls: list[list[Any]] = []

    def capture(backends: list[Any]):
        calls.append(backends)
        return nullcontext()

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", capture)
    context = fim_attention._efficient_sdpa_context(torch)
    assert calls == [[torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]]
    with context:
        pass


def test_unknown_plan_attention_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing or unsupported"):
        fim_attention.install_q25_fim_attention("sdpa")
