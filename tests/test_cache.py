"""CPU-only Qwen3.5 cache proof: tiny random hybrid model, no weights download.

Compares full forward over A+B against cached continuation (A, then B with
past_key_values). Uses tolerant comparison (kernels/order may differ).
Also verifies cache branch isolation via deepcopy.
"""

import copy

import pytest
import torch

from tinycomplete.model.cache_probe import cache_bytes, cache_seq_length, describe_cache
from tinycomplete.model.tiny_qwen import TINY_LAYER_PATTERN, build_tiny_model

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    return build_tiny_model(seed=0)


A_TOKENS = [10, 20, 30, 40, 50, 60, 70, 80]
B_TOKENS = [81, 82, 83, 84]


def test_layer_pattern_is_hybrid():
    assert TINY_LAYER_PATTERN == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


def test_cached_continuation_matches_full_forward(model):
    model.eval()
    A = torch.tensor([A_TOKENS])
    B = torch.tensor([B_TOKENS])
    with torch.no_grad():
        full = model(torch.cat([A, B], dim=1), use_cache=True)
        prefix = model(A, use_cache=True)
        continued = model(B, past_key_values=prefix.past_key_values, use_cache=True)
    want = full.logits[:, len(A_TOKENS) :, :]
    got = continued.logits
    assert want.shape == got.shape
    assert torch.allclose(want, got, atol=1e-4, rtol=1e-4), (
        f"max diff {(want - got).abs().max().item()}"
    )


def test_cache_holds_recurrent_and_kv_state(model):
    model.eval()
    A = torch.tensor([A_TOKENS])
    with torch.no_grad():
        out = model(A, use_cache=True)
    desc = describe_cache(out.past_key_values)
    assert desc["num_layers"] == 4
    linear = [layer for layer in desc["layers"] if "recurrent_states" in layer]
    full = [layer for layer in desc["layers"] if "keys" in layer or "values" in layer]
    assert len(linear) == 3, f"expected 3 recurrent layers, got {desc}"
    assert len(full) == 1, f"expected 1 KV layer, got {desc}"
    assert cache_seq_length(out.past_key_values) == len(A_TOKENS)
    assert cache_bytes(out.past_key_values) > 0


def test_cache_grows_with_appended_length(model):
    model.eval()
    with torch.no_grad():
        short = model(torch.tensor([A_TOKENS]), use_cache=True)
        b_short = cache_bytes(short.past_key_values)
        long = model(torch.tensor([A_TOKENS + B_TOKENS]), use_cache=True)
        b_long = cache_bytes(long.past_key_values)
    assert b_long > b_short, "KV layer must grow with sequence length"


def test_cache_branching_is_isolated(model):
    model.eval()
    A = torch.tensor([A_TOKENS])
    branch_a = torch.tensor([[81, 82]])
    branch_b = torch.tensor([[91, 92, 93]])
    with torch.no_grad():
        prefix = model(A, use_cache=True)
        cache_a = copy.deepcopy(prefix.past_key_values)
        cache_b = copy.deepcopy(prefix.past_key_values)
        out_a = model(branch_a, past_key_values=cache_a, use_cache=True)
        out_b = model(branch_b, past_key_values=cache_b, use_cache=True)
        fresh = model(A, use_cache=True)
        ref_b = model(branch_b, past_key_values=fresh.past_key_values, use_cache=True)
    # Branch B is unaffected by branch A having been explored.
    assert torch.equal(out_b.logits, ref_b.logits)
    assert out_a.logits.shape[1] == 2


def test_naive_cache_reuse_without_copy_diverges(model):
    """Documents WHY branching needs a copy: the cache object mutates in place."""
    model.eval()
    A = torch.tensor([A_TOKENS])
    branch_a = torch.tensor([[81, 82]])
    branch_b = torch.tensor([[91, 92, 93]])
    with torch.no_grad():
        prefix = model(A, use_cache=True)
        _ = model(branch_a, past_key_values=prefix.past_key_values, use_cache=True)
        reused = model(branch_b, past_key_values=prefix.past_key_values, use_cache=True)
        fresh = model(A, use_cache=True)
        ref = model(branch_b, past_key_values=fresh.past_key_values, use_cache=True)
    assert not torch.allclose(reused.logits, ref.logits, atol=1e-4, rtol=1e-4)
