# Qwen3.5 Cache Probe Findings

Date: 2026-09-16. Environment: CPU-only, torch 2.14.0+cpu, transformers 5.17.0.
Model: tiny random `Qwen3_5ForCausalLM` (vocab 256, hidden 64, 4 layers in
3×`linear_attention` + 1×`full_attention` pattern), no weights downloaded.
Config/behavior cross-checked against installed source
`transformers/models/qwen3_5/modular_qwen3_5.py` (Qwen3.5 = Qwen3Next family:
Gated DeltaNet linear layers + standard full-attention layers).

## Question

Can an editor session be advanced using only new events (cached continuation)
rather than re-processing the whole context? Tested as: full forward over
A+B vs forward over A with cache, then B with `past_key_values`.

## Result: YES — continuation matches full forward

- `max abs logit diff: 1.19e-07` (mean 1.96e-08), well within `atol=rtol=1e-4`.
- `model.eval()`, seeded init, `torch.no_grad()`.
- CPU runs on reference PyTorch kernels (`causal_conv1d` and
  `flash-linear-attention` not installed — correct but slower; expected on CPU).

## Cache anatomy (`DynamicCache`, transformers `cache_utils`)

| layer | type | state | shape (toy cfg) | dtype | bytes @ seq 8 |
|---|---|---|---|---|---|
| 0–2 | `LinearAttentionLayer` (Gated DeltaNet) | `conv_states` | [1, 96, 4] | fp32 cpu | 1536 each |
| 0–2 | same | `recurrent_states` | [1, 2, 16, 16] = heads×k_dim×v_dim | fp32 cpu | 2048 each |
| 3 | `DynamicLayer` (full attention) | `keys`, `values` | [1, 2 kv-heads, seq, 16] | fp32 cpu | 1024 each @ seq 8 |

- Recurrent state is **fixed-size** in sequence length — the property the
  append-only session design relies on.
- KV state grows linearly: total cache 12800 B @ 8 tokens → 13312 B @ 10
  tokens (+256 B/token in this toy config: K+V × 2 heads × 16 dim × fp32).
- `cache.get_seq_length()` drives position offsets on continuation, so new
  tokens need no manual position bookkeeping.

## Branching: deepcopy is safe and exactly isolated

- `copy.deepcopy(cache)` works on CPU; continuing branch B from a copy while
  branch A was explored elsewhere gives **bit-identical** logits to a fresh
  continuation (diff 0.0).
- Naive reuse without a copy **diverges** (max diff 0.133): the cache object
  mutates in place on every forward, so candidate exploration
  (`cache at X → append A vs append B`) MUST copy first.

## Implication for the editor design

Per-keystroke (or per-event-batch) inference can advance the DeltaNet
recurrent state + KV cache with only new tokens. Candidate branches need one
deepcopy per candidate — cheap at these state sizes, to be re-measured at
0.8B scale on GPU. The recurrent state must NOT be treated as lossless repo
storage: exact facts get re-injected from deterministic indexes (see
`docs/architecture.md`).

## Reproduce

`uv run pytest -q tests/test_cache.py`
`src/tinycomplete/model/tiny_qwen.py`, `src/tinycomplete/model/cache_probe.py`
