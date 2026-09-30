# Changing editor state replay, revision 5

This is a local CPU latency and format diagnostic for the existing Sweep 1.5B Q4_K_M and Q8_0 artifacts. It does not change the installed Qwen service, editor configuration, model weights, or training state. The complete run artifacts remain under `/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/changing_state_replay_v5/20260930T114536Z-06f1e61035/`.

## Run identity and scope

- Frozen plan self-hash: `06f1e61035bc7eb4578827b10c75e4fd2ecb129796facb67fe1e238dc48bb415`; plan file SHA-256: `ae465ebb19a83354d0054f3bf29b8bae32474a9264b6c1a8ced05260806bd494`.
- Frozen 24-state fixture SHA-256: `3056d31b499277fcc188c0f9cb68ec93d02028c96a69c985ae62f8de24cdddcc`.
- Prompt policy: preserve the publisher prompt and field order, adding one LF after the updated-file header. The request body and generated output were not repaired or normalized. The older no-LF snapshot results use different fixtures and are not a causal before/after comparison.
- The suite has 24 deterministic synthetic states, each requested twice per precision: fresh open (6), append (2), earlier edit (2), reject then divergent typing (3), file switch (3), return to file A (3), near-cursor replacement (2), and typed matching prefix (3). This is not a replay of real human sessions.
- There were 96 completed requests: 48 per precision. All 48 responses per precision had a terminal event and observed EOS; none reached the 512-token output cap. Both research server processes exited, as recorded in their measurement files. The installed q25 service remained running and unchanged.
- The test suite reported `quality_evidence: false`. There are no human decisions or accepted-training examples in this run.

## Latency

Times are request-path milliseconds from the local completion request to the first token or completed full-file response. The valid-action latency is the same completed request interval restricted to outputs that mapped into the editable range; it does not include a live editor debounce or claim display time. Percentiles use nearest rank.

| Precision | Load to health (s) | First token median / p95 (ms) | 8 output tokens median / p95 (ms) | 16 output tokens median / p95 (ms) | Completed file median / p95 (ms) | Valid mapped action median / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.404 | 5,069 / 19,390 | 5,419 / 19,685 | 5,805 / 20,018 | 6,900 / 22,315 | 8,366 / 32,482 |
| Q8_0 | 0.592 | 6,044 / 23,638 | 6,526 / 24,046 | 6,998 / 24,525 | 8,681 / 27,826 | 9,965 / 39,399 |

The run took 1,038.439 seconds from its frozen invocation deadline start through summary creation. Q4 ran first for 462.636 seconds; Q8 then ran for 573.044 seconds. The tokenization endpoint was used before each request and its time is recorded separately (median 15.4 ms for Q4 and 15.3 ms for Q8).

The request latencies are far above the prototype's 500 ms median and 1,000 ms p95 goals. This suite does not include the editor's 250 ms debounce or measure last-keystroke-to-render time. It therefore cannot establish interactive latency, and the measured request times do not pass those performance goals.

## Actual token lengths and cache evidence

The pinned model tokenizer measured the same request lengths for both precisions. Each bucket has 16 requests (8 states × 2 passes):

| Nominal bucket | Actual tokens, min / median / p95 / max |
|---|---:|
| 512 | 435 / 566 / 726 / 726 |
| 1,024 | 1,105 / 1,237.5 / 1,399 / 1,399 |
| 2,048 | 1,740 / 1,875.5 / 2,034 / 2,034 |

Across all requests the actual input-token min / median / p95 / max was 435 / 1,237.5 / 2,000 / 2,034. Every input plus the 512-token generation ceiling stayed below the explicit 3,072-token context limit. The 2,048 bucket uses the pinned adjacent editor modules; it does not pad with arbitrary text.

The runtime used one slot, `cache_prompt=true`, no context shifting, no optional RAM snapshots, and zero context checkpoints. The recorded token-ID longest common prefix against the immediately preceding request was positive for 47/48 requests per precision (median 1,010, p95 1,726, max 1,813). The runtime's actual `server_timings.cache_n` had the same distribution; `server_timings.prompt_n` (recomputed prompt tokens) had median 193, p95 842, max 1,178. This confirms reuse for this exact ordered replay, not that every editor transition will achieve the same reuse.

Do not interpret the legacy summary fields `backend_tokens_cached_median` or `server_tokens_cached` as cache-hit tokens. The pinned runtime reports retained sequence length there. The backend reuse evidence is `server_timings.cache_n`; recomputed work is `server_timings.prompt_n`.

## Output mapping and automatic display suitability

| Precision | EOS-complete outputs | In editable range | Out of range | Canonical `replace` | `no_edit` / deletion | One-line display eligible |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 48/48 | 38/48 | 10/48 | 38 | 0 / 0 | 0/38 |
| Q8_0 | 48/48 | 32/48 | 16/48 | 32 | 0 / 0 | 0/32 |

Every mapped replacement contained a newline. Thus the initial one-line automatic display policy would suppress all mapped actions in this diagnostic. Mapping into the editable byte range is only a serialization/range result; it does not show that the replacement is correct, useful, or wanted. There is no functional-quality denominator here, and the `replace` counts must not be called successful suggestions. No no-edit or deletion behavior was demonstrated.

The two precisions received identical effective input-prompt hashes for all 48 matched state/repetition pairs. Raw whole-file output hashes and output-token-ID hashes matched in 32/48 pairs and differed in 16/48. Among the 32 pairs where both outputs mapped, the canonical actions were exactly equal in 26 and different in 6; null/unmapped pairs were excluded from this comparison. Six additional pairs mapped only for Q4, none mapped only for Q8, and 10 were unmapped for both. These results show that the quantized variants sometimes emit different bytes. They do not show that Q4's higher range-mapping count means higher quality.

## Host and memory

The inference host was `crabcake`, an AMD Ryzen 5 PRO 3400GE (4 cores / 8 threads), Linux x86_64, with 14,580,572,160 bytes total RAM. The runtime was llama.cpp `f072b103714dfa1eee531f80b24512faf38e3dd2`, CPU-only (`--n-gpu-layers 0`, AVX2 backend), four generation and prompt threads, context 3,072, batch 256, microbatch 64, one slot, F16 K/V cache, no warmup, no repacking, no speculative model, and no optional saved state cache. The service used its own loopback port and was stopped after each precision.

| Precision | Service tree peak / retained RSS (bytes) | Service tree peak / retained PSS (bytes) | Anonymous RSS | File-backed RSS | Service swap | Classification |
|---|---:|---:|---:|---:|---:|---|
| Q4_K_M | 1,017,802,752 | 1,011,790,848 | 123,277,312 | 894,525,440 | 0 | Below the 1.5 GiB threshold |
| Q8_0 | 1,671,602,176 | 1,665,670,144 | 123,236,352 | 1,548,365,824 | 0 | Research-only: RSS exceeds 1.5 GiB by 60,989,440 bytes |

The candidate server process tree is included in these values. The Python/uv measurement client and the separate resident q25 predictor are outside the candidate server tree; these numbers are not whole-host RAM use. The q25 service stayed resident at about 472 MB RSS and accumulated only 0.03 CPU seconds during Q4 and 0.02 CPU seconds during Q8. Other user applications and services remained active, so this is a local host measurement, not a bare-metal or normal-LazyVim controlled session. Available RAM stayed above 11.4 GB. The host's I/O pressure counter deltas were 1.65/1.60 seconds (Q4 some/full) and 2.51/2.42 seconds (Q8 some/full); these are host-wide accumulated counters, not predictor-specific stalls.

Available host RAM remained above 11.4 GB. The machine had about 1.81 GB of pre-existing swap occupancy; during the Q4 and Q8 windows, global counters recorded 44 and 11 pages read in and zero pages written out. The candidate services themselves reported zero swapped memory. Memory-pressure counter deltas were under 1 ms per window. These observations do not show predictor thrashing. CPU-frequency means were 3.03 GHz (Q4) and 2.96 GHz (Q8); reported CPU thermal peaks were 76.4 °C and 82.6 °C. The host has Radeon Vega graphics, but this replay did not use GPU inference and did not measure VRAM.

## Limits and interpretation

- Q4 used an 883,289,056-byte Q4_K_M artifact; Q8 used the pinned 1,537,269,856-byte Q8_0 artifact. The Q4 file is a local requantization of the Q8 artifact. Their exact hashes are in `artifact_manifest.json`.
- The Q8 service crossed the 1.5 GiB predictor limit. Q4 stayed below it, but did not produce any single-line display-eligible action in this suite.
- This run cannot establish model correctness, human acceptance, no-edit calibration, usefulness, production accuracy, or a quality advantage for either precision. It also cannot isolate the LF prompt change from suite/state differences in older results.
- No editor service was changed, no model was downloaded or converted, and no training or personalization ran.

See `HANDOFF.md` for artifact locations, frozen identities, test gates, and the required separation from earlier snapshot results.
