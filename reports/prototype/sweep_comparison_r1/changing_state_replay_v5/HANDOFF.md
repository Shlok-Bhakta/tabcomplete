# Handoff: changing-state local replay v5

The 96-request local replay is complete and its candidate server processes exited. The installed q25 service on port 19093 was not changed. Full prediction files, runtime logs, measurements, observability bundle, metadata, and summary are retained under:

`/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/changing_state_replay_v5/20260930T114536Z-06f1e61035/`

Use `interpretation.md` for the measured results and `artifact_manifest.json` for exact SHA-256 identities. Prediction files retain raw response text and are not copied into the report. The fixture is deterministic and synthetic; its typing, rejection, and navigation transitions are not human feedback.

## Frozen identities and verification

- Plan v5 self-hash `06f1e61035bc7eb4578827b10c75e4fd2ecb129796facb67fe1e238dc48bb415`; file SHA-256 `ae465ebb19a83354d0054f3bf29b8bae32474a9264b6c1a8ced05260806bd494`.
- Fixture SHA-256 `3056d31b499277fcc188c0f9cb68ec93d02028c96a69c985ae62f8de24cdddcc`.
- Runner SHA-256 `850d778ed30af436bb167730fcd24ba3b1729e40672264d8658a2458ef9b36c8`; targeted test SHA-256 `b01a50e397695595f9a1326bc51f31d170290e34110e437de81183eeb1aa1d56`.
- The final targeted gate before the run was `uv run pytest -q tests/test_measure_sweep_local.py`: 21 passed. Targeted Ruff and mypy checks passed.
- The runner verified all 96/96 expected requests, actual terminal events, memory limits, and clean process exit. No model body was repaired. The run summary has `quality_evidence=false`.

## Review guidance

Keep this input-policy-v2 run separate from the previous no-LF full-file snapshot suite and from any GPU evaluation. Fixtures and prompt inputs differ, so the two results do not establish an LF-related quality improvement. The current run's range mapping is not functional correctness: all in-range actions were multiline replacements and none is display eligible for the one-line automatic policy. Functional scoring must use its own frozen evaluator and report edit-required/no-edit denominators separately.

For cache reporting, use each request's `server_timings.cache_n` as actual runtime reuse and `server_timings.prompt_n` as recomputed prompt work. The legacy `server_tokens_cached` value is retained sequence length, not a cache-hit count. Token-ID common-prefix measurements compare the actual chronological predecessor.

The Q8 predictor exceeds the 1.5 GiB threshold and is researcher-only under this measurement. Q4 is under the memory threshold, but neither precision has demonstrated a usable one-line suggestion here. Do not change the live provider or editor on the strength of this latency diagnostic.
