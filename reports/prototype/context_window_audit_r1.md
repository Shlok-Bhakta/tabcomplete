# ThinkPad FIM context-window audit R1

## Status and scope

Source audit, CPU tokenizer counterfactual, and an isolated opt-in implementation on branch `prototype/q25-fim-context-window-r1`, based on `baacd9a5803471b193a3b821af88b77c3bae15c3`. No model, training fixture, or deployed ThinkPad configuration was changed. The user completed the declarative NixOS switch after the FIM update. The existing served policy remains `q25-fim-psm-cursor-to-line-end-bounded640-256-v2`.

The alignment policy is implemented and passes focused deterministic tests. It has not been benchmarked with inference or evaluated for edit quality. It is disabled by default and must remain undeployed until a paired public-data comparison establishes its tradeoff. Work stops at this reviewable state because the user requested a handoff and pause after the current Kaggle run.

## Measured bottleneck

Evidence: [ThinkPad changing-state samples](thinkpad_fim_update_r1/runtime_fim_four_threads.json) and [runtime summary](thinkpad_fim_update_r1/runtime_summary.json). The i7-8650U target used four physical-core threads, CPU inference, the embedded direct-FIM Q4_K_M artifact, F16 KV state, one slot, and no optional state snapshots. Two repetitions of four deterministic public synthetic editor states ran while existing user applications remained present. No competing-process trace proves a controlled interval.

| Transition condition | Actual prompt tokens | Cached input tokens | Recomputed input tokens | Prefill ms | Completed client request ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| First uncached state | 643 | 0 | 643 | 3652.5 | 3709.0 |
| Append changing total prefix token count | 643 | 1 | 642 | 3733.8 | 3763.3 |
| Near-cursor replacement changing prefix token count | 643 | 1 | 642 | 3741.7 | 3794.8 |
| Near-cursor replacement retaining the cropped start | 643 | 640 | 3 | 40.1 | 94.1 |
| Same complete prompt repeated | 643 | 642 | 1 | 22.9 | 79.6 |

Generation took approximately 21–52 ms. The large changed-state penalty comes from rebuilding almost the whole prompt after the left edge of the last-640-token window moves. This slice is too small to characterize everyday quality or latency distributions. Client request timing excludes editor debounce and final display time.

Selected-service peak RSS was 592,429,056 bytes; peak PSS 588,277,760 bytes. Service swap was zero, host swap-in/out deltas were zero, and memory-pressure totals did not increase in the measurement window. Editor and curl-helper peaks were not measured separately. The service retained the measured peak RSS/PSS after requests.

## Current implementation and correctness boundaries

- `tools/tabcomplete_engine/src/fim_v1.rs`: `crop_context` tracks native token-piece byte offsets and crops the right side by `pieces.len() - 640`. It rounds cuts to UTF-8 scalar boundaries and retokenizes until the actual retained count fits. `set_bounded_window` preserves original model-hole and apply ranges. The source text is never normalized or repaired.
- `tools/tabcomplete_engine/src/main.rs`, FIM preparation near lines 746–786: prefixes are source bytes before the cursor; suffixes are source after the physical line ending. The model prompt remains exact PSM order. Prefix 640 plus suffix 256 plus three control tokens is at most 899 prompt tokens. A further 96-token output reservation is required within both configured input and total context limits.
- The recorded layout is `q25-fim-psm-bounded-v2`, with policy `q25-fim-psm-cursor-to-line-end-bounded640-256-v2`. Existing larger full-source buckets do not mean the actual model consumes 2,048 tokens under these limits.
- The context digest includes full-source identity, prompt identity, cursor, byte ranges, actual token counts, tokenizer contract, layout, and policy. A new policy can bind its identity through this existing mechanism.
- The request cache near `main.rs:913` compares actual token IDs. It removes sequence state after the common prefix and prefills the remainder. Previously generated text is removed when it differs from the next source-derived request. Repository changes and explicit cache disabling clear state.
- Ordinary client cancellation (`reply.is_closed()` or a failed streaming send) returns success and retains already decoded input state. Genuine processing errors clear the cache. A separate cancellation-retention change is unnecessary for this proposal.
- One active slot and no saved snapshots means file A→B→A still loses the prior A sequence after incompatible B input. Prefix alignment does not solve that case.

## Minimal proposed opt-in policy

For a full-prefix token count `N`, keep the existing source window when `N <= 640`. Otherwise choose the absolute token index:

```text
raw_start = N - 640
aligned_start = ceil(raw_start / 128) * 128
```

The alignment is relative to the entire source prefix, not the current cropped substring. For example, `N = 641..768` starts at full-source token 128, retaining approximately 513..640 tokens. At `N = 769` it advances to token 256. Token count remains an actual retokenized count, never an estimate.

Resolve the chosen native piece offset to a valid UTF-8 boundary and retokenize the substring exactly. Document and count any boundary/BPE fallback needed to stay within 640 tokens. Preserve the suffix policy, original edit ranges, control tokens, complete physical-line semantics, and EOS contract. If the aligned cut would remove any part of an editable-line prefix retained by the control policy, fall back to the control crop and record `editable_line_prefix`. This also preserves the control tail of an individual line longer than 640 tokens. If retokenization exceeds the budget, record `retokenized_budget` and use the control window.

The candidate discards up to 127 extra prefix tokens. It can remove a relevant import, declaration, scope boundary, or user intent. This is a context-information tradeoff. Lower recomputation alone is not proof that the candidate improves the product.

Implemented identities:

- Runtime option: `--fim-context-window sliding-v2|aligned128-v1`, default `sliding-v2`.
- Candidate context policy: `q25-fim-psm-cursor-to-line-end-bounded640-256-align128-v1`.
- Keep the existing PSM layout and wire protocol; add the explicit option to runtime configuration hashing, health metadata, and editor package policy metadata.

No prompt source reordering, new tokenizer stack, model update, or training is needed to test this idea.

## Tokenizer-only counterfactual

The existing synthetic ThinkPad replay source has 45 public Python helper functions followed by an `add` function. Its four prefix variants are `return a +`, `return a + b`, `return a *`, and `return a +`. The existing local tokenizer JSON was loaded with `tokenizers.Tokenizer`, with no model loading or downloads. Tokenizer-file SHA-256: `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8`.

The ASCII fixture allows character offsets to correspond directly to bytes. Each retained substring was retokenized, and full PSM token prefixes were compared. This is a deterministic token-level simulation, not native-runtime or quality evidence.

| State | Full prefix tokens | Sliding start / retained | Aligned start / retained | Sliding recomputed prompt tokens | Aligned theoretical recomputed prompt tokens |
| --- | ---: | --- | --- | ---: | ---: |
| Initial `return a +` | 665 | 25 / 640 | 128 / 537 | 643 | 540 |
| Append ` b` | 666 | 26 / 640 | 128 / 538 | 642 | 3 |
| Replace `+ b` with `*` | 665 | 25 / 640 | 128 / 537 | 642 | 3 |
| Replace `*` with `+` | 665 | 25 / 640 | 128 / 537 | 3 | 3 |

Alignment removes an additional 102–103 source tokens, or 341–343 ASCII characters, in these cases. No latency number is assigned to the candidate. Native token-piece behavior and task correctness still require verification.

## Cross-language/API constraints

The original `tools/trajectory_collector/nvim/lua/tabcomplete_trajectory/fim_v1.lua` required the old policy exactly in `verify_prepared`. It also verifies source ranges, UTF-8 boundaries, original hole/apply ranges, line ending, tokenizer identity, and token counts. `predict.lua` repeated identity checks in `validate_editor_context`. A Rust-only policy change would correctly fail these checks.

The isolated implementation adds an optional expected-policy argument to `verify_prepared` and `fim_context_window` in the configured `allowed_models[alias]` specification. Missing configuration keeps `sliding-v2`. Only the two known policy mappings are allowed. Candidate health must explicitly match configured window and policy; the legacy control health remains compatible. The shared digest already includes the policy, so candidate requests receive new hashes. Request/generation identity guards and apply-time checks are unchanged. A supplied allowlist now replaces the previous allowlist and clears cached model identity, preventing a removed opt-in field from surviving a deep-merge on configuration reload.

`scripts/prepare_q25_fim.py` contains the frozen training preparation crop. Do not change it for this runtime experiment. `scripts/evaluate_q25_fim_native.py` validates native context against that frozen prompt and hardcodes the old policy in its digest. An aligned server must fail the current control validation. Add a separately versioned paired experiment with explicit policy-aware expected contexts; never rewrite existing fixtures or quietly relax the validator. `src/tinycomplete/data/fim.py` implements a different general missing-middle task and is not the insertion point for this proposal.

## Required checks before deployment

1. Freeze an opt-in comparison plan and fixed public development source states before producing candidate outputs. Preserve full source, cursor, target replacement, and file grouping. Record paired retained ranges, prompt token IDs, source hashes, and discarded-context counts. Do not use sealed tests or the new training study's final holdout for cache-policy tuning.
2. Keep old-policy Python, Rust, and Lua goldens byte-identical. Add shared candidate goldens for 640/641/768/769-token boundaries, LF/CRLF/EOF, empty files, Unicode with token pieces splitting a scalar, combining marks, whitespace, long editable lines, and alignment fallback. Wrong policy/digest/range responses must still fail.
3. Measure identical-prompt repeats separately from changing states, with at least two repetitions and a longer confirmation replay. Include append, replacement, paste, deletion, earlier incompatible edit, A→B→A, cancellation, and bucket-crossing transitions. Record actual native cached/recomputed tokens and prefill/generation/client/display stage times.
4. Compare paired EOS termination, output length, canonical actions, useful edits, and functional checks where available. Report affected long-prefix cases separately from unchanged short cases. Report uncertainty and meaningful action differences. Codec-valid output alone is not useful completion evidence.
5. Retain the service memory cap and measure RSS/PSS, retained memory, swap activity, and pressure. Avoid concurrent compilation during a claimed controlled run. Preserve existing user applications for a separately labeled normal-workload run.
6. Only after comparison, make any selected policy a declarative Nix/Home Manager setting with rollback metadata. Existing ThinkPad configuration and model remain unchanged by this audit.

## Implementation verification and handoff

- Rust release unit tests: **49 passed, 0 failed**. Includes existing default goldens, 640/641/768/769 boundaries, split-byte Unicode, current-line fallback including overlong lines, retokenization fallback, exact native-piece byte checks, CRLF/EOF unchanged application ranges, policy-bound digests, and CLI known-value/default serialization checks.
- Full existing Neovim headless suite: **94 passed, 0 failed**. Includes configured candidate opt-in, control/candidate mismatch, missing candidate identity, unknown policy, policy-bound digest, unchanged source-range checks, and removal of an earlier allowlist opt-in. All generated editor outcomes are synthetic.
- `cargo fmt --check` and `git diff --check` passed. No new Python implementation or training fixtures were modified. Repository-wide Python/Bun/type checks and model-quality inference were not run for this isolated experimental code change.
- Initial Rust compilation failed because this host's libclang lacked its standard-header search path. The successful command uses the existing GCC13 standard-header directory. No packages or dependencies were installed. Existing vendored llama.cpp warnings remain.
- Existing native Cargo target reuse increased its measured size from 993,644,544 to 1,009,238,016 bytes at the end of testing. No weights were copied. The SSD had more than31GB free at preflight. The shared release model-serving binary was not rebuilt or replaced; release tests compile a separate test executable.
- Existing observability CLI returned historical run `run-b36031da-89cf-4648-be59-f541c7787dfd`, trace `ad0c7da5d4352fde724a4f95f63fca1e`. Its bounded failures query returned `no_runs`. It does not provide telemetry for the ThinkPad replay window; authoritative local runtime samples remain the evidence for this audit. No monitoring infrastructure changed.
- No Nix/Home Manager option, source pin, deployed service, or ThinkPad configuration was changed. A future declarative option must configure both the CLI and matching editor allowlist. No automatic rollout is authorized by these tests.

Repeat the completed focused checks:

```sh
BINDGEN_EXTRA_CLANG_ARGS='-I/usr/lib/gcc/x86_64-linux-gnu/13/include' \
CARGO_TARGET_DIR=/mnt/ssd/tabcomplete-q25-code-cpt-r2/fim-native-build-cache \
cargo test --release --offline --manifest-path tools/tabcomplete_engine/Cargo.toml
cargo fmt --check --manifest-path tools/tabcomplete_engine/Cargo.toml
cd tools/trajectory_collector/nvim
nvim --headless -u NONE -l tests/run.lua
```

Pending work is the frozen paired public quality and real changed-state latency comparison described above. No speedup or task-quality result is claimed for `aligned128-v1`. ThinkPad continues serving the working default policy.
