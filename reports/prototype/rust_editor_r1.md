# Rust editor prototype on ThinkPad

This iteration moves local inference and bounded context construction into one Rust service. It reuses the trajectory collector, its SQLite database, and the Neovim plugin. Suggestions remain experimental. Applying an edit requires explicit acceptance, and automatic personalization is disabled.

## What the measurements establish

The target is `shlokthinkpad`, an i7-8650U with four physical cores, eight logical CPUs, AVX2, approximately 23.2 GiB RAM, and Intel UHD 620 graphics. All measurements here use CPU inference on that ThinkPad. Crabcake controls the benchmark through an owned SSH tunnel. Existing browser, editor, and desktop applications remain running. No competing build or conversion runs during the inference windows.

The pinned native engine uses `llama-cpp-2 0.1.157` and `llama-cpp-sys-2 0.1.158`, built for x86-64-v3. Four generation threads and four prompt threads beat the tested two-thread alternatives. The service has one active slot, a 2,304-token total context, a 1,024-token input budget, batch 256, microbatch 64, FP16 KV storage, and no optional saved context snapshots. Generated text is removed from cached state before the next editor request. Cache reuse follows actual token prefixes.

| Configuration | Changed-state median / p95 | Identical-repeat median / p95 | Development exact actions |
| --- | --- | --- | --- |
| Qwen, four generation/four prompt threads, trained layout | 2,404 / 3,683 ms | 172 / 300 ms | 9/24 |
| Qwen, four/four, cursor metadata last | 926 / 3,620 ms | 159 / 233 ms | 9/24 |
| Qwen, two/four, trained layout | 2,490 / 3,748 ms | 213 / 326 ms | 9/24 |
| Qwen, two/two, trained layout | 4,019 / 5,327 ms | 203 / 347 ms | 9/24 |
| Sweep Q4, four/four, strict window | 12,458 / 16,212 ms | 5,953 / 9,185 ms | 1/24 |

Each completed setting has 48 changed-state requests, 48 identical repeats, and 24 development cases. Sweep hit its 900-second deadline after 91 completed requests. A fingerprint-checked continuation finished the missing requests under a new frozen plan. Its combined results include that boundary and are not an uninterrupted replay. The cursor-last experiment moves only the cursor metadata line, preserves the other information, and recounts the selected model's input tokens. All 24 development canonical actions match the control exactly. Changed requests reuse 12,695 input tokens and recompute 9,677 with that layout, versus 8,993 reused and 13,379 recomputed in the control. These measurements exclude the editor's 250 ms debounce and display callback. Identical repeats are not everyday edit latency.

The baseline retained 553.7 MiB RSS and 549.9 MiB PSS. Its anonymous portion was 82.4 MiB and its file-backed PSS was 467.4 MiB. Peak RSS was also 553.7 MiB. There were no process swap bytes, no swap-in/out activity, no major-fault increment, and no measured memory-pressure stall increment during that comparison. Endpoint memory snapshots do not establish peak PSS between them.

This is a latency improvement, not a demonstrated quality improvement. The 24 development cases are related public/synthetic contract checks. Qwen gets three of six replacements and all six deletions exactly, zero of six insertions, and zero of six no-edit cases. The automatic prefix guard withholds deletion and replacements that remove typed bytes. It reduces the display scope; it does not repair the model or establish human acceptance.

Sweep returned 28/48 canonical actions in each replay condition and 46/48 actual EOS terminations. In the development slice it produced 2/24 canonical actions and 1/24 exact action, versus Qwen's 24/24 canonical, 24/24 EOS, and 9/24 exact. Sweep got one of six replacement cases exactly; insertion and deletion mappings are unsupported, and all six no-edit cases were invalid. This is not evidence of good no-edit behavior. Its retained RSS was 941.0 MiB and its process swap was zero.

The default selection was locked before deployment confirmation. Qwen wins this comparison on useful replacement outcomes, validity, measured changed-state latency, and RAM. Sweep remains an optional experimental picker model. Neither model has established production accuracy or human acceptance.

## Installed serving confirmation

The final installed binary SHA-256 is `9341ac3dbf74e0ecf42d82d24bf0b5f0ee0da7327415f1dbdcb83392ae753e95`. A new frozen plan confirms the same 24 transitions, two repetitions, and 24 development cases through the actual `tabcomplete-engine.service`. Rust now constructs the cursor-last prompt directly, with no controller reordering or extra tokenization RPC. Its actual runtime configuration hash is `5109a74b23b8a01d41bdb8e6c81be6fe54c48bd2639581c3687fc6e1cf6ba1ec`.

Changed-state completed-action latency is 874 ms median and 2,689 ms p95. Same-prompt repeats are 160 ms median and 196 ms p95. All 96 replay requests have valid canonical actions and actual EOS. The 24 development actions remain 9/24 exact, with the same per-action breakdown. Selection was already locked; this confirmation did not select new settings. Final figures are one sequential window on a normal desktop workload, not an uncertainty bound or guaranteed everyday latency.

Retained Qwen RSS is 555.8 MiB, PSS 551.8 MiB, anonymous memory 84.3 MiB, and file-backed PSS 467.5 MiB. Swap and swap-in/out are zero, and major faults and memory-pressure totals do not increase during the confirmation. Process lifetime peak RSS is 956.9 MiB, which includes the earlier Sweep picker check; it is not a Qwen-only peak measurement. The earlier Qwen-only screening peak was 553.7 MiB. There is no resident tokenizer/helper process. The native model/context load was 281 ms, excluding prior model hash verification and backend startup, and the first confirmation request took 2,580 ms for 430 input tokens. Filesystem-cold loading was not measured.

The final confirmation accidentally omitted the observability enable flag and therefore has scientific results but no new SigNoz run ID. Earlier screened settings and the pre-install failure have imported scoped traces, and duplicate offline import was verified. This omission is preserved, not repaired by inventing retrospective spans.

## Models and delivery

The two existing Q4_K_M artifacts total 1,374,688,864 bytes. Both are imported into immutable Nix store paths. Only one model is resident at a time. No new weights were downloaded, no weights were retrained or requantized, and no model files are committed or published.

| Model | SHA-256 | Bytes |
| --- | --- | --- |
| Existing adapted Qwen2.5-Coder-0.5B | `4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb` | 491,399,808 |
| Sweep next-edit 1.5B Q4_K_M | `936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4` | 883,289,056 |

Qwen's foundation/tokenizer revision is `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`. The current adapted export is the existing public-functional LR 1e-5 checkpoint. Sweep's source revision is `409016591c6c1a94f545f22328a85a3516118f34`. Both foundations have Apache-2.0 licenses. The Qwen adaptation remains private. Sweep's existing Q4 derives from the publisher's Q8 export, so historical Q4/Q8 comparisons are not an F16 quantization ablation.

The Nix bundle delivers the Rust executable, matching Lua plugin, registry, and model symlinks together. Weights remain a separate mmap file rather than bytes embedded in the executable. Startup neither downloads weights nor extracts a second copy. `:TabCompleteModel q25` and `:TabCompleteModel sweep` switch the single service and persist the selected alias.

## Boundaries and feedback

The Rust worker owns tokenization, context selection, model state, and generation. A busy claim returns HTTP 409 rather than queuing stale jobs. Its channel holds at most one command; the SSE channel holds eight events. Cancellation checks occur between prefill batches and generation tokens. A native decode already running cannot be interrupted by this implementation.

Context comes from the current file, actual recent edit history, and a bounded set of same-repository open/recent buffers. The engine does not scan a repository or call Git per prediction. It uses deterministic recency/import/definition heuristics. This is a small context selector, not a demonstrated repository reasoning system.

Raw request, generation, display, acceptance, dismissal, delta, undo, and anchor events stay in `/mnt/ssd/collector-data/collector.sqlite` on Crabcake through the existing collector API. Existing schema versions 1, 2, and 3 and prediction projections remain authoritative. No database replacement or destructive migration is needed for this pivot. A WAL-consistent SQLite backup passed integrity checking before installation.

A shown proposal dismissed by typing records the observation immediately. Later classification uses actual buffer deltas. Exact matching typing is distinct from a rejection; a later edit is not automatically a correction or a preference. Scripted verification uses `synthetic=true` and never sets human review merely because a request completed. Sequence gaps and unobserved delayed outcomes remain explicit.

## TurboQuant

The installed pinned llama.cpp source contains no TurboQuant implementation. Google's published work concerns KV compression and includes H100 kernel results; it does not establish a speedup for this small CPU workload. A separate fork/kernel integration is not justified for this deployment. FP16 KV storage is the supported control. Q4 weight quantization is a different setting. See [Google Research's TurboQuant article](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/).


## Reload and deployment cache regression

The real SQLite audit found that `:edit!` discarded unsaved edits without a captured delta. Neovim can detach line callbacks during this reload. The collector now reconciles its retained same-file shadow into a full-buffer `edit_delta`, with `change_origin=buffer_reload` and EOL/format metadata. Reload invalidation is an editor change, never an implicit typing rejection. New regression tests include Unicode, a chained edit after reload, and proposal dismissal without a false rejection.

The first installation of this fix still loaded the old Lua code. Nix-generated old/new plugin specs were both 1,749 bytes with timestamp 1, so Neovim's disk bytecode cache reused the older compiled spec. Actual module source inspection proved the mismatch. The installer now backs up and removes only the owned spec's exact cache key, including on rollback. Home Manager also invalidates that key after linking. Other cache files remain untouched. A regression reproduces the same-size/same-time collision and checks idempotence, rollback, and preservation of unrelated cache entries.

The final editor driver checks actual loaded module paths and SHA-256 before sending input. Prior UI-only passes do not establish collection correctness; their strict replay failures remain preserved. See `lua_cache_failure.json` and the detailed verification reports.

## Installed interaction and final verification

The ThinkPad's normal LazyVim wrapper now starts in persistent automatic experimental mode. Type, pause, and a completed eligible proposal appears without `:TabCompletePredict`. Alt+l accepts it. Continuing to type clears it immediately. Alt+Backspace explicitly dismisses it; Tab's existing completion/snippet mapping is preserved. `:TabCompleteMode off` cancels pending work and removes proposals. `:TabCompleteMode automatic` enables suggestions again. `:TabCompleteStatus` reports the uncalibrated quality status, model, request state, collector connectivity, and disabled personalization. `:TabCompleteModel` opens the picker; `:TabCompleteModel sweep` selects the optional Sweep artifact.

The final real persistent Neovim controller passed all eight scenarios on the installed 0.12.3 wrapper. It checked loaded source paths and hashes before input, triggered real Qwen inference by typing and pausing, accepted with Alt+l, undid only the acceptance, dismissed another proposal by divergent typing, typed a matching proposal, switched files, cancelled stale work, exercised the model picker, and restarted the editor. Automatic mode persisted, and the restarted editor inferred again. All scripted decisions are synthetic and unreviewed. Headless Neovim had no attached visual UI, so this establishes programmatic preview/event behavior, not visual human inspection.

The final main session is `3b127a5a-24e9-40bf-813f-58d14a8b873e`. Its 473 ordered events and six projections pass the strict database audit with 23 anchors, 31 deltas, zero replay mismatches, and zero unanchored deltas. Acceptance is recorded exactly once, undo restores the pre-state, and actual buffer deltas support the typing classification. The matching-typing case starts as an immutable partial-match observation and reaches an exact match through 19 linked deltas. The restart-only session `c00d6eb1-3b45-45c3-9532-f9e1c0722569` also replays cleanly. It does not repeat the other scenarios, so the full-coverage verifier reports missing scenario coverage for that session rather than a replay failure. Duplicate retries acknowledge 473/473 and 41/41 events with zero new ingestions and unchanged events, projections, blobs, and global counts.

For a concrete linked example, synthetic prediction `924bb649-6a5c-4554-802f-a89a0a32c074` references request event `ac271231-21cc-4b5e-b5ee-aad66c66322b`, display event `58714fde-bf2c-4260-92df-d43aebe6d8ad`, typing dismissal `9869e6b1-80dd-4c10-bc50-c2d06304db15`, and causal delta `ac6b1574-9339-4a5b-90cc-c15aaef43b38`. Its later delta `dd972d01-bff5-4e7d-bcc1-18d9aaf665c9` remains a separate observation. Navigation breaks the continuity needed to call later work a corresponding alternative. Context, raw response, canonical action, pre-state hash, and resolved server file identity all verify against the existing database.

In the first real automatic editor request, last input to request state was 250 ms, context/request-event emission added 23 ms, request event to first token was 2,243 ms, and first token to completed action added 298 ms. Last input to the shown event was 2,814 ms. Backend prompt processing was 2,232.602 ms and generation was 294.903 ms. These are one observed first request, not a median. The independent changed-state benchmark above provides the median and p95.

The live service was explicitly restarted after the editor window closed. PID 244220 restored saved alias `q25`, the expected model digest, and runtime hash `6e58a95c8f5fba5581784636cf5e042a33a14e93b7c764f681eb2ff4c10c54ba`. This hash differs from the serving-confirmation hash because the bundle paths changed for the collector reload fix; model, native binary, and compute settings match. The permanent Nix source pin is `f96d32f7e79945aea432a64ee88775313a58a713`, with fetch hash `sha256-d/sQ3Ikq0LX1lZkRZsGTQxa3xX2yNzkE5ZYzxlGfZjs=`. Focused Nix evaluation has zero failed assertions and one plugin target. The installed service/config were updated narrowly; unrelated dirty Nix changes remain untouched, and no full system activation was run. Open a fresh normal Neovim session to load the installed code; existing editor processes were not killed.

## Tests, readiness, and rollback

Actual repository verification passed 826 Python tests, 75 Lua tests, nine Rust tests locally and in the native Nix build, 59 collector-server Bun tests, 38 analysis Bun tests, and 16 gateway Bun tests. TypeScript checks passed for server and gateway. Ruff, Rust fmt/clippy with warnings denied, and mypy over 83 source files passed. The SQLite verifier adds 13 deterministic tests. The bytecode-cache collision regression passed installation, rollback, idempotence, and unrelated-cache preservation. The final real editor run passed eight of eight scenarios. Counts and exact commands are in `tests.json`; stub tests do not substitute for the real-model verification.

The refreshed private feedback export uses wire version 6 and reports 75 sessions, 107 proposals, zero candidate preference pairs, and zero defensible pairs. Four historical sequence gaps remain flagged. Automatic personalization is disabled and readiness is false. The clean synthetic audit proves the collection path, not human feedback or completed preference learning. No GPU allocation, training run, paid call, weight download, or requantization occurred in this pivot. Both candidate artifacts fit the laptop's 2 GiB allowance, and the service loads only one. Cleanup reclaimed approximately 3.5 GiB of this worktree's compiler cache while preserving checkpoints.

Rollback instructions and private backup locations are in [ROLLBACK.md](rust_editor_r1/ROLLBACK.md). The immutable bundle and its model paths remain stable. Scientific results, failure histories, source identities, final audit receipts, and hashes are in the adjacent report directory.

Remaining limitations are model accuracy, uncalibrated useful-display coverage, latency above the former 500 ms median goal, no visual human inspection, and no filesystem-cold measurement. They do not prevent explicitly opted-in experimental suggestions. No automatic edits or training are enabled.
