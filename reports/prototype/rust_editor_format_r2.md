# Editor diff preview, embedded models, and ThinkPad CPU profiling

Work starts at `72744ed` on `prototype/rust-editor-format-r2`. This iteration fixes the preview and packaging, measures the existing CPU engine, and preserves experimental opt-in. It does not train or change model weights.

## Escaped Rust output

The screenshot's literal `\\\"` bytes came from the model. The collector's SHA-verified raw SSE payload and canonical action contain the same replacement bytes. Transport decoded JSON once. Removing every backslash would corrupt legitimate strings and is not an acceptable fix.

The current model context quotes source as JSON while its action contract requires raw source text. A frozen public regression experiment compared the current prompt with an explicit reminder to decode context quoting. Both had 0/8 exact quoted-Rust actions. The reminder reduced parse-valid outcomes from 5/8 to 4/8, while the separate 24 development actions stayed 9/24 exact. We retained the current prompt.

A bounded Rust syntax guard now applies the exact proposed action to the exact prepared source in memory and rejects newly introduced parse-error fragments. It preserves genuine backslashes, Unicode, raw strings, and unrelated existing parse errors. It performs no source repair. This guard cannot prove semantic correctness or resolve missing enum variants. Invalid output remains evidence in the existing collector, not a human rejection.

## Preview and acceptance

The preview marks removed source bytes with red strikethrough and added bytes with a green background. It uses UTF-8-safe shared prefix/suffix boundaries. Cursor completions appear inline; full-line insertion uses an adjacent virtual line. Empty-line deletion has a visible marker. Preview drawing changes neither source text nor changedtick.

Accept in insert mode with `Alt+l`, represented by `<M-l>`, or run `:TabCompleteAccept`. Continuing to type dismisses the displayed proposal. `:TabCompleteMode off` cancels pending work. Tab and snippet mappings remain unchanged. Automatic experimental suggestions require explicit acceptance to alter code, and automatic personalization remains disabled.

## Embedded model packaging

The native loader reads the GGUF at an aligned internal offset of `/proc/self/exe`. Each executable maps its own inode; it does not extract a weight file or duplicate the model into an anonymous buffer. The small wrapper exposes the pinned native FILE-pointer API with a retained owned FILE and no transmute.

| Executable | GGUF SHA-256 | GGUF bytes | Protocol |
| --- | --- | ---: | --- |
| Qwen | `4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb` | 491,399,808 | single-line-edit-v1 |
| Sweep | `936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4` | 883,289,056 | sweep-full-file-v1 |

Both use existing Q4_K_M weights. The two final ThinkPad Nix payload executables total 1,393,039,654 bytes. Their runtime closure is 1,441,657,936 NAR bytes and contains neither external GGUF inputs nor the unembedded engine. Existing raw model inputs and old Nix generations remain preserved; their storage is additional. No weights are committed or published.

Home Manager selects one fixed executable through `modelVariant`. Status reports the selected identity and declarative selection. A fixed executable does not offer a runtime model picker. Only one model serves requests.

## Measured CPU work

The target is `shlokthinkpad`, an i7-8650U with four physical cores, eight logical CPUs, AVX2, and about 23.2 GiB RAM. Inference is CPU-only. The existing native engine already uses x86-64-v3, greedy decoding, one resident slot, a bounded context, and verified token-prefix KV reuse. It saves no optional context snapshots.

The screenshot request spent 1,714.518 ms processing 321 input tokens and 332.521 ms generating 13 tokens. The 64-token microbatch requires six native decode calls for that uncached prompt. This motivates the bounded microbatch and prompt-thread screen. Identical-prompt repeats are reported separately from changed editor states.

Performance results and final declarative configuration are recorded in the accompanying machine-readable receipts. A temporary benchmark restores the unchanged original service; it does not activate Home Manager or NixOS configuration.

## Verification and limits

The scientific result files are authoritative. Scripted decisions are synthetic and do not establish human attention, acceptance, or model quality. The collector remains the existing SQLite database at `/mnt/ssd/collector-data/collector.sqlite`; this change does not reset it or introduce a new production database.

The project and target instructions reserve `nrs` activation for the user. Configuration can be prepared and evaluated here, but the running old generation stays active until that activation. Existing mode settings and rollback generations remain available.

## ThinkPad latency results

The frozen v2 screen completed six settings, with 48 changed-state and 24 identical-prompt requests per setting: 432 completed requests. All completed requests reached actual EOS and decoded valid canonical actions. These are synthetic runtime checks, not task correctness. Actual prepared inputs were 430–500 tokens; the 512/1,024/2,048 fixture labels describe source buckets before context selection.

| Configuration | Changed-state median | Changed-state p95 | Peak RSS | Action differences versus baseline |
| --- | ---: | ---: | ---: | ---: |
| 4 decode / 4 prompt threads, microbatch 64 | 795 ms | 2,619 ms | 554 MiB | Reference |
| Microbatch 128 | 951 ms | 3,126 ms | 559 MiB | 2/72 |
| Microbatch 256 | 704 ms | 3,097 ms | 569 MiB | 2/72 |
| 8 prompt threads | 1,269 ms | 6,472 ms | 555 MiB | See paired receipt |
| 2 decode threads | 863 ms | 3,554 ms | 554 MiB | See paired receipt |
| Native weight repacking enabled | 827 ms | 3,158 ms | 583 MiB | 6/72 |

These times run from client context construction through completed response parsing. They exclude the 250 ms editor debounce and do not measure last keystroke to visible proposal. Backend baseline medians were 604 ms prompt processing and 111 ms generation. CPU processing of changed context is the largest measured stage; median context construction was 36 ms. Stage medians are not additive per-request timing decompositions.

Verified prefix reuse supplied 12,695 cached and 9,677 recomputed input tokens across the 48 baseline changed-state requests. Identical-prompt repeats supplied 10,824 cached tokens and recomputed 24, with 206 ms client median and 334 ms p95. That repeat condition must not be advertised as ordinary editing latency.

The baseline uses all four physical cores. Eight logical prompt threads oversubscribed the service's four-core CPU allowance and performed worse, with substantial measured throttling. Larger microbatches changed two action outcomes; microbatch 256 improved the median but worsened p95. Repacking increased anonymous memory and did not improve completed-action latency. No experimental setting was promoted. The final declarative configuration retains 4/4 threads, microbatch 64, batch 256, FP16 KV state, context 2,304, input limit 1,024, one slot, and zero saved snapshots. Qwen's output cap remains the existing 64, below the 96-token ceiling.

Baseline peak PSS was 550 MiB, including about 83 MiB anonymous and 468 MiB file-backed PSS. Sampled process swap and host swap-in/out increments were zero, memory PSI remained zero, and no OOM events occurred. File pages can be charged outside the service cgroup, so process RSS/PSS, rather than the small cgroup charge alone, is authoritative for model memory. The final Nix Qwen smoke sampled 555 MiB RSS; Sweep sampled 937 MiB, one process at a time.

Browser and user workloads were left running. Thermal-zone readings reached 99–100°C without verified sensor names, and clocks varied. The deadline stopped the final baseline drift check before a complete receipt, so thermal drift remains unresolved. The native instruction profiler was unavailable; these measurements provide stage timing and Linux resource samples, not instruction-level cache-miss traces. They do not establish an isolated controlled-desktop comparison or an acceleration-backend comparison.

### Plan integrity incident

The first screen's runner was formatted and its plan rewritten after freezing. Two completed preliminary settings (144 requests) are preserved but excluded. Original frozen bytes could not be recovered, and the manifest records that limitation. A new v2 plan fixed the Linux counter parser and froze the runner again, retaining the original deadline. No extra time was granted. The incident and incomplete final drift check are recorded in `performance_freeze_incident.json` and `performance_summary_v2.json`.

## End-to-end editor and database evidence

The actual final Nix Qwen executable and updated source-loaded plugin passed `tests/automatic_real.lua` on ThinkPad Neovim 0.12.3. Four automatic requests produced three displayed proposals. The installed Alt+l callback accepted one, undo restored its pre-state, divergent typing dismissed another, an exact typed match was recorded, and a file switch cancelled an unseen request. A headless insert-mode seam and scripted buffer changes are explicitly synthetic. No GUI appearance or human acceptance is claimed.

Session: `85f0fdf1-06a0-419d-8e31-9f5e9a50d2ad`.
Typing dismissal: `42cbd097-d535-424d-82ca-bcaa3e08bdde`.
The authoritative SQLite audit links requests, generated actions, displays, decisions, and exact subsequent deltas; every proposal remains unreviewed with `human_verified=false`.

The session contains 46 raw events, four projections, 15 anchors, and eight deltas. The existing read-only Bun replay command exits 0: all deleted bytes agree and the rebuilt 484 bytes equal the final anchor. Redelivery of all 46 events ingested zero and skipped 46 duplicates, leaving projections and global counts unchanged. All 199 database blobs passed decompression/SHA verification. SQLite integrity is `ok`, with zero foreign-key issues. Schema 3 and classifier 3 already support these observations; no migration or new production database was needed.

The broader existing synthetic-session auditor exits 1 only because this three-display smoke does not include its additional partial-match-then-eventual-full-match episode. That coverage failure is preserved in `database_evidence.json`; the replay, identity, payload, acceptance, undo, dismissal, and deduplication checks all pass. Unit coverage includes partial-match behavior, but is not claimed as that missing persistent-session episode.

## Tests, monitoring, and storage

| Verification | Actual result |
| --- | --- |
| Repository Python | 826 passed, two reported deprecation warnings |
| Ruff / mypy | Passed / 84 source files passed |
| Rust release / strict Clippy | 18 passed / passed |
| Bun collector / analysis | 59 passed / 38 passed |
| TypeScript collector / analysis | Both passed |
| Neovim full suite | 76 passed locally and 76 on ThinkPad |
| Additional Lua scripts | Five passed, plus model-picker checks |
| Actual Nix build | Passed, private payload reference checks passed |
| Actual model automatic smoke | Exit 0; synthetic decisions |
| Read-only SQLite replay | Exit 0, zero mismatches |

The first Python run lacked existing ignored historical fixtures; linking the existing immutable fixtures enabled the complete green rerun. The first remote Lua staging omitted a shared fixture; preserving repository-relative paths enabled the complete green rerun. Neither failure was hidden as a passing run.

Existing SigNoz received 139 quote-probe spans and 1,012 CPU spans. Quote run `run-34e1f2a99de323a6ed43d948f143ea41` has retrieved public input/output artifacts with matching hashes. The actual faulty user request was not found in SigNoz; its collector payload is authoritative. CPU telemetry verification queried a bounded first page, not a claimed full paginated audit. No private source was sent to external APIs and no prompts were duplicated into metric labels.

Research scratch including compiler outputs and temporary conversions remained about 2.93 GB. Final target model payloads remain below 2 GiB; retained source GGUF inputs and old generations are additional. No model download, training, cloud allocation, paid call, or automatic personalization occurred.

## Declarative activation and rollback

Only `~/nixos-config/pkgs/tabcomplete-engine/default.nix` and `~/nixos-config/home/features/tabcomplete/default.nix` were changed. Unrelated file digests match before/after. They pin code `53aec2f845ac6c0f2840831079d4d68e0fa7b06a`, select Qwen, enable experimental automatic suggestions, retain quality validation false, and keep training disabled. Full Home Manager evaluation reports zero failed assertions. Both final payloads were built and tested on the target, but the new generation has not been activated.

Run the usual `nrs` on the ThinkPad and restart Neovim to activate this preview and embedded engine. This respects the target's user-only activation instruction. The original automatic experimental service was restored active after every temporary benchmark and smoke; no persistent service unit was edited imperatively.

Before the change, configuration and the active service unit were backed up at `/home/shlok/.local/state/tabcomplete-install-backups/editor-format-r2-20261002T035122Z`. Restore the two owned files from that directory, run the usual user-owned activation, and restart Neovim to roll back. A WAL-consistent collector backup is `/mnt/ssd/collector-data/backups/pre-editor-format-r2-20261002T033940Z.sqlite`; no database restore is necessary for code rollback.

Remaining limits are model quality on quoted Rust, lack of GUI visual inspection, the unactivated generation, the incomplete thermal drift check, and the persistent partial-to-full feedback episode absent from this smoke. The current automatic prefix guard suppresses destructive replacements in automatic mode; manual `:TabCompletePredict` uses the full replacement/deletion preview. None of these results establish production accuracy.
