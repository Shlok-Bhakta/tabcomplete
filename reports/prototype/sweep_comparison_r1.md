# Sweep next-edit 1.5B comparison

## Scope and current evidence

This campaign evaluates the user-authorized Sweep checkpoint against the existing
strict 200-case and line 180-case code suites, its publisher's next-edit prompt,
and local editor-state replay. The installed q25 automatic experimental service
and collector remain unchanged. No training, teacher, draft model, or other
model weights were used in this comparison.

Both local replays are complete. The fourth Kaggle allocation reached ERROR,
observed at 2026-09-30 12:11:59 UTC, without scientific predictions. A shorter
retry is being prepared under a new frozen plan and the existing four-hour cap.
This report does not mark the overall comparison or useful-model objective
complete.

## Model and precision identities

Source: [sweepai/sweep-next-edit-1.5B](https://huggingface.co/sweepai/sweep-next-edit-1.5B/tree/409016591c6c1a94f545f22328a85a3516118f34),
immutable revision `409016591c6c1a94f545f22328a85a3516118f34`, Apache-2.0.
The checkpoint has 1,445,014,016 GGUF tensor elements in 339 tensors, a dense
28-layer Qwen2 architecture. Its own tokenizer and EOS ID 43816 are used.
No per-token perplexity comparison across tokenizers is made.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Publisher Q8_0 v2 | 1,537,269,856 | `1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da` |
| Local Q4_K_M | 883,289,056 | `936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4` |

The publisher repository has no unquantized weights. Q4 was requantized from
the exact publisher Q8 artifact with the pinned llama.cpp quantizer. Results
therefore measure Q8-to-Q4 deployment differences, not FP16-to-Q4 loss.
Quantization took 50.947 seconds with four CPU threads. The manifests record
all downloads, remote sizes, runtime hashes, and conversion provenance.

The Hugging Face GGUF bridge is documented in the
[Hub guide](https://huggingface.co/docs/hub/en/ollama).
The user's `ollama run hf.co/sweepai/sweep-next-edit-1.5B:Q8_0` form addresses
that bridge. A chat conversation through it is a different task from Sweep's
trained prompt. This campaign uses one pinned llama.cpp runtime and avoids a
duplicate Ollama weight download. The publisher also has a direct
[Ollama listing](https://ollama.com/sweepai/sweep-next-edit), with the command
`ollama run sweepai/sweep-next-edit`. Its registry artifact identity was not
verified against our pinned Hub file, so the listing is not another tested
precision result.

## Completed local baseline

Before the replay, both precisions loaded, tokenized a fixed 28-token synthetic
prompt, generated six tokens, and reached actual EOS. The default CPU Q4 smoke
retained 1,516,704 KiB RSS, including 650,996 KiB anonymous memory. Disabling
optional weight repacking reduced it to 913,712 KiB RSS, with 40,076 KiB
anonymous memory and identical smoke output. Q8's default smoke retained
1,552,508 KiB RSS. Thus Q4 file size alone did not establish a small resident
service. The smoke ran amid research activity and is compatibility evidence,
not controlled latency or task quality. The replay below uses `--no-repack`.

Inference ran on crabcake, AMD Ryzen 5 PRO 3400GE, four physical cores and eight
logical threads, approximately 13.6 GiB RAM. The binary was CPU-only. This was
the desktop with ordinary applications and the resident q25 service present,
not an empty controlled machine. Initial Q4 measurements overlapped artifact
staging, which is retained as a limitation. Q8 ran after that staging finished.

Runtime revision `f072b103714dfa1eee531f80b24512faf38e3dd2` used four generation
and prompt threads, one slot, context 3,072, batch 256, microbatch 64, F16 KV,
zero optional cache RAM, zero context checkpoints, no idle cache, no context
shift, no draft model, no warmup, and no weight repacking. The active slot state
remained available for normal cache reuse.

The baseline made 192 requests. Per precision it measured 24 independent source
snapshots with two repetitions and immediate same-prompt probes. It was not a
chronological editor trajectory. Inputs ranged from 113 to 1,107 actual tokens,
median 319, so it did not cover the intended 2,048-token bucket.

| Precision | Changed-prompt completed-response median / p95 | Peak and retained RSS | Peak and retained PSS | EOS / changed requests | Safe mapped actions |
|---|---|---|---|---|---|
| Q4_K_M | 9,554.612 / 30,681.616 ms | 1,017,303,040 B, 0.947 GiB | 1,011,405,824 B, 0.942 GiB | 48/48 | 0/48 |
| Q8_0 | 12,086.885 / 36,600.393 ms | 1,671,397,376 B, 1.557 GiB | 1,665,375,232 B, 1.551 GiB | 48/48 | 0/48 |

These are completed full-file response times. Valid canonical-action latency is
undefined because no output mapped. Q8 exceeded the 1.5 GiB service limit by
60,784,640 bytes, approximately 58 MiB, before counting any additional helpers.
The separate resident q25 service used approximately 450 MiB RSS and had almost
no CPU activity during the measurement windows. Its memory is not folded into
Sweep's predictor process measurement.

For Q4, process swap remained zero; host swap-in was 175 pages and swap-out zero.
Nonzero background swap occupancy alone does not establish thrashing. Memory
PSI increased by 13,775 microseconds for some pressure and 11,135 for full
pressure. I/O PSI increased by 3,072,214 and 2,960,345 microseconds respectively.
Maximum recorded CPU temperature was 77.5 degrees C, mean sampled frequency
approximately 3,020 MHz, and minimum available RAM 11,114,774,528 bytes.
Detailed raw measurements preserve both precision windows and sampling limits.

Backend `timings.cache_n` and `timings.prompt_n` establish actual input-state
reuse. Their baseline changed-request medians were 2 reused and 317 recomputed
tokens; immediate repeats had medians 318 reused and one recomputed token.
The server's post-request `tokens_cached` includes retained generation state
and is not itself evidence of reused input. Immediate repeats were much faster,
but are not everyday changed-state latency.

## Completed baseline quality diagnosis

Both precisions terminated with EOS below the 512-token full-file cap. Every
output started with LF, while the source files did not. The publisher's exact
prompt ends at the updated-file header without LF. The score did not strip any
generated byte. A separate diagnostic that ignored the first LF still found
incompatible or absent immutable suffixes in most outputs; only two of 24 states
per precision had a mismatch explained by the separator alone.

No patch reached the functional source oracle, so functional edit correctness
is unassessed. Zero mapped false-positive edits is not a zero false-positive
rate. Null-action equality also does not establish valid-action consistency.
The [complete interpretation](sweep_comparison_r1/local_replay_interpretation_v3.md)
and its JSON manifest retain the exact historical producer/evaluator identities.

## Separate chronological replay

The [new frozen plan](sweep_comparison_r1/changing_state_plan-v5.json) used 24
actual synthetic editor transitions, two repetitions, and both precisions.
Its `publisher-header-lf-input-v2` policy appends one LF to the input header,
preserving all other prompt information. It does not modify generated output.
The [revision note](sweep_comparison_r1/changing_state_revision.md) records the
distinct fixture/prompt identity, interrupted seven-request attempt, and strict
90-minute aggregate local budget. All 96 requests completed, 48 per precision.
Both research server processes exited; the installed q25 process remained
running.

| Precision | Completed-file response median / p95 | Range-valid actions | Valid-action median / p95 |
|---|---|---|---|
| Q4_K_M | 6,899.9 / 22,314.9 ms | 38/48 | 8,365.8 / 32,482.5 ms |
| Q8_0 | 8,681.1 / 27,825.8 ms | 32/48 | 9,965.2 / 39,399.1 ms |

These latency populations differ. Range validity does not prove that an edit
is useful or task-correct. All mapped actions were replacements containing a
newline. There were zero no-edit or deletion actions, and zero mapped actions
eligible for the existing initial one-line automatic display. Each original
editable range itself contained zero LF bytes. Multiline output was retained,
never truncated into a one-line suggestion. All requests ended with EOS and
none hit the cap. No functional gold labels or human feedback were loaded for
this replay. Q8 peak RSS was
1,671,602,176 bytes, again above the 1.5 GiB limit. The new prompt policy and
fixtures both differ from the baseline; the table does not measure a matched
causal gain from adding LF.

Actual immediately preceding request tokens shared a prefix in 47/48 requests
per precision, median 1,010 tokens, p95 1,726, maximum 1,813. Backend input reuse
had median 1,010 tokens and recomputed input median 193. This reflects changing
states, not identical-prompt repetitions. Legacy post-request counters retain
their original meanings.

All 48 precision pairs used identical effective prompt hashes. Raw full-file
and output-token-ID hashes matched in 32/48 pairs and differed in 16/48. Of
32 pairs where both actions mapped, 26 canonical actions matched and six
differed. Six pairs mapped only in Q4 and ten mapped in neither. Null actions
are excluded from action agreement. This measures numerical precision
sensitivity, not a functional quality drop or proof that Q4 is better.
The [full interpretation and artifact manifest](sweep_comparison_r1/changing_state_replay_v5/interpretation.md)
provide exact timing, cache, memory, pressure, and fingerprint details.

## Kaggle execution and limits

The first three allocations produced no scientific predictions. The first
failed CUDA configuration; the next two failed startup evidence guards after
successful compilation. The third's retained diagnostics prove CUDA offload,
but its device-count log was suppressed when a GPU option initialized CUDA
before the later verbosity flag. An ordered-argument regression test reproduced
the defect. The corrected runner sets verbosity first and keeps both guards.

Attempt three independently recorded 29/29 layers offloaded to CUDA0, Tesla T4,
with 1,396.39 MiB Q8 device weights, 224 MiB KV, and 8.69 MiB compute buffers.
Those prove placement for that startup, not a completed benchmark or measured
transient VRAM peak. Its failed guard and all three failure manifests remain
preserved. No setup failure is counted as a model quality failure.

Fourth allocation `shlokbhakta/tabcomplete-sweep-comparison-r1-final` submitted
at 2026-09-30 11:33:55.280142 UTC under
[plan 8](sweep_comparison_r1/plan-v8.json). Authenticated quota at 11:33:23 UTC
reported 43.46 remaining of 45 account GPU-hours, 1.54 used, renewal
2026-10-03 00:00:00. There were no other active jobs. Those are observations at
that time, not a current balance.

The frozen comparison caps aggregate conservative session bounds at four hours,
one allocation at a time, and reserves 20 minutes of the two-hour final session
for finalization. Prior bounds total 7,065.335220 seconds; adding the final
7,200-second limit gives 14,265.335220 seconds. These bounds include queue and
observation delays and are not exact billed GPU usage. No Kaggle training ran.

The fourth worker ran 1,896.346900338 seconds. Its logs again prove 29/29
CUDA0 layer offload, with two enumerated T4 devices, while the one-time
initializer message remained absent. The original guard still rejected it.
The [new evidence revision](sweep_comparison_r1/backend_evidence_revision_v2.md)
records an actual-log-backed parser fix with mandatory inventory and offload
checks. Three regression tests failed before the fix; all 22 targeted tests
passed afterward. The new interpretation does not create predictions from
the failed allocation.

The GPU runner preserves the existing raw-causal strict200 and line180 prompts,
96-token ceiling and scoring rules. They are task-mismatch code controls for a
full-file next-edit model. It also runs the fixed publisher next-edit prompts.
No chat template, parser rescue, changed fixture, sealed training input, or
hidden-test-selected truncation is used. CUDA host RAM and VRAM are measured
separately; snapshots are not claimed as transient VRAM peaks.

## Selection and operational state

Current evidence does not justify replacing the installed q25 service with
Sweep. Q4 fits the process memory limit, but the original local task produced no
safely applicable actions and multi-second full-file latency. Q8 exceeds the
memory limit. The ongoing experiments may refine that conclusion; neither
precision is promoted because of model size or release date.

The existing desktop installation remains automatic experimental, quality
uncalibrated, with Alt+l acceptance and `:TabCompleteMode off`. Source changes
still require explicit acceptance. The existing collector database, raw edit
history, rollback configuration, and disabled automatic personalization remain
intact. This campaign records no genuine human acceptance and performs no RL.

The baseline observability bundle imported 500 spans and its controller score
bundle two. Actual paginated run, request, and trace queries passed. Example
request `request-968a672e-06ef-48ea-a855-17ee4f51267d` belongs to baseline run
`run-8a8683a51d167629819c24ff871b3e4a`; its generation trace is
`eb64ded111f1a34a96ae0515192b658d`. Content capture was disabled for that baseline,
so no captured-payload retrieval is claimed. GPU content capture is authorized
only for the public/synthetic fixtures and still requires actual verification.

The [pre-chronological report](sweep_comparison_r1/report_before_chronological_replay.cb30931.md)
is archived verbatim from commit `cb30931`. Historical source, fixture, plan,
submission, and result identities are preserved in the versioned companion
directory and Git history.

## Repository verification snapshot

After the local replay, the actual workspace Python suite passed 694 tests
with one skip in 117.56 seconds. Ruff passed and expanded mypy passed across
85 source files. These counts precede the four new CUDA-inventory regression
cases and the forthcoming shortened-worker changes; the relevant revised
runner suite has separately passed 22 tests.

Fresh Bun results were gateway 16, collector server 56, and analysis 38 tests,
all passed. All three TypeScript checks passed. Lua collector tests passed
56/56; predictor safety, automatic/SSE state-machine, and isolated two-process
slot scripts also passed. Those are integration regressions with synthetic
transport and decisions, not model accuracy or human acceptance. Private logs
are retained in the final-verification artifact directory. The Python snapshot
includes preserved uncommitted research work and is not misrepresented as a
clean checkout-only CI run.
