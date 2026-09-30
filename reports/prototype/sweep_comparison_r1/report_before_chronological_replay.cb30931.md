# Sweep Q8 and Q4 comparison

## Status

The user authorized this specific model, Q4 comparison, the existing benchmarks, and free Kaggle comparisons on 2026-09-30. No Sweep model has replaced the installed automatic q25 provider. The first GPU allocation failed during CMake configuration before model loading. The tested CUDA allocator fix compiled successfully in attempt2. Its benchmark stopped at the logging-based GPU guard before generating predictions. The local Q4/Q8 replay is complete,192requests; quality scoring and a new bounded recovery plan remain pending.

## Identity and format

The official [model repository](https://huggingface.co/sweepai/sweep-next-edit-1.5B) is pinned to `409016591c6c1a94f545f22328a85a3516118f34`. It declares Apache-2.0 and supplies Q8_0 GGUF, not an original F16 checkpoint. Actual GGUF inspection finds 1,445,014,016 tensor elements, Qwen2 architecture and 28 blocks. Q4_K_M is a requantization of that Q8 artifact. Results must describe this provenance.

The [publisher prompt implementation](https://huggingface.co/sweepai/sweep-next-edit-1.5B/blob/409016591c6c1a94f545f22328a85a3516118f34/run_model.py) uses related files, reversible recent diffs, original and current file content, and an updated-file header. It predicts the whole updated file. Neither a chat template nor the existing compact next-edit codec reproduces that task. The prompt implementation has been checked byte-for-byte against its pinned source.

The [Hugging Face Ollama bridge](https://huggingface.co/docs/hub/en/ollama) supports `ollama run hf.co/sweepai/sweep-next-edit-1.5B:Q8_0`. This does not establish correct next-edit behavior with Ollama’s default chat interface. The comparison uses pinned llama.cpp native completion and each GGUF’s tokenizer. No extra Ollama weight copy was downloaded.

## Completed compatibility diagnostics

Both exact Q8 and Q4 artifacts loaded on crabcake CPU, tokenized the fixed synthetic smoke, generated six tokens and ended with the actual GGUF EOS token. The smoke prompt contains 28 input tokens and runtime context512. It is a compatibility observation, not quality evidence.

| Diagnostic | Retained predictor RSS | Anonymous RSS | Swapped process memory |
| --- | ---: | ---: | ---: |
| Q8_0 default CPU repacking | 1,552,508 KiB | 40,048 KiB | 0 |
| Q4_K_M default CPU repacking | 1,516,704 KiB | 650,996 KiB | 0 |
| Q4_K_M with --no-repack | 913,712 KiB | 40,076 KiB | 0 |

The Q4 file is 883,289,056 bytes versus Q8’s 1,537,269,856. Default CPU repacking retains an anonymous weight copy, so file size alone substantially overstates the service-memory saving. Disabling optional repacking yielded about892MiB RSS on this tiny smoke with identical output. Normal-context memory, controlled latency and numerical action differences still need replay validation. The smoke timings were collected while other research processes were active and must not be presented as controlled latency.

## Comparison design

The unchanged200-case strict causal and180-case line continuation suites keep their96-token ceiling and scoring rules. These raw causal tests are task-mismatch controls for this next-edit-trained model. The24 fixed public/synthetic editor states use the publisher’s trained prompt and512-token full-file output ceiling, with two repetitions per precision. Worker inputs omit gold actions; history is reconstructed only from an exact reversible recorded synthetic delta. Full-file mapping requires unchanged bytes outside the editable range. No trimming or parser repair is allowed; cap hits and absent terminal events remain incomplete.

Planv1 is preserved as a pre-execution freeze. Review found missing strict-fixture enforcement, next-edit observability hooks and repeated model-file hashing; v2 preserved the same prompt/data/decoding and fixed these issues before any quality output. Allocation amendments v3 and v4 preserve those scientific settings and document the failed setup and bounded retry. The GPU worker privately stages the exact local Q4 artifact, avoiding assumptions about byte-identical conversion by a different GPU build.

## Existing product

The actual active `tabcomplete-predictor.service` was verified at2026-09-30T07:36:31Z, MainPID3343901, started05:01:13Z, NRestarts1. The installed q25 artifact still matches `a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626`. Automatic experimental suggestions remain enabled, quality uncalibrated, acceptanceAlt+l, off command`:TabCompleteMode off`, automatic personalization disabled. This snapshot does not imply uninterrupted process uptime or a Sweep deployment.

## Actual GPU setup failure

The private initial job was submitted at 2026-09-30T08:17:42.592904Z and reached ERROR. Its worker ran for 54.991668288 seconds, failing because CMake could not resolve `CUDA::cuda_driver`. No model was loaded and no quality prediction was produced. The fix resolves the existing driver via `ldconfig`, or selects the pinned runtime's supported CUDA allocator without virtual memory management. It still requires verified GPU offload and does not install drivers. The focused worker and submission tests pass, 39 tests in 0.67 seconds.

Authenticated quota was 44.59 GPU-hours remaining before attempt 1 and 44.57 at 08:34:03.962691Z after its failure, with renewal 2026-10-03. These balances are observed rounded values. Exact provider wall time is unknown. The interval from the pre-submit quota observation through the authenticated terminal observation bounds attempt 1 at 1,352.229974 seconds. Adding the two-hour retry gives a conservative 8,552.229974-second total below the four-hour campaign cap. Fresh quota and active-job checks remain enforced at submission.

## Submission metadata reconciliation

The first retry push was rejected, leaving a deliberately ambiguous `submission_started` ledger. Authenticated exact-status and full account notebook listing found no retry notebook; the initial notebook retained its previous execution timestamp. The retry title did not resolve to its requested slug. Installed Kaggle CLI source documents this mismatch. The original state and metadata were preserved privately, the title was corrected to match the retry slug, and a new fresh all-job/quota gate was started before retrying submission. The discarded API error text prevents a confirmed causal claim about the title. This administrative retry does not count as a GPU allocation unless the provider actually launches a job.

The actual retry was submitted at 2026-09-30T09:06:53.261529Z from committed source `fd6e548ae7cd572654314c0b4906dfb6bdc13bb5`. The fresh authenticated quota observation at 09:06:18.669494Z reports 44.57 GPU-hours remaining and no active jobs before allocation. Plan v5 preserves the scientific comparison and pins the final setup worker. The job is restricted to two session hours, including a 20-minute finalization reserve.

## Completed local replay

Crabcake completed 192 requests, 24 fixed public/synthetic source states × two repetitions × changed-state and immediate-repeat requests × two precisions, in 28 minutes 16.5 seconds. Both owned servers exited. This is varied-state replay, not an observation of ordinary human editing. Input lengths ranged from 113 to 1,107 tokens, median 319. It does not establish 2,048-token performance.

| Precision | Changed-prompt median / p95 | Exact-repeat median / p95 | Peak and retained predictor RSS | Retained PSS | Terminal / requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M, no repacking | 9.55 / 30.68 s | 2.82 / 9.29 s | 0.947 GiB | 0.942 GiB | 96 / 96 |
| Q8_0 | 12.09 / 36.60 s | 4.06 / 13.20 s | 1.557 GiB | 1.550 GiB | 96 / 96 |

The Q4 window initially overlapped private artifact staging; Q8 followed it. Desktop applications and the resident q25 service stayed present. The q25 process accumulated 0.03 and 0.05 CPU seconds respectively during the windows. Its roughly450MiB RSS is reported separately from each candidate, not included as candidate memory. No response hit the output cap. Immediate repeated output hashes matched46/48 pairs per precision, so cache execution is not claimed bitwise identical.

The correct backend cache fields are `timings.cache_n` for reused prompt tokens and `timings.prompt_n` for recomputed prompt tokens. Their changed-state medians were2 and317; exact repeats318 and1. Native `tokens_cached` instead reports post-request retained sequence tokens, and `tokens_evaluated` the full prompt size. The original counters and telemetry are preserved; their presence alone is not evidence of useful cache reuse.

Q4 remains within the memory policy. Q8 exceeds1.5GiB by60,784,640bytes and is a research measurement rather than an automatic-provider choice. Q4's process swap was zero. Host swap-in was175pages with no swap-out during its window; memory-pressure totals were13,775microseconds some /11,135full. Nonzero background swap occupancy is not characterized as thrashing. Both completed-action latency distributions are much slower than the interactive targets. These measurements establish compatibility and resource use, not useful edit quality.

## Actual second allocation failure

Attempt2 compiled the pinned CUDA runtime successfully, using its supported allocator without virtual memory management. It ran for1,948.321623seconds and stopped in `comparison_quality` after model health became ready. The guard could not find CUDA-device or offloaded-layer messages. Pinned `common/log.cpp` maps GGML info messages to verbosity4, while this server defaults to3. Thus missing logs do not establish CPU fallback. No quality predictions were generated. The guard remains mandatory. Plan v6 freezes verbosity4 and persisted startup diagnostics; plan v7 corrects its loader revision field before allocation. Both historical attempts remain preserved. The third job was submitted at10:31:15.194904Z from f1ede76 and verified RUNNING. Its immediately preceding authenticated quota observation,10:30:44.320548Z, reports44.03GPU-hours remaining,0.97used of45, renewalOctober3 and no active jobs before launch. Its two-hour deadline includes20minutes for finalization. The conservative prior bounds total4,734.404287seconds; adding this deadline gives11,934.404287below the14,400-second campaign cap. No quality outcome is inferred from submission.

## Local edit quality and prompt boundary

The frozen controller adapter scored48changed-state requests per precision,24states with two repetitions, using the unchanged gold and sandbox rules. Both Q4 and Q8 terminated48/48with EOS, but0/48mapped into the editable region. There were no cap hits. This strict control produced no safely usable actions and provides no basis for claiming useful edit quality or quantization equivalence.

All raw responses began with LF while their source files did not. A diagnostic examination found only2/24states per precision where that was the sole range mismatch. Most other responses changed or omitted the immutable suffix, or lacked enough body bytes to contain both immutable spans. No output was stripped, repaired, or rescored. These EOS responses were below the output cap, so the body omissions are not runtime cutoffs.

All192genuine local request IDs were recovered one-to-one by case, precision, request kind and timestamp, with no missing or ambiguous join. Exact repeated response hashes matched46/48per precision. The legacy summary's48identical actions are `None == None`; there were zero valid mapped-action pairs. The authoritative score directory is `/mnt/ssd/tabcomplete-product-r2/sweep_comparison_r1/local_replay_scoring_v2`, summary SHA`c234335964b0cb5e39a47bc2c63d3149a13b73a6e7f3162f884b6d5ba24272d9`.

A separately versioned input experiment will put one LF after the updated-file header before generation, preserve the publisher's field order, and require the same strict output mapping. Its outcomes must remain separate from this control. The unexecuted transition plans and original fixtures are preserved.

## Actual third allocation failure

Attempt3 ran2,027.052669427seconds before the same device-count guard stopped it. Retained startup diagnostics independently verify29/29layers offloaded to CUDA0,TeslaT4. Q8 device buffers were1,396.39MiBweights,224MiBKV and8.69MiBcompute; those are allocated device buffers, not CPU RSS or measured transient VRAM peaks. No quality predictions were generated. CUDA initialization happened when earlier GPU options were parsed, before the later verbosity4 flag. The next tested runner places that flag first and retains both backend guards. Another attempt requires a new allocation amendment and fits the same four-hour cap only with the verified prior bounds.
