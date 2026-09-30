# Sweep Q8 and Q4 comparison

## Status

The user authorized this specific model, Q4 comparison, the existing benchmarks, and free Kaggle comparisons on 2026-09-30. No Sweep model has replaced the installed automatic q25 provider. The first GPU allocation failed during CMake configuration before model loading. A tested CUDA driver-library lookup fix is being prepared for the one bounded retry. Ordinary-context CPU replay has now started; quality and replay results are still pending.

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
