# Automatic experimental suggestions and feedback, R1

## Installed result

The `crabcake` desktop runs the already selected adapted `q25-coder` Q4_K_M GGUF through the resident, CPU-only `llama-server` on `127.0.0.1:19093`. The GGUF is 397,807,232 bytes and has SHA-256 `a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626`. The binary SHA-256 is `064d67d7cc3d2a3fbbed39e6f1b1e779b481a01c23f55902ac1e5599ced1180c` (llama.cpp source revision `f072b10`). No model weights were downloaded or changed.

The installed LazyVim config opts into `automatic` mode while retaining `automatic_quality_validated=false` and `automatic_personalization_enabled=false`. It survives an editor restart. `<M-l>` accepts an eligible displayed proposal; `<M-BS>` explicitly dismisses it. `:TabCompleteMode off` cancels pending work and hides the proposal. Manual, shadow, and off modes remain available. The existing Tab mapping and collector configuration remain in place. Suggestions are drawn with an extmark or manual multiline preview; the editor never changes code until acceptance.

The automatic path waits 250 ms after relevant insert activity, rechecks the current state, tokenizes through the resident server, and sends a bounded 96-token compact-action request. One request is active per editor, one latest state can wait, and a shared local lock prevents multiple Neovim processes from queuing simultaneous server jobs. The client parses SSE records incrementally and requires a terminal EOS action before display. New input immediately removes a visible proposal and cancels an unseen client request; the next state waits for callback and lock release. The server slot released within 0.105 s of closing a client after the first output token in the cancellation probe. Three consecutive transport or output failures trigger a bounded backoff.

## Runtime comparison

The actual host is `crabcake`, AMD Ryzen 5 PRO 3400GE (4 physical cores, 8 threads), 13 GiB RAM, integrated Radeon Vega with `/dev/dri`; the pinned binary reports no usable GPU backend. The service uses four generation and prompt threads, context 2,304, one slot, batch 256, microbatch 64, optional cache budget 128 MiB, FP16 KV, and no speculative model. The system had 1.5 GiB swap occupancy when inventoried; the predictor itself had zero swapped bytes. Memory and I/O pressure `avg10` were 0.00 during the baseline measurement. Swap occupancy alone is not evidence of thrashing.

The fixed synthetic replay has 24 changed editor states over 512, 1,024, and 2,048 target-token buckets, with two repetitions. Actual token lengths and per-request backend timings are in the result JSON. The initial suite revision exceeded the context limit and stopped at HTTP 400; suite v2 reduced only synthetic filler and preflighted exact token counts plus the 96-token reserve. The failed attempt remains in the experiment log. All reported successful configurations use suite v2.

| Runtime setting | Changed-state median | Changed-state p95 | Peak service RSS | Retained RSS | Output hash differences from baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 generation / 4 prompt threads, batch 256/64, FP16 KV (installed) | 1.100 s | 11.959 s | 635 MiB | 618 MiB | 0/48 |
| 2/4 threads, batch 256/64, FP16 KV | 1.556 s | 13.817 s | 598 MiB | 598 MiB | 0/48 |
| 4/2 threads, batch 256/64, FP16 KV | 2.228 s | 20.851 s | 620 MiB | 620 MiB | 0/48 |
| 4/4 threads, batch 128/32, FP16 KV | 1.331 s | 16.644 s | 619 MiB | 619 MiB | 0/48 |
| 4/4 threads, batch 256/64, Q8 KV | 1.285 s | 15.919 s | 551 MiB | 551 MiB | 0/48 |

The original FP16 setting stayed within the 1.5 GiB service budget and was fastest on changed states, so the installer retained it. Q8 KV saved roughly 84 MiB at the observed peak but increased median and tail latency. The 48 equal output hashes are a narrow synthetic regression observation, not proof of model quality. Same-prompt cached requests completed in 0.362, 0.382, and 0.482 s across the three baseline buckets; the backend reported a median 960 cached input tokens in changed-state replay. Those same-prompt numbers are not everyday edit latency. In the real selected-model synthetic editor smoke, prompt lengths were 132–134 tokens, backend `cache_n` was 131–133, and last edit to completed display was 402, 375, and 386 ms, including the debounce. The smoke is short and favorable to prefix reuse.

The installed llama.cpp binary exposes standard FP16, Q8, and Q4 KV storage but no TurboQuant implementation or flag. [Google Research describes TurboQuant as KV-cache compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/); an [upstream llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) describes work outside this pinned binary. Integrating a fork or a new kernel was not justified for this bounded, already-under-budget workload. Q8 KV tested here is standard cache quantization, not TurboQuant.

## Existing database and verified feedback

The existing collector remains on `100.100.163.102:8787` with SQLite at `/mnt/ssd/collector-data/collector.sqlite`. A WAL-consistent backup was created at `/mnt/ssd/collector-data/backups/collector-before-automatic-20260925T191646Z.sqlite` before migration. The additive schema v2 migration applied successfully and rebuilt 58 historical prediction projection rows. Current raw events, blobs, file versions, and anchors were retained. A disposable copy rebuilt all 70 projection rows byte-identically and reapplied no migration. Replaying a delivered dismissal event against the live API returned `ingested=0`, `skipped_duplicate=1`; its raw event and projection remained single rows.

The last anchored synthetic smoke session is `d849500c-abda-4559-894c-36d3a94f97f9`. It used the real Q4 model without `:TabCompletePredict` and stored three automatic requests and displays:

| Prediction | Outcome | Request, display, and aftermath |
| --- | --- | --- |
| `f05aa50b-d6a4-4f7c-a994-5a1fe9329da7` | Accepted, then undone | Request, generated action, display, acceptance, edit and undo events are linked. |
| `913cfca4-08c0-46cf-b0a8-89e1289f697a` | Implicit typing dismissal | Display and dismissal link to the terminating `edit_delta`; later edits remain separate observations. |
| `f9929cce-7af0-456f-83cb-289e128ea93d` | Typed match | Matching typed text is recorded separately from rejection. |

The existing read-only replay reconstructed `main.py` from 10 anchors and 7 deltas with zero unanchored deltas or mismatches; the rebuilt and final files were both 63 bytes. All seven distinct context, action, and raw-response blobs referenced by that session were retrieved from SQLite and matched their SHA-256. An earlier synthetic run began typing before its initial asynchronous anchor and was correctly left unverified by replay. The exporter now flags missing pre-state anchors and sequence gaps before preference derivation. The current versioned export has zero defensible preference pairs; it lives under the collector data directory and is not committed. No acceptance rate or human preference claim is made from scripted actions.

The collector now stores generated action and dismissal events in the same ordered, idempotent event stream. A compact `prediction_projection` table is rebuildable from raw events and links request, display, terminal outcome, file identity, hashes, and sequence. It resolves a client file identity through its repository instead of treating that string as the server integer file ID. The exporter retains five-second, thirty-second, next-save, and subsequent-edit observations with censored and ambiguity flags. It never interprets all nonacceptance as dislike. `synthetic=false` does not set `human_verified=true`; automatic preference training remains disabled.

The existing observability CLI returned a healthy gateway and `no_runs` for the prior 24 hours at deployment time. This editor smoke therefore has collector session and prediction IDs, but no SigNoz run or trace ID. Scientific replay files and the collector database are the evidence for its behavior.

## Verification and limits

Automated gates: Python `239 passed, 1 skipped`; Ruff clean; Mypy clean (117 source files); collector Bun `56 passed`; analysis Bun `38 passed`; collector Lua suite `56 passed`; predictor safety, automatic state/SSE, and real-model headless smoke passed. Server and analysis TypeScript checks passed. The deterministic automatic tests cover debounce, idle no-edit, stale and late responses, divergent and matching typing, clean acceptance undo, focus loss, model outage and backoff, SSE UTF-8 chunk splits, malformed/incomplete output, and off cancellation. Existing collector tests cover spool loss/retry, delta reconstruction, Unicode, paste, and exclusion behavior. The actual startup reported automatic experimental mode and `<M-l>` after restarting Neovim.

Headless extmark state and command behavior were checked; a human did not visually inspect or accept a proposal. A synthetic external lock holder verified that a second editor waits and sends one latest-state request after release. Normal browser/LSP workload latency and two simultaneous real editor processes were not separately timed. The replay measures scripted synthetic edits, not production task accuracy. The p95 includes intentionally long capped generations, so the 500 ms / 1,000 ms goals are not met by the 24-state runtime replay. The user explicitly opted into experimental suggestions; stale-state and explicit-acceptance safeguards remain active.

All measurements and IDs are in [`automatic_feedback_r1/`](automatic_feedback_r1/). The original model artifact, personal data, and raw feedback were not committed.
