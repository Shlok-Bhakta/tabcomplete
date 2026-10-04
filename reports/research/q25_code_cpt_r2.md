# Qwen2.5 code continued-pretraining experiment

This experiment tests the user's request to teach the tiny model more code
before training completion or next-edit behavior. It starts from the untouched
Qwen2.5-Coder-0.5B Base checkpoint. The working ThinkPad editor model is preserved.
The experiment has not established improved model quality merely by preparing
data or passing trainer tests.

The [frozen plan](q25_code_cpt_r2/plan.json) binds the source pool, model and
tokenizer hashes, fixture hashes, split policy, decoding, runtime choices,
training schedule and limits. The initial cap is 8,000,000 nonpadding input
tokens and one pass. Four language allocations cover Python, Rust, TypeScript
and Go. The actual token counts may be lower if a clean source allocation is
exhausted. No duplicates will be added to fill a quota.

## Source and interpretation

The existing private research source pool contains original public code from
Stack dedup revision `17cad72c886a2858e08d4c349a00d6466f54df63`.
It is retokenized with Qwen2.5's own tokenizer. Historical Qwen3.5 token IDs are
not reused. The original reserved repository buckets 0 through 29 remain
excluded. Buckets 970 through 999 become a new Qwen2.5 development reservation;
they were historical training-pool repositories, not the earlier reserved test
set. Repository aliases and exact content identities must remain separate across
the new training and development splits.

The data records retain per-file license metadata. This experiment restricts
that metadata to MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD, Unlicense
and CC0-1.0. Dataset metadata is not a blanket Apache license. Research inputs and
candidate weights remain private. No external teacher calls or new model weight
downloads are needed.

The [Stack dedup dataset card](https://huggingface.co/datasets/bigcode/the-stack-dedup/blob/main/README.md)
describes approximately 1.5 TB of source. This bounded
subset is an ablation, not an entire-corpus campaign. Qwen2.5-Coder already has
extensive code pretraining, so extra raw code may have little effect on editor
intent. The unchanged 200-case causal suite and 180-case continuation suite are
code diagnostics. They do not establish next-edit ability. A future matched
completion adaptation must compare the untouched base with this CPT candidate
using the same ordered training examples and separate development data.

## Budgets and recovery

The authenticated observation at 2026-10-03 23:27:58 UTC reported 45.00 account
GPU-hours available, renewal 2026-10-10, with no active kernels among 43 checked.
The controller refreshes quota and job state immediately before allocation.
This observation does not guarantee a future balance.

The campaign reserves at most 20 aggregate session wall-hours and conservatively
40 account GPU-hours. Each private Kaggle allocation has a four-hour hard limit,
with at least 30 minutes reserved for saving and final evaluation, increased
using measured evaluation and checkpoint-save time. One GPU notebook allocation
may run at a time. Training uses one T4; a second visible device is not treated as
free quota. Failures, preparation inside the allocation, evaluations and saving
consume the session budget. Renewed allocations cannot be consumed automatically.

Resumable state includes master weights, optimizer, scheduler, loss scaler, RNG
and the completed-batch data cursor. Checkpoints are written atomically and
marked complete with their hash and fingerprint. An export alone is not a
training resume state. Only q25-owned obsolete checkpoints may be removed after
the replacement is complete. Existing research checkpoints are preserved.
New research artifacts, including temporary writes, have a 12 GiB cap.

## Verification status

CPU preparation completed with 7,872,512 training input tokens in 7,688 blocks
and 131,072 development input tokens in 128 blocks. There are 6,121 training
feeder documents and 120 development feeder documents. The Go allocation ran
short after filtering, so the corpus retains its actual size. Independent checks
found zero content or repository-alias overlap between splits. All 86 raw source
records overlapping benchmark repositories were excluded before tokenization.
The provenance and hashes are in `preparation_verification.json`.

The CPU trainer preflight resolves 481 updates for one pass and 7,864,824 causal
target tokens. The first source token in each block has no preceding position to
predict it. The private input dataset upload passed remote path and size checks;
the worker verifies its content hashes before training. No model weights were
downloaded. The worker reuses the existing private dataset containing the
identical untouched Qwen checkpoint.

The logical corpus cap is 8 million input tokens. The separate processed-token
cap is 12 million, including a 4 million allowance for replaying an interrupted
uncommitted tail. This allowance does not authorize an extra pass over the
source corpus. Every resumed session carries that counter forward.

The complete repository verification passed 862 Python tests, Ruff, mypy across
85 source files, 16 gateway Bun tests and 59 collector Bun tests. Both service
type checks passed. Initial Python failures came from two ignored historical
artifacts missing in the isolated worktree; the exact existing files were
restored without changing the tests. The subsequent full run passed.

Attempt 1 was submitted from commit `fce293d0074d38817810434d74051da243d30e2d`
at 2026-10-04 00:23:07 UTC. Its actual private notebook is
`shlokbhakta/tabcomplete-q25-code-cpt-r2-attempt-1`. Authenticated status reported
`RUNNING`. The immediate pre-allocation observation reported 45.00 account
GPU-hours remaining, no active notebooks, and the unchanged October 10 renewal.
Kaggle used the title-derived slug despite the requested short ID. The receipt
was reconciled against the returned URL and authenticated status; no duplicate
allocation was submitted. The controller now records returned notebook URLs.
This reference correction does not change data, prompts, model or training.

Kaggle has not exposed progress/output artifacts during the running session.
Completed training tokens, quality changes and reload checks remain unknown
until their respective execution records exist.

The existing SigNoz CLI query for the historical failed kernel identifier
`tabcomplete-one-line-instinct-pilot-r1-retry` returned `no_runs`. Its private
worker status and scientific artifacts remain available, but a training trace
was not recovered under that identifier. New disconnected worker telemetry uses
the existing offline mode and will be imported after artifact retrieval.

Automatic personalization remains disabled. No CPT checkpoint is automatically
promoted into the editor.
