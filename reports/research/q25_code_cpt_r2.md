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
lists 1.5 TB for v1.0, 3 TB for v1.1 and 2.7 TB for v1.2 of the
near-deduplicated source. The reused source revision still matches the current
Hub revision at this check. This bounded subset is an ablation, not an
entire-corpus campaign. [Qwen's model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
reports 5.5 trillion training tokens for the coder family. Extra raw code may
have little effect on editor intent. The unchanged 200-case causal suite and 180-case continuation suite are
code diagnostics. They do not establish next-edit ability. A future matched
completion adaptation must compare the untouched base with this CPT candidate
using the same ordered training examples and separate development data.

## Matched completion follow-up

The CPU preparation plan freezes a supervised FIM comparison after the raw-code
pass. One arm starts from the untouched Qwen checkpoint; the other starts from
the verified completed CPT export. Both use the same ordered examples, one pass,
response-only loss, and supervised EOS. This is synthetic source completion,
not observed next-edit intent.

Preparation uses actual source lines or the remainder after a UTF-8 cursor.
It keeps the original PSM markers and never truncates labels. The reserved
development pool contains only 120 eligible source documents, yielding 240
distinct states. It is not duplicated to reach the requested 512-state target.

A CPU serving check found two revision-2 training prompts whose decomposed
Unicode was normalized by Qwen's NFC tokenizer. Revision 3 requires exact
prompt and target token/string round trips before deterministic selection.
The old prepared files and failure evidence are preserved. This correction
preceded every FIM model output and does not change CPT data or benchmark fixtures.

Each FIM allocation has a three-hour deadline including setup, evaluation and
saving. The shared campaign cap is 32 million processed input tokens, 20 reserved
session hours and 40 conservatively charged account GPU-hours. Raw CPT retains
its own 12-million processed-token cap. The controller preserves budget for both
completion arms and accounts for interrupted tails without summing the same
cumulative cursor twice. These limits implement the user's separate authorization
for further free Kaggle work; they do not extend the historical model-data-r2
exception or consume a renewed allocation.

| Path | Completed training | Development completion | General code regression |
| --- | --- | --- | --- |
| Untouched Qwen → FIM | Pending | Pending | Pending |
| Qwen → raw-code CPT → FIM | Pending | Pending | Pending |

The final training plan must bind the actual completed CPT artifact and runtime
before either FIM allocation. Existing tests and CPU data checks are not evidence
of a model improvement.

The full training plan is now frozen at revision 3. Both actual initializer
preflights passed all 4,336 prepared states, with 1,776,908 total training input
tokens and 34,405 supervised target/EOS tokens per arm, and 256 batches of 16.
Revisions 1 and 2 are preserved. Revision 3 corrects the runtime report
checksum before GPU allocation. CPU preflight found that Qwen's reserved FIM markers
are ordinary added tokens, despite having fixed control IDs, and Transformers 5
normalizes RoPE and disabled-window configuration fields. The corrected checks
compare exact token IDs and canonical architecture meaning while preserving
file-hash verification. Generation now rejects all Qwen added control tokens.
This revision preceded every FIM model generation; it changes neither source
states nor model weights, tokenizer IDs, benchmark fixtures or output ceiling.
The old private input dataset remains historical. The revised dataset is
`shlokbhakta/tabcomplete-q25-fim-r2-inputs-r3`.

The Python 3.11 GPU dependency lock contains 71 wheel-only packages. A complete
fresh CUDA dependency installation would exceed the temporary-storage allowance.
The worker must reuse the host's 15 NVIDIA distributions only after verifying
their exact locked versions and native-library architecture. Incompatible host
libraries cause a recorded failure before training. Python-dependent PyTorch
and Triton extensions must be installed for CPython 3.11, never reused from 3.13.

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

Attempt 1 completed on 2026-10-04, observed at 02:43:28 UTC. The worker took
8,323.96 seconds including setup, generation, training and saving. Training
consumed 7,872,512 input tokens and 7,864,824 target tokens, with 481 completed
updates, no skipped updates and no replayed tail. The largest checkpoint save
took 12.46 seconds. The collected full resumable checkpoint passed its hash and
cursor checks. Its SHA-256 is
`5d8d57f869965dd0cfd39c4cebd194ce4a85ab504ec791ed620b380c2d1b0c8f`.

Held-out source NLL decreased from 1.15449 to 1.13026 on the same 128 blocks and
130,944 scored target tokens. The raw continuation suite changed from 16 to 17
exact matches out of 180, and syntax passes from 146 to 155. These small changes
do not establish better editor intent. Full causal scoring and paired outcomes
are recorded separately; the FIM arms remain pending.

| Fixed diagnostic | Untouched Qwen | After raw-code CPT |
| --- | --- | --- |
| Causal functional test pass | 10/200 | 10/200 |
| Causal compile pass | 13/200 | 20/200 |
| Causal exact match | 1/200 | 0/200 |
| Line exact match | 16/180 | 17/180 |
| Line syntax pass | 146/180 | 155/180 |

The functional cases have four CPT wins and four losses. Line exact matches have
two wins and one loss; the paired case bootstrap interval for the rate change
is approximately -1.11 to +2.78 percentage points. Syntax improved on this fixed
line suite, with ten wins and one loss. These case-level intervals describe
synthetic diagnostics and do not justify a population or human-edit claim.
All eight current compiler-container image digests match the frozen environment.
The [paired comparison](q25_code_cpt_r2/cpt_quality_comparison.json) includes
input/output hashes, denominators and the tokenizer-serialization caveat.

The actual worker used Python 3.13.15, despite the repository's Python 3.11
requirement. This was a platform setup error and is recorded rather than
reported as compliant. Both upcoming FIM arms require the same pinned Python
3.11.15 environment. The CPT runtime otherwise reports PyTorch 2.11.0+cu128,
CUDA 12.8, Transformers 5.17.0 and bitsandbytes 0.50.2. One Tesla T4 performed
training; a second visible T4 was unused.

An authenticated quota refresh at 2026-10-04 03:06:36 UTC reported 42.69 of 45
account GPU-hours remaining, 2.31 used, renewal 2026-10-10, and no active jobs.
It is an observation, not authorization to use a renewed balance. Allocation
still requires another live quota and active-job refresh.

The verified FP16 inference export weighs 1,260,367,152 bytes, SHA-256
`15df09d25a5c39610e6c15148850a02295a32d228e1674871e13beddb25a8ccb`.
Its stored tensor count includes two byte-identical copies of the tied embedding
and output head, each 272,269,312 bytes. The reloaded model still has 494,032,768
logical parameters. All serialized weights are FP16. Actual file bytes remain
included in storage accounting; checkpoint estimates use unique logical weights.
This artifact has not been quantized, deployed or measured as a laptop service.

The existing SigNoz CLI query for the historical failed kernel identifier
`tabcomplete-one-line-instinct-pilot-r1-retry` returned `no_runs`. Its private
worker status and scientific artifacts remain available, but a training trace
was not recovered under that identifier. The completed CPT offline bundle was
imported through the existing CLI. Run `run-73c890ceb76492f727219afff8f0ce0c`
returned 979 records over ten pages, with complete pagination and status `ok`.
The bundle contained 3,331 imported spans. Prompt content capture was disabled.

Automatic personalization remains disabled. No CPT checkpoint is automatically
promoted into the editor.

The completion follow-up code passed a final full run of 966 Python tests,
Ruff and mypy across 90 source files. Gateway tests passed 16 cases and collector
tests passed 59 cases; both service type checks passed. These are implementation
checks. Actual completion training and model-quality results are still pending.

## Matched completion implementation verification

The revision-3 completion follow-up passed 992 Python tests (two dependency
deprecation warnings), Ruff and mypy across 90 source files. The actual Bun
results are 16 gateway and 59 collector tests, with both type checks passing.
Both CPU initializer preflights validate all 4,336 examples against the same
plan and tokenizer IDs. See `q25_code_cpt_r2/verification-fim-r3.json`.

The worker downloads no new model weights. It bootstraps a pinned Python 3.11
environment and reuses host NVIDIA libraries only after exact version and ELF
checks. Setup, owned runtime files and atomic checkpoint writes count toward
the deadline and storage budget. An incompatible host library fails before
training with a bounded inventory. The CPT arm mounts only its completed export;
partial runs save resumable state without producing an unused inference export.
The 10 GiB output cap and 12 GiB aggregate artifact cap both remain enforced.

## First completion allocation setup failure

Kernel `shlokbhakta/tc-q25-fim-r2-0-a1`, pinned to `a78a5f6`, stopped after
12.176 worker seconds before training. All 15 expected NVIDIA distribution
versions matched; the NVSHMEM file inventory failed the strict platform check.
Fourteen other distributions passed ELF checks. The worker recorded 1.057 GB
of combined input/setup/output artifacts and 20.94 GB free space. Training
input tokens and quality outputs are zero. The existing telemetry query
returned `no_runs`; no training run ID existed before setup stopped.

The controller collected a verified zero-work receipt. A retry must retain
this lineage, refresh authenticated quota and fix the file-layout check using
actual vendor evidence. The runtime lock, examples and model weights stay
unchanged. See `q25_code_cpt_r2/fim_setup_failure_untouched_a1.json`.

The vendor-wheel RECORD confirms native NVSHMEM bootstrap/transport plugins
without a `lib` filename prefix. The worker now permits those two prefixes only
in that vendor distribution and directory, retaining exact versions, ELF checks
and Python ABI exclusions. This setup-only fix changes no training data,
weights, prompt, decoding or runtime package selection. The corrected worker
passed 993 Python tests, Ruff and mypy; the last 992-test run remains historical.
The retry uses the same frozen plan and records a new pinned worker commit.
