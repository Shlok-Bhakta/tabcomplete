# One-line observed-edit GPU pilot r1

This is a separately frozen, narrow supervised pilot. It does not replace the
plan-11 main campaign or its 20,000 accepted-state, 100-source-group, and
25-reviewed-mechanism gates. The installed q25 Q4 editor model is unchanged.

## Source and scope

The source is the pinned original TypeScript training shard of
[Continue Instinct](https://huggingface.co/datasets/continuedev/instinct-data).
Continue [describes the original TypeScript examples](https://blog.continue.dev/instinct)
as edits captured from its team while working on its open-source code.
The dataset repository declares Apache-2.0. Per-file license, source commit,
and edit session identifiers are unavailable in the rows, so those properties
are unverified. Only private research packaging is permitted for this pilot.
The native test split remains sealed. Training and development are separated
by source file path; this is still only one publisher-reported project and
cannot establish cross-repository transfer.

The frozen converter accepted 930 of 4,371 source rows: 720 training states
from 47 paths and 210 development states from 20 disjoint paths. The other
rows failed the strict one-line or history checks. Training contains 528,870
prompt tokens and 9,405 supervised response and EOS tokens, for 538,275
nonpadding input tokens in one pass. Development contains 157,462 prompt and
target tokens combined. These are actual q25 tokenizer counts.

Plan revision 2 records a pre-comparison source-formatting change. A second
deterministic conversion produced identical train and development JSONL hashes,
counts, filters, and token exposure. The converter and manifest hashes changed
because formatting and a type annotation changed source bytes. The earlier
plan is retained as `plan_preformat_audit.json`; no model result was inspected
when the revision was made.

Attempt 1 ran on a Tesla T4 for 506 seconds and failed during the fixed
synthetic calibration part of the untouched baseline. The Kaggle environment
was missing `tree-sitter-language-pack`; the private log identifies the import
failure. The evaluator had not written a baseline result, and training had
zero updates and zero input-token exposure. Plan revision 3 pins the package
from the repository lockfile, records the failure evidence, refreshes quota,
and assigns distinct private retry IDs. The model, data, prompt, and scoring
rules are unchanged.

The first retry upload stopped before dataset creation because its dataset
slug was 51 characters; Kaggle permits at most 50. Plan revision 4 shortens
only that private slug and adds a pre-upload validation check. It consumed no
additional GPU time or training tokens.

Kaggle rejected the first retry kernel push because its metadata title did
not resolve to the retry kernel ID. The staged private metadata title was
corrected to the exact kernel slug; the same quota checked submission path
then started the retry notebook. The scientific worker code and input manifest
were unchanged. The builder now derives the title from the kernel ID.

The converter accepts a state only when the marked editable region and the
assistant replacement reconstruct a cursor-aligned, one-line edit byte for
byte. It also requires a uniquely located earlier same-file edit in the
visible history, then checks that the selected model tokenizer retains that
history under the fixed 1,024-token prompt budget. Rows remain marked
`inferability_reviewed=false`; historical accepted-label counts stay zero.

## Frozen training contract

The student is the untouched `Qwen/Qwen2.5-Coder-0.5B` revision
`8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`. Only its verified local
configuration, tokenizer, and safetensors file are staged. The pilot trains
one supervised pass at peak learning rate `1e-5`, response-only loss,
32-example effective batch, FP32 master weights with FP16 compute, and a
2,000,000 nonpadding input-token ceiling. The frozen plan and dataset manifest
record the exact exposure before GPU allocation.

The T4 session deadline is 120 minutes, including evaluation and saves, with
20 minutes reserved for finalization. The controller refreshes authenticated
quota and active jobs before submission. The job writes a complete resumable
state and a separate inference export. Weights are not published or copied to
the live editor by this run.

The authenticated quota at freeze was 45 of 45 Kaggle account GPU-hours,
observed 2026-09-29 06:23 UTC, with renewal 2026-10-03 00:00 UTC and no active
job. Two old notebook references returned HTTP 404 on direct status queries;
the plan records that evidence. Unknown status for any other job blocks
allocation.

## Evaluation rule

The untouched baseline and adapted export use the same file-heldout
development shard and greedy 64-token action ceiling. Valid serialization,
real EOS termination, and byte-exact after-state reconstruction are reported
separately. A fixed synthetic calibration set checks keep and codec behavior;
its score is not a human-edit quality estimate. Any later promotion requires
independent evidence beyond this pilot.

## Results

The retry ran on a Kaggle Tesla T4 from worker commit
`cb4b18872416a6ee1ccae31d3e35703d9c9f8ab9`. It completed 23 of 23
updates, one pass over 720 rows, and 538,275 nonpadding training input tokens.
No update was skipped. Mean example loss fell from 3.48 on the first update to
0.94 on the last. One training state occurred three times, leaving 718 distinct
training states; the duplicate was confined to the training split.

| Frozen development result | Untouched q25 F16 | Adapted q25 F16 |
| --- | ---: | ---: |
| Valid actions and real EOS | 0/210 | 210/210 |
| Exact reconstructed after-state | 0/210 | 9/210 (4.3%) |
| Replacement cases correct | 0/182 | 9/182 |
| Insertion cases correct | 0/9 | 0/9 |
| Deletion cases correct | 0/19 | 0/19 |
| Synthetic calibration edit success | 0/12 | 1/12 |
| Synthetic keep cases correct | 0/4 | 0/4 |

The 9 exact development wins are paired with zero baseline wins. A 10,000 draw
bootstrap resampling the 20 heldout file paths gives a 95% interval of
1.7–7.8 percentage points for the exact-after improvement on this dataset.
It cannot establish transfer beyond the one source project. All 210 adapted
development actions were replacements. The model proposed edits for all four
synthetic keep cases, replaced a line with its existing text 29 times, and
directly reversed the latest same-row edit once. Serialization improved;
useful next-edit behavior remains weak. The live editor model was not changed.

On the T4, generation-only completed-action latency was 1,921 ms median and
1,981 ms p95 before adaptation, versus 443 ms median and 684 ms p95 after.
The untouched model ran to the 64-token cap on every case; the adapted model
stopped at 13 output tokens median. These times exclude editor scheduling,
prompt construction, and CPU deployment conversion. Peak allocated CUDA
tensor memory was 1.07 GiB in both evaluation processes; retained allocation
was 0.94 GiB. Post-evaluation process RSS was 2.30 GiB for the baseline and
1.65 GiB for the adapted export. Those are Kaggle process measurements, not
laptop predictor-service RAM or total GPU VRAM.

The 2,980,477,490-byte resumable checkpoint has SHA-256
`e6479970c6f11cf5e568b6a238f893d6f1ac98ade4729671928852b4d5aecc0d`.
The 1,260,367,152-byte F16 export has SHA-256
`128e9bce231d82ee6ac075cae60895da3464e5a5cfe7e4377aa8dea932032ea1`.
Both downloaded hashes match Kaggle's manifest. A separate process reloaded
the export for adapted evaluation. The checkpoint reload inspection confirmed
weights, optimizer, scheduler, scaler, RNG states, epoch, position, and exact
token counters; its [machine-readable record](one_line_gpu_pilot_r1/checkpoint_reload_verification.json)
contains the field and count checks.

The worker's last verification step failed because `save_pretrained` rewrote
`tokenizer.json`, changing its byte hash. An independent CPU check found
identical prompt text and token IDs on all 210 frozen development states,
an identical vocabulary ID map and special-token map, and identical encoding
and decoding for all 210 observed adapted action strings. This does not prove
identical behavior for every possible token sequence. The offline artifact verifier passed
every checkpoint and export hash while reporting the worker's failed final
state explicitly. The worker code now checks tokenizer behavior and each
export's own recorded byte hash. This post-evaluation issue does not change
the measured actions, and it is not a reason to deploy the weak model.

Authenticated quota after the attempts, observed 2026-09-29 07:31 UTC, was
0.41 of 45 GPU-hours used and 44.59 hours remaining, with no active jobs.
The research SSD held about 8.3 GB of new pilot files, including temporary
private bundles, checkpoint, and export; it stayed below the 12 GiB cap.
No paid compute, teacher call, model publication, or live-editor change was
made. The private monitoring identifiers are Kaggle kernels
`shlokbhakta/tabcomplete-one-line-instinct-pilot-r1` (failed before training)
and `shlokbhakta/tabcomplete-one-line-instinct-pilot-r1-retry` (trained and
evaluated). SigNoz's live run stream was not used for the offline Kaggle job.

The [result summary](one_line_gpu_pilot_r1/result_summary.json),
[tokenizer parity](one_line_gpu_pilot_r1/tokenizer_parity.json),
[checkpoint reload check](one_line_gpu_pilot_r1/checkpoint_reload_verification.json),
[artifact manifest](one_line_gpu_pilot_r1/artifact_manifest.json), and
[quota record](one_line_gpu_pilot_r1/quota_after.json) are the compact
authoritative records. Raw evaluation actions, data shards, weights, and
checkpoint remain private off Git.
