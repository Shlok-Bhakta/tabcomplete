# One-line observed-edit GPU pilot r1

This is a separately frozen, narrow supervised pilot. It does not replace the
plan-11 main campaign or its 20,000 accepted-state, 100-source-group, and
25-reviewed-mechanism gates. The installed q25 Q4 editor model is unchanged.

## Source and scope

The source is the pinned original TypeScript training shard of
[Continue Instinct](https://huggingface.co/datasets/continuedev/instinct-data).
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

Pending the authorized GPU run. Numerical results, exact hashes, quota use,
failure evidence, and the final disposition will be appended after retrieval.
