# Commit-sequence authoring queue

This is a bounded review queue built from the existing, hash-pinned
`bigcode/commitpackft` download. It is not a training dataset or evidence that
the reconstructed changes are inferable next edits.

## Existing pool and why no rows were accepted for training

The frozen source revision is `fc56fe33c030c6daa414c2b112c932b8eed085e6`.
The four JSONL files contain 69,893 records and 170,308,911 bytes; the downloaded
README adds 17,288 bytes, for 170,326,199 bytes total. The previous exact
reconstruction produced 17,308 one-line candidates, then selected 4,096 rows
for local review. Its original split audit was 2,842 train, 674 development,
and 580 `test_new_repo` rows across 2,754 connected groups.

The earlier route correctly accepted zero rows for training. Commit diffs do
not reveal the editor's action order; histories were constructed in ascending
file order. Repository license metadata did not verify the license of each file
at the pinned commit. A 32-case development-only blind diagnostic had 9 literal
target matches, 8 unknown judgments, and 12 cases where the reviewer considered
no edit supportable. That small diagnostic did not establish functional quality
or inferability, so it cannot justify promoting the reconstructed targets.

## Frozen source-review queue

Plan v4 was frozen before selecting the queue. It binds the dataset files,
candidate artifact, source manifest, split audit, grouping/reconstruction code,
selector, tests, deterministic seed, and caps. The queue selector skipped the
580 reserved rows by their serialized split field before JSON decoding. It
selected one candidate from each of 128 connected source groups: 96 rows retain
the train assignment and 32 retain development. No held-out rows were used.

Selection was round-robin across language/action strata. Within a stratum it
prioritized pre-state contexts with prior edits and visible declarations, types,
comments/docstrings, or guards. These are review-priority signals only; they do
not establish task intent. The 128 selected rows include 39 Python, 35
TypeScript, 35 Go, and 19 Rust states; actions include 50 replacements, 41
insertions, and 37 deletions. Visible-context flags overlap: 105 have a nearby
declaration, 92 a typed contract, 74 nearby documentation/comment text, and 101
a guard/error cue.

The queue index contains identifiers, split assignments, source URLs and
revision metadata, code/action hashes, and context-signal counts. It exports no
source body or target text. The SSD directory is mode 0700 and its files are
mode 0600. There were no provider calls or new downloads; all training-accepted
flags remain false and quality evidence remains false.

## Frozen identities

- Plan file: `commit_sequence_authoring_queue_plan_v4.json`
- Plan file SHA-256: `de0f9ed37d9fd3cf193ee741d3a422d51333cf4173ae884f115ac447e40a8a79`
- Canonical plan identity: `3585d3d6f2f4d3ef4149c6c4425068253897df3fa83826deefe44b17cf25c7f0`
- Input candidate artifact SHA-256: `23921525b61ca798d4c85d927a04c2682f30056d1d7b59a832fb8a1d2ca0e2cb`
- Private queue index: `/mnt/ssd/tabcomplete-product-r2/commitpackft/authoring-queue-v4/review_index.jsonl`
- Queue index SHA-256: `294aadad794507c9caa8ed013973dfd38011a73f2281ba840c5c2111a01c2de7`
- Queue manifest SHA-256: `8ee43c12b0a91096d78ebf850e831c7fd6c1db99c4af678cb96158a683b930a1`

Plans v1-v3 are preserved as superseded records. V1's split-token matcher did
not account for whitespace in the parent JSONL; v2 then rejected valid EOF
insertions; v3 selected rows but used `str.splitlines` and did not replay each
action through the canonical edit contract. V4 uses `physical_lines`,
`EditState`, `EditAction`, and `apply_action` to check valid byte-exact replay,
including EOF insertion positions.

## Verification and next gate

Targeted verification passed: 6 pytest cases, Ruff, and mypy for the selector
and its tests. The tests cover held-out rows skipped before JSON decoding,
group/split isolation, deterministic balanced selection, metadata-only export,
plan/input fingerprint binding, canonical EOF insertion, CRLF, and empty files.

Before any teacher call or training, verify source licenses and exact file bytes
at each selected public commit. Then build separate source-only authoring inputs,
with no action labels or future edits exposed; retain development rows for
tuning. Require a fresh blind inferability review and independent behavioral
oracles with wrong-action controls. Keep synthetic history labeled synthetic;
do not treat these candidates as observed editor chronology or no-edit labels.
