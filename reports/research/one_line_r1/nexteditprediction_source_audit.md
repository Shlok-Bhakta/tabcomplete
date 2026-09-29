# NextEditPrediction train-source audit

Observed 2026-09-29 05:35 UTC. This is a metadata and structural audit, not a
training or quality result. The public train file from
[`lurf21/NextEditPrediction`](https://huggingface.co/datasets/lurf21/NextEditPrediction/tree/3769eeb29fed82088166f80358b2a5b8445f9a9e)
was fetched at immutable revision
`3769eeb29fed82088166f80358b2a5b8445f9a9e`. Its 23,434,175 bytes matched
the publisher's LFS SHA-256
`0f5b71a5f8e1544f09c3bb7f59585b12267ab85d703ff1d2e91747726c7cec57`.
The separate test file was not downloaded or inspected.

The train file has 3,211 rows and 3,211 distinct commit fields. Its language
counts are Python 1,056, JavaScript 832, Java 655, C 220, Go 192, C++ 130,
and TypeScript 126; **Rust is absent**. The row's `license` field is mixed:
1,663 MIT, 775 Apache-2.0, 320 BSD-3-Clause, 119 BSD-2-Clause, 162
AGPL-3.0, and smaller other or unknown groups. These are metadata values, not
verified source-file grants. There is no dataset card declaring a separate
data license in the Hub metadata observed here. The GitHub code repository's
Apache-2.0 license alone does not settle the rights to each source snippet.

Comparing each row's `incomplete_new_contents` with `new_contents` using
Python `difflib.SequenceMatcher(autojunk=False)` found one change hunk in all
3,211 rows; 1,458 were a one-line insertion. This describes the publisher's
construction, not an observed editor sequence. The published
[pipeline](https://github.com/lurf21/NextEditPrediction#data-pipeline) removes
the last edit chunk from a commit-derived example and uses a model to judge
chunk relatedness. Its four prompt markers are `original_code`, `edits_diff`,
`current_version`, and `next_version`. No row contains an observed cursor or
within-commit editor chronology. A commit message exists as metadata, and the
first two rendered prompts did not include it; this audit did not validate
every prompt for leakage or independently review target inferability.

Decision: retain this as a source-format reference only. Even all 3,211 rows
would be below the frozen 20,000-state training floor, and the file has only
126 TypeScript and 192 Go rows, no Rust, no observed editor order, and no
verified file-level source grants. The 1,458 one-line insertions are not 1,458
accepted labels. No GPU session, teacher call, or model update followed.
