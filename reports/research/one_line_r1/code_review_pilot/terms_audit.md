# GitHub Code Review terms-first pilot

Observed 2026-09-28 03:21 UTC. The audit stopped at its license gate before fetching any training, validation, or test rows. No source code, review comments, Parquet shards, model weights, or provider outputs were downloaded.

## Pinned source and evidence

- Dataset: [ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview).
- Dataset revision from the publisher's Hub metadata API: `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca`.
- The pinned [README](https://huggingface.co/datasets/ronantakizawa/github-codereview/blob/c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca/README.md) is 5,416 bytes, SHA-256 `c99f37c0225412333fd90e377e785f1f7794d21433794525986a10fb66f9b666`.
- Its current front matter says `license: other`. The repository file listing has no LICENSE or separate terms file. An [older README revision](https://huggingface.co/datasets/ronantakizawa/github-codereview/commit/db754216268fb61b8a3ea82595e37d500585dd89) used `license: mit`; that historical field does not resolve the current `other` field. A [public license-clarification discussion](https://huggingface.co/datasets/ronantakizawa/github-codereview/discussions/2) provides no publisher clarification in the visible thread.
- The current card claims the source repositories use permissive licenses, but it lists no per-row license, source-file notice, or exact source commit. Its row schema identifies repository, PR number, and path. Those fields could guide later source verification; they do not themselves verify file-level provenance.

## Scope and stop result

The publisher reports 355,807 total rows across 37 languages, including Python, TypeScript, Rust, and Go. It describes 90% train, 5% test, and 5% validation splits grouped by repository. Each positive row pairs a human reviewer comment with before/after code chunks. The review comment is task evidence that a normal editor prediction would lack unless the user had it in visible context; removing it may make the change uninferable. The card publishes no per-language row counts or exact one-line action yield.

| Measure | Result |
| --- | ---: |
| Train rows fetched | 0 |
| Validation or test rows fetched | 0 |
| Source-code bytes downloaded | 0 |
| README metadata bytes fetched | 5,416 |
| Sampled repository groups | 0 |
| Structurally valid one-line candidates | Not measured |
| Blind-inferable candidates | Not measured |
| Accepted training labels | 0 |

The present record cannot estimate an exact one-line or blind-inferable yield. Treating card counts or viewer examples as accepted labels would conceal both the license gap and the unknown yield. The bounded 100-row pilot can resume only after the dataset's current use terms are explicit and each sampled source file can be tied to a compatible license at a pinned source revision. It must use train rows only, at most 25 per target language, across repository groups, under the 100 MiB transfer cap; no held-out row can be used for selection.
