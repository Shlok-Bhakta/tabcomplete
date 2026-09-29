# Public edit-sequence pilot outcome

Observed 2026-09-28 UTC. This is a bounded source-data investigation for the frozen one-line campaign. It made no model calls, GPU allocations, training updates, or live editor changes. The accepted training/development/test counts remain **0/0/0**.

## Pinned Git chronology pilots

I ran the same 40 previously frozen MIT repository seeds, ten each in Python, TypeScript, Rust, and Go, with at most 32 first-parent commits per repository. The [v1 plan](public_git_sequence_plan_v1.json) tested observed cross-commit order. The [v2 plan](public_git_sequence_plan_v2.json) tested two actual one-line identifier substitutions in one commit, explicitly marking their proposed editor order and intermediate state as synthetic. The source set was not replaced after seeing failures.

| Route | Repositories | Proxied bytes | Structural candidates | Independently accepted labels |
| --- | ---: | ---: | ---: | ---: |
| Observed cross-commit pair | 40 | 2,982,311 | 1 | 0 |
| Same-commit repeated identifier | 40 | 2,982,974 | 2 | 0 |

The cross-commit candidate, from `theflakes/MimeType`, changed display formatting before a later unrelated `NONE` to `none` replacement. The visible earlier edit did not establish a defensible reason for the later one. The two v2 candidates are narrower and more coherent:

* `git-synthetic-order/bbe15642613715a197f1366b`: `vasjaj/modem` at `62038d758711ba16363c425ddf3b1f96893a6026`. A Go import owner changed from `warthog618` to `vasjaj` in one line; a second import in that commit had the same substitution.
* `git-synthetic-order/48403a0996470ce64c69c714`: `khmseu/Mechanic` at `5dd7cb460253939849e94a4a850580eb94faa791`. One TypeScript assertion call changed from `assertType` to `assert`; a second call in that commit had the same substitution.

The [mechanical audit](public_git_sequence_review_v2.json) fetched each child file independently from its pinned public GitHub commit and matched its Git blob ID. It matched the pinned license bytes, applied the proposed action byte exactly, parsed the constructed pre-state and post-action state with the language parser, and checked the exact local Qwen2.5-Coder tokenizer budget. Inputs used 804 and 770 tokens; actions including EOS used 15 and 16 tokens. Both prompts retained one prior edit and did not literally contain the complete target wire action. The candidate artifact SHA-256 is `e4dc592d56782571a2fd4f47bbbaa3ad34655d16fc62d6f3fe4d6be35e0b068a`.

Neither candidate is a witnessed editor trajectory. Both intermediate buffers and first-to-second orders are constructed from one Git commit, and the commits can contain other changes. A mechanical source check does not establish a blind, student-view reason to edit or a functional objective. The TypeScript file has no file-level SPDX notice; the pinned repository MIT license is recorded, but source-file scope still requires review before acceptance. These two are **unreviewed candidates, not training labels**. The predeclared v2 expansion gate required at least 24 accepted candidates across 40 repositories, at least five per language; even treating both as accepted would fail it.

## Ordered agent-trace replayability

The [pinned Open-SWE-Traces source](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) contains agent trajectories, not human editor keystrokes. Its task source, [SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2/blob/main/README.md), supplies a `base_commit` field, but a correct reconstruction also needs every prior file mutation in tool order. The [frozen replayability screen](public_trace_replayability_plan_v1.json) rechecked the same five fixed 20-row blocks as the prior structural probe. It transferred 28,594,303 response bytes. Among 100 trajectories, there were 24 one-line `str_replace_editor` calls. Fifteen had a preceding editor call in the same file, and **all 15 also had earlier `bash` calls**. The optimistic shell-free count was zero. This is a necessary-condition screen, not a proof that shell-free calls would have been valid.

The [single-row reconstruction attempt](public_trace_reconstruction_result_v1.json) examined `firecracker-microvm__firecracker-containerd-308` at trace row 324. Its one-line call followed 43 shell calls, including `git checkout`, `git restore`, and `gofmt -w`. Its exact pre-edit state cannot be obtained by replaying only `str_replace_editor` calls. The [Hugging Face viewer filter and search endpoints](https://github.com/huggingface/dataset-viewer/blob/main/docs/source/filter.md) returned HTTP 500 for its base-task lookup; the complete source Parquet file is 428,839,266 bytes, above this pilot's 16 MiB transfer cap. I did not guess a base commit, execute any agent shell command, or reconstruct from the final patch. This route produced zero pre-states and zero labels.

## Decision

The one bounded Git expansion and the trace replayability test did not meet the frozen data gate. No 20,000-state, 100-group, 25-mechanism training bank is available from these inspected routes. A larger scrape would multiply unverified or constructed states without fixing the missing editor-order and objective evidence. The current campaign therefore does **not** allocate a GPU or start the full-weight pilot. The small verified mechanics remain useful for format and extractor tests, with their synthetic-order provenance intact.

The practical source of trustworthy future *human* sequences remains the already installed editor collector: proposals, visible decisions, and subsequent deltas can accumulate as the user edits. Those records should be reserved for temporal personal evaluation until there are enough reconstructable observations. They must not be inflated with unrelated later edits or agent traces. A separate small synthetic training pilot would require an explicit revision to the real-source fraction and quality claims; this result does not silently make that change.

## Verification

The focused chronology and replayability tests passed **14/14**. The CPU environment does not have PyTorch installed, so the literal full `uv run pytest -q` command failed during collection of five PyTorch-dependent files. Excluding those five files and one further PyTorch-dependent case in `test_model_data_r2.py`, the remaining repository run passed **341 tests**, with **11 skipped** and **1 explicitly deselected**. `uv run ruff check src scripts tests`, Ruff format checks on the changed Python files, and mypy on 76 source files passed. No GPU-dependent training or inference test is claimed to have passed locally.
