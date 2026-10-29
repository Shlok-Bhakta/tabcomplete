# tabcomplete-nes

CPU next-edit engine for [Sweep next-edit](https://huggingface.co/sweepai/sweep-next-edit-0.5B)
models, with a workspace context engine and a Neovim client.

## Pieces

| file | role |
|---|---|
| `src/model.rs` | llama.cpp runner: KV prefix reuse, prompt-lookup speculative decoding (drafts copied from the current window), early-stop splice, token log-probabilities |
| `src/sweep.rs` | Sweep prompt format and post-processing, ported from Sweep's reference server (`sweep-autocomplete` 0.1.1); jump-block search |
| `src/history.rs` | edit history from buffer snapshots, coalesced by proximity and pause |
| `src/index.rs` | workspace index: parallel gitignore-aware walk, tree-sitter tags queries (Rust, Python, TS/TSX, JS, Go, C, C++, Java, Lua, Nix), mtime-based incremental refresh |
| `src/retrieval.rs` | definition snippets for identifiers near the cursor and in recent edits; sticky selection to keep the KV prefix |
| `src/service.rs` | request flow: local window prediction, then a jump-to-edit prediction when the window has nothing; confidence gate |
| `nvim/lua/tabcomplete_nes` | client: ghost text, replacement previews, jump hints, Tab accept (LazyVim `ai_nes`), collector telemetry |

Prompt order is stable-first (recent other file, original file, retrieval,
edit history, then the `original/`, `current/`, `updated/` window), so a
keystroke usually re-evaluates only the window and the newest edit.

## Run

```sh
cargo build --release   # RUSTFLAGS="-C target-cpu=native" for local builds
tabcomplete-nes --model sweep-next-edit-0.5b.q8_0.gguf serve --port 19094
```

`GET /health`, `POST /v1/predict` (path, content, cursor, diagnostics,
clipboard), `POST /v1/observe` (snapshots, saves, idle warm-up). The Nix
package and Home Manager module live in `nix/nes/`.

`POST /v1/predict?stream=1` answers with newline-delimited JSON:
`{"event":"provisional","edits":[...]}` lines while the cursor line is being
decoded (ghost text at the cursor, each extending the previous one), then
`{"event":"final", ...}` with the normal response body. Provisional text can
still be retracted by the final answer (confidence gate on later tokens, a
line that turns into a replacement); `timings.first_edit_ms` is when the
shown edit was first known and `timings.provisional_retracted` flags the
retractions. Client tests: `NES_CLIENT=tools/nes_engine/nvim nvim --headless
--clean -l tools/nes_engine/nvim/tests/stream.lua` (and `type_through.lua`).

Format flags (`--line-numbers`, `--diff-format`, `--prefill`,
`--cursor-marker`, `--lines-before/after`, `--whole-file-lines`,
`--chunk-lines`, `--retrieval-chunks`, `--early-stop`, `--min-confidence`)
exist so the evaluation can compare prompt variants without rebuilding.

## Evaluate

```sh
python3 -I eval/build_collector_cases.py --out cases.jsonl   # from the trajectory collector (read-only)
python3 -I eval/build_zeta_cases.py --out zeta.jsonl eval.jsonl train.jsonl
python3 -I eval/run_eval.py --engine target/release/tabcomplete-nes --model M.gguf \
    --cases cases.jsonl --name label --out rows.jsonl -- [engine flags]
python3 -I eval/rescore.py --cases cases.jsonl rows.jsonl
```

Collector cases replay real editing: the state at a typing pause, the earlier
bursts as history, and the next bursts as the target. `rescore.py` counts an
edit as helpful when accepting it moves the file closer to where the user was
shortly after, since the next burst alone undercounts predictions that are
right but early. `eval/kaggle/` runs the same sweep on a Kaggle GPU.

Observed with the 0.5B Q8 model: the literal `<|cursor|>` marker is not a
special token in this model and can flip predictions; Sweep's reference
prompt with greedy decoding matches stock llama.cpp byte for byte.

## Results so far (2026-10-26)

Real trajectories (263 cases from the collector, Rustlings exercises),
ThinkPad T480s CPU, scored with `rescore.py`:

| prompt format | helpful | harmful | precision | typing saved |
|---|---|---|---|---|
| Sweep server format (default) | 46% | 24% | 64% | 33% |
| server format without `<|cursor|>` | 37% | 18% | 67% | 26% |
| model-card format, jump on | 40% | 27% | 58% | 28% |

Exact match with the very next burst is about 10% for all three; exact match
with a state the user reached within a minute is about 20%.

Latency on the i7-8650U (4 decode / 6 prompt threads), warm cache: about
2 s for a local suggestion (roughly 250 new prompt tokens at 190 tok/s plus
0.6 s of decode with every draft accepted), about 3 s when the jump stage
also runs. A cold file costs a few seconds more; opening a buffer warms its
prefix. The workspace index of the Rustlings repository (188 files, 735
definitions) builds in 25 ms.

The minimum changed-token log-probability separates helpful from harmful
suggestions only weakly (AUC 0.63), so the confidence gate is off by default.
The Kaggle CUDA build works with one GPU but replays still hit a CUDA error
whose message is hidden by `void_logs()`; enable llama.cpp logging before
the next GPU attempt.

## Speed work (2026-10-27, ThinkPad, Q4_0)

Data: 714 real cases rebuilt from the collector (Rustlings and NNFS); quality
on a 179-case subset, latency on a 90-case keystroke replay (predictions at
the last keystrokes of each typing burst, timing the pause-point request).

| change | effect |
|---|---|
| Q4_0 instead of Q8_0 | ~23% faster prefill/decode on AVX2 (repacked layout); Q4_K_M is slower than Q8 here |
| hide the in-progress edit from the history section | KV cache survives typing; prompt tokens per keystroke 894 → 673 |
| confidence gates (-1.0, multi-line -0.5) | wrong suggestions 28% → 13% on 179 cases |
| 3-line rewrite window (Sweep uses 5) | faster and fewer wrong rewrites: harmful 28% → 19%, helpful 34% → 35% |
| one splice anchor line | ~55 ms at equal quality |
| previous completion as first draft source | keystroke p50 1.49 s → 1.13 s, draft acceptance 52% → 65% |
| prefetch after showing an edit | the next suggestion after Tab is served from cache (~20 ms) |
| retrieval off | 9% faster but harmful 19% → 25%; kept on |

Shipped config, keystroke replay: p50 1.15 s (performance profile), 1.0 s
(turbo-lite), 3.3 s (balanced: TLP disables turbo boost on that profile).

Did not help: larger drafts (32/64 tokens; every verified position pays the
151k-vocabulary output layer), forcing flash attention (auto already enables
it), 64-token prefill chunks in sequential replay, prefilling through the
cursor line, a 2-line window. The UHD 620 iGPU through Vulkan is 2-5x
slower than the CPU (no int8 dot products or matrix units).

Prefill and decode are ~99.8% of request time; the context engine, index,
tokenizer and HTTP together take ~3.5 ms.

## Caching work (2026-10-28, ThinkPad, Q4_0, keystroke replay)

Stage medians are the stable signal; whole-request p50 drifts ±10-15%
between identical runs (thermals), so close calls were rerun interleaved.

| change | effect |
|---|---|
| suffix KV reuse (scratch sequence + shift) | prefill 481 → 216 ms, p50 1.17 → 0.92 s, quality equal |
| one 4096-cell context per file slot (4 slots) | prefill 246 → 158 ms vs a shared 12288-cell cache; returning to a file reuses 320-500 cached tokens (0.7-1.2 s instead of 2.4-3.4 s); ~90 MB per extra slot |
| sticky recent-file snippet | file switches keep their cached prefix |
| same-line answer reuse | leaving insert mode or moving along a line answers from cache |
| 120-line file context (was 240) | equal quality on current files, bounds depth for large files |

Primitive costs at 900 cached tokens (`tabcomplete-nes bench`): one-token
decode 20-25 ms (14 ms with no context), a 17-token verify 164-200 ms, of
which full-vocabulary logits are ~44 ms; llama.cpp's cache shift re-rotates
every cell, ~50 ms at 4096 cells and ~120 ms at 12288.

Rejected: q8_0 KV cache (slower), OpenMP thread pool (no gain), 6 decode
threads (slower: decode is memory-bound), 3 history hunks (the oldest hunk
churns the prefix, so more tokens per keystroke), one retrieval chunk
(slightly less accurate), prefilling through the cursor line (equal).

Remaining large cost: attention over the cached context and full-vocabulary
logits at every verified draft position. Scoring a token shortlist instead
needs a llama.cpp patch (the graph always builds the full output projection).

## Verification cost work (2026-10-28 night, ThinkPad, keystroke replay)

A per-batch log (`stats.batches`: draft source, match length, proposed,
accepted, ms) gave the cost model on this CPU: a decode batch costs about
23 ms + 9.5 ms per extra token, so a draft only pays when it yields more
than one token per ~23 ms. Drafts started from 1-2 token matches produced
27-40 ms per token; 4-token matches 13 ms.

| change | effect |
|---|---|
| drafts need a 3-token match (`--draft-min-match 3`) | 16.0 -> 14.4 ms per generated token, decode p50 496 -> 457 ms |
| vectorized argmax / log-sum-exp (polynomial exp) | 0.69 -> 0.29 ms per verified position |
| output layer limited to ASCII tokens (`vocab/`) | 14.9 -> 13.5 ms per token; identical suggestions on 90 replay and 179 quality cases |
| diagnostics/clipboard after the history (`--context-order volatile-late`) | live LSP churn no longer re-evaluates the history (replay has no diagnostics) |

The tied embedding/output matrix is 145 MB of the 346 MB Q4_0 file and
every verified position pays a 151,936-row projection. Pruning it needs a
small llama.cpp patch (`vocab/llama-short-output.patch`: the qwen2 loader
accepts a shorter `output.weight`, the graph pads the logits) and a GGUF
rewrite (`vocab/prune_output_vocab.py`) that moves kept tokens to the
front of the vocabulary.

Rejected: a 63k-token shortlist from a code corpus (12.6 ms per token but
fewer helpful edits: the model copies window identifiers whose tokens the
corpus never had), a Q4_0 token embedding (~12% faster decode, 3 points
fewer helpful edits on 179 cases), Q5_0 embedding (no gain), prefilling
the window through the cursor (helpful 26% -> 20%, and the cursor line
appears twice in the prompt so suffix reuse fails), draft growth caps of
32/48 (equal to 16), retrieval after the history (no clear gain).

For suggestions that are shown, p50 is ~0.92 s; their prompts change more
(a new history hunk at the typing pause), so prefill is about half of it.

Current defaults on the 179 quality cases: helpful 36%, harmful 15%,
precision 68%, typing saved 27%.

## Cache stability, kernels and streaming (2026-10-29 night, ThinkPad, desktop stopped)

Keystroke replay, 90 Rust cases, shown-suggestion latency (what the user waits for):

| configuration | shown p50 | overall p50 |
|---|---|---|
| start of night (desktop running) | 869 ms | 869 ms |
| same, desktop stopped | 804 ms | 760 ms |
| + stable history/window, full-prompt warm, KV-shift range patch | 430 ms | 602 ms |
| + flash attention off (default) | 343 ms | 482 ms |

Python/YAML (46 cases): overall p50 490 -> 357 ms.

| change | effect |
|---|---|
| `--stable-history` (hide before capping, live group skips filters, positional revert) | shown prefill 363 -> 103 ms; 761 cases: helpful 26.4% -> 27.5%, wrong 16.0% -> 15.0% |
| `--stable-window` (line-based cursor mapping, chunk hysteresis) | accuracy-neutral; stops cache resets on files over 120 lines |
| `--warm-full-prompt` (observes warm the whole local prompt, interruptible) | shown p50 557 -> 446 ms after observed-only edits |
| KV shift rotates only shifted cells (`vocab/llama-kv-shift-range.patch`) | shift+copy 50 -> 22 ms, p50 -3%, outputs unchanged within replay noise |
| flash attention off | 17-token verify 170 -> 124 ms (CPU FA is a scalar per-row loop below 64 rows); 761 cases: helpful equal, wrong +1 point |
| splice at a line boundary | fixes garbled text after the first hunk (27/179 completions) |
| streamed provisional ghost text (`/v1/predict?stream=1`, client `stream`) | 10% of shown suggestions appear earlier, by ~60-230 ms |
| service without `Nice`/`CPUQuota` | ggml threads spin at per-op barriers; one preempted thread stalls all |

No gain: persistent pinned threadpools (any poll), per-call small-batch thread counts, f32 KV cache, mmap off, context 2048, draft min-match 2/4 or 32-token drafts with FA off. The perf profile with FA on: Q4_0 GEMM 42%, FA kernel + f16 dot + expf 27%, thread sync 11%.

Replays are mildly nondeterministic (some prompt pieces depend on wall-clock timing): two identical runs differ on 27 of 483 responses, so arm differences of 2-5 cases on 90-179 cases are noise. Decide accuracy questions on all 761 cases.

## Accuracy fixes (2026-10-29, ThinkPad, all 761 cases)

A live session in a new file (`dekit.yaml`) had an empty file section in
every prompt: the original snapshot is taken when a file is first seen and
only refreshed on a save after 15 minutes, so the model saw nothing but the
last three history hunks. Three fixes, each behind a flag:

- `--refresh-original` (default on): the tracker keeps the text before each
  recent edit group, the prompt reports the oldest group it shows, and the
  snapshot moves to the text just before that group once it is clearly stale
  (12 lines of drift no shown hunk covers, or under half the current lines).
  The file section opens the prompt, so this re-evaluates nearly all of it;
  it happens rarely, and the new prompt is warmed while idle.
- `--insert-min-confidence -1.3` (default): insertions and appends that keep
  the existing text get a looser gate; `--multiline-min-confidence` is now -0.4.
- `--protect-live-edit` (default off): edits that would delete or replace 3+
  characters typed in the live edit group are trimmed to the lines they add,
  or withheld; trimmed edits keep the gate of the untrimmed edit.

| arm (761 cases) | shown | helpful | harmful | precision | saved |
|---|---|---|---|---|---|
| baseline (flags off) | 350 | 209 | 130 | 0.597 | 141.8 |
| baseline rerun | 346 | 206 | 129 | 0.595 | 140.3 |
| `--refresh-original` | 345 | 213 | 124 | 0.617 | 142.1 |
| `--insert-min-confidence -1.3` | 371 | 223 | 137 | 0.601 | 144.4 |
| `--multiline-min-confidence -0.4` | 339 | 208 | 121 | 0.614 | 140.7 |
| `--protect-live-edit` | 319 | 194 | 114 | 0.608 | 132.6 |
| refresh + insert -1.3 + multi-line -0.4 (new defaults) | 358 | 222 | 128 | 0.620 | 144.2 |
| new defaults + protect | 329 | 209 | 112 | 0.635 | 136.1 |

The `dekit.yaml` replay (36 cases, history from the empty file): baseline 7
helpful / 8 harmful, refresh 13 / 6, new defaults 13 / 5, + protect 13 / 4.
Keystroke replay (`real_sub8k`, 90 cases, 241 requests): p50/p90 446/1399 ms
before, 445/1414 ms with every flag on. Protecting the live edit trades
helpful for harmful edits about one for one on the 761 cases, so it is off.

## Jump stage (2026-10-29, ThinkPad, Q4_0)

The jump stage runs when the local window has nothing and the request allows
jumps. It cost ~2.1 s per run, of which ~1.8 s was prefill (~350 tokens at
~5 ms each: the newest hunks and the jump windows) and ~0.3 s decode. 116
of its 139 runs on the 761 cases found nothing. Per jump reason:

| reason (761 cases, jump on) | runs | ms/run | shown | helpful | harmful |
|---|---|---|---|---|---|
| block_after_cursor | 70 | 2060 | 9 | 0 | 9 |
| identifier_match | 42 | 2168 | 3 | 0 | 3 |
| deleted_line_match | 27 | 1307 | 6 | 2 | 4 |

New defaults: `--jump-reasons deleted_line_match,diagnostic` (no case has
diagnostics), `--jump-scratch` (the jump prompt is evaluated in the scratch
sequence, so the local cache survives and no re-warm follows), and
`--jump-shared-history` (the jump prompt repeats the local history, then the
hunks the local window hid: -16% jump prefill tokens). They also include
`--jump-prefill-chunk 64` (cancellable sooner, same throughput) and
`--jump-reuse-local` (leaving insert mode after a keystroke that found
nothing runs only the jump stage).

| arm | keystroke p50/p90 | leave-insert p50/p90 | 761 helpful / harmful |
|---|---|---|---|
| jump off | 369 / 1200 ms | | 29.4% / 17.0% |
| jump on, old defaults | 449 / 2768 ms | 137 / 2374 ms | 30.7% / 18.5% |
| jump on, new defaults | 364 / 1297 ms | 2 / 420 ms | 29.3% / 17.5% |

Keystroke latency is for the scored request of each case (jump allowed).
Leave-insert (`run_eval.py --leave`) puts a local request for the same
state before it. Per case on the 761, the jump edits add 2 helpful and 16
harmful with the old defaults, and 2 and 4 with the new ones. The other
differences between the arms are in local answers. With the old defaults the
jump prompt replaced the local cache, so later local answers came from a
different cache state. With `--jump-scratch` and every reason allowed, the
local answers matched jump off exactly. The 3-8
local flips between gated arms happen before any jump ran in those replays:
they are timing noise from the replay's idle warm-ups.

Not done: delivering jump suggestions after the local answer. A jump runs
only when the local answer is empty, and the client shows nothing for an
empty answer, so a late event would change nothing the user sees. The
engine-side cost of a jump is the time the next keystroke waits for it to
cancel, which the 64-token jump prefill chunk bounds.
