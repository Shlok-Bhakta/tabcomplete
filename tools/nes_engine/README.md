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
