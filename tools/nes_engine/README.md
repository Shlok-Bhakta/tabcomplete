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
