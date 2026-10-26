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
