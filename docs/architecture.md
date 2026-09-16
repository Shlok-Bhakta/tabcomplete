# Architecture

```text
editor
   │
   ├── event stream
   ├── source-of-truth files
   ├── future Tree-sitter/LSP retrieval
   │
   ▼
context serializer
   │
   ▼
Qwen3.5 hybrid model
   ├── recurrent DeltaNet state
   └── attention KV cache
   │
   ▼
next editable-region prediction
```

## Event stream (`src/tinycomplete/protocol/`)

Thirteen typed, immutable event kinds (`events.py`): open/close/snapshot,
insert/delete/replace, cursor, diagnostics, retrieved context, completion
accept/partial/reject, predict. Positions are **byte offsets into UTF-8**,
boundary-checked on replay (`replay.py`). A `SNAPSHOT` rebases the
source-of-truth; everything else folds forward. Replay is deterministic and
validates sequence numbers and bounds.

## Serializer (`protocol/serialize.py`)

Line-based tags (`<F>`, `<S>`, `<C>`, `<D>`, `<X>`, `<A>`, trailing `<P>`),
backslash-escaped, round-trippable. A realistic session serializes to
260 chars / 111 Qwen tokens (`docs/serialization.md`). Per-keystroke cost
is one delta, not the whole context.

## Data (`src/tinycomplete/data/`)

- **FIM** (`fim.py`): seeded PSM/SPM holes in 8 syntactic classes, formatted
  with native Qwen sentinels; SPM keeps the live prefix adjacent to generation.
- **Next-edit** (`static_edits.py`): editable region + disjoint recent edits
  applied to context; target is the future rewrite or first-class `NOOP`.
- **Git-mined** (`git_edits.py`): parent→child hunks + enclosing syntax nodes
  + cumulative synthetic states. Labeled `git_synthetic` with an explicit
  not-human-order disclaimer, always.

## Model (`src/tinycomplete/model/`)

Tiny random hybrid (3 linear + 1 full) proves the cache mechanics on CPU:
continuation ≡ full forward (1.2e-07), fixed-size recurrent state per layer,
linear KV growth, deepcopy-branch isolation (`reports/cache.md`).

## Teacher (`src/tinycomplete/teacher/`)

Narrow region-rewrite interface (`base.py`), rule-based prompt (`prompt.py`),
OpenRouter (Muse Spark Contributor, structured output) and DeepSeek adapters,
hard budget gate (`budget.py`), per-candidate validation with rejects kept
(`validate.py`), staged fake-by-default pipeline (`generate.py`).

## Eval (`src/tinycomplete/eval/`)

Independent metrics only — exact/normalized exact, prefix length, edit
distance, lengths, parse success, noop accuracy — plus a latency harness
(prefill/decode tok/s, TTFT, RAM/VRAM). Raw predictions stored as JSONL.

## Train (`src/tinycomplete/train/` + notebook)

`train.py` (lazy GPU imports) + `notebooks/qwen35_colab.ipynb`: LoRA smoke
first, full-weight config as OOM probe, throughput in tokens per compute unit.

## Why the recurrent state is not storage

DeltaNet recurrence is lossy by construction — a fixed-size matrix cannot
retain an entire repository. The design treats it as *working memory* of the
live session while exact repository facts (file contents, symbols, types)
are re-injected deterministically from indexes (files on disk → Tree-sitter →
LSP) at each prediction point. The cache proof shows the mechanics; retrieval
depth is the next experiment after the first LoRA smoke.
