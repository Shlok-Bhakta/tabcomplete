# tinycomplete — tiny local next-edit code model (experiment infrastructure)

## Project goal

Build the experimental infrastructure for a tiny, local, privacy-preserving
code next-edit/autocomplete engine. We are NOT training the final model here;
we are building enough verified infrastructure to run a real Colab training
experiment with confidence.

## Architecture hypothesis

```text
editor
   │
   ├── event stream (append-only, typed)
   ├── source-of-truth files
   ├── future Tree-sitter/LSP retrieval
   │
   ▼
context serializer (compact tags, <P> last)
   │
   ▼
Qwen3.5 hybrid model
   ├── recurrent DeltaNet state (fixed-size per layer)
   └── attention KV cache (grows with context)
   │
   ▼
next editable-region prediction
```

**Core experiment: represent an editor session as an append-only event stream
so the model's recurrent/KV cache advances with only new events instead of
re-processing the whole context after every keystroke.** Proven on CPU with a
tiny hybrid model: cached continuation matches full forward to 1.2e-07
(see `reports/cache.md`).

The recurrent state is NOT permanent lossless repository storage — exact
facts are re-injected from deterministic indexes (files, Tree-sitter, LSP).

## Why Qwen3.5-0.8B

- Hybrid text architecture: 24 LM layers = 18 Gated DeltaNet linear-attention
  + 6 full softmax attention (3-linear/1-full × 6).
- Native FIM tokenizer tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`),
  long context, recurrent state + KV cache — exactly the split an append-only
  session design needs.
- Small enough (0.8B) for L4 LoRA smokes; Unsloth supports the family.

## Local CPU setup

```bash
uv sync --group dev        # Python 3.11, transformers 5.17 (has Qwen3_5 classes)
bash scripts/bootstrap_cpu.sh   # CPU-only torch, kept OUT of uv.lock
bash scripts/smoke.sh      # ruff + pytest + harness check
```

Local machine is CPU-only by policy. Never install CUDA locally; never
provision cloud resources from here. GPU training happens only in a
human-launched Colab runtime (`notebooks/qwen35_colab.ipynb`).

## Test command

```bash
uv run ruff check .
uv run pytest -q
```

## Synthetic-data command

```bash
uv run tinycomplete synthetic --provider fake --states 12 --seed 1 \
  --out data/generated/teacher_fake.jsonl
uv run tinycomplete budget-status
```

## Budget safety

Paid generation needs `ALLOW_PAID_SYNTHETIC=1` AND caps
(`MAX_TOTAL_SPEND_USD=2.00`, `MAX_PAID_EXAMPLES=2000`,
`MAX_CANDIDATES_PER_STATE=3`), with persistent atomic accounting in
`data/generated/budget.json`. Without the flag, the full pipeline runs on
the fake provider and stops before any paid request. Secrets live only in
`OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` env vars — never in logs, reports,
or git. Only public/synthetic fixture code is ever sent to teachers.

## Colab workflow

See `docs/colab.md`. L4 24GB preferred (bf16); T4 16GB fallback (fp16,
watch GDN NaNs). LoRA smoke first (100 steps @ 2048), full-weight config is
a 15-step OOM probe only.

## Current limitations

- No GPU run yet: cache proof used a tiny random model on CPU; 0.8B LoRA
  smoke awaits a human Colab session.
- No paid teacher labels ($0.00 spent; no keys present) — training data is
  static FIM + synthetic next-edit + git-mined trajectories.
- Baselines (Falcon-H1, Granite-4.0-H, RWKV-7) are interface-planned only
  (`reports/baselines.md`); Qwen first by policy.
- Tree-sitter grammars: python only; other languages fall back to
  regex/line heuristics.
