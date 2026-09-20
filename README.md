# tinycomplete: tiny local next-edit code model

## Project goal

Build a tiny, local, privacy-preserving code next-edit/autocomplete engine.
Stage 1 has produced a full-weight code-specialized Qwen3.5-0.8B checkpoint;
Stage 2 will train next-edit behavior separately.

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

The recurrent state is NOT permanent lossless repository storage. Exact
facts are re-injected from deterministic indexes (files, Tree-sitter, LSP).

## Why Qwen3.5-0.8B

- Hybrid text architecture: 24 LM layers = 18 Gated DeltaNet linear-attention
  + 6 full softmax attention (3-linear/1-full × 6).
- Native FIM tokenizer tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`),
  long context, recurrent state + KV cache, exactly the split an append-only
  session design needs.
- Small enough (0.8B) for full-weight two-T4 FSDP training.

## Local CPU setup

```bash
uv sync --group dev        # Python 3.11, transformers 5.17 (has Qwen3_5 classes)
bash scripts/bootstrap_cpu.sh   # CPU-only torch, kept OUT of uv.lock
bash scripts/smoke.sh      # ruff + pytest + harness check
```

Local machine is CPU-only by policy. Never install CUDA locally or provision
persistent cloud resources. GPU jobs use the checked-in Kaggle kernels.

## Test command

```bash
uv run ruff check .
uv run pytest -q
```

## Executable code benchmark

The fixed 200-case suite covers nine languages, includes 20 multi-file cases, and
scores syntax, compilation, and hidden behavioral tests. Generated code runs in
rootless Podman or Docker containers with no network and fixed resource limits.

```bash
# Prove every gold completion works before scoring a model.
uv run python scripts/evaluate_code_benchmark.py \
  --gold --backend container --workers 1 \
  --output-dir outputs/code_benchmark/gold

# Generate model completions, then execute them in the same fixtures.
uv run python scripts/generate_code_predictions.py \
  --model-path /path/to/checkpoint --max-new-tokens 96 \
  --output outputs/code_benchmark/model/predictions.jsonl
uv run python scripts/evaluate_code_benchmark.py \
  --predictions outputs/code_benchmark/model/predictions.jsonl \
  --backend container --workers 1 \
  --output-dir outputs/code_benchmark/model/results
```

## Local browser playground

Run the base model and promoted checkpoint behind the local editor UI:

```bash
uv run tinycomplete-playground \
  --model base=Qwen/Qwen3.5-0.8B-Base \
  --model stage1=/path/to/tokens-005000000
```

Open `http://127.0.0.1:8765`. The editor shows ghost completions, accepts them with
Tab, rejects with Escape, accepts repository context files, switches checkpoints,
and records explicit feedback locally. Bind `--host 0.0.0.0` only when you intend
to expose the server to trusted devices on your LAN.

For a GGUF checkpoint, start a local OpenAI-compatible server and point the same
playground at it:

```bash
llama-server -m /path/to/tabcomplete-code-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 --reasoning off
uv run tinycomplete-playground \
  --server-url http://127.0.0.1:8080 \
  --server-model tabcomplete-q4 --server-label "TabComplete Code Q4"
```

Qwen3.5 keeps its native MTP block outside the causal checkpoint used for Stage-1
training. Prepare an indexed export directory before running llama.cpp's converter
so that the MTP sidecar is preserved:

```bash
uv run python scripts/prepare_qwen35_gguf_source.py \
  /path/to/tokens-005000000 outputs/models/stage1-gguf-source
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  outputs/models/stage1-gguf-source \
  --outfile outputs/models/tabcomplete-code-f16.gguf --outtype f16
/path/to/llama-quantize outputs/models/tabcomplete-code-f16.gguf \
  outputs/models/tabcomplete-code-q4_k_m.gguf Q4_K_M
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
`OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` env vars, never in logs, reports,
or git. Only public/synthetic fixture code is ever sent to teachers.

## GPU workflow

See `docs/kaggle.md` and `reports/code_cpt/overnight.md`. The Stage-1 run used two
T4s, FP16 autocast over FP32 master weights, FSDP full sharding, sequence length
2,048, and an 8-bit AdamW optimizer.

## Current result and limitations

- The promoted 5.014M-token checkpoint reduced held-out code NLL by 1.071% across
  all nine core languages. General-text NLL increased 2.733%.
- The executable completion benchmark is synthetic and intentionally unpublished
  before this model run, but 200 cases remain too small for a final product claim.
- Stage 1 trains code token prediction only. It does not teach FIM or next-edit
  behavior.
- Transformers preserves native MTP weights in a sidecar because its causal Qwen3.5
  class does not currently load that module.
- Baselines such as Falcon-H1, Granite, and RWKV remain future comparisons.
