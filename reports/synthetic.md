# Synthetic Data Report

Date: 2026-09-16. Paid gate: `ALLOW_PAID_SYNTHETIC` unset → **no paid calls made**.

## Paid API usage

- OpenRouter requests: 0, spend: $0.00
- DeepSeek requests: 0, spend: $0.00
- Total paid examples: 0
- `~/Hermes` does not exist on this machine; no DeepSeek key file was
  located (search scope restricted to `~/Hermes` per policy; no filesystem
  hunt performed). `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` absent from env.
- OpenRouter model-id verification and live pricing were therefore NOT
  recorded — they are mandatory prerequisites before any future paid run
  (see `src/tinycomplete/teacher/openrouter.py::verify_model_id`).

## Fake-provider pipeline (full path, $0)

`uv run tinycomplete synthetic --provider fake --states 12 --seed 1
--out data/generated/teacher_fake.jsonl`

- states: 12 (10 stage-A + 2 stage-B), request ok-rate 12/12
- accepted candidates: 20, rejected (stored with reasons): 4
- spend: $0.0000; budget file `data/generated/budget.json` untouched by fakes
- artifacts (gitignored): `data/generated/teacher_fake.jsonl`,
  `data/generated/teacher_fake.summary.json`

## Validation enforced per candidate

JSON/schema parse, region applicability, UTF-8, changed-unless-noop,
tree-sitter error-count check (python), rewrite size caps (8 KB absolute,
4× region), anti-reversal of the preceding edit, anti-duplication of the
region. Rejects are kept for future preference data.

## Staging policy for a future paid run

A (10) → require 100% request success → B (50×3) → require ≥90% schema
success → C (up to caps/budget). Caps: $2.00 total, 2000 examples,
3 candidates/state, ≤4 attempts/request with exponential backoff.
Record live OpenRouter pricing + model listing in this file first.
