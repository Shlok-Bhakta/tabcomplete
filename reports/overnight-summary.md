# Overnight Summary

## Completed

- Bootstrap: uv Python 3.11 project, transformers 5.17 (Qwen3.5 classes present),
  CPU torch via `scripts/bootstrap_cpu.sh` (kept out of `uv.lock`), ruff+mypy+pytest green.
- Editor event protocol: 13 typed immutable events, byte-offset UTF-8 convention,
  deterministic replay with seq/bounds validation.
- Serialization: compact tag format, escaping round-trip, realistic session doc
  (260 chars / 111 Qwen tokens), PREDICT always last.
- FIM + next-edit: seeded PSM/SPM in 8 hole classes with native Qwen FIM tokens,
  tree-sitter alignment, first-class NOOP targets, region-rewrite format.
- Git synthetic miner: temp-repo extraction, added-file handling, enclosing nodes,
  converging intermediates, `git_synthetic` + not-human-order provenance.
- Cache proof: tiny hybrid model continuation ≡ full forward (1.2e-07);
  fixed-size recurrent state + linear KV growth; deepcopy branch isolation.
- Teacher layer: protocol, rule prompt, OpenRouter (structured) + DeepSeek adapters,
  budget gate, per-candidate validation, staged fake pipeline (12 states, 20/4).
- Evaluator: independent metrics + CPU latency harness.
- Colab harness: 8-cell notebook, `train.py`, LoRA + full smoke configs,
  `colab_smoke.py` gate, version report.
- Docs: README, architecture, serialization, colab workflow, baselines plan.

## Blocked

- Paid teacher stages (OpenRouter/DeepSeek): BLOCKED by design — `ALLOW_PAID_SYNTHETIC`
  unset, no keys in env, no `~/Hermes` key file exists. Fake pipeline covers the path.
- GPU LoRA smoke: BLOCKED on environment (CPU-only local; needs human Colab run).
- Nothing else blocked. No upstream CPU bug encountered.

## Tests

`uv run pytest -q` → **54 passed** (11.63s).
`uv run ruff check .` → **All checks passed!**
`uv run mypy src` → **Success: no issues found in 28 source files**.
Incomplete-code search (`TODO|FIXME|NotImplementedError|placeholder` in src/tests) →
no matches. `git diff --check` → clean.

## Qwen Cache Findings

Continuation over A→B matches full A+B forward to max abs diff 1.19e-07.
Per linear layer: fixed-size recurrent `[1,2,16,16]` fp32 + conv `[1,96,4]`;
KV layer `[1,2,seq,16]` grows +256 B/token (toy cfg). `copy.deepcopy` branches
are bit-identical (0.0); naive in-place reuse diverges (0.133) — candidates
must copy first. CPU runs reference torch kernels (no fla/causal-conv1d).
Full detail: `reports/cache.md`.

## Synthetic Data Findings

Static FIM/SPM + next-edit + git trajectories all deterministic and validated.
Fake teacher run: 12/12 requests ok, 20 accepted / 4 rejected-with-reasons.
Validation rejects syntax-breaking, oversized, reversal, and duplication cases.
Paid path implemented but unexecuted by gate. Full detail: `reports/synthetic.md`.

## Paid API Usage

OpenRouter requests: 0
OpenRouter estimated/reported spend: $0.00
DeepSeek requests: 70 states / 180 candidates (deepseek-flash)
DeepSeek estimated/reported spend: $0.022 (peak-price + 50% margin estimate)
Total paid examples: 70

## Colab Harness

`notebooks/qwen35_colab.ipynb` + `src/tinycomplete/train/train.py` +
`configs/qwen35_lora_smoke.yaml` (100 steps @2048) +
`configs/qwen35_full_smoke.yaml` (15-step OOM probe @1024).
Unsloth Qwen3.5 support verified against current docs (transformers v5+,
FastLanguageModel, LoRA q/k/v/o+mlp, GDN fp16 caution → prefer L4/bf16).
Versions table + first-run checklist: `reports/colab.md`.

## Dependency Versions

transformers 5.17.0, torch 2.14.0+cpu (local only), pydantic 2.x, httpx,
tenacity, orjson, PyYAML, tree-sitter 0.26 + language-pack 1.20, unidiff 1.0.1,
huggingface-hub, tokenizers, safetensors, pytest 9.1.1, ruff 0.16.7, mypy.
Python 3.11 via uv. See `uv.lock` for the full pin set.

## Files Added

AGENTS.md, README.md, pyproject.toml, uv.lock, .gitignore, .python-version,
configs/{qwen35_lora_smoke,qwen35_full_smoke,teacher.example.toml}.yaml/toml,
docs/{OVERNIGHT_STATE,serialization,architecture,colab}.md,
notebooks/qwen35_colab.ipynb, reports/{cache,synthetic,colab,baselines}.md,
scripts/{bootstrap_cpu,smoke,colab_smoke}.{sh,py},
src/tinycomplete/{cli,protocol/{events,replay,serialize},data/{schema,fim,static_edits,git_edits},model/{tiny_qwen,cache_probe},teacher/{base,prompt,budget,validate,fake,openrouter,deepseek,generate},eval/{metrics,latency},train/train}.py,
tests/{test_bootstrap,protocol,serialize,fim,git_edits,cache,teacher,budget,eval}.py,
tests/fixtures/sample.py.

## Known Problems

- None failing. Watch items: T4 fp16 GDN NaN risk (use L4/bf16); OpenRouter model
  id + pricing must be re-verified live before any paid run; `data/generated/`
  and `checkpoints/` are gitignored by design.

## Recommended First Command Tomorrow

`bash scripts/smoke.sh` — then open `notebooks/qwen35_colab.ipynb` on an L4 runtime.
