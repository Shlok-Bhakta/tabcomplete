# Overnight State — Tiny Local Next-Edit Code Model

Durable source of truth. Updated immediately after each milestone.
Subagents: read AGENTS.md before doing work.

Legend: NOT_STARTED | IN_PROGRESS | PASS | BLOCKED

## 0. Startup / inspection
- Status: PASS
- Files changed: AGENTS.md, docs/OVERNIGHT_STATE.md
- Verification command: `test -s AGENTS.md && command -v uv && git status --short || true`
- Verification result: PASS (uv 0.12.3, python3.11 via uv available, codex exec available; repo dir was empty, not a git repo yet)
- Blocker: none
- Next milestone: bootstrap

## 1. Bootstrap (uv Python 3.11 project, deps, ruff+pytest baseline)
- Status: PASS
- Files changed: pyproject.toml, .python-version, .gitignore, README.md (stub), scripts/bootstrap_cpu.sh, src/tinycomplete/**/__init__.py, tests/test_bootstrap.py, uv.lock
- Verification command: `uv run ruff check . && uv run pytest -q` + qwen35 import check
- Verification result: PASS — ruff clean; pytest 2 passed; `from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig` OK on transformers 5.17.0; torch 2.14.0+cpu (CPU-only, installed outside lockfile via scripts/bootstrap_cpu.sh)
- Blocker: none
- Next milestone: editor event protocol

## 2. Editor event protocol + deterministic replay
- Status: PASS
- Files changed: src/tinycomplete/protocol/events.py, src/tinycomplete/protocol/replay.py, src/tinycomplete/protocol/__init__.py, tests/test_protocol.py
- Verification command: `uv run pytest -q tests/test_protocol.py`
- Verification result: PASS — 12 passed (open/snapshot/insert, insert+delete, replace, unicode incl. mid-sequence rejection, multi-file, cursor, snapshot rebase, deterministic replay x2, bad seq rejected, OOB rejected, immutability, dict round-trip)
- Blocker: none
- Next milestone: serialization

## 3. Model serialization
- Status: PASS
- Files changed: src/tinycomplete/protocol/serialize.py, tests/test_serialize.py, docs/serialization.md
- Verification command: `uv run pytest -q tests/test_serialize.py` (+ round-trip via test_protocol)
- Verification result: PASS — 15 passed (serialize+protocol); realistic session: 260 chars / 111 Qwen tokens (tokenizer Qwen/Qwen3.5-0.8B-Base from hub, no weights); escaping round-trip OK; PREDICT always last line
- Blocker: none
- Next milestone: FIM + next-edit

## 4. FIM + next-edit data
- Status: PASS
- Files changed: src/tinycomplete/data/schema.py, src/tinycomplete/data/fim.py, src/tinycomplete/data/static_edits.py, src/tinycomplete/data/__init__.py, tests/test_fim.py
- Verification command: `uv run pytest -q tests/test_fim.py`
- Verification result: PASS — 8 passed (all 8 hole types cover source; byte-identical determinism; PSM/SPM with native Qwen `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>` tokens verified present in hub tokenizer; tree-sitter alignment for identifier/import/function_body; NO_EDIT first-class target observed; region bounds verified against current file)
- Blocker: none
- Next milestone: git synthetic miner

## 5. Git synthetic edit trajectories
- Status: PASS
- Files changed: src/tinycomplete/data/git_edits.py, tests/test_git_edits.py
- Verification command: `uv run pytest -q tests/test_git_edits.py`
- Verification result: PASS — 3 passed (temp-repo extraction: non-py skipped, added-file handled, enclosing nodes incl. function_definition, intermediates converge on child text, future labels exposed, determinism, linear history; provenance git_synthetic + not-human-order disclaimer on every record)
- Blocker: none
- Next milestone: cache proof

## 6. CPU-only Qwen3.5 cache proof
- Status: PASS
- Files changed: src/tinycomplete/model/tiny_qwen.py, src/tinycomplete/model/cache_probe.py, tests/test_cache.py, reports/cache.md
- Verification command: `uv run pytest -q tests/test_cache.py`
- Verification result: PASS — 6 passed. Continuation vs full forward max diff 1.19e-07; 3 recurrent layers (fixed-size [1,2,16,16] state) + 1 KV layer (grows +256B/token toy cfg); deepcopy branches bit-identical; naive reuse diverges (0.133). No upstream CPU bug encountered (reference torch kernels used).
- Blocker: none
- Next milestone: teacher layer

## 7. Teacher interface + prompt + OpenRouter/DeepSeek + budget + validation
- Status: PASS (extended 2026-09-17: DeepSeek stages A+B live)
- Files changed: src/tinycomplete/teacher/{base,prompt,budget,validate,fake,openrouter,deepseek,generate,__init__}.py, src/tinycomplete/cli.py, configs/teacher.example.toml, scripts/deepseek_stage_a.py, tests/test_teacher.py, tests/test_budget.py, reports/synthetic.md
- Verification command: `uv run pytest -q tests/test_teacher.py tests/test_budget.py`
- Verification result: PASS — 16 passed. LIVE: DeepSeek flash 60/60 req ok, 180/180 schema, 148 acc/32 rej, ~$0.022 of $2.00, 70/2000 examples. Fixes: thinking disabled, JSON extraction, prompt schema skeleton, quota only on success. ToS §4.2 permits distillation. Stage C awaits user go-ahead.
- Blocker: none
- Next milestone: evaluator

## 8. Evaluator (metrics + latency)
- Status: PASS
- Files changed: src/tinycomplete/eval/{metrics,latency,__init__}.py, tests/test_eval.py
- Verification command: `uv run pytest -q tests/test_eval.py`
- Verification result: PASS — 5 passed (exact/normalized, prefix len, kitten/sitting=3, parse success/fail, noop accuracy 0.5 + None, aggregate rates, JSONL store, CPU benchmark stats incl. VRAM None)
- Blocker: none
- Next milestone: colab harness

## 9. Colab training harness (notebook + train.py + configs)
- Status: PASS
- Files changed: src/tinycomplete/train/{train,__init__}.py, notebooks/qwen35_colab.ipynb, configs/qwen35_{lora,full}_smoke.yaml, scripts/{colab_smoke,smoke}.py, docs/colab.md, reports/colab.md, tests/fixtures/sample.py
- Verification command: `uv run python scripts/colab_smoke.py --help` + full run
- Verification result: PASS — both YAML configs load (lora 100 steps @2048, full 15 steps @1024); formatting pipeline OK; real Qwen token stats (n=12 median=94 p95=126 max=126); notebook JSON valid (config→runtime→deps→dataset→lora→save→reload/eval→throughput). Unsloth Qwen3.5 support verified against current official docs (transformers v5+, FastLanguageModel, GDN/fla notes). No GPU runtime launched (by design).
- Blocker: none
- Next milestone: docs + final gate

## 10. Docs + scripts + final verification + overnight-summary + public gh repo
- Status: PASS
- Files changed: README.md, docs/architecture.md, reports/baselines.md, reports/overnight-summary.md
- Verification command: `uv run ruff check . && uv run pytest -q && uv run mypy src`
- Verification result: PASS — ruff clean; pytest 54 passed; mypy 28 files clean; TODO-search empty; git diff --check clean; public repo https://github.com/Shlok-Bhakta/tabcomplete created and pushed
- Blocker: none
- Next milestone: done

## Subagent strategy note
- `codex exec` exists (native non-interactive mechanism confirmed via `codex --help`).
- Decision: root proceeds alone using this file as persistent context to avoid two agents
  editing the same files simultaneously. Short-lived `codex exec` may be used for
  read-only research only if needed; all edits verified by root.
