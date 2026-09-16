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
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_git_edits.py`
- Verification result:
- Blocker:
- Next milestone: cache proof

## 6. CPU-only Qwen3.5 cache proof
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_cache.py`
- Verification result:
- Blocker:
- Next milestone: teacher layer

## 7. Teacher interface + prompt + OpenRouter/DeepSeek + budget + validation
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_teacher.py tests/test_budget.py`
- Verification result:
- Blocker:
- Next milestone: evaluator

## 8. Evaluator (metrics + latency)
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q` (full suite)
- Verification result:
- Blocker:
- Next milestone: colab harness

## 9. Colab training harness (notebook + train.py + configs)
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run python scripts/colab_smoke.py --help` / config load check
- Verification result:
- Blocker:
- Next milestone: docs + final gate

## 10. Docs + scripts + final verification + overnight-summary + public gh repo
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run ruff check . && uv run pytest -q && git diff --check && git status --short`
- Verification result:
- Blocker:
- Next milestone: done

## Subagent strategy note
- `codex exec` exists (native non-interactive mechanism confirmed via `codex --help`).
- Decision: root proceeds alone using this file as persistent context to avoid two agents
  editing the same files simultaneously. Short-lived `codex exec` may be used for
  read-only research only if needed; all edits verified by root.
