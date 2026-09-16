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
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_protocol.py`
- Verification result:
- Blocker:
- Next milestone: serialization

## 3. Model serialization
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_protocol.py` (round-trip covered there) + manual char/token counts
- Verification result:
- Blocker:
- Next milestone: FIM + next-edit

## 4. FIM + next-edit data
- Status: NOT_STARTED
- Files changed:
- Verification command: `uv run pytest -q tests/test_fim.py`
- Verification result:
- Blocker:
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
