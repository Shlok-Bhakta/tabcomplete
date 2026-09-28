"""Frozen structured smoke identity checks; no provider calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_one_line_structured_smoke.py"
_SPEC = importlib.util.spec_from_file_location("one_line_structured_smoke_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_frozen_plan8_spec_prompt_and_schema() -> None:
    plan_sha = runner.sha_bytes(runner.PLAN.read_bytes())
    spec = runner.verify_frozen(plan_sha)
    assert plan_sha == "404c848388359a6f6771ea8692634684f9de009c67347a7982fdb9a8713281c2"
    assert spec["request_id"] == "structured8-synthetic-001"
    assert runner.sha_bytes(spec["prompt"].encode()) == runner.PROMPT_SHA
    assert runner.sha_bytes(runner.canonical_bytes(spec["schema"])) == runner.SCHEMA_SHA


def test_frozen_plan8_rejects_changed_plan_and_spec(tmp_path: Path) -> None:
    root = tmp_path
    path = root / runner.SPEC_REL
    path.parent.mkdir(parents=True)
    spec = json.loads((runner.ROOT / runner.SPEC_REL).read_text())
    spec["prompt"] += " changed"
    path.write_text(json.dumps(spec))
    plan = json.loads(runner.PLAN.read_text())
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = runner.sha_bytes(plan_path.read_bytes())
    with pytest.raises(ValueError, match="spec changed"):
        runner.verify_frozen(plan_sha, plan_path=plan_path, root=root)
    with pytest.raises(ValueError, match="plan-8 hash mismatch"):
        runner.verify_frozen("0" * 64, plan_path=plan_path, root=root)
