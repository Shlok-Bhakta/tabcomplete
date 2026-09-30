from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_local_sweep_replay.py"
SPEC = importlib.util.spec_from_file_location("score_local_sweep_replay", SCRIPT)
assert SPEC and SPEC.loader
local_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_score)


def _span(name: str, request_id: str, case: str, precision: str, kind: str, start: int) -> dict:
    return {
        "name": name,
        "start_time_unix_nano": start,
        "attributes": {
            "tabcomplete.request_id": request_id,
            "tabcomplete.case_id": case,
            "tabcomplete.quantization": precision,
            "tabcomplete.cache.condition": kind,
        },
    }


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).isoformat()


def test_exact_row_adapter_preserves_response_bytes_and_runtime_semantics() -> None:
    prompt = {"case_id": "python/case", "prompt_sha256": "prompt-sha"}
    raw = {
        "case_id": "python/case",
        "precision": "q4_k_m",
        "model_sha256": "model-sha",
        "plan_sha256": "local-plan-sha",
        "prompt_sha256": "prompt-sha",
        "raw_output": "  value = 'é'  \n",
        "output_sha256": local_score.base.digest_bytes("  value = 'é'  \n".encode()),
        "input_tokens": 3,
        "actual_terminal_observed": True,
        "hit_output_cap": False,
        "finish_reason": "eos",
        "output_file_mapping": {"mapping": "within_editable_range"},
        "output_token_count": 5,
        "repetition": 1,
        "completed_response_ms": 8.0,
    }
    adapted = local_score.adapt(raw, prompt, "q4_k_m", "model-sha", "local-plan-sha", "req-real")

    assert adapted["raw_output"] == raw["raw_output"]
    assert adapted["output_sha256"] == raw["output_sha256"]
    assert adapted["explicit_terminal"] is True
    assert adapted["hit_token_cap"] is False
    assert adapted["observability_ids"] == {"case_id": "python/case", "request_id": "req-real"}


def test_adapter_rejects_changed_bytes_instead_of_repairing() -> None:
    prompt = {"case_id": "python/case", "prompt_sha256": "prompt-sha"}
    row = {
        "case_id": "python/case",
        "precision": "q8_0",
        "model_sha256": "model-sha",
        "plan_sha256": "local-plan-sha",
        "prompt_sha256": "prompt-sha",
        "raw_output": "x \n",
        "output_sha256": local_score.base.digest_bytes(b"x\n"),
        "input_tokens": 2,
        "actual_terminal_observed": True,
        "hit_output_cap": False,
        "finish_reason": "eos",
        "output_file_mapping": {},
        "output_token_count": 2,
        "repetition": 0,
    }
    with pytest.raises(ValueError, match="raw response bytes/hash mismatch"):
        local_score.adapt(row, prompt, "q8_0", "model-sha", "local-plan-sha", None)


def test_request_ids_join_by_state_kind_and_timestamp_not_row_order(tmp_path: Path) -> None:
    t0 = 1_790_000_000_000_000_000
    rows = [
        {
            "precision": "q4_k_m",
            "case_id": "python/case",
            "repetition": 1,
            "request_kind": "immediate_same_prompt_repeat",
            "request_started_at_utc": _iso(t0 + 2_000_000_000),
        },
        {
            "precision": "q4_k_m",
            "case_id": "python/case",
            "repetition": 0,
            "request_kind": "changed_state",
            "request_started_at_utc": _iso(t0),
        },
    ]
    spans = [
        _span(
            "model.generate",
            "repeat-id",
            "python/case",
            "Q4_K_M",
            "immediate_same_prompt_repeat",
            t0 + 2_100_000_000,
        ),
        _span(
            "request.start", "changed-id", "python/case", "Q4_K_M", "changed_state", t0 + 50_000_000
        ),
        _span(
            "request.start",
            "repeat-id",
            "python/case",
            "Q4_K_M",
            "immediate_same_prompt_repeat",
            t0 + 2_020_000_000,
        ),
        _span(
            "model.generate",
            "changed-id",
            "python/case",
            "Q4_K_M",
            "changed_state",
            t0 + 100_000_000,
        ),
    ]
    path = tmp_path / "offline.jsonl"
    path.write_text("".join(json.dumps(span) + "\n" for span in spans), encoding="utf-8")

    joined, counts = local_score.join_ids(rows, path)

    assert joined[("q4_k_m", "python/case", 0, "changed_state")] == "changed-id"
    assert joined[("q4_k_m", "python/case", 1, "immediate_same_prompt_repeat")] == "repeat-id"
    assert counts["matched"] == 2
    assert counts["unmatched"] == counts["ambiguous"] == 0


def test_ambiguous_timestamp_join_stays_unknown(tmp_path: Path) -> None:
    t0 = 1_790_000_000_000_000_000
    rows = [
        {
            "precision": "q8_0",
            "case_id": "rust/case",
            "repetition": 0,
            "request_kind": "changed_state",
            "request_started_at_utc": _iso(t0),
        }
    ]
    spans = [
        _span("request.start", "left", "rust/case", "Q8_0", "changed_state", t0 - 500_000_000),
        _span("model.generate", "left", "rust/case", "Q8_0", "changed_state", t0 - 400_000_000),
        _span("request.start", "right", "rust/case", "Q8_0", "changed_state", t0 + 500_000_000),
        _span("model.generate", "right", "rust/case", "Q8_0", "changed_state", t0 + 600_000_000),
    ]
    path = tmp_path / "offline.jsonl"
    path.write_text("".join(json.dumps(span) + "\n" for span in spans), encoding="utf-8")

    joined, counts = local_score.join_ids(rows, path)

    assert joined == {}
    assert counts["ambiguous"] == 1
    assert counts["matched"] == 0
