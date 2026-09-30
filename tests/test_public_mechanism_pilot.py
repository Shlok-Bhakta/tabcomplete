"""Local-only invariants for the bounded public author/solver/reviewer runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.observability.context import RunContext
from tinycomplete.one_line.contract import EditAction, encode_action
from tinycomplete.one_line.data import parse_author_response
from tinycomplete.one_line.pilot_roles import build_reviewer_prompt
from tinycomplete.one_line.teacher import TeacherCompletedResponseError, TeacherResponse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_public_mechanism_pilot.py"
SPEC = importlib.util.spec_from_file_location("public_mechanism_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([0] if add_special_tokens else [])


def _source_row() -> dict[str, Any]:
    source = "def f(x, y):\n    result = x.old\n    return y.old\n"
    return {
        "id": "public-source/test",
        "student_state_seed": {
            "file_id": "f.py",
            "filetype": "python",
            "source": source,
        },
        "authoring_metadata": {
            "source_repo": "example/f",
            "source_aliases": ["example/f"],
            "source_revision": "a" * 40,
            "source_path": "f.py",
            "source_license": "MIT",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "license_sha256": "b" * 64,
            "authoring_focus": None,
            "source_provenance_verified": True,
            "file_license_scope_unverified_without_notice": True,
        },
    }


def _candidate() -> dict[str, Any]:
    row = _source_row()
    response = {
        "prior_edit": {
            "row": 1,
            "old_text": "    result = x.old",
            "new_text": "    result = x.new",
        },
        "target_row": 2,
        "action": {"kind": "R", "text": "    return y.new"},
        "intent_evidence": "The same field spelling is used at both visible sites.",
        "objective": {
            "kind": "field_name_consistency",
            "description": "Keep the copied field name consistent.",
            "checks": [],
        },
    }
    candidate = parse_author_response(json.dumps(response), row, ByteTokenizer())
    candidate["provenance"]["author_actor_id"] = pilot.AUTHOR_ACTOR
    return candidate


def test_author_prompt_drops_focus_tags_but_keeps_hashed_source_evidence() -> None:
    row = _source_row()
    row["authoring_metadata"]["authoring_focus"] = "DO NOT LEAK THIS HIDDEN FOCUS"
    item = {
        "source_row": row,
        "selected_seed": {
            "detected_source_evidence": [
                {
                    "family": "local_helper_reuse",
                    "evidence": [{"line": 1, "role": "declaration", "text": "def f(x, y):"}],
                }
            ]
        },
    }
    prompt = pilot._author_prompt(item)
    assert "DO NOT LEAK THIS HIDDEN FOCUS" not in prompt
    assert '"author_focus": null' in prompt
    assert '"text": "def f(x, y):"' in prompt
    assert "not an intended edit" in prompt


def test_solver_action_uses_complete_canonical_wire_and_requires_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    item = {"source_id": "public-source/test"}
    tok = ByteTokenizer()
    wire = encode_action(EditAction(**candidate["action"]))
    event = {"finish_reason": "stop", "request_id": "solver-request"}
    monkeypatch.setattr(
        pilot,
        "_output",
        lambda _event: f"<FINAL_ACTION>\n{wire}\n</FINAL_ACTION>",
    )
    action, parsed_wire, n_tokens = pilot._solver_action(item, candidate, event, tok)
    assert action == EditAction(**candidate["action"])
    assert parsed_wire == wire
    assert n_tokens == len(tok.encode(wire, add_special_tokens=False)) + 1

    event["finish_reason"] = "length"
    assert pilot._solver_action(item, candidate, event, tok) == (None, None, None)


def test_reviewer_evidence_hash_matches_the_canonical_wire_prompt() -> None:
    tok = ByteTokenizer()
    candidate = _candidate()
    action_wire = encode_action(EditAction(**candidate["action"]))
    reviewer_text = json.dumps(
        {"retain": True, "ambiguous": False, "reason": "Visible identifier propagation."}
    )
    events = {
        role: {"session_id": f"session-{role}-0001", "response_id": f"response-{role}"}
        for role in ("author", "solver", "reviewer")
    }
    events["author"]["output_sha256"] = candidate["provenance"]["author_response_sha256"]
    case = {
        "candidate": candidate,
        "author_event": events["author"],
        "solver_event": events["solver"],
        "review_event": events["reviewer"],
    }
    payload = json.loads(pilot._role_evidence_bytes(case, action_wire, 8, reviewer_text, tok))
    expected_prompt = build_reviewer_prompt(candidate, action_wire)
    assert (
        payload["reviewer"]["prompt_sha256"] == hashlib.sha256(expected_prompt.encode()).hexdigest()
    )
    assert payload["solver"]["wire"] == action_wire


def test_independent_identifier_control_does_not_accept_a_wrong_author_action() -> None:
    candidate = _candidate()
    objective = pilot._objective_from_control(candidate)
    assert objective is not None
    assert objective["checker"] == "visible-identifier-copy-v1"
    assert objective["expected_action"] == candidate["action"]

    candidate["action"] = {"kind": "replace_line", "text": "    return y.other"}
    assert pilot._objective_from_control(candidate) is None


def test_recovery_plan_freezes_only_uncalled_seeds_and_remaining_call_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    active = [
        {
            "source_id": f"source-{index}",
            "source_row": _source_row(),
            "selected_seed": {"detected_source_evidence": []},
        }
        for index in range(8)
    ]
    scope = {
        "selection_plan_sha256": "a" * 64,
        "selected_seed_lines_sha256": "b" * 64,
        "file_scope_review_sha256": "c" * 64,
        "selection_rules_sha256": "d" * 64,
        "selected_count": 12,
        "dispositions": [],
        "file_license_policy": "test-only",
    }
    original_ids = [f"original-{index:03}" for index in range(178)]
    original = {
        "ledger": {
            "baseline_totals": {"calls": 178, "input_tokens": 100, "output_tokens": 200},
            "baseline_request_ids": original_ids,
            "baseline_request_ids_sha256": hashlib.sha256(
                pilot.canonical_json(original_ids)
            ).hexdigest(),
        }
    }
    prior_run = tmp_path / "prior-execution"
    prior_run.mkdir()
    (prior_run / "preflight_plan.json").write_text(json.dumps(original), encoding="utf-8")
    failed_sources = ["prior-a", "prior-b", "prior-c"]
    failures = [
        {"source_id": source, "request_id": pilot._role_request_id("author", source)}
        for source in failed_sources
    ]
    historical_ids = sorted(entry["request_id"] for entry in failures)
    historical = {
        "ambiguous_request_ids": historical_ids,
        "ambiguous_source_ids": sorted(failed_sources),
        "prior_run_v1_plan_sha256": "1" * 64,
        "prior_run_v1_events_sha256": "2" * 64,
        "prior_run_v2_plan_sha256": "3" * 64,
        "prior_run_v2_events_sha256": "4" * 64,
        "diagnostic_request_id": pilot.TRANSPORT_DIAGNOSTIC_REQUEST_ID,
        "diagnostic_result_sha256": "5" * 64,
    }
    ledger_rows = {request_id: {} for request_id in original_ids}
    ledger_rows.update({request_id: {} for request_id in historical_ids})
    ledger_rows[pilot.TRANSPORT_DIAGNOSTIC_REQUEST_ID] = {"input": 5, "output": 2}
    current_totals = {"calls": 182, "input_tokens": 6_000, "output_tokens": 7_000}

    monkeypatch.setattr(pilot, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(pilot, "PRIOR_RUN_DIR", prior_run)
    monkeypatch.setattr(pilot, "_source_rows_and_license_scope", lambda: (active, scope))
    monkeypatch.setattr(
        pilot,
        "_load_historical_failures_and_ledger",
        lambda **_kwargs: (failures, ledger_rows, current_totals, historical),
    )
    monkeypatch.setattr(pilot, "_assert_public_prompt", lambda *_args: None)
    monkeypatch.setattr(pilot, "_runtime_fingerprint", lambda: {"runtime": "fixed-test"})
    ledger = pilot.TeacherUsageLedger(pilot.LEDGER)
    plan = pilot.build_plan(ledger, tokenizer=ByteTokenizer())
    assert plan["provider"]["max_planned_calls"] == 24
    assert plan["provider"]["request_calls_max"] == 32
    assert plan["provider"]["historical_calls_consumed"] == 4
    assert plan["provider"]["remaining_calls_at_freeze"] == 32
    assert plan["provider"]["campaign_total_calls_max"] == 36
    assert plan["eligible_seed_ids"] == [f"source-{index}" for index in range(8)]
    assert len(plan["carried_ambiguous_request_ids"]) == 3
    assert plan["training"]["started"] is False
    assert plan["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()

    over = active + [
        {
            "source_id": f"source-extra-{index}",
            "source_row": _source_row(),
            "selected_seed": {"detected_source_evidence": []},
        }
        for index in range(3)
    ]
    monkeypatch.setattr(pilot, "_source_rows_and_license_scope", lambda: (over, scope))
    with pytest.raises(ValueError, match="32-call recovery ceiling"):
        pilot.build_plan(ledger, tokenizer=ByteTokenizer())


def test_first_new_ambiguous_failure_opens_breaker_but_history_alone_does_not() -> None:
    plan = {"carried_ambiguous_request_ids": ["old-request-1", "old-request-2"]}
    event_sets = {"author": {}, "solver": {}, "reviewer": {}}
    assert not pilot._has_new_ambiguous_failure(plan, event_sets)
    event_sets["author"]["current-c"] = {
        "request_id": "new-request",
        "source_id": "current-c",
        "http_status": 500,
        "failure_status": "ambiguous_transport_failure",
    }
    assert pilot._has_new_ambiguous_failure(plan, event_sets)

    class NoCallClient:
        def run_role(self, **_kwargs: Any) -> None:
            pytest.fail("fresh ambiguous failure must stop provider requests")

    pilot._call_role_if_missing(
        role="author",
        item={"source_id": "current-d"},
        prompt="public fixture prompt",
        system="fixed system",
        purpose="student_label",
        output_schema=None,
        plan=plan,
        client=NoCallClient(),  # type: ignore[arg-type]
        ledger=object(),  # type: ignore[arg-type]
        event_sets=event_sets,
        run_context=RunContext.new(campaign_id="test-campaign"),
    )
    assert "current-d" not in event_sets["author"]


def test_completed_overrun_is_durable_terminal_event_and_opens_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot, "RUN_DIR", tmp_path)
    monkeypatch.setattr(pilot, "client_prompt_tokens", lambda _prompt: [1])
    item = {"source_id": "public-source-overrun"}
    plan = {"plan_sha256": "a" * 64}
    response = TeacherResponse(
        content='{"retain":false,"ambiguous":true,"reason":"synthetic"}',
        session_id="session-overrun",
        response_id="message-overrun",
        model_id=pilot.MODEL_ID,
        input_tokens=120,
        output_tokens=9,
        reasoning_tokens=4,
        total_tokens_reported=133,
        cached_read_tokens=0,
        cached_write_tokens=0,
        finish_reason="stop",
        cost_usd_reported=0.0,
    )

    class CompletedOverrunClient:
        def run_role(self, **kwargs: Any) -> TeacherResponse:
            evidence_hash = kwargs["persist_completed_response"](response)
            raise TeacherCompletedResponseError(
                request_id=kwargs["request_id"],
                response=response,
                failure_status="completed_budget_overrun",
                evidence_sha256=evidence_hash,
                ledger_reconciled=True,
            )

    event = pilot._capture_response(
        client=CompletedOverrunClient(),  # type: ignore[arg-type]
        ledger=object(),  # type: ignore[arg-type]
        plan=plan,
        role="reviewer",
        item=item,
        prompt="Public synthetic prompt.",
        purpose="automated_score",
        system_instruction="Fixed review instruction.",
        output_schema=None,
    )

    assert event["failure_status"] == "completed_budget_overrun"
    assert event["provider_completed"] is True
    assert event["retry_permitted"] is False
    assert event["output_available"] is False
    assert event["response_evidence_sha256"]
    loaded = pilot._load_events("reviewer", plan, [item])
    assert loaded[item["source_id"]]["failure_status"] == "completed_budget_overrun"
    with pytest.raises(ValueError, match="unavailable"):
        pilot._output(event)
    event_sets = {"author": {}, "solver": {}, "reviewer": {item["source_id"]: event}}
    no_historical_failures = {"carried_ambiguous_request_ids": []}
    assert not pilot._has_new_ambiguous_failure(no_historical_failures, event_sets)
    assert pilot._has_new_role_blocker(no_historical_failures, event_sets)


def test_reviewer_uses_separate_structured_only_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    source_id = candidate["id"]
    item = {
        "source_id": source_id,
        "source_row": _source_row(),
        "selected_seed": {"detected_source_evidence": []},
    }
    event_sets = {
        "author": {source_id: {"output_available": True}},
        "solver": {source_id: {"output_available": True}},
        "reviewer": {},
    }
    clients: list[bool] = []
    calls: list[tuple[str, bool]] = []

    class StubClient:
        def __init__(
            self, _ledger: object, *, structured_output_only: bool = False, **_kwargs: Any
        ):
            self.structured_output_only = structured_output_only
            clients.append(structured_output_only)

        def __enter__(self) -> StubClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(pilot, "OpenCodeTeacherClient", StubClient)
    monkeypatch.setattr(pilot, "_load_candidate", lambda *_args: (candidate, None))
    monkeypatch.setattr(pilot, "_solver_action", lambda *_args: (None, "N", 1))
    monkeypatch.setattr(pilot, "build_reviewer_prompt", lambda *_args: "fixed reviewer prompt")
    monkeypatch.setattr(
        pilot,
        "_call_role_if_missing",
        lambda **kwargs: calls.append((kwargs["role"], kwargs["client"].structured_output_only)),
    )
    pilot._run_role_passes(
        {"carried_ambiguous_request_ids": []},
        [item],
        ByteTokenizer(),
        object(),  # type: ignore[arg-type]
        event_sets,
        RunContext.new(campaign_id="test-campaign"),
    )
    assert clients == [False, True]
    assert calls == [("author", False), ("solver", False), ("reviewer", True)]


def test_missing_reviewer_payload_is_reported_without_loading_or_accepting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    source_id = candidate["id"]
    item = {"source_id": source_id, "source_row": _source_row()}
    action = EditAction(**candidate["action"])
    monkeypatch.setattr(pilot, "_load_candidate", lambda *_args: (candidate, None))
    monkeypatch.setattr(
        pilot,
        "_solver_action",
        lambda *_args: (action, encode_action(action), 8),
    )
    monkeypatch.setattr(
        pilot, "_review_verdict", lambda _event: (None, "reviewer_provider_not_stopped")
    )
    monkeypatch.setattr(pilot, "_objective_from_control", lambda _candidate: {"counts": []})
    monkeypatch.setattr(
        pilot,
        "check_candidate_acceptance",
        lambda *_args, **_kwargs: pytest.fail("missing reviewer cannot reach acceptance"),
    )
    event_sets = {
        "author": {source_id: {"output_available": True}},
        "solver": {source_id: {"output_available": True}},
        "reviewer": {
            source_id: {
                "output_available": False,
                "failure_status": "ambiguous_transport_failure",
                "failure_type": "TeacherTransportError",
                "http_status": None,
                "failure_stage": None,
            }
        },
    }
    cases, summary = pilot._assess_candidates(
        [item], event_sets["author"], event_sets["solver"], event_sets["reviewer"], ByteTokenizer()
    )
    assert len(cases) == 1
    assert summary["training_accepted"] == 0
    assert summary["mechanically_supported_candidates"] == 1
    assert summary["by_source"][source_id]["status"] == "reviewer_output_unavailable"
    assert summary["by_source"][source_id]["reviewer_failure_type"] == "TeacherTransportError"

    unreviewed_cases, unreviewed_summary = pilot._assess_candidates(
        [item], event_sets["author"], event_sets["solver"], {}, ByteTokenizer()
    )
    assert len(unreviewed_cases) == 1
    assert unreviewed_summary["parsed_author_candidates"] == 1
    assert unreviewed_summary["valid_solver_actions"] == 1
    assert unreviewed_summary["by_source"][source_id]["status"] == "reviewer_not_called"
