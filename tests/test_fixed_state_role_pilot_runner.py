from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tinycomplete.one_line.contract import EditState, RecentEdit
from tinycomplete.one_line.fixed_state_roles import build_fixed_state_prompt
from tinycomplete.one_line.teacher import MODEL_ID, TeacherResponse, TeacherUsageLedger

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_fixed_state_role_pilot.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("fixed_state_role_pilot_runner", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([0] if add_special_tokens else [])


def _state() -> EditState:
    return EditState(
        file_id="public/example.py",
        filetype="python",
        source="def value(x):\n    return x\n",
        target_row=1,
        cursor_col=len(b"    return x"),
        history=(RecentEdit(1, "    ", "    return x"),),
        relevant=(),
    )


def _private_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def _packet(path: Path, *, add_answer: bool = False) -> None:
    tokenizer = ByteTokenizer()
    state = _state()
    prompt = build_fixed_state_prompt(state, tokenizer)
    row: dict[str, Any] = {
        "candidate_id": "public-seed-candidate-001",
        "seed_id": "public-seed-001",
        "state": asdict(state),
        "prompt": prompt.user_prompt,
        "context_sha256": prompt.context_sha256,
    }
    if add_answer:
        row["answer"] = "must not enter role packet"
    _private_file(path, (json.dumps(row, ensure_ascii=False) + "\n").encode())


def _freeze(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    packet = tmp_path / "packet.jsonl"
    _packet(packet)
    run_dir = tmp_path / "phase"
    ledger = tmp_path / "ledger.jsonl"
    plan = runner.freeze_plan(
        packet_path=packet,
        run_dir=run_dir,
        ledger_path=ledger,
        tokenizer=ByteTokenizer(),
        runtime={"runtime": "test-only"},
    )
    return plan, run_dir, ledger


def _release_bytes(plan: dict[str, Any], *, plan_sha256: str | None = None) -> bytes:
    ids = [case["request_ids"][role] for case in plan["cases"] for role in runner.CALL_ROLES]
    release = {
        "schema": runner.RELEASE_SCHEMA,
        "plan_sha256": plan_sha256 or plan["plan_sha256"],
        "baseline_ledger_sha256": plan["ledger"]["baseline_sha256"],
        "baseline_request_ids_sha256": plan["ledger"]["baseline_request_ids_sha256"],
        "planned_request_ids": ids,
        "max_calls": len(ids),
        "max_wall_seconds": runner.MAX_PHASE_WALL_SECONDS,
        "reserve_input_tokens": runner.RESERVE_INPUT_TOKENS,
        "reserve_output_tokens": runner.RESERVE_OUTPUT_TOKENS,
        "global_caps": {
            "calls": runner.MAX_CALLS,
            "input_tokens": runner.MAX_INPUT_TOKENS,
            "output_tokens": runner.MAX_OUTPUT_TOKENS,
        },
        "released_by": "root",
        "released_at_utc": "2026-09-30T00:00:00Z",
    }
    return runner._canonical(release) + b"\n"


def test_freeze_plan_binds_one_to_eight_exact_answer_free_states_without_calls(
    tmp_path: Path,
) -> None:
    plan, run_dir, ledger = _freeze(tmp_path)
    assert plan["packet"]["case_count"] == 1
    assert plan["provider"]["planned_calls"] == 3
    assert plan["qualification_boundary"]["functional_oracle_status"] == (
        "not_evaluated_by_role_runner"
    )
    assert plan["qualification_boundary"]["training_started"] is False
    assert (
        len({case["request_ids"][role] for case in plan["cases"] for role in runner.CALL_ROLES})
        == 3
    )
    assert not ledger.exists()
    assert (run_dir / "plan.json").stat().st_mode & 0o777 == 0o600
    assert (run_dir / "inputs.jsonl").stat().st_mode & 0o777 == 0o600


def test_freeze_rejects_answer_bearing_packet_and_never_makes_provider_call(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "bad.jsonl"
    _packet(packet, add_answer=True)
    with pytest.raises(ValueError, match="answer-free row schema"):
        runner.freeze_plan(
            packet_path=packet,
            run_dir=tmp_path / "phase",
            ledger_path=tmp_path / "ledger.jsonl",
            tokenizer=ByteTokenizer(),
            runtime={"runtime": "test-only"},
        )


def test_release_plan_mismatch_fails_before_client_construction(tmp_path: Path) -> None:
    plan, run_dir, ledger = _freeze(tmp_path)
    release_path = tmp_path / "release.json"
    payload = _release_bytes(plan, plan_sha256="0" * 64)
    _private_file(release_path, payload)
    constructed = False

    def client_factory(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError("provider client must not be constructed")

    with pytest.raises(ValueError, match="does not authorize"):
        runner.execute_plan(
            run_dir=run_dir,
            release_path=release_path,
            release_sha256=runner._sha(payload),
            tokenizer=ByteTokenizer(),
            runtime={"runtime": "test-only"},
            ledger_path_override=ledger,
            client_factory=client_factory,
        )
    assert not constructed


def test_release_digest_is_mandatory_and_checked_before_client(tmp_path: Path) -> None:
    plan, run_dir, ledger = _freeze(tmp_path)
    release_path = tmp_path / "release.json"
    payload = _release_bytes(plan)
    _private_file(release_path, payload)
    with pytest.raises(ValueError, match="reviewed digest"):
        runner.execute_plan(
            run_dir=run_dir,
            release_path=release_path,
            release_sha256="f" * 64,
            tokenizer=ByteTokenizer(),
            runtime={"runtime": "test-only"},
            ledger_path_override=ledger,
            client_factory=lambda *_args, **_kwargs: pytest.fail("must not create provider client"),
        )


def test_event_chain_detects_mutation_and_duplicate_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase"
    run_dir.mkdir(mode=0o700)
    event = {
        "plan_sha256": "a" * 64,
        "candidate_id": "public-candidate-001",
        "role": "author",
        "request_id": "fixed-state-role-v1-author-000000000000000000000000",
        "status": "ambiguous_transport_failure",
    }
    events: list[dict[str, Any]] = []
    runner._append_event(run_dir, event, events)
    replayed, by_id = runner._read_events(run_dir, "a" * 64)
    assert len(replayed) == 1
    assert len(by_id) == 1
    mutated = json.loads((run_dir / "events.jsonl").read_text())
    mutated["status"] = "completed_valid_action"
    (run_dir / "events.jsonl").write_text(json.dumps(mutated) + "\n")
    (run_dir / "events.jsonl").chmod(0o600)
    with pytest.raises(ValueError, match="event chain"):
        runner._read_events(run_dir, "a" * 64)


def test_fake_execution_persists_each_role_once_and_resumes_without_recalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, run_dir, ledger_path = _freeze(tmp_path)
    release_path = tmp_path / "release.json"
    release_bytes = _release_bytes(plan)
    _private_file(release_path, release_bytes)

    # Keep this integration test local and deterministic while exercising the
    # real runner, ledger, durable response callback, event chain, and resume.
    monkeypatch.setattr(runner, "run_scope", lambda *_args: nullcontext(None))
    monkeypatch.setattr(runner, "_enable_offline_observability", lambda *_args: None)
    monkeypatch.setattr(runner, "operation", lambda *_args, **_kwargs: nullcontext(None))
    persisted_before_settlement: list[str] = []
    settled_ids: list[str] = []
    calls: list[str] = []
    simulate_exit_after_first_settlement = [True]

    class FakeClient:
        def __init__(self, ledger: TeacherUsageLedger) -> None:
            self.ledger = ledger

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def run_role(self, **kwargs: Any) -> TeacherResponse:
            request_id = kwargs["request_id"]
            calls.append(request_id)
            self.ledger.reserve(
                request_id,
                input_tokens=kwargs["reserve_input_tokens"],
                max_output_tokens=kwargs["reserve_output_tokens"],
            )
            role = (
                "reviewer"
                if kwargs["purpose"] == "automated_score"
                else ("author" if "-author-" in request_id else "solver")
            )
            content = (
                '{"retain":true,"ambiguous":false,"reason":"Independent agreement."}'
                if role == "reviewer"
                else "N"
            )
            response = TeacherResponse(
                content=content,
                session_id=f"fake-session-{len(calls):02d}",
                response_id=f"fake-response-{len(calls):02d}",
                model_id=MODEL_ID,
                input_tokens=10,
                output_tokens=max(1, len(content)),
                reasoning_tokens=0,
                total_tokens_reported=None,
                cached_read_tokens=0,
                cached_write_tokens=0,
                finish_reason="stop",
                cost_usd_reported=None,
            )
            receipt_sha = kwargs["persist_completed_response"](response)
            assert isinstance(receipt_sha, str)
            response_path, receipt_path = runner._response_paths(run_dir, request_id)
            assert response_path.is_file() and receipt_path.is_file()
            persisted_before_settlement.append(request_id)
            self.ledger.settle(
                request_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens + response.reasoning_tokens,
            )
            settled_ids.append(request_id)
            if simulate_exit_after_first_settlement[0]:
                simulate_exit_after_first_settlement[0] = False
                raise RuntimeError("simulated process exit after durable settlement")
            return response

    def fake_factory(ledger: TeacherUsageLedger) -> FakeClient:
        return FakeClient(ledger)

    with pytest.raises(RuntimeError, match="simulated process exit"):
        runner.execute_plan(
            run_dir=run_dir,
            release_path=release_path,
            release_sha256=runner._sha(release_bytes),
            tokenizer=ByteTokenizer(),
            runtime={"runtime": "test-only"},
            ledger_path_override=ledger_path,
            client_factory=fake_factory,
        )
    assert len(calls) == 1
    assert persisted_before_settlement == calls
    assert settled_ids == calls
    assert runner._read_event_log(run_dir, plan) == ([], {})

    # Restart recovers the settled first response from its durable receipt and
    # continues with only the two outstanding roles.
    first = runner.execute_plan(
        run_dir=run_dir,
        release_path=release_path,
        release_sha256=runner._sha(release_bytes),
        tokenizer=ByteTokenizer(),
        runtime={"runtime": "test-only"},
        ledger_path_override=ledger_path,
        client_factory=fake_factory,
    )
    assert first["phase_status"] == "complete"
    assert first["provider_calls_made"] == 3
    assert first["cases"][0]["role_evidence_accepted"] is True
    assert first["cases"][0]["functional_status"] == "not_evaluated_by_role_runner"
    assert first["training_started"] is False
    assert len(calls) == 3
    assert persisted_before_settlement == calls
    assert settled_ids == calls
    events, by_id = runner._read_event_log(run_dir, plan)
    assert len(events) == len(by_id) == 3
    assert events[0]["recovered_from_persisted_receipt"] is True
    settled = TeacherUsageLedger(ledger_path).totals()
    assert settled.calls == 3
    assert settled.input_tokens == 30

    # A process restart can rebuild its result from durable receipts/events;
    # completed requests are never sent a second time.
    resumed = runner.execute_plan(
        run_dir=run_dir,
        release_path=release_path,
        release_sha256=runner._sha(release_bytes),
        tokenizer=ByteTokenizer(),
        runtime={"runtime": "test-only"},
        ledger_path_override=ledger_path,
        client_factory=lambda *_args: pytest.fail("resume must not construct a provider"),
    )
    assert resumed["phase_status"] == "complete"
    assert resumed["provider_calls_made"] == 3
    assert len(calls) == 3
