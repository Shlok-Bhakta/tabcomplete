from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from typing import Any

import pytest

from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    RecentEdit,
    apply_action,
    encode_action,
)
from tinycomplete.one_line.fixed_state_roles import (
    HISTORY_ORDER,
    ROLE_PROTOCOL_SCHEMA,
    SOURCE_TYPE,
    CompletedFixedStateRole,
    build_fixed_state_prompt,
    build_fixed_state_review_prompt,
    build_fixed_state_role_evidence,
    fixed_state_protocol_bindings,
    verify_fixed_state_role_evidence,
)
from tinycomplete.one_line.teacher import MODEL_ID, TeacherResponse


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8")) + ([self.eos_token_id] if add_special_tokens else [])


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha(payload.encode("utf-8"))


def _candidate() -> dict[str, Any]:
    state = EditState(
        file_id="public_sample.py",
        filetype="python",
        source="def answer(value):\n    return value\n",
        target_row=1,
        cursor_col=len(b"    return value"),
        history=(RecentEdit(1, "    ", "    return value"),),
        relevant=(),
    )
    action = EditAction("replace_line", "    return value + 1")
    row: dict[str, Any] = {
        "id": "public-repo/candidate-001",
        "candidate_id": "public-repo/candidate-001",
        "source_type": SOURCE_TYPE,
        "history_order": HISTORY_ORDER,
        "human_chronology_observed": False,
        "context_policy": "single-line-context-v2",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": apply_action(state, action),
        "split": "train",
        "source_repo": "example/repo",
        "source_revision": "a" * 40,
        "source_path": "src/example.py",
        "source_license": "MIT",
        "source_license_sha256": "b" * 64,
        "source_group_id": "example/repo@parent",
        "session_or_commit": "parent-revision-001",
        "task_family_id": "public-prefix-function",
        "template_id": "return-expression-001",
        "human_edit_order_observed": False,
    }
    bindings = fixed_state_protocol_bindings(
        {
            **row,
            "context_sha256": build_fixed_state_prompt(state, ByteTokenizer()).context_sha256,
            "history_sha256": _canonical_sha([asdict(edit) for edit in state.history]),
            "prompt": build_fixed_state_prompt(state, ByteTokenizer()).user_prompt,
        },
        ByteTokenizer(),
    )
    row.update(
        context_sha256=bindings["context_sha256"],
        history_sha256=bindings["history_sha256"],
        prompt=build_fixed_state_prompt(state, ByteTokenizer()).user_prompt,
    )
    return row


def _role_record(candidate: dict[str, Any]) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    state = EditState.from_mapping(candidate["state"])
    student = build_fixed_state_prompt(state, tokenizer)
    action_wire = encode_action(EditAction(**candidate["action"]))
    review = build_fixed_state_review_prompt(state, action_wire, action_wire, tokenizer)
    verdict = json.dumps(
        {"retain": True, "ambiguous": False, "reason": "Both actions fit the visible code state."},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    common = {
        "model_id": MODEL_ID,
        "finish_reason": "stop",
        "provider_input_tokens": 130,
        "provider_output_tokens": 22,
        "provider_reasoning_tokens": 10,
    }
    author = {
        **common,
        "actor_id": "author-role-001",
        "session_id": "session-author-001",
        "request_id": "request-author-001",
        "response_id": "response-author-001",
        "prompt_sha256": student.prompt_sha256,
        "response_sha256": _sha(action_wire.encode("utf-8")),
        "wire": action_wire,
        "wire_sha256": _sha(action_wire.encode("utf-8")),
        "q25_supervised_tokens_including_eos": len(
            tokenizer.encode(action_wire, add_special_tokens=False)
        )
        + 1,
    }
    solver = {
        **common,
        "actor_id": "solver-role-002",
        "session_id": "session-solver-002",
        "request_id": "request-solver-002",
        "response_id": "response-solver-002",
        "prompt_sha256": student.prompt_sha256,
        "response_sha256": _sha(action_wire.encode("utf-8")),
        "wire": action_wire,
        "wire_sha256": _sha(action_wire.encode("utf-8")),
        "q25_supervised_tokens_including_eos": len(
            tokenizer.encode(action_wire, add_special_tokens=False)
        )
        + 1,
    }
    reviewer = {
        **common,
        "actor_id": "review-role-003",
        "session_id": "session-review-003",
        "request_id": "request-review-003",
        "response_id": "response-review-003",
        "prompt_sha256": review.prompt_sha256,
        "response_sha256": _sha(verdict.encode("utf-8")),
        "verdict_json": verdict,
        "verdict_sha256": _sha(verdict.encode("utf-8")),
        "q25_supervised_tokens_including_eos": len(
            tokenizer.encode(verdict, add_special_tokens=False)
        )
        + 1,
    }
    return {
        "schema": ROLE_PROTOCOL_SCHEMA,
        "candidate_id": candidate["id"],
        "protocol": fixed_state_protocol_bindings(candidate, tokenizer),
        "author": author,
        "solver": solver,
        "reviewer": reviewer,
    }


def _bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_author_and_blind_solver_get_same_action_free_fixed_prompt() -> None:
    tokenizer = ByteTokenizer()
    candidate = _candidate()
    state = candidate["state"]
    input_with_private_fields = {
        **state,
        "action": {"kind": "replace_line", "text": "GOLD MUST NOT APPEAR"},
        "tests": ["PRIVATE TEST MUST NOT APPEAR"],
        "oracle": {"answer": "DO NOT INCLUDE"},
    }
    author = build_fixed_state_prompt(input_with_private_fields, tokenizer)
    solver = build_fixed_state_prompt(state, tokenizer)
    assert author == solver
    assert "GOLD MUST NOT APPEAR" not in author.user_prompt
    assert "PRIVATE TEST MUST NOT APPEAR" not in author.user_prompt
    assert "DO NOT INCLUDE" not in author.user_prompt
    assert author.history_sha256 == candidate["history_sha256"]
    assert author.user_prompt == candidate["prompt"]


def test_reviewer_prompt_binds_both_actions_without_objective_or_tests() -> None:
    candidate = _candidate()
    tokenizer = ByteTokenizer()
    state = EditState.from_mapping(candidate["state"])
    wire = encode_action(EditAction(**candidate["action"]))
    reviewer = build_fixed_state_review_prompt(state, wire, wire, tokenizer)
    assert json.dumps(wire) in reviewer.user_prompt
    assert "oracle" not in reviewer.user_prompt.lower()
    assert "test" not in reviewer.user_prompt.lower()
    assert "objective" not in reviewer.user_prompt.lower()


def test_fixed_state_evidence_accepts_exact_roles_and_reports_no_functional_claim() -> None:
    candidate = _candidate()
    payload = _bytes(_role_record(candidate))
    decision = verify_fixed_state_role_evidence(candidate, payload, tokenizer=ByteTokenizer())
    assert decision.accepted
    assert decision.reason == "fixed_state_role_evidence_verified"
    assert decision.evidence["functional_status"] == "not_evaluated_by_role_verifier"
    assert decision.evidence["human_chronology_observed"] is False


def test_fixed_state_rejects_role_action_disagreement_even_with_valid_prompt_binding() -> None:
    candidate = _candidate()
    record = _role_record(candidate)
    other = "N"
    record["solver"]["wire"] = other
    record["solver"]["wire_sha256"] = _sha(other.encode())
    record["solver"]["response_sha256"] = _sha(other.encode())
    record["solver"]["q25_supervised_tokens_including_eos"] = len(other) + 1
    review = build_fixed_state_review_prompt(
        EditState.from_mapping(candidate["state"]),
        record["author"]["wire"],
        other,
        ByteTokenizer(),
    )
    record["reviewer"]["prompt_sha256"] = review.prompt_sha256
    decision = verify_fixed_state_role_evidence(
        candidate, _bytes(record), tokenizer=ByteTokenizer()
    )
    assert not decision.accepted
    assert decision.reason == "fixed_state_actions_disagree_with_row"


def test_fixed_state_rejects_state_prompt_history_and_chronology_tampering() -> None:
    candidate = _candidate()
    payload = _bytes(_role_record(candidate))
    changed = copy.deepcopy(candidate)
    changed["state"]["source"] += "# changed\n"
    assert not verify_fixed_state_role_evidence(
        changed, payload, tokenizer=ByteTokenizer()
    ).accepted
    changed = copy.deepcopy(candidate)
    changed["history_order"] = "observed"
    assert not verify_fixed_state_role_evidence(
        changed, payload, tokenizer=ByteTokenizer()
    ).accepted
    changed = copy.deepcopy(candidate)
    changed["human_chronology_observed"] = True
    assert not verify_fixed_state_role_evidence(
        changed, payload, tokenizer=ByteTokenizer()
    ).accepted


def test_fixed_state_rejects_incomplete_capped_and_duplicate_role_receipts() -> None:
    candidate = _candidate()
    record = _role_record(candidate)
    record["solver"]["finish_reason"] = "length"
    assert not verify_fixed_state_role_evidence(
        candidate, _bytes(record), tokenizer=ByteTokenizer()
    ).accepted


def test_response_assembler_emits_verifiable_receipts_from_completed_responses() -> None:
    candidate = _candidate()
    tokenizer = ByteTokenizer()
    state = EditState.from_mapping(candidate["state"])
    wire = encode_action(EditAction(**candidate["action"]))
    reviewer = build_fixed_state_review_prompt(state, wire, wire, tokenizer)
    verdict = json.dumps(
        {"retain": True, "ambiguous": False, "reason": "Both actions fit the visible code."},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    student = build_fixed_state_prompt(state, tokenizer)

    def call(role: str, request_id: str, prompt_hash: str, content: str) -> CompletedFixedStateRole:
        response = TeacherResponse(
            content=content,
            session_id=f"session-{role}-0001",
            response_id=f"response-{role}-0001",
            model_id=MODEL_ID,
            input_tokens=120,
            output_tokens=12,
            reasoning_tokens=4,
            total_tokens_reported=136,
            cached_read_tokens=0,
            cached_write_tokens=0,
            finish_reason="stop",
            cost_usd_reported=None,
        )
        return CompletedFixedStateRole(
            actor_id=f"actor-{role}-0001",
            request_id=request_id,
            prompt_sha256=prompt_hash,
            response=response,
        )

    payload = build_fixed_state_role_evidence(
        candidate,
        author=call("author", "request-author-0001", student.prompt_sha256, wire),
        solver=call("solver", "request-solver-0001", student.prompt_sha256, wire),
        reviewer=call("reviewer", "request-reviewer-0001", reviewer.prompt_sha256, verdict),
        tokenizer=tokenizer,
    )
    decision = verify_fixed_state_role_evidence(candidate, payload, tokenizer=tokenizer)
    assert decision.accepted
    assert decision.evidence["functional_status"] == "not_evaluated_by_role_verifier"


def test_response_assembler_rejects_non_stop_action_without_repair() -> None:
    candidate = _candidate()
    tokenizer = ByteTokenizer()
    state = EditState.from_mapping(candidate["state"])
    wire = encode_action(EditAction(**candidate["action"]))
    review = build_fixed_state_review_prompt(state, wire, wire, tokenizer)
    student = build_fixed_state_prompt(state, tokenizer)

    def call(
        role: str,
        request_id: str,
        prompt_hash: str,
        content: str,
        finish: str,
    ) -> CompletedFixedStateRole:
        response = TeacherResponse(
            content=content,
            session_id=f"session-{role}-0002",
            response_id=f"response-{role}-0002",
            model_id=MODEL_ID,
            input_tokens=120,
            output_tokens=12,
            reasoning_tokens=4,
            total_tokens_reported=136,
            cached_read_tokens=0,
            cached_write_tokens=0,
            finish_reason=finish,
            cost_usd_reported=None,
        )
        return CompletedFixedStateRole(
            actor_id=f"actor-{role}-0002",
            request_id=request_id,
            prompt_sha256=prompt_hash,
            response=response,
        )

    with pytest.raises(ValueError, match="did not stop normally"):
        build_fixed_state_role_evidence(
            candidate,
            author=call("author", "request-author-0002", student.prompt_sha256, wire, "length"),
            solver=call("solver", "request-solver-0002", student.prompt_sha256, wire, "stop"),
            reviewer=call(
                "reviewer",
                "request-reviewer-0002",
                review.prompt_sha256,
                '{"retain":true,"ambiguous":false,"reason":"looks good"}',
                "stop",
            ),
            tokenizer=tokenizer,
        )

    record = _role_record(candidate)
    long_wire = "R\t" + ("x" * 70)
    record["solver"]["wire"] = long_wire
    record["solver"]["wire_sha256"] = _sha(long_wire.encode())
    record["solver"]["response_sha256"] = _sha(long_wire.encode())
    record["solver"]["q25_supervised_tokens_including_eos"] = len(long_wire) + 1
    assert not verify_fixed_state_role_evidence(
        candidate, _bytes(record), tokenizer=ByteTokenizer()
    ).accepted

    record = _role_record(candidate)
    record["reviewer"]["request_id"] = record["solver"]["request_id"]
    assert not verify_fixed_state_role_evidence(
        candidate, _bytes(record), tokenizer=ByteTokenizer()
    ).accepted


def test_fixed_state_protocol_requires_full_history_and_truthful_new_source_type() -> None:
    candidate = _candidate()
    candidate["source_type"] = "muse_author_public_candidate"
    try:
        fixed_state_protocol_bindings(candidate, ByteTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("the new protocol must reject legacy source types")

    state = EditState(
        file_id="empty.py",
        filetype="python",
        source="pass\n",
        target_row=0,
        cursor_col=0,
        history=(),
    )
    try:
        build_fixed_state_prompt(state, ByteTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("fixed public-history protocol must not accept missing history")
