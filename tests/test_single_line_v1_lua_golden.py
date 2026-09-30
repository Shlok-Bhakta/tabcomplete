from __future__ import annotations

import json
from pathlib import Path

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION, serialize_state_bounded
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    apply_action,
    decode_action,
    encode_action,
)

FIXTURE = Path(__file__).parent / "fixtures" / "single_line_edit_v1_golden.json"


class Utf8ByteBudgetTokenizer:
    """Predictable tokenizer stand-in for testing greedy item selection parity."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.encode("utf-8")) + int(add_special_tokens)))


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_golden_context_prompt_matches_python_reference() -> None:
    fixture = _load()
    assert fixture["context_policy_version"] == CONTEXT_POLICY_VERSION
    for case in fixture["contexts"]:
        state = EditState.from_mapping(case["state"])
        actual = serialize_state_bounded(state, tokenizer=None)
        assert actual.text == case["expected_prompt"], case["name"]
        assert list(actual.included_rows) == case["expected_rows"], case["name"]


def test_golden_bounded_context_selection_uses_exact_order() -> None:
    fixture = _load()
    tokenizer = Utf8ByteBudgetTokenizer()
    for case in fixture["bounded_contexts"]:
        state = EditState.from_mapping(case["state"])
        actual = serialize_state_bounded(
            state,
            tokenizer=tokenizer,
            max_input_tokens=case["max_input_tokens"],
        )
        assert actual.text == case["expected_prompt"], case["name"]
        assert list(actual.included_rows) == case["expected_rows"], case["name"]
        assert actual.input_tokens == case["expected_input_tokens"], case["name"]
        assert actual.included_history == case["expected_history_count"], case["name"]
        assert actual.included_relevant == case["expected_relevant_count"], case["name"]


def test_golden_action_wire_and_byte_application_match_python_reference() -> None:
    fixture = _load()
    for case in fixture["actions"]:
        state = EditState.from_mapping(case["state"])
        action = EditAction(**case["action"])
        assert encode_action(action) == case["wire"], case["name"]
        assert apply_action(state, action) == case["after_source"], case["name"]


def test_golden_decoder_requires_eos_exact_header_and_the_64_token_limit() -> None:
    fixture = _load()
    for case in fixture["decode_cases"]:
        decoded = decode_action(
            case["wire"],
            terminated=case["stop_type"] == "eos",
            generated_tokens=case["generated_tokens"],
        )
        assert decoded.status == case["status"], case["name"]
        expected = case["action"]
        if expected is None:
            assert decoded.action is None, case["name"]
        else:
            assert decoded.action == EditAction(**expected), case["name"]
