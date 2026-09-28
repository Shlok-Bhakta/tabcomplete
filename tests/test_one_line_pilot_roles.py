"""The three Muse pilot roles must receive only their declared information."""

from __future__ import annotations

import hashlib
import json

import pytest

from tinycomplete.one_line.pilot_roles import (
    AUTHOR_SCHEMA,
    REVIEW_SCHEMA,
    build_author_prompt,
    build_author_prompt_v2,
    build_blind_solver_prompt,
    build_reviewer_prompt,
    parse_review_response,
)


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(text.encode())


def source_row() -> dict:
    source = "def primary(x):\n    return x.value\n\ndef secondary(x):\n    return x.value\n"
    return {
        "id": "synthetic-seed",
        "student_state_seed": {
            "file_id": "lib/example.py",
            "filetype": "python",
            "source": source,
        },
        "authoring_metadata": {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_license": "MIT",
            "source_provenance_verified": True,
            "authoring_focus": "sibling change",
            "private_note": "never leave author metadata",
        },
    }


def candidate() -> dict:
    return {
        "state": {
            "file_id": "lib/example.py",
            "filetype": "python",
            "source": (
                "def primary(x):\n    return x.value_id\n\ndef secondary(x):\n    return x.value\n"
            ),
            "target_row": 4,
            "cursor_col": 0,
            "history": [
                {"row": 1, "old_text": "    return x.value", "new_text": "    return x.value_id"}
            ],
            "relevant": [],
        },
        "action": {"kind": "replace_line", "text": "    return x.value_id"},
        "after_source": "hidden future answer",
        "provenance": {
            "requested_focus": "private author hint",
            "objective": {"kind": "history_consistency", "description": "update sibling"},
        },
    }


def test_author_prompt_contains_only_declared_seed() -> None:
    prompt = build_author_prompt(source_row())
    assert "sibling change" in prompt
    assert "private_note" not in prompt
    assert "never leave author metadata" not in prompt
    assert "return x.value" in prompt
    assert set(AUTHOR_SCHEMA["required"]) == {
        "prior_edit",
        "target_row",
        "action",
        "intent_evidence",
        "objective",
    }


def test_author_prompt_rejects_changed_or_unverified_source() -> None:
    row = source_row()
    row["student_state_seed"]["source"] += "# changed\n"
    with pytest.raises(ValueError, match="identity"):
        build_author_prompt(row)
    row = source_row()
    row["authoring_metadata"]["source_provenance_verified"] = False
    with pytest.raises(ValueError, match="provenance"):
        build_author_prompt(row)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "α\r\n\r\nβ\nlast",
            [(0, "α", "CRLF"), (1, "", "CRLF"), (2, "β", "LF"), (3, "last", "none")],
        ),
        ("one\n", [(0, "one", "LF")]),
        ("", []),
    ],
)
def test_author_prompt_v2_numbers_contract_lines_and_eof(source: str, expected: list) -> None:
    row = source_row()
    row["student_state_seed"]["source"] = source
    row["authoring_metadata"]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    prompt = build_author_prompt_v2(row)
    instruction, payload = prompt.split("\n", 1)
    data = json.loads(payload)
    assert "zero-based" in instruction
    assert "source" not in data
    assert data["row_index_base"] == 0
    assert data["physical_line_count"] == len(expected)
    assert data["eof_insert_row"] == len(expected)
    lines = data["source_lines_zero_based"]
    assert [(x["row"], x["text"], x["terminator"]) for x in lines] == expected
    terminators = {"LF": "\n", "CRLF": "\r\n", "none": ""}
    rebuilt = "".join(x["text"] + terminators[x["terminator"]] for x in lines)
    assert rebuilt == source


def test_author_prompt_v2_rejects_lone_cr_and_changed_source() -> None:
    row = source_row()
    row["student_state_seed"]["source"] = "x\ry"
    row["authoring_metadata"]["source_sha256"] = hashlib.sha256(b"x\ry").hexdigest()
    with pytest.raises(ValueError, match="lone CR"):
        build_author_prompt_v2(row)
    row["student_state_seed"]["source"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        build_author_prompt_v2(row)


def test_solver_prompt_has_no_author_answer_or_focus() -> None:
    prompt = build_blind_solver_prompt(candidate(), CharacterTokenizer())
    assert "return x.value_id" in prompt
    assert "private author hint" not in prompt
    assert "hidden future answer" not in prompt
    assert "history_consistency" not in prompt
    assert "update sibling" not in prompt


def test_reviewer_prompt_has_only_declared_public_case() -> None:
    prompt = build_reviewer_prompt(candidate(), "N")
    assert "R\\t    return x.value_id" in prompt
    assert "history_consistency" in prompt
    assert "hidden future answer" not in prompt
    assert "private author hint" not in prompt
    assert set(REVIEW_SCHEMA["required"]) == {"retain", "ambiguous", "reason"}


def test_reviewer_parser_rejects_prose_duplicates_and_extra_fields() -> None:
    result = parse_review_response(
        json.dumps({"retain": True, "ambiguous": False, "reason": "visible edit"})
    )
    assert result.retain and not result.ambiguous
    for text in (
        'prose {"retain":true,"ambiguous":false,"reason":"x"}',
        '{"retain":true,"retain":false,"ambiguous":false,"reason":"x"}',
        '{"retain":true,"ambiguous":false,"reason":"x","gold":"N"}',
        '{"retain":1,"ambiguous":false,"reason":"x"}',
    ):
        with pytest.raises(ValueError):
            parse_review_response(text)
