from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_instinct_one_line_pilot import (
    _extract_excerpt,
    _safe_relative_ts_path,
    _secret_like,
    convert_messages,
    split_for_path,
)

from tinycomplete.one_line.contract import EditAction, EditState, apply_action


def _messages(
    *,
    source: str,
    target: str,
    old: str = "  const value = 0;",
    new: str = "  const value = 1;",
    path: str = "src/example.ts",
) -> list[dict[str, str]]:
    region = target.replace("|", "<|user_cursor_is_here|>")
    marked_source = source.replace(
        "<TARGET>",
        "<|editable_region_start|>\n" + region + "<|editable_region_end|>\n",
    )
    user = (
        "### User Edits:\n"
        f'User edited file "{path}"\n\n'
        "```diff\n@@ -1 +1 @@\n"
        f"-{old}\n+{new}\n```\n"
        "### User Excerpt:\n\n"
        f'"{path}"\n\n'
        f"{marked_source}"
    )
    return [
        {"role": "system", "content": "synthetic test"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ""},
    ]


def _example(
    *,
    before: str = "function f() {\n",
    target: str = "  const value = 1;|\n",
    after: str = "  return value;\n}\n",
    old: str = "  const value = 0;",
    new: str = "  const value = 1;",
    answer: str = "  const value = 2;\n",
) -> tuple[list[dict[str, str]], str]:
    source = before + "<TARGET>" + after
    messages = _messages(source=source, target=target, old=old, new=new)
    messages[2]["content"] = answer
    return messages, before + target.replace("|", "") + after


def _convert(messages: list[dict[str, str]], tokenizer: Any | None = None):
    row, outcome, token_counts = convert_messages(messages, 7, tokenizer)
    return row, outcome, token_counts


def test_replacement_replays_exactly_and_keeps_history_unreviewed() -> None:
    messages, _ = _example()
    row, outcome, _ = _convert(messages)

    assert outcome == "accepted"
    assert row is not None
    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    assert action == EditAction("replace_line", "  const value = 2;")
    assert apply_action(state, action) == row["after_source"]
    assert row["validation"]["replay_verified"] is True
    assert row["validation"]["inferability_reviewed"] is False
    assert row["source_type"] == "continue_instinct_observed"
    assert row["source_license"] == "Apache-2.0"
    assert state.history[0].new_text == "  const value = 1;"


def test_marker_offset_updates_second_duplicate_line_not_first() -> None:
    before = "  const value = 1;\nfunction f() {\n"
    messages, _ = _example(
        before=before,
        old="function g() {",
        new="function f() {",
    )
    row, outcome, _ = _convert(messages)

    assert outcome == "accepted"
    assert row is not None
    state = EditState.from_mapping(row["state"])
    assert state.source.count("  const value = 1;") == 2
    assert state.target_row == 2
    assert row["after_source"].splitlines()[0] == "  const value = 1;"
    assert row["after_source"].splitlines()[2] == "  const value = 2;"


def test_insertion_and_deletion_are_exact_single_line_actions() -> None:
    insertion_messages, _ = _example(
        target="  return value;|\n",
        old="  return oldValue;",
        new="  return value;",
        after="}\n",
        answer="  if (value == null) return;\n  return value;\n",
    )
    inserted, outcome, _ = _convert(insertion_messages)
    assert outcome == "accepted"
    assert inserted is not None
    assert inserted["action"] == {
        "kind": "insert_before",
        "text": "  if (value == null) return;",
    }
    assert (
        apply_action(EditState.from_mapping(inserted["state"]), EditAction(**inserted["action"]))
        == inserted["after_source"]
    )

    deletion_messages, _ = _example(
        target="  return value;|\n",
        old="  return oldValue;",
        new="  return value;",
        after="}\n",
        answer="",
    )
    deleted, outcome, _ = _convert(deletion_messages)
    assert outcome == "accepted"
    assert deleted is not None
    assert deleted["action"] == {"kind": "delete_line", "text": None}
    assert (
        apply_action(EditState.from_mapping(deleted["state"]), EditAction(**deleted["action"]))
        == deleted["after_source"]
    )


def test_utf8_cursor_column_is_a_byte_offset() -> None:
    before = "function f() {\n"
    messages, _ = _example(
        before=before,
        target='  const label = "é";|\n',
        old='  const label = "x";',
        new='  const label = "é";',
        answer='  const label = "🚀";\n',
    )
    row, outcome, _ = _convert(messages)
    assert outcome == "accepted"
    assert row is not None
    state = EditState.from_mapping(row["state"])
    line = state.source.splitlines()[state.target_row]
    expected_prefix = '  const label = "é";'
    assert line == expected_prefix
    assert state.cursor_col == len(expected_prefix.encode())
    assert state.cursor_col > len(expected_prefix)


@pytest.mark.parametrize("path", ["../evil.ts", "/tmp/x.ts", "src/x.py", "src\\x.ts", "./x.ts"])
def test_only_safe_relative_typescript_paths(path: str) -> None:
    assert not _safe_relative_ts_path(path)


def test_excerpt_parser_preserves_an_intentional_leading_blank_line() -> None:
    excerpt = _extract_excerpt('### User Excerpt:\n\n"src/a.ts"\n\n\nconst x = 1;')
    assert excerpt == ("src/a.ts", "\nconst x = 1;")


def test_secret_filter_rejects_obvious_credentials_without_returning_them() -> None:
    assert _secret_like(["const token = 'ghp_" + "a" * 36 + "';"])
    messages, _ = _example()
    messages[2]["content"] = "const privateKey = 'sk-" + "b" * 32 + "';\n"
    row, outcome, _ = _convert(messages)
    assert row is None
    assert outcome == "secret_like_material"


class _CharTokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) for char in text] + ([1] if add_special_tokens else [])


def test_context_that_loses_all_history_under_token_budget_is_rejected() -> None:
    messages, _ = _example(old="x" * 1500)
    row, outcome, token_counts = _convert(messages, _CharTokenizer())
    assert row is None
    assert outcome == "token_budget_dropped_history"
    assert token_counts == {}


def test_split_is_path_deterministic_and_has_only_two_values() -> None:
    assert split_for_path("src/example.ts") == split_for_path("src/example.ts")
    assert split_for_path("src/example.ts") in {"train", "development"}
