from __future__ import annotations

import pytest

from tinycomplete.one_line.context import (
    CONTEXT_POLICY_VERSION,
    serialize_state,
    serialize_state_bounded,
)
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    RecentEdit,
    apply_action,
    decode_action,
    encode_action,
    physical_lines,
)


def state(source: str, row: int = 0, col: int = 0) -> EditState:
    return EditState("public/example.py", "python", source, row, col)


@pytest.mark.parametrize(
    ("source", "row", "action", "expected"),
    [
        ("", 0, EditAction("keep"), ""),
        ("", 0, EditAction("insert_before", "x"), "x"),
        ("", 0, EditAction("insert_before", ""), "\n"),
        ("a", 0, EditAction("replace_line", "b"), "b"),
        ("a", 0, EditAction("replace_line", ""), "\n"),
        ("a\n", 0, EditAction("replace_line", ""), "\n"),
        ("a\n", 0, EditAction("insert_before", ""), "\na\n"),
        ("a\n", 1, EditAction("insert_before", ""), "a\n\n"),
        ("a", 1, EditAction("insert_before", ""), "a\n\n"),
        ("a\nb", 1, EditAction("delete_line"), "a\n"),
        ("foo\n\nbar", 2, EditAction("delete_line"), "foo\n\n"),
        ("a\nb\n", 1, EditAction("delete_line"), "a\n"),
        ("a\r\nb\r\n", 0, EditAction("delete_line"), "b\r\n"),
        ("a\r\nb", 1, EditAction("replace_line", "c"), "a\r\nc"),
        ("a\r\nb", 1, EditAction("insert_before", "c"), "a\r\nc\r\nb"),
        ("a\r\nb", 2, EditAction("insert_before", "c"), "a\r\nb\r\nc"),
        ("α\nβ", 1, EditAction("replace_line", "λ\tμ"), "α\nλ\tμ"),
    ],
)
def test_byte_exact_application(source: str, row: int, action: EditAction, expected: str) -> None:
    assert apply_action(state(source, row), action).encode() == expected.encode()


def test_line_model_excludes_phantom_line_and_preserves_mixed_terminators() -> None:
    assert physical_lines(b"") == ()
    assert len(physical_lines(b"a\n")) == 1
    assert [line.terminator for line in physical_lines(b"a\r\nb\n")] == [b"\r\n", b"\n"]
    assert apply_action(state("a\r\nb\n", 1), EditAction("replace_line", "c")) == "a\r\nc\n"
    with pytest.raises(ValueError, match="lone CR"):
        state("a\rb")


def test_invalid_rows_and_utf8_cursor_boundaries() -> None:
    with pytest.raises(ValueError, match="splits a UTF-8"):
        state("é\n", col=1)
    with pytest.raises(ValueError, match="outside"):
        state("abc", row=2)
    with pytest.raises(ValueError, match="EOF"):
        state("abc", row=1, col=1)
    with pytest.raises(ValueError, match="existing"):
        apply_action(state("", 0), EditAction("delete_line"))
    with pytest.raises(ValueError, match="existing"):
        apply_action(state("a", 1), EditAction("replace_line", "b"))


@pytest.mark.parametrize(
    ("action", "wire"),
    [
        (EditAction("keep"), "N"),
        (EditAction("delete_line"), "D"),
        (EditAction("replace_line", ""), "R\t"),
        (EditAction("replace_line", " \tλ"), "R\t \tλ"),
        (EditAction("insert_before", ""), "I\t"),
        (EditAction("insert_before", "    x = 1"), "I\t    x = 1"),
    ],
)
def test_wire_round_trip(action: EditAction, wire: str) -> None:
    assert encode_action(action) == wire
    decoded = decode_action(wire, terminated=True, generated_tokens=4)
    assert decoded.status == "ok" and decoded.action == action


@pytest.mark.parametrize("wire", ["", "n", "R", "R x", "N\n", "I\tx\ny", "D\t", "R\tx\r"])
def test_malformed_wire(wire: str) -> None:
    assert decode_action(wire, terminated=True).status == "malformed"


def test_unterminated_and_capped_wire_is_never_an_action() -> None:
    assert decode_action("N", terminated=False).status == "incomplete"
    assert decode_action("R\tfoo", terminated=False, generated_tokens=64).action is None
    assert decode_action("N", terminated=True, generated_tokens=65).status == "incomplete"


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return ([1] if add_special_tokens else []) + [ord(character) + 2 for character in text]


def test_context_has_target_history_and_no_label_metadata() -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(80))
    edit = EditState(
        "public/example.py",
        "python",
        source,
        40,
        0,
        history=(RecentEdit(40, "old", "new"),),
        relevant=("import math",),
    )
    context = serialize_state_bounded(edit, CharacterTokenizer(), max_input_tokens=800)
    assert context.input_tokens is not None and context.input_tokens <= 800
    assert 40 in context.included_rows
    assert "old=" in context.text and "new=" in context.text
    assert "Action:" in context.text
    assert "R\t, I\t" in context.text
    assert "<TAB>" not in context.text
    assert CONTEXT_POLICY_VERSION == "single-line-context-v2"
    assert "source_license" not in context.text
    assert "gold_action" not in context.text
    assert serialize_state(edit, CharacterTokenizer(), max_input_tokens=800) == context.text


def test_context_rejects_oversize_mandatory_line_without_truncation() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        serialize_state(state("x" * 1500), CharacterTokenizer(), max_input_tokens=1024)
