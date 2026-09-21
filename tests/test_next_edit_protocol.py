from __future__ import annotations

from tinycomplete.eval.next_edit_protocol import (
    NextEditAction,
    apply_next_edit_action,
    parse_next_edit_action,
    serialize_next_edit_action,
)


def test_no_edit_is_distinct_from_empty_replacement() -> None:
    source = "value = broken\n"
    start = len(b"value = ")
    end = start + len(b"broken")

    no_edit = parse_next_edit_action('{"action":"no_edit"}')
    deletion = parse_next_edit_action('{"action":"replace","text":""}')

    assert no_edit.status == "ok"
    assert deletion.status == "ok"
    assert apply_next_edit_action(source, start, end, no_edit.action) == source
    assert apply_next_edit_action(source, start, end, deletion.action) == "value = \n"


def test_replace_supports_insert_and_preserves_whitespace() -> None:
    insertion = NextEditAction(action="replace", text="  x\n")
    assert apply_next_edit_action("ab", 1, 1, insertion) == "a  x\nb"

    whitespace = parse_next_edit_action('{"action":"replace","text":"  \\n"}')
    assert whitespace.status == "ok"
    assert whitespace.action.text == "  \n"
    assert apply_next_edit_action("aXXb", 1, 3, whitespace.action) == "a  \nb"


def test_unicode_regions_use_byte_offsets() -> None:
    source = "héllo 🌍"
    start = len("héllo ".encode())
    end = len(source.encode("utf-8"))
    action = NextEditAction(action="replace", text="world")

    assert apply_next_edit_action(source, start, end, action) == "héllo world"


def test_parser_distinguishes_malformed_truncated_and_overgeneration() -> None:
    malformed = parse_next_edit_action("not json", finish_reason="stop", max_tokens=96)
    truncated = parse_next_edit_action(
        '{"action":"replace","text":"unfinished',
        finish_reason="length",
        generated_tokens=96,
        max_tokens=96,
    )
    overgenerated = parse_next_edit_action(
        '{"action":"no_edit"} trailing', finish_reason="length", max_tokens=96
    )

    assert malformed.status == "malformed"
    assert truncated.status == "truncated"
    assert truncated.hit_token_cap is True
    assert overgenerated.status == "overgeneration"


def test_serializer_is_versioned_and_does_not_trim_text() -> None:
    assert serialize_next_edit_action(NextEditAction(action="no_edit")) == (
        '{"action":"no_edit"}'
    )
    assert serialize_next_edit_action(NextEditAction(action="replace", text=" \n")) == (
        '{"action":"replace","text":" \\n"}'
    )


def test_parser_rejects_missing_or_extra_action_fields() -> None:
    assert parse_next_edit_action('{"action":"replace"}').status == "malformed"
    assert parse_next_edit_action('{"action":"no_edit","text":""}').status == "malformed"
    assert parse_next_edit_action('{"action":"unknown"}').status == "malformed"
