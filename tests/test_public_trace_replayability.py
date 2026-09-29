"""The replayability screen never equates a one-line call with an accepted edit."""

import runpy
from pathlib import Path

screen_messages = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/probe_public_trace_replayability.py")
)["screen_messages"]


def _call(name: str, arguments: dict[str, str]) -> dict:
    return {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def test_prior_editor_same_file_is_only_optimistically_replayable() -> None:
    messages = [
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "a.go", "old_str": "x", "new_str": "y"},
        ),
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "a.go", "old_str": "a", "new_str": "b"},
        ),
    ]
    counts = screen_messages(messages)
    assert counts["one_line_calls"] == 2
    assert counts["optimistic_shell_free_one_line_sequences"] == 1


def test_prior_shell_call_blocks_exact_replay_even_when_editor_calls_follow() -> None:
    messages = [
        _call("bash", {"command": "gofmt -w a.go"}),
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "a.go", "old_str": "x", "new_str": "y"},
        ),
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "a.go", "old_str": "a", "new_str": "b"},
        ),
    ]
    counts = screen_messages(messages)
    assert counts["optimistic_shell_free_one_line_sequences"] == 0
    assert counts["one_line_with_prior_editor_and_bash"] == 1


def test_different_file_does_not_supply_same_file_history() -> None:
    messages = [
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "a.go", "old_str": "x", "new_str": "y"},
        ),
        _call(
            "str_replace_editor",
            {"command": "str_replace", "path": "b.go", "old_str": "a", "new_str": "b"},
        ),
    ]
    counts = screen_messages(messages)
    assert counts["one_line_with_prior_editor_same_file"] == 0
