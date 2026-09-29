"""Check exact mapping without treating a Zeta row as accepted source evidence."""

import runpy
from pathlib import Path

CLASSIFY = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/probe_zeta_one_line_mapping.py")
)["classify"]


def _row(before: str, after: str) -> dict[str, str]:
    return {
        "input": "```demo.py\nx = 1\n<|editable_region_start|><|user_cursor_is_here|>"
        + before
        + "<|editable_region_end|>\n```",
        "output": "```demo.py\nx = 1\n<|editable_region_start|>"
        + after
        + "<|editable_region_end|>\n```",
        "events": "User edited demo.py",
    }


def test_one_line_replacement_replays_exactly() -> None:
    status, detail = CLASSIFY(_row("y = 1\n", "y = 2\n"))
    assert status == "one_line_exact"
    assert detail["action"] == "replace_line"
    assert detail["repo_identity_available"] is False
    assert detail["cursor_on_target_row"] is True


def test_outside_region_change_rejected() -> None:
    row = _row("y = 1\n", "y = 2\n")
    row["output"] = row["output"].replace("x = 1", "x = 2")
    assert CLASSIFY(row)[0] == "outside_region_changed"


def test_multiple_line_edits_rejected() -> None:
    assert CLASSIFY(_row("y = 1\nz = 1\n", "y = 2\nz = 2\n"))[0] == "not_one_line_action"
