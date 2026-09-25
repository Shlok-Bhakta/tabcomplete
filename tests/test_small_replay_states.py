import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_small_model import sha, source_states  # noqa: E402


def test_frozen_interactive_states_cover_editor_transitions() -> None:
    states = source_states()
    assert len(states) == 12
    assert Counter(row["source_target"] for row in states) == {512: 4, 1024: 4, 2048: 4}
    assert {row["operation"] for row in states} >= {
        "fresh_open", "append_chars", "replace_near_cursor",
        "reject_then_type_different", "edit_earlier_in_file",
        "switch_to_other_file", "return_to_first_file",
    }
    assert [row["file_id"] for row in states[4:7]] == [
        "src/example.py", "tests/other.py", "src/example.py"
    ]
    for row in states:
        assert sha(row["prompt"].encode()) == row["state_sha256"]
    assert "return value + 2" in states[4]["prompt"]
    assert "return value + 2" in states[6]["prompt"]
    assert states[5]["prompt"].startswith("# file: tests/other.py\n")
    assert states[6]["prompt"].startswith("# file: src/example.py\n")
