from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_small_model.py"
SPEC = importlib.util.spec_from_file_location("selected_replay_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_replay_rejects_changed_prompt_even_with_matching_file_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    row = {"id": "one", "prompt": "Unicode λ", "state_sha256": replay.sha("Unicode λ".encode()),
           "operation": "append", "file_id": "a.py", "source_target": 512}
    path.write_text(json.dumps(row) + "\n")
    assert replay.load_frozen_states(path, replay.file_sha(path)) == [row]
    row["prompt"] = "different"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="prompt hash"):
        replay.load_frozen_states(path, replay.file_sha(path))


def test_replay_rejects_fixture_changes_and_duplicate_states(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    row = {"id": "one", "prompt": "N", "state_sha256": replay.sha(b"N"),
           "operation": "no_edit", "file_id": "a.py", "source_target": 512}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        replay.load_frozen_states(path, "0" * 64)
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        replay.load_frozen_states(path, replay.file_sha(path))
