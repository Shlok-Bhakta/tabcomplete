from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_one_line_gpu_pilot.py"
SPEC = importlib.util.spec_from_file_location("evaluate_one_line_gpu_pilot_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_development_input_requires_exact_replay_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    row = {
        "id": "dev-1",
        "split": "development",
        "state": {
            "file_id": "core/example.ts",
            "filetype": "typescript",
            "source": "const n = 1;\n",
            "target_row": 0,
            "cursor_col": 0,
        },
        "action": {"kind": "replace_line", "text": "const n = 2;"},
        "after_source": "const n = 2;\n",
    }
    path.write_text(json.dumps(row) + "\n")
    digest = evaluator.sha256_file(path)
    assert evaluator.load_rows(path, digest) == [row]
    with pytest.raises(ValueError, match="SHA-256"):
        evaluator.load_rows(path, "0" * 64)
    row["after_source"] = "const n = 3;\n"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="does not replay"):
        evaluator.load_rows(path, evaluator.sha256_file(path))
