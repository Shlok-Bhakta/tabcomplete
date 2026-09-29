from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_one_line.py"
SPEC = importlib.util.spec_from_file_location("train_one_line_pilot_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def _row(index: int) -> dict[str, object]:
    source = f"const value = {index};\n"
    return {
        "id": f"instinct-{index}",
        "split": "train",
        "source_type": "continue_instinct_observed",
        "source_license": "Apache-2.0",
        "state": {
            "file_id": f"core/example-{index}.ts",
            "filetype": "typescript",
            "source": source,
            "target_row": 0,
            "cursor_col": 0,
            "history": [],
            "relevant": [],
        },
        "action": {"kind": "replace_line", "text": f"const value = {index + 1};"},
        "after_source": f"const value = {index + 1};\n",
        "validation": {"replay_verified": True, "inferability_reviewed": False},
    }


def test_pilot_phase_accepts_replayed_rows_without_weakening_main_gate(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [_row(index) for index in range(128)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = trainer.sha256_file(path)
    loaded = trainer.load_training_rows(path, digest, phase="pilot", minimum_main_train=20_000)
    assert len(loaded) == 128
    assert trainer.peak_learning_rate("pilot", None) == 1e-5
    with pytest.raises(ValueError, match="inferability-reviewed"):
        trainer.load_training_rows(path, digest, phase="main", minimum_main_train=20_000)


def test_pilot_rejects_unverified_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = _row(0)
    row["validation"] = {"replay_verified": False}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="replay-verified"):
        trainer.load_training_rows(
            path, trainer.sha256_file(path), phase="pilot", minimum_main_train=20_000
        )
