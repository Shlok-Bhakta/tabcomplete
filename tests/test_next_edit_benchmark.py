"""Next-edit benchmark contract and functional scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinycomplete.eval.code_benchmark import Prediction
from tinycomplete.eval.next_edit_benchmark import (
    NextEditCase,
    build_next_edit_prompt,
    evaluate_next_edit_prediction,
    summarize_next_edit_results,
)


def _case(**overrides) -> NextEditCase:
    current = "def add(a, b):\n    return a - b\n"
    region = "return a - b"
    start = len(b"def add(a, b):\n    ")
    values = {
        "id": "python/fix-add",
        "language": "python",
        "path": "solution.py",
        "current": current,
        "region_start": start,
        "region_end": start + len(region.encode()),
        "expected": "return a + b",
        "action": "replace",
        "recent_edits": ["- replace helper name: sum_two -> add"],
        "context_files": {"README.md": "add returns the sum"},
        "check": {
            "compile": ["python3", "-m", "py_compile", "solution.py"],
            "test": ["python3", "tests.py"],
            "files": {"tests.py": "from solution import add\nassert add(2, 3) == 5\n"},
        },
    }
    values.update(overrides)
    return NextEditCase.model_validate(values)


def test_prompt_marks_only_current_region_and_never_leaks_gold_or_tests():
    case = _case(expected="return a + b  # SECRET_GOLD")
    prompt = build_next_edit_prompt(case)
    assert "SECRET_GOLD" not in prompt
    assert "from solution import add" not in prompt
    assert "[[EDIT]]return a - b[[/EDIT]]" in prompt
    assert prompt.endswith(f"<P {case.path} {case.region_start}>\n")
    assert prompt.index("README.md") < prompt.index(f"<file {case.path}>")


def test_replacement_is_applied_and_scored_functionally(tmp_path: Path):
    case = _case()
    result = evaluate_next_edit_prediction(
        case,
        Prediction(case_id=case.id, completion="return a + b"),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.action_correct is True
    assert result.exact_match is True
    assert result.character_edit_distance == 0
    assert result.valid_patch is True
    assert result.compile.status == "pass"
    assert result.test.status == "pass"


def test_noop_is_first_class_and_keeps_current_file(tmp_path: Path):
    case = _case(
        id="python/noop-add",
        current="def add(a, b):\n    return a + b\n",
        expected="",
        action="noop",
    )
    result = evaluate_next_edit_prediction(
        case,
        Prediction(case_id=case.id, completion=""),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.predicted_action == "noop"
    assert result.action_correct is True
    assert result.test.status == "pass"
    summary = summarize_next_edit_results([result])
    assert summary["action_accuracy"] == 1.0
    assert summary["noop_precision"] == 1.0
    assert summary["noop_recall"] == 1.0
    assert summary["noop_f1"] == 1.0
    assert summary["valid_patch_rate"] == 1.0
    assert summary["by_gold_action"]["noop"]["test_pass_rate"] == 1.0


def test_case_rejects_invalid_byte_regions_and_action_contract():
    with pytest.raises(ValueError, match="outside current file"):
        _case(region_end=10_000)
    with pytest.raises(ValueError, match="empty expected"):
        _case(action="noop")
    with pytest.raises(ValueError, match="non-empty"):
        _case(expected="")
    with pytest.raises(ValueError, match="target path"):
        _case(context_files={"solution.py": "leak"})
    with pytest.raises(ValueError, match="collide"):
        _case(context_files={"tests.py": "leak"})


def test_post_build_run_command_is_executed(tmp_path: Path):
    case = _case(
        check={
            "test": [
                "python3",
                "-c",
                "from pathlib import Path; Path('built.py').write_text('raise SystemExit(7)')",
            ],
            "run": ["python3", "built.py"],
        }
    )
    result = evaluate_next_edit_prediction(
        case,
        Prediction(case_id=case.id, completion=case.expected),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.test.status == "fail"
    assert result.test.returncode == 7
    assert result.valid_patch is False


def test_expected_stdout_is_a_behavioral_gate(tmp_path: Path):
    case = _case(
        check={
            "test": ["python3", "-c", "print('wrong')"],
            "expected_stdout": "right\n",
        }
    )
    result = evaluate_next_edit_prediction(
        case,
        Prediction(case_id=case.id, completion=case.expected),
        work_root=tmp_path,
        execution_backend="trusted-host",
    )
    assert result.test.status == "fail"
    assert result.test.stdout == "wrong\n"
    assert result.test.stderr == "stdout did not match expected output"
